# 02 — Rule-Based Routing

> *The simplest router is often the best — check conditions in order and send to the right model.*

---

## 2.1 What Is Rule-Based Routing?

Rule-based routing makes model selection decisions using explicit, hand-crafted conditions:

```python
def route(request: str) -> str:
    if token_count(request) > 50_000:
        return "gemini-1.5-pro"       # Only model with enough context
    elif is_code_task(request):
        return "claude-3-5-sonnet"    # Best coder
    elif is_simple_task(request):
        return "gpt-4o-mini"          # Cheap for simple tasks
    else:
        return "gpt-4o"               # Default quality model
```

✅ **Advantages**: predictable, debuggable, no added latency, no extra API calls  
❌ **Limitations**: brittle (rules need manual updates), can't handle novel patterns

---

## 2.2 Rule Types

### Rule Type 1: Token Count / Context Length

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")

def route_by_context(messages: list[dict]) -> str:
    """Route based on total input token count."""
    total_tokens = sum(len(enc.encode(m.get("content", ""))) for m in messages)
    
    if total_tokens > 800_000:
        return "gemini-1.5-pro"         # 2M context
    elif total_tokens > 100_000:
        return "claude-3-5-sonnet"      # 200k context
    elif total_tokens > 64_000:
        return "gpt-4o"                 # 128k context
    else:
        return "gpt-4o-mini"            # Most tasks fit here
```

### Rule Type 2: Keyword / Pattern Matching

```python
import re

CODE_KEYWORDS = {"python", "javascript", "function", "class", "def ", "bug", "debug", "code", "algorithm", "implement"}
CREATIVE_KEYWORDS = {"write a story", "poem", "creative", "imagine", "fiction", "narrative"}
MATH_KEYWORDS = {"calculate", "equation", "integral", "derivative", "proof", "solve for"}

def route_by_keywords(text: str) -> str:
    """Route by detecting keywords in the user's text."""
    text_lower = text.lower()
    
    if any(kw in text_lower for kw in CODE_KEYWORDS):
        return "claude-3-5-sonnet"   # Best for code
    elif any(kw in text_lower for kw in MATH_KEYWORDS):
        return "o1-mini"             # Best for math/reasoning
    elif any(kw in text_lower for kw in CREATIVE_KEYWORDS):
        return "claude-3-5-sonnet"   # Best for creative writing
    else:
        return "gpt-4o-mini"         # Default cheap model
```

### Rule Type 3: Cost Ceiling

```python
PRICING = {
    "gpt-4o-mini":       (0.15e-6, 0.60e-6),
    "claude-3-5-haiku":  (0.80e-6, 4.00e-6),
    "gpt-4o":            (5.00e-6, 15.00e-6),
    "claude-3-5-sonnet": (3.00e-6, 15.00e-6),
}

def route_by_budget(
    estimated_input_tokens: int,
    expected_output_tokens: int,
    max_cost_usd: float
) -> str:
    """Route to the best model that fits within the cost ceiling."""
    quality_order = ["claude-3-5-sonnet", "gpt-4o", "claude-3-5-haiku", "gpt-4o-mini"]
    
    for model in quality_order:
        pi, po = PRICING[model]
        cost = estimated_input_tokens * pi + expected_output_tokens * po
        if cost <= max_cost_usd:
            return model
    
    return "gpt-4o-mini"  # Cheapest as final fallback
```

### Rule Type 4: Latency Budget

```python
# Approximate median latency per model (from benchmarks)
MODEL_LATENCY_MS = {
    "gemini-2.0-flash":   500,
    "claude-3-5-haiku":   800,
    "gpt-4o-mini":       1200,
    "gpt-4o":            2000,
    "claude-3-5-sonnet": 3000,
    "o1":               20000,
}

def route_by_latency(max_latency_ms: int) -> str:
    """Return fastest model that still meets latency budget."""
    # Sort by latency ascending
    sorted_models = sorted(MODEL_LATENCY_MS.items(), key=lambda x: x[1])
    
    for model, latency in sorted_models:
        if latency <= max_latency_ms:
            return model
    
    return sorted_models[0][0]  # Return fastest if nothing fits
```

---

## 2.3 Rule Engine — Composable Rules

Rather than nested if/else, compose rules as a priority-ordered list:

```python
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class RoutingRule:
    name: str
    condition: Callable[[dict], bool]   # Takes request context, returns True if rule applies
    model: str                           # Which model to route to
    priority: int                        # Lower = higher priority (evaluated first)
    reason: str                          # Human-readable description for logging

class RuleEngine:
    """
    Priority-ordered rule engine for model routing.
    Rules are evaluated in priority order; first match wins.
    """
    
    def __init__(self, default_model: str = "gpt-4o-mini"):
        self.rules: list[RoutingRule] = []
        self.default = default_model
    
    def add_rule(self, rule: RoutingRule):
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)
    
    def route(self, context: dict) -> tuple[str, str]:
        """Returns (model, reason)."""
        for rule in self.rules:
            if rule.condition(context):
                return rule.model, f"Rule '{rule.name}': {rule.reason}"
        return self.default, "No rules matched — using default model"
```

---

## 2.4 Building a Practical Rule Engine

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")

def build_production_rule_engine() -> RuleEngine:
    engine = RuleEngine(default_model="gpt-4o-mini")
    
    # Rule 1: Long context → needs large context model
    engine.add_rule(RoutingRule(
        name="long_context",
        condition=lambda ctx: ctx.get("input_tokens", 0) > 100_000,
        model="claude-3-5-sonnet",
        priority=1,
        reason="Input exceeds 100k tokens — needs 200k context model"
    ))
    
    # Rule 2: Real-time requirement → fast model
    engine.add_rule(RoutingRule(
        name="realtime",
        condition=lambda ctx: ctx.get("max_latency_ms", float("inf")) < 1000,
        model="claude-3-5-haiku",
        priority=2,
        reason="Latency budget < 1s — route to fastest model"
    ))
    
    # Rule 3: Coding task → best coding model
    engine.add_rule(RoutingRule(
        name="coding",
        condition=lambda ctx: any(kw in ctx.get("text", "").lower()
                                  for kw in ["def ", "class ", "function", "debug", "code", "algorithm", "implement"]),
        model="claude-3-5-sonnet",
        priority=3,
        reason="Coding keywords detected — route to best coding model"
    ))
    
    # Rule 4: Budget constraint
    engine.add_rule(RoutingRule(
        name="budget",
        condition=lambda ctx: ctx.get("max_cost_usd", float("inf")) < 0.001,
        model="gpt-4o-mini",
        priority=4,
        reason="Budget ceiling < $0.001/call — use cheapest model"
    ))
    
    return engine
```

---

## 2.5 Rule Priority Design

```
Priority  Rule                    Reason
────────────────────────────────────────────────────────────────
1         Hard constraints        Context limit, privacy — MUST be first
2         Latency requirements    Real-time SLA — critical to honor early
3         Quality requirements    Complex task needing flagship model
4         Cost constraints        Budget ceiling — apply after quality check
5         Task-type rules         Code, math, creative — apply when no harder rule
6         Default                 Fallback when no rule matches
```

**Rule order matters**: put the most critical constraints (context, privacy) first. A cost rule after a quality rule means you'll pay if quality is needed — that's correct.

---

## 📌 Key Takeaways

1. **Rule-based routing** = explicit conditions → deterministic, debuggable
2. **Priority order matters**: hard constraints (context > latency > quality > cost)
3. **RuleEngine class** = composable, extensible; add rules without touching routing logic
4. **4 rule types**: token count, keyword matching, cost ceiling, latency budget
5. **Always log which rule matched** — essential for debugging misrouted requests
6. **Rules should be data-driven**: load rule configs from a YAML/JSON file for easy tuning
