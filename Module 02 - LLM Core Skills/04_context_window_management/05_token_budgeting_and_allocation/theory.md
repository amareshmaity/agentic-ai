# 05 — Token Budgeting and Allocation

> *Treat your context window like RAM — allocate deliberately, track exactly, enforce hard limits.*

---

## 5.1 The Token Budget Problem

A 128k context window sounds enormous, but real agents consume tokens fast:

```python
# Typical gpt-4o-mini agent call budget breakdown
total_context = 128_000  # tokens

# Input budget breakdown:
system_prompt     =   500  # tokens
tool_definitions  =   600  # 5 tools × 120 tokens each
conversation_hist = 8_000  # 40 turns × 200 tokens/turn
current_messages  =   300  # current user query + system additions

# Output budget (reserved):
max_output        = 4_096  # max_tokens setting

# ──────────────────────────────────
total_used = 500 + 600 + 8_000 + 300 + 4_096  # = 13,496 tokens
# 13k / 128k = 10.5% — lots of room!
# But in a 500-step agent run: could easily hit the ceiling
```

Token budgeting = **proactively planning and enforcing** how tokens are distributed.

---

## 5.2 The Five Budget Sections

Every agent call has five categories of token spending:

```python
from dataclasses import dataclass

@dataclass
class ContextBudget:
    context_limit:    int    # Model's maximum context window
    system_budget:    int    # Tokens allocated for system prompt
    tools_budget:     int    # Tokens allocated for tool definitions  
    history_budget:   int    # Tokens allocated for conversation history
    current_budget:   int    # Tokens for the current user message
    output_budget:    int    # Reserved for the model's output (max_tokens)
    
    @property
    def total_input_budget(self) -> int:
        return self.system_budget + self.tools_budget + self.history_budget + self.current_budget
    
    @property
    def safety_margin(self) -> int:
        return self.context_limit - self.total_input_budget - self.output_budget
    
    @property
    def is_valid(self) -> bool:
        return self.safety_margin >= 0 and all(v > 0 for v in [
            self.system_budget, self.tools_budget, self.history_budget, 
            self.current_budget, self.output_budget
        ])

# Example balanced budget for gpt-4o-mini
AGENT_BUDGET = ContextBudget(
    context_limit  = 128_000,
    system_budget  =   2_000,   # 1.6% — system prompt
    tools_budget   =   2_000,   # 1.6% — tool schemas
    history_budget =  50_000,   # 39%  — conversation history
    current_budget =   5_000,   # 3.9% — current user message
    output_budget  =   8_000,   # 6.3% — model output
)
# Safety margin: 61,000 tokens — plenty of headroom
```

---

## 5.3 Token Budget Enforcer

```python
import tiktoken
from openai import OpenAI

client = OpenAI()

class TokenBudgetEnforcer:
    """
    Enforces strict token budgets for each section of the context.
    Raises warnings and truncates sections that exceed their budget.
    """
    
    def __init__(self, budget: ContextBudget, model: str = "gpt-4o-mini"):
        self.budget = budget
        self.model = model
        self.enc = tiktoken.encoding_for_model(model)
    
    def count(self, text: str) -> int:
        return len(self.enc.encode(text))
    
    def count_messages(self, messages: list[dict]) -> int:
        return sum(3 + self.count(str(m.get("content", ""))) for m in messages) + 3
    
    def enforce_system(self, system_prompt: str) -> str:
        """Truncate system prompt to budget."""
        tokens = self.count(system_prompt)
        if tokens <= self.budget.system_budget:
            return system_prompt
        # Truncate at token boundary
        token_ids = self.enc.encode(system_prompt)[:self.budget.system_budget - 50]
        truncated = self.enc.decode(token_ids)
        print(f"⚠️  System prompt truncated: {tokens} → ~{self.budget.system_budget} tokens")
        return truncated + "\n[TRUNCATED]"
    
    def enforce_history(self, history: list[dict]) -> tuple[list[dict], int]:
        """Trim history to fit within history budget."""
        total = self.count_messages(history)
        if total <= self.budget.history_budget:
            return history, total
        
        # Trim from oldest, keep newest
        kept = []
        used = 3
        dropped = 0
        for msg in reversed(history):
            t = 3 + self.count(str(msg.get("content", "")))
            if used + t <= self.budget.history_budget:
                kept.append(msg)
                used += t
            else:
                dropped += 1
        
        if dropped > 0:
            print(f"⚠️  History trimmed: dropped {dropped} oldest messages")
        
        kept.reverse()
        return kept, used
    
    def enforce_tools(self, tools: list[dict]) -> list[dict]:
        """Remove tools that exceed the tools budget."""
        import json
        total = 15  # Base overhead
        kept = []
        for tool in tools:
            t = self.count(json.dumps(tool))
            if total + t <= self.budget.tools_budget:
                kept.append(tool)
                total += t
            else:
                print(f"⚠️  Tool '{tool['function']['name']}' removed (budget exceeded)")
        return kept
    
    def build_enforced_messages(
        self,
        system_prompt: str,
        history: list[dict],
        current_message: str,
        tools: list[dict] = None
    ) -> tuple[list[dict], list[dict]]:
        """Return (enforced_messages, enforced_tools) within budget."""
        
        # Enforce each section
        safe_system = self.enforce_system(system_prompt)
        safe_history, _ = self.enforce_history(history)
        safe_tools = self.enforce_tools(tools or [])
        
        messages = [
            {"role": "system", "content": safe_system},
            *safe_history,
            {"role": "user", "content": current_message}
        ]
        
        return messages, safe_tools
    
    def usage_report(self, messages: list[dict], tools: list[dict] = None) -> dict:
        import json
        msg_tokens = self.count_messages(messages)
        tool_tokens = self.count(json.dumps(tools)) + 15 if tools else 0
        total_input = msg_tokens + tool_tokens
        return {
            "message_tokens": msg_tokens,
            "tool_tokens":    tool_tokens,
            "total_input":    total_input,
            "output_reserved":self.budget.output_budget,
            "total_used":     total_input + self.budget.output_budget,
            "pct_of_limit":   round((total_input + self.budget.output_budget) / self.budget.context_limit * 100, 1),
            "headroom":       self.budget.context_limit - total_input - self.budget.output_budget
        }
```

---

## 5.4 Dynamic Budget Reallocation

When one section needs more tokens, reallocate from others:

```python
def reallocate_budget(
    budget: ContextBudget,
    extra_system_tokens: int = 0,
    extra_history_tokens: int = 0
) -> ContextBudget:
    """
    Dynamically adjust budget when sections need more tokens.
    Borrows from sections with surplus (history is usually most flexible).
    """
    from copy import copy
    new_budget = copy(budget)
    
    # System needs more → borrow from history
    if extra_system_tokens > 0:
        borrow = min(extra_system_tokens, new_budget.history_budget // 4)
        new_budget.system_budget += borrow
        new_budget.history_budget -= borrow
    
    # History needs more → borrow from output reserve
    if extra_history_tokens > 0:
        borrow = min(extra_history_tokens, new_budget.output_budget // 2)
        new_budget.history_budget += borrow
        new_budget.output_budget -= borrow
    
    if not new_budget.is_valid:
        return budget  # Revert if reallocation makes it invalid
    
    return new_budget
```

---

## 5.5 Per-Tool Budget — Removing Low-Priority Tools

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class PrioritizedTool:
    schema: dict         # The tool definition
    priority: int        # 1 (highest) to 5 (lowest)
    description: str     # Human-readable description

def select_tools_by_budget(
    tools: list[PrioritizedTool],
    token_budget: int,
    model: str = "gpt-4o-mini"
) -> list[dict]:
    """
    Select highest-priority tools that fit within token budget.
    Lower priority number = higher priority (1 = must include).
    """
    import json
    enc = tiktoken.encoding_for_model(model)
    
    # Sort by priority (ascending — 1 first)
    sorted_tools = sorted(tools, key=lambda t: t.priority)
    
    selected = []
    used_tokens = 15  # Base overhead
    
    for tool in sorted_tools:
        tool_tokens = len(enc.encode(json.dumps(tool.schema)))
        if used_tokens + tool_tokens <= token_budget:
            selected.append(tool.schema)
            used_tokens += tool_tokens
        else:
            print(f"  ⚠️  Budget full — '{tool.description}' (priority {tool.priority}) excluded")
    
    return selected
```

---

## 5.6 Budget Dashboard — Real-Time Monitoring

```python
def print_budget_dashboard(report: dict, budget: ContextBudget):
    """Visual token budget dashboard."""
    pct = report['pct_of_limit']
    bar_full = 40
    bar_used = int(pct / 100 * bar_full)
    bar = '█' * bar_used + '░' * (bar_full - bar_used)
    
    color = '🟢' if pct < 60 else ('🟡' if pct < 85 else '🔴')
    
    print(f"\n{'─'*55}")
    print(f"  Context Budget Dashboard              {color} {pct}% used")
    print(f"  [{bar}]")
    print(f"{'─'*55}")
    print(f"  Messages:     {report['message_tokens']:>7,} tokens")
    print(f"  Tools:        {report['tool_tokens']:>7,} tokens")
    print(f"  Output Rsv:   {report['output_reserved']:>7,} tokens")
    print(f"  Total Used:   {report['total_used']:>7,} / {budget.context_limit:,}")
    print(f"  Headroom:     {report['headroom']:>7,} tokens")
    print(f"{'─'*55}")
```

---

## 📌 Key Takeaways

1. **Five sections**: system + tools + history + current + output — each needs a budget
2. **TokenBudgetEnforcer**: validates and truncates each section to its allotted tokens
3. **Dynamic reallocation**: borrow from history when system prompt grows
4. **Prioritized tools**: drop lowest-priority tools first when budget is tight
5. **Monitor w/ `usage_report()`**: check `pct_of_limit` and `headroom` after every call
6. **Set budgets before running** — like memory allocation in software engineering  
7. **Start with 60% target utilization** — leave headroom for unexpected long output
