# 07 — The LLM Selection Framework

> *A systematic, repeatable process for choosing the right model — every time.*

---

## 7.1 The Problem with Ad-Hoc Selection

Without a framework, teams pick models based on:
- What they heard about at a conference
- What the founder used at their previous company
- Whatever demo they last saw
- "We should use GPT-4 because it's the best"

This leads to: wrong model for the task, budget overruns, poor scalability, and painful migrations.

A framework makes selection **data-driven, reproducible, and auditable**.

---

## 7.2 The 6-Step Selection Framework

```
Step 1: Define the task profile
Step 2: Apply hard constraints first  
Step 3: Shortlist by task fit
Step 4: Evaluate cost and latency
Step 5: Run proof-of-concept tests
Step 6: Monitor and adapt
```

---

## 7.3 Step 1 — Define the Task Profile

Before looking at any model, define your task clearly:

```python
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class TaskRequirements:
    # Functional requirements
    task_category: Literal['reasoning','coding','extraction','summarization',
                            'classification','qa','chat','creative','multimodal']
    complexity: Literal['simple','moderate','complex']
    
    # Non-functional requirements
    max_latency_ms: Optional[int] = None       # None = no requirement
    max_cost_per_call_usd: Optional[float] = None  # e.g., 0.001 = $0.001/call
    
    # Context requirements
    min_context_tokens: int = 4_096
    requires_multimodal: bool = False           # Vision, audio, etc.
    
    # Operational requirements
    privacy_required: bool = False              # Data must stay on-premises
    high_availability: bool = True             # Need SLA guarantees
    
    # Volume
    calls_per_day: int = 1_000
    
    # Quality bar (0-1)
    min_quality_score: float = 0.80

# Example: customer support chatbot
SUPPORT_BOT_REQUIREMENTS = TaskRequirements(
    task_category="chat",
    complexity="moderate",
    max_latency_ms=3_000,      # 3 seconds max
    max_cost_per_call_usd=0.002,  # $2/1000 calls
    min_context_tokens=16_000,
    calls_per_day=10_000,
    min_quality_score=0.85,
)
```

---

## 7.4 Step 2 — Apply Hard Constraints

Hard constraints eliminate models before any evaluation:

```python
HARD_CONSTRAINTS = [
    # Constraint function: (req, model_spec) → bool (True = passes, False = eliminated)
    
    ("Privacy",       lambda r, m: not r.privacy_required or m.get("on_premise", False)),
    ("Context",       lambda r, m: m["context_k"] * 1000 >= r.min_context_tokens),
    ("Multimodal",    lambda r, m: not r.requires_multimodal or m.get("multimodal", False)),
    ("Cost",          lambda r, m: r.max_cost_per_call_usd is None or 
                                   m["typical_cost_per_call"] <= r.max_cost_per_call_usd),
    ("Latency",       lambda r, m: r.max_latency_ms is None or 
                                   m.get("median_latency_ms", 0) <= r.max_latency_ms),
]

def apply_hard_constraints(requirements: TaskRequirements, model_registry: dict) -> list[str]:
    """Return models that pass ALL hard constraints."""
    passed = []
    for model_name, spec in model_registry.items():
        if all(check(requirements, spec) for _, check in HARD_CONSTRAINTS):
            passed.append(model_name)
    return passed
```

---

## 7.5 Step 3 — Score by Task Fit

After hard filtering, score remaining candidates:

```python
# Task-to-capability weights: how important is each capability for this task?
TASK_WEIGHTS = {
    "reasoning":    {"reasoning": 0.40, "instruction": 0.30, "coding": 0.10, "speed": 0.10, "cost": 0.10},
    "coding":       {"reasoning": 0.20, "instruction": 0.20, "coding": 0.40, "speed": 0.10, "cost": 0.10},
    "extraction":   {"reasoning": 0.10, "instruction": 0.40, "coding": 0.10, "speed": 0.10, "cost": 0.30},
    "summarization":{"reasoning": 0.10, "instruction": 0.30, "coding": 0.00, "speed": 0.10, "cost": 0.50},
    "classification":{"reasoning": 0.10, "instruction": 0.30, "coding": 0.00, "speed": 0.20, "cost": 0.40},
    "chat":         {"reasoning": 0.20, "instruction": 0.30, "coding": 0.10, "speed": 0.30, "cost": 0.10},
}

# Model capability scores (0-100)
MODEL_SCORES = {
    "gpt-4o":            {"reasoning": 88, "instruction": 92, "coding": 85, "speed": 70, "cost": 30},
    "gpt-4o-mini":       {"reasoning": 76, "instruction": 88, "coding": 78, "speed": 82, "cost": 95},
    "claude-3-5-sonnet": {"reasoning": 87, "instruction": 94, "coding": 92, "speed": 65, "cost": 45},
    "claude-3-5-haiku":  {"reasoning": 74, "instruction": 88, "coding": 76, "speed": 88, "cost": 88},
    "gemini-1.5-flash":  {"reasoning": 72, "instruction": 80, "coding": 72, "speed": 90, "cost": 98},
    "llama-3.1-70b":     {"reasoning": 80, "instruction": 84, "coding": 78, "speed": 75, "cost": 80},
}

def score_models(task_category: str, candidates: list[str]) -> list[tuple[str, float]]:
    weights = TASK_WEIGHTS.get(task_category, TASK_WEIGHTS["chat"])
    scores = []
    for model in candidates:
        if model not in MODEL_SCORES:
            continue
        raw = MODEL_SCORES[model]
        score = sum(raw.get(cap, 50) * w for cap, w in weights.items())
        scores.append((model, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)
```

---

## 7.6 Step 4 — Calculate Full Cost

```python
def monthly_cost(model: str, daily_calls: int, avg_input: int = 2000, avg_output: int = 400) -> float:
    pricing = {
        "gpt-4o":            (5.00e-6,  15.00e-6),
        "gpt-4o-mini":       (0.15e-6,   0.60e-6),
        "claude-3-5-sonnet": (3.00e-6,  15.00e-6),
        "claude-3-5-haiku":  (0.80e-6,   4.00e-6),
        "gemini-1.5-flash":  (0.075e-6,  0.30e-6),
        "llama-3.1-70b":     (0.52e-6,   0.75e-6),
    }
    p_in, p_out = pricing.get(model, (1e-6, 3e-6))
    per_call = avg_input * p_in + avg_output * p_out
    return per_call * daily_calls * 30  # monthly
```

---

## 7.7 Step 5 — PoC Evaluation Scorecard

```python
@dataclass
class EvaluationResult:
    model: str
    task_score:  float   # Quality of output: 0-100
    latency_ms:  float   # Actual measured latency
    cost_usd:    float   # Actual cost per call
    pass_rate:   float   # % of test cases passing validation
    notes: str = ""

def run_evaluation(model: str, test_cases: list[dict]) -> EvaluationResult:
    """Run N test cases and return scored EvaluationResult."""
    from openai import OpenAI
    import time
    
    client = OpenAI()
    scores, latencies, costs = [], [], []
    passed = 0
    
    for tc in test_cases:
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": tc["prompt"]}],
            max_tokens=tc.get("max_tokens", 200)
        )
        latency = (time.perf_counter() - t0) * 1000
        output = r.choices[0].message.content
        
        # Score quality
        quality = tc.get("score_fn", lambda x: 70)(output)
        passed += 1 if tc.get("pass_fn", lambda x: True)(output) else 0
        
        latencies.append(latency)
        scores.append(quality)
        # cost: simplified
        costs.append(r.usage.total_tokens * 0.15e-6)
    
    return EvaluationResult(
        model=model,
        task_score=sum(scores)/len(scores),
        latency_ms=sum(latencies)/len(latencies),
        cost_usd=sum(costs)/len(costs),
        pass_rate=passed/len(test_cases)*100
    )
```

---

## 7.8 Step 6 — Monitoring in Production

Never set-and-forget your model selection:

```python
# Metrics to track in production
PRODUCTION_METRICS = {
    "quality_drift":       "Track quality score weekly — models update silently",
    "cost_drift":          "Pricing changes; recalculate monthly",
    "latency_p99":         "Track 99th percentile latency, not just average",
    "error_rate":          "Rate limit hits, timeouts, API errors",
    "new_models":          "New releases every 3-6 months — re-evaluate quarterly",
}

# Re-evaluate trigger conditions
REEVALUATE_WHEN = [
    "Quality score drops > 5% week-over-week",
    "A new model releases scoring > 10% better on your eval suite",
    "Monthly cost exceeds 120% of budget",
    "API error rate > 1%",
    "Provider announces price change",
]
```

---

## 7.9 Complete Selection Scorecard Template

```
╔═══════════════════════════════════════════════════════════════╗
║              LLM Selection Scorecard                         ║
╠══════════════════╦═══════════╦═══════════╦═══════════════════╣
║ Criterion        ║  Weight   ║  Score    ║  Notes           ║
╠══════════════════╬═══════════╬═══════════╬═══════════════════╣
║ Task fit quality ║   30%    ║  __/100   ║                  ║
║ Instruction follow║  25%    ║  __/100   ║                  ║
║ Latency (p50)    ║   20%    ║  __/100   ║  __ ms           ║
║ Cost per call    ║   15%    ║  __/100   ║  $__             ║
║ Reliability/SLA  ║   10%    ║  __/100   ║  99.9% uptime?   ║
╠══════════════════╩═══════════╩═══════════╩═══════════════════╣
║ TOTAL SCORE      ║           ║  __/100   ║                  ║
╠══════════════════════════════════════════════════════════════╣
║ Hard constraints passed?   □ Privacy  □ Context  □ Cost     ║
║ PoC validation pass rate?  ___% (minimum: 85%)              ║
║ Recommended model:          _______________________________  ║
║ Budget option:              _______________________________  ║
║ Re-evaluate date:           _______________________________ ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 📌 The 7-Rule Selection Checklist

1. ✅ Define task profile **before** looking at models
2. ✅ Apply privacy/context/cost hard constraints **first**
3. ✅ Score remaining candidates by **task-specific weights**
4. ✅ Calculate **monthly cost at expected volume**, not just per-call
5. ✅ Run **PoC on your real data**, not generic benchmarks
6. ✅ Set a **re-evaluation date** (3-6 months)
7. ✅ Document your decision for the team (**the scorecard is your audit trail**)
