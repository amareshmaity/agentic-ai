# 03 — Cost and Latency Analysis

> *In production, the best model is the cheapest one that meets your quality bar — not the most capable one.*

---

## 3.1 The Real Cost — More Than Just Per-Token Pricing

Token pricing is only part of total cost:

```
Total Cost of Ownership (TCO) =
    Input token cost
  + Output token cost
  + Rate limit overage / premium tier cost
  + Retry cost (failed calls × retry multiplier)
  + Embedding cost (if using RAG context)
  + Latency cost (latency × engineer time × SLA risk)
  + Vendor lock-in cost (migration effort if provider changes pricing)
```

---

## 3.2 Token Pricing Reference (Early 2025)

| Model | Input (per 1M) | Output (per 1M) | Ratio to GPT-4o |
|---|---|---|---|
| `gemini-1.5-flash` | $0.075 | $0.30 | 0.02× |
| `gpt-4o-mini` | $0.150 | $0.60 | 0.03× |
| `gemini-2.0-flash` | $0.100 | $0.40 | 0.02× |
| `claude-3-5-haiku` | $0.80 | $4.00 | 0.16× |
| `deepseek-v3` | $0.27 | $1.10 | 0.05× |
| `llama-3.1-70b` (API) | $0.52 | $0.75 | 0.10× |
| `claude-3-5-sonnet` | $3.00 | $15.00 | 0.60× |
| `gemini-1.5-pro` | $3.50 | $10.50 | 0.70× |
| `gpt-4o` | $5.00 | $15.00 | 1.00× |
| `o1` | $15.00 | $60.00 | 3.00× |

**Key insight**: GPT-4o-mini costs ~33× less than GPT-4o with ~80-85% quality.

---

## 3.3 Total Cost at Scale — Real Production Math

```python
# Monthly cost estimate for a typical RAG chatbot
SCENARIOS = {
    "Low volume (1k users)": {
        "daily_conversations": 500,
        "avg_turns": 5,
        "avg_input_tokens": 3000,   # System + history + user
        "avg_output_tokens": 400,
    },
    "Medium volume (10k users)": {
        "daily_conversations": 5_000,
        "avg_turns": 5,
        "avg_input_tokens": 3000,
        "avg_output_tokens": 400,
    },
    "High volume (100k users)": {
        "daily_conversations": 50_000,
        "avg_turns": 5,
        "avg_input_tokens": 3000,
        "avg_output_tokens": 400,
    },
}

MODELS = {
    "gpt-4o-mini":        {"input": 0.150e-6, "output": 0.600e-6},
    "claude-3-5-haiku":   {"input": 0.800e-6, "output": 4.000e-6},
    "gpt-4o":             {"input": 5.000e-6, "output": 15.000e-6},
    "claude-3-5-sonnet":  {"input": 3.000e-6, "output": 15.000e-6},
}

for scenario_name, s in SCENARIOS.items():
    monthly_calls = s["daily_conversations"] * s["avg_turns"] * 30
    print(f"\n{scenario_name}: {monthly_calls:,} API calls/month")
    for model, pricing in MODELS.items():
        monthly_cost = monthly_calls * (
            s["avg_input_tokens"] * pricing["input"] +
            s["avg_output_tokens"] * pricing["output"]
        )
        print(f"  {model:<22} ${monthly_cost:>8,.0f}/month")
```

---

## 3.4 Latency Metrics — What to Measure

```python
from dataclasses import dataclass

@dataclass
class LatencyMetrics:
    ttft_ms:        float   # Time to first token — critical for streaming UX
    total_ms:       float   # Total end-to-end latency
    tokens_out:     int     # Tokens generated
    throughput_tps: float   # Output tokens per second
    
    @property
    def chars_per_second(self) -> float:
        return self.throughput_tps * 4  # ~4 chars per token

# Published approximate benchmarks (early 2025, may vary by region/load)
LATENCY_BENCHMARKS = {
    "gemini-2.0-flash":  LatencyMetrics(ttft_ms=50,  total_ms=800,  tokens_out=200, throughput_tps=300),
    "claude-3-5-haiku":  LatencyMetrics(ttft_ms=80,  total_ms=1200, tokens_out=200, throughput_tps=180),
    "gpt-4o-mini":       LatencyMetrics(ttft_ms=120, total_ms=1500, tokens_out=200, throughput_tps=150),
    "gpt-4o":            LatencyMetrics(ttft_ms=200, total_ms=2500, tokens_out=200, throughput_tps=100),
    "claude-3-5-sonnet": LatencyMetrics(ttft_ms=300, total_ms=3500, tokens_out=200, throughput_tps=90),
    "o1":                LatencyMetrics(ttft_ms=3000, total_ms=20000, tokens_out=500, throughput_tps=40),
}
```

---

## 3.5 Batch Processing — Dramatic Cost Reduction

OpenAI and Anthropic offer batch APIs at 50% cost reduction for non-real-time tasks:

```python
# OpenAI Batch API — 50% discount, 24-hour completion window
from openai import OpenAI

client = OpenAI()

# Create batch of requests
batch_requests = [
    {
        "custom_id": f"request-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"Summarize: {document}"}],
            "max_tokens": 200
        }
    }
    for i, document in enumerate(["doc1 text...", "doc2 text..."])
]

# Submit batch (returns immediately, processes in background)
# batch_response = client.batches.create(
#     input_file_id=file_id,
#     endpoint="/v1/chat/completions",
#     completion_window="24h"
# )

# Use batch when:
# - Processing thousands of documents
# - No real-time requirement
# - Cost is the primary constraint
# Savings: 50% cost reduction
```

---

## 3.6 Rate Limits — Know Your Ceiling

Rate limits constrain throughput in production. Always plan for them:

```python
# OpenAI rate limits (tier 1 — default new accounts)
RATE_LIMITS = {
    "gpt-4o": {
        "RPM": 500,     # Requests per minute
        "TPM": 30_000,  # Tokens per minute
        "RPD": 10_000,  # Requests per day
    },
    "gpt-4o-mini": {
        "RPM": 500,
        "TPM": 200_000,
        "RPD": 10_000,
    }
}

def max_throughput_per_second(model: str, avg_tokens: int = 1000) -> float:
    """Maximum sustained throughput given rate limits."""
    limits = RATE_LIMITS.get(model, {})
    rpm_limit = limits.get("RPM", 60) / 60   # Requests per second
    tpm_limit = limits.get("TPM", 60_000) / 60 / avg_tokens  # Requests per second by token limit
    return min(rpm_limit, tpm_limit)

for model in RATE_LIMITS:
    rps = max_throughput_per_second(model)
    print(f"{model}: max ~{rps:.1f} requests/sec")
```

---

## 3.7 Caching — Reduce Cost with Prompt Caching

Both OpenAI and Anthropic cache input tokens that repeat across calls:

```python
# Prompt caching reduces cost when:
# - System prompt is long and repeated across many calls
# - A reference document is included in every call

# OpenAI: automatic, 50% cache discount on input tokens
# Anthropic: explicit cache_control markers, 90% cache discount

# Example with Anthropic caching:
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "text",
#                 "text": long_reference_document,
#                 "cache_control": {"type": "ephemeral"}  # Mark for caching
#             },
#             {"type": "text", "text": user_question}
#         ]
#     }
# ]
# 
# First call: full input price
# Subsequent calls (within 5 min): 90% cheaper on cached portion

# Cost impact:
# 10k token system prompt, 1000 calls/day, gpt-4o-mini:
# Without caching: 10,000 × $0.15/1M × 1000 = $1.50/day
# With caching:    10,000 × $0.075/1M × 1000 = $0.75/day (50% savings)
```

---

## 3.8 The Cost–Quality Decision Matrix

```
                  LOW COST REQUIRED    COST FLEXIBLE
                ┌───────────────────┬───────────────────┐
HIGH QUALITY    │ gpt-4o-mini with  │ claude-3-5-sonnet │
REQUIRED        │ smart prompting   │ or gpt-4o         │
                ├───────────────────┼───────────────────┤
QUALITY         │ gemini-flash or   │ gpt-4o-mini       │
FLEXIBLE        │ llama-3.1-8b      │ (overkill)        │
                └───────────────────┴───────────────────┘
```

---

## 📌 Key Takeaways

1. **Mini/flash/haiku models are 10-50× cheaper** for 80-85% quality — use them by default
2. **Output tokens cost more** than input — optimize for short, dense responses
3. **Batch API = 50% off** for non-real-time bulk processing — always use for large jobs
4. **Prompt caching** cuts costs whenever you repeat a long system prompt or reference doc
5. **Rate limits constrain throughput** — design around them, not just around cost
6. **Calculate monthly cost before choosing** — $500/month vs $15,000/month matters
7. **Start with mini/flash, upgrade only when measured quality gap justifies cost**
