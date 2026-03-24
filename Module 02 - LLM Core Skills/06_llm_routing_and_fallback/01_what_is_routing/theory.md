# 01 — What Is LLM Routing?

> *Routing is the infrastructure layer that decides which model handles which request — and what to do when that model fails.*

---

## 1.1 The Problem Without Routing

Without a router, you hardcode one model everywhere:

```python
# ❌ Naive — one model for everything
response = openai.chat.completions.create(
    model="gpt-4o",   # Same model regardless of task complexity
    messages=[...]
)
# Problems:
# - Expensive: $5/1M tokens for a task gpt-4o-mini handles fine
# - Fragile: If OpenAI is down → your entire app is down
# - Rigid: Can't adapt to traffic spikes, budget limits, or new models
```

With a router:

```python
# ✅ With routing
response = router.complete(messages=[...])
# Router decides:
# - This is a simple classification → use gpt-4o-mini (cheap)
# - This is complex reasoning → use gpt-4o (quality)
# - Rate limit hit → fallback to claude-3-5-haiku
# - Both providers down → return cached response or graceful error
```

---

## 1.2 Routing Taxonomy

```
LLM Routing Strategies
├── Static Routing          — Always send task X to model Y
├── Rule-Based Routing      — If token_count > 1000: use model A else model B
├── Semantic Routing        — Classify intent by embedding → route to specialized model
├── Cost-Based Routing      — Stay within per-call budget ceiling
├── Latency-Based Routing   — Choose fastest available model
└── Load Balancing          — Distribute across multiple providers to avoid rate limits

Fallback Strategies
├── Linear Fallback         — Try A → if fails → B → if fails → C
├── Parallel Fallback       — Fire A and B simultaneously, use whichever responds first
├── Circuit Breaker         — Stop calling a failing provider temporarily
└── Cache Fallback          — Return cached response if all live calls fail
```

---

## 1.3 Why Routing Matters at Each Scale

### Development / Prototype
- **Route by cost**: use mini model during development to save budget
- **Single fallback**: if your one model is down, return a helpful error

### Production (1k-100k calls/day)
- **Route by task type**: simple tasks → cheap model, complex → flagship
- **Multi-provider fallback**: OpenAI down → Anthropic backup
- **Rate limit routing**: spread load across API keys or providers

### High Scale (100k+ calls/day)
- **Load balancing**: distribute across multiple API keys and providers
- **Latency routing**: choose the fastest responding model in real-time
- **Circuit breaker**: detect provider degradation, redirect traffic automatically

---

## 1.4 The Router Contract

Every router should fulfill this contract:

```python
class LLMRouter:
    def complete(
        self,
        messages: list[dict],
        task_hint: str = None,        # Optional signal about task type
        max_cost_usd: float = None,   # Optional budget ceiling
        max_latency_ms: int = None,   # Optional latency ceiling
        **kwargs
    ) -> RouterResponse:
        """
        Returns a normalized response regardless of which model was used.
        Handles routing, retry, and fallback internally.
        Caller never needs to know which model ran.
        """
        ...

@dataclass
class RouterResponse:
    content: str           # The LLM's response text
    model_used: str        # Which model actually handled it (for logging)
    provider: str          # Which provider was used
    input_tokens: int      # Tokens consumed (for cost tracking)
    output_tokens: int
    latency_ms: float
    fallback_used: bool    # True if primary model failed
    attempt_count: int     # How many tries it took
```

---

## 1.5 Routing vs Fallback vs Load Balancing

These three concepts are related but distinct:

| Concept | Purpose | Trigger |
|---|---|---|
| **Routing** | Proactive — send right task to right model | Every request |
| **Fallback** | Reactive — recover from failure | Error / rate limit |
| **Load balancing** | Preventive — spread load before hitting limits | High volume |

They work together in a production router:
1. **Router** selects the model based on task, cost, latency rules
2. **Load balancer** picks which API key/provider instance to use
3. **Fallback** kicks in when the chosen model/provider fails

---

## 1.6 Real Failure Scenarios You Must Handle

```python
# Failures your router must handle:
FAILURE_TYPES = {
    "rate_limit":       "HTTP 429 — too many requests; retry after backoff or switch model",
    "server_error":     "HTTP 500/503 — provider down; switch to fallback provider",
    "timeout":          "Request took too long; retry or use faster model",
    "context_length":   "Input too long for model; switch to larger-context model",
    "content_filter":   "Content blocked; try different model with lighter filter",
    "invalid_response": "JSON parse failure or empty content; retry or fallback",
    "quota_exceeded":   "Monthly quota hit; switch to secondary API key or provider",
}
```

---

## 📌 Key Takeaways

1. **Routing = proactive model selection** based on task, cost, and latency
2. **Fallback = reactive recovery** when the selected model fails
3. **Load balancing = preventive distribution** to avoid rate limits
4. **A router abstracts all three** — callers never write retry/fallback logic themselves
5. **Never hardcode one model** in production — even the best APIs have outages
6. **The router contract**: same input/output shape regardless of which model ran
7. **Start simple**: linear fallback chain + rule-based routing covers 80% of cases
