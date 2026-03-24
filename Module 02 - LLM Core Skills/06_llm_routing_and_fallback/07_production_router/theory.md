# 07 — Production Router

> *Combine routing, fallback, load balancing, and observability into one hardened class.*

---

## 7.1 What Makes a Router "Production-Grade"?

A production router must handle:

| Requirement | Implementation |
|---|---|
| Multi-provider support | Fallback chain across OpenAI, Anthropic, Google |
| Rate limit recovery | Retry with exponential backoff + jitter |
| Provider failure | Circuit breaker + failover |
| Task-based routing | Rule engine → model assignment |
| Load distribution | Multi-key balancing |
| Observability | Structured logs, metrics, cost tracking |
| Graceful degradation | Cache fallback + safe error response |

---

## 7.2 The Full Architecture

```
User Request
    │
    ▼
┌───────────────────────────────────────────────────────┐
│                   ProductionRouter                     │
│                                                       │
│  1. Rule Engine ──→ select model group                │
│        │                                              │
│        ▼                                              │
│  2. Load Balancer ──→ select deployment/API key        │
│        │                                              │
│        ▼                                              │
│  3. Call with retry (exponential backoff)             │
│        │                                              │
│        ├── Success ──→ 6. Log + return                │
│        │                                              │
│        └── Failure ──→ 4. Classify error              │
│                              │                        │
│                    RETRYABLE │ FAILOVER               │
│                         ↙   │    ↘                   │
│                    Retry     │   5. Fallback chain     │
│                              │      (circuit breaker)  │
│                              │         │               │
│                              └─────────→ 7. Cache      │
│                                                       │
│  6. Structured logging (latency, cost, model, error)  │
└───────────────────────────────────────────────────────┘
```

---

## 7.3 The ProductionRouter Class

```python
import os, time, random, json, hashlib, logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable
from collections import defaultdict, deque

from openai import OpenAI, RateLimitError, APIStatusError, APITimeoutError

logger = logging.getLogger(__name__)


# ── Data types ───────────────────────────────────────────────────────────

@dataclass
class RouterResponse:
    content: str
    model_used: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    fallback_used: bool
    attempt_count: int
    cost_usd: float
    from_cache: bool = False
    routing_reason: str = ""


@dataclass
class RoutingRule:
    name: str
    condition: Callable[[dict], bool]
    model_group: str   # Maps to a model group in the config
    priority: int
    reason: str


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 60
    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = field(default=0, init=False)
    last_failure: datetime | None = field(default=None, init=False)

    def can_proceed(self) -> bool:
        if self.state == CircuitState.OPEN:
            if self.last_failure and datetime.now() > self.last_failure + timedelta(seconds=self.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True

    def success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def failure(self):
        self.failure_count += 1
        self.last_failure = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit OPEN: {self.name}")


# ── ProductionRouter ─────────────────────────────────────────────────────

class ProductionRouter:
    """
    Production-grade LLM router with:
    - Rule-based model selection
    - Multi-provider fallback chain
    - Circuit breakers per provider
    - Retry with exponential backoff
    - Request/response caching
    - Structured cost and latency logging
    """

    # Model pricing: (input_price_per_token, output_price_per_token)
    PRICING = {
        "gpt-4o":                    (5.00e-6, 15.00e-6),
        "gpt-4o-mini":               (0.15e-6,  0.60e-6),
        "claude-3-5-sonnet-20241022":(3.00e-6, 15.00e-6),
        "claude-3-5-haiku-20241022": (0.80e-6,  4.00e-6),
        "gemini-1.5-flash":          (0.075e-6, 0.30e-6),
        "gemini-1.5-pro":            (1.25e-6,  5.00e-6),
    }

    def __init__(self, model_groups: dict[str, list[str]], default_group: str = "standard"):
        """
        model_groups: e.g. {
            "premium":  ["gpt-4o", "claude-3-5-sonnet-20241022"],
            "standard": ["gpt-4o-mini", "claude-3-5-haiku-20241022"],
            "fast":     ["gemini-1.5-flash", "gpt-4o-mini"],
        }
        """
        self.model_groups = model_groups
        self.default_group = default_group
        self.rules: list[RoutingRule] = []
        self.breakers: dict[str, CircuitBreaker] = {
            model: CircuitBreaker(name=model)
            for models in model_groups.values()
            for model in models
        }
        self._cache: dict[str, str] = {}
        self._clients: dict[str, OpenAI] = {}
        self._metrics: list[dict] = []   # In-memory metrics (use Prometheus/Datadog in prod)

    def add_rule(self, rule: RoutingRule) -> "ProductionRouter":
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)
        return self

    def _get_client(self, model: str) -> OpenAI:
        """Get (or create) provider client for a model."""
        if "claude" in model:
            # Anthropic via litellm or their SDK (simplified: use openai-compat wrapper)
            # In production: from anthropic import Anthropic; return Anthropic()
            return self._clients.setdefault("openai", OpenAI())
        return self._clients.setdefault("openai", OpenAI())

    def _call_model(self, model: str, messages: list[dict], **kwargs) -> dict:
        """Make a single model call. Returns content + token counts."""
        client = self._get_client(model)
        # Use gpt-4o-mini as stand-in for non-OpenAI models in this demo
        call_model = model if model.startswith("gpt") else "gpt-4o-mini"
        response = client.chat.completions.create(
            model=call_model, messages=messages, max_tokens=kwargs.get("max_tokens", 500)
        )
        return {
            "content": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "model": call_model,
        }

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pi, po = self.PRICING.get(model, (5e-6, 15e-6))
        return pi * input_tokens + po * output_tokens

    def _cache_key(self, messages: list[dict]) -> str:
        return hashlib.md5(json.dumps(messages, sort_keys=True).encode()).hexdigest()

    def _select_model_group(self, context: dict) -> tuple[str, str]:
        """Apply rules to select a model group."""
        for rule in self.rules:
            try:
                if rule.condition(context):
                    return rule.model_group, f"Rule '{rule.name}': {rule.reason}"
            except Exception:
                pass
        return self.default_group, "Default group"

    def complete(self, messages: list[dict], context: dict | None = None, **kwargs) -> RouterResponse:
        """
        Main entry point. Routes the request through the full pipeline.
        """
        start = time.time()
        ctx = context or {}
        cache_key = self._cache_key(messages)

        # 1. Select model group via rules
        group, routing_reason = self._select_model_group(ctx)
        model_chain = self.model_groups.get(group, self.model_groups[self.default_group])

        attempt = 0
        last_error = None

        # 2. Try each model in the chain
        for i, model in enumerate(model_chain):
            breaker = self.breakers[model]

            if not breaker.can_proceed():
                logger.info(f"Skipping {model} — circuit open")
                continue

            # 3. Retry loop for rate limits
            for retry in range(3):
                attempt += 1
                try:
                    result = self._call_model(model, messages, **kwargs)
                    breaker.success()

                    latency = (time.time() - start) * 1000
                    cost = self._compute_cost(model, result["input_tokens"], result["output_tokens"])

                    # Store in cache
                    self._cache[cache_key] = result["content"]

                    # Log metrics
                    self._metrics.append({
                        "ts": datetime.now().isoformat(),
                        "model": model,
                        "latency_ms": round(latency, 1),
                        "cost_usd": cost,
                        "fallback_used": i > 0,
                        "attempts": attempt,
                        "routing_reason": routing_reason,
                    })

                    return RouterResponse(
                        content=result["content"],
                        model_used=model,
                        provider="openai" if "gpt" in model else "anthropic",
                        input_tokens=result["input_tokens"],
                        output_tokens=result["output_tokens"],
                        latency_ms=round(latency, 1),
                        fallback_used=(i > 0),
                        attempt_count=attempt,
                        cost_usd=cost,
                        routing_reason=routing_reason,
                    )

                except RateLimitError:
                    if retry < 2:
                        delay = min(1.0 * (2 ** retry) + random.uniform(0, 0.5), 30)
                        logger.warning(f"{model} rate limited. Retry in {delay:.1f}s")
                        time.sleep(delay)
                    else:
                        breaker.failure()
                        last_error = "rate_limit"
                        break

                except Exception as e:
                    breaker.failure()
                    last_error = str(e)
                    break  # Move to next model

        # 4. Cache fallback
        cached = self._cache.get(cache_key)
        if cached:
            latency = (time.time() - start) * 1000
            return RouterResponse(
                content=cached, model_used="cache", provider="cache",
                input_tokens=0, output_tokens=0, latency_ms=round(latency, 1),
                fallback_used=True, attempt_count=attempt, cost_usd=0,
                from_cache=True, routing_reason=routing_reason
            )

        raise Exception(f"All models failed. Last error: {last_error}")

    def print_metrics_summary(self):
        """Print a summary of all requests processed."""
        if not self._metrics:
            print("No requests yet.")
            return
        total_cost = sum(m["cost_usd"] for m in self._metrics)
        avg_latency = sum(m["latency_ms"] for m in self._metrics) / len(self._metrics)
        fallbacks = sum(1 for m in self._metrics if m["fallback_used"])
        print(f"\n📊 Metrics Summary ({len(self._metrics)} requests)")
        print(f"   Total cost:    ${total_cost:.5f}")
        print(f"   Avg latency:   {avg_latency:.0f}ms")
        print(f"   Fallbacks:     {fallbacks}/{len(self._metrics)}")
        print(f"   Avg attempts:  {sum(m['attempts'] for m in self._metrics) / len(self._metrics):.1f}")
```

---

## 7.4 Wiring Up the Rules

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o-mini")

def build_router() -> ProductionRouter:
    router = ProductionRouter(
        model_groups={
            "premium":  ["gpt-4o", "claude-3-5-sonnet-20241022"],
            "standard": ["gpt-4o-mini", "claude-3-5-haiku-20241022"],
            "fast":     ["gemini-1.5-flash", "gpt-4o-mini"],
        },
        default_group="standard"
    )

    (
        router
        .add_rule(RoutingRule(
            name="long_context",
            condition=lambda ctx: ctx.get("input_tokens", 0) > 64_000,
            model_group="premium",
            priority=1,
            reason="Large input requires high-context model"
        ))
        .add_rule(RoutingRule(
            name="realtime",
            condition=lambda ctx: ctx.get("max_latency_ms", float("inf")) < 1000,
            model_group="fast",
            priority=2,
            reason="Latency budget under 1s"
        ))
        .add_rule(RoutingRule(
            name="high_stakes",
            condition=lambda ctx: ctx.get("task") in {"contract_review", "medical", "legal"},
            model_group="premium",
            priority=3,
            reason="High-stakes task requires flagship model"
        ))
        .add_rule(RoutingRule(
            name="budget",
            condition=lambda ctx: ctx.get("max_cost_usd", float("inf")) < 0.005,
            model_group="standard",
            priority=4,
            reason="Budget ceiling applies"
        ))
    )

    return router
```

---

## 7.5 Production Deployment Checklist

```
✅ Before going to production:

Routing
□ Rule engine configured and tested for your task types
□ Fallback chain covers all providers you have access to
□ Circuit breakers tuned (threshold, recovery timeout)
□ Retry limits and backoff delays configured

Monitoring
□ Latency logged per model (p50, p95, p99)
□ Error rate tracked per provider
□ Cost tracked per model and per endpoint
□ Circuit state changes alerted (PagerDuty, Slack)

Reliability
□ Cache fallback configured for critical paths
□ Graceful error message when all fallbacks fail
□ Load test done to verify actual rate limits

Operations
□ API keys rotated regularly
□ Budget alerts set (avoid surprise bills)
□ Fallback chain tested with injected failures
```

---

## 📌 Key Takeaways

1. **Production router = rules + balancing + fallback + circuit breaker + cache**
2. **Separation of concerns**: routing logic, call logic, and retry logic are separate
3. **Circuit breakers** protect from flooding degraded providers
4. **Structured metrics** are essential — log model, latency, cost, fallback_used per call
5. **Cache as last resort** — for read-heavy use cases, prevents total failure
6. **Test with injected failures** — chaos testing ensures your fallback actually works
7. **LiteLLM Router** handles most of this for you — build custom only when needed
