# 05 — Load Balancing

> *Spread the load before you hit the limit — not after.*

---

## 5.1 What Is Load Balancing for LLMs?

Load balancing distributes requests across multiple model instances, API keys, or providers so that:
- **No single key gets rate limited** during traffic spikes
- **Latency improves** by parallelizing across providers
- **Availability increases** — if one key/provider is degraded, others absorb the load

```
              200 req/min
              │
              ▼
     ┌─────────────────┐
     │   Load Balancer │
     └─────────────────┘
       │        │        │
       ▼        ▼        ▼
  Key A (0)  Key B (0)  Key C (0)   ← Round-robin assigns
  → 67/min   → 67/min   → 67/min   ← Below each key's limit
```

---

## 5.2 Round-Robin Balancing

The simplest strategy — cycle through available endpoints in order:

```python
from itertools import cycle

class RoundRobinBalancer:
    def __init__(self, endpoints: list[str]):
        self._cycle = cycle(endpoints)
        self.endpoints = endpoints

    def next(self) -> str:
        return next(self._cycle)
```

✅ Zero config, perfectly even distribution  
❌ Doesn't account for endpoint health or current load

---

## 5.3 Weighted Round-Robin

Give more traffic to higher-capacity endpoints:

```python
class WeightedBalancer:
    """
    Endpoints with higher weight receive proportionally more requests.
    Example: weights=[3, 2, 1] → endpoint A gets 50%, B gets 33%, C gets 17%
    """

    def __init__(self, endpoints: list[str], weights: list[int]):
        assert len(endpoints) == len(weights)
        # Expand endpoints by weight
        self._pool = []
        for endpoint, weight in zip(endpoints, weights):
            self._pool.extend([endpoint] * weight)
        self._cycle = cycle(self._pool)

    def next(self) -> str:
        return next(self._cycle)
```

Use when:
- One API key has higher rate limits than others
- One provider is cheaper and should handle more traffic

---

## 5.4 Least-Latency Balancing

Route to the endpoint with the lowest recent average latency:

```python
from collections import deque
import statistics

class LeastLatencyBalancer:
    """
    Tracks rolling average latency per endpoint.
    Routes to the endpoint with the lowest recent latency.
    """

    def __init__(self, endpoints: list[str], window: int = 10):
        self.endpoints = endpoints
        self.latencies: dict[str, deque] = {
            ep: deque(maxlen=window) for ep in endpoints
        }

    def record_latency(self, endpoint: str, latency_ms: float):
        self.latencies[endpoint].append(latency_ms)

    def next(self) -> str:
        """Return endpoint with lowest average latency."""
        avg_latencies = {}
        for ep in self.endpoints:
            history = self.latencies[ep]
            avg_latencies[ep] = statistics.mean(history) if history else float("inf")
        return min(avg_latencies, key=avg_latencies.get)
```

---

## 5.5 Rate-Limit Aware Balancing

Track requests per endpoint and avoid those near their limit:

```python
import time
from collections import defaultdict

class RateLimitAwareBalancer:
    """
    Tracks requests per minute per endpoint.
    Skips endpoints that are near their rate limit.
    """

    def __init__(self, endpoints: list[str], rpm_limits: dict[str, int]):
        self.endpoints = endpoints
        self.rpm_limits = rpm_limits
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _current_rpm(self, endpoint: str) -> int:
        """Count requests in the last 60 seconds."""
        now = time.time()
        cutoff = now - 60
        self._requests[endpoint] = [t for t in self._requests[endpoint] if t > cutoff]
        return len(self._requests[endpoint])

    def _available_capacity(self, endpoint: str) -> int:
        limit = self.rpm_limits.get(endpoint, 60)
        return max(0, limit - self._current_rpm(endpoint))

    def next(self) -> str | None:
        """Return endpoint with most capacity, or None if all are saturated."""
        available = {
            ep: self._available_capacity(ep)
            for ep in self.endpoints
        }
        best = max(available, key=available.get)
        if available[best] == 0:
            return None  # All saturated
        self._requests[best].append(time.time())
        return best
```

---

## 5.6 Combining Load Balancing with Fallback

The full flow in production:

```python
class LoadBalancedRouter:
    """
    1. Load balancer selects the endpoint with capacity
    2. Makes the call with retry on rate limit
    3. Falls back to alternative if primary is saturated or fails
    """

    def __init__(
        self,
        primary_endpoints: list[str],
        fallback_chain: list[str],
        rpm_limits: dict[str, int]
    ):
        self.balancer = RateLimitAwareBalancer(primary_endpoints, rpm_limits)
        self.fallback_chain = fallback_chain

    def complete(self, messages: list[dict]) -> dict:
        # Try load-balanced endpoint first
        endpoint = self.balancer.next()
        if endpoint:
            try:
                return self._call(endpoint, messages)
            except Exception as e:
                print(f"Primary endpoint {endpoint} failed: {e}")

        # Fallback chain
        for model in self.fallback_chain:
            try:
                result = self._call(model, messages)
                result["fallback_used"] = True
                return result
            except Exception as e:
                print(f"Fallback {model} failed: {e}")

        raise Exception("All endpoints and fallbacks exhausted")

    def _call(self, model: str, messages: list[dict]) -> dict:
        import time
        from openai import OpenAI
        client = OpenAI()
        start = time.time()
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=150)
        return {
            "content": resp.choices[0].message.content,
            "model": model,
            "latency_ms": round((time.time() - start) * 1000, 1),
            "fallback_used": False,
        }
```

---

## 5.7 Multi-Key Setup Pattern

For high-volume production, use multiple API keys per provider:

```python
import os

# In .env:
# OPENAI_API_KEY_1=sk-...
# OPENAI_API_KEY_2=sk-...
# OPENAI_API_KEY_3=sk-...

def load_api_keys(prefix: str) -> list[str]:
    """Load all keys matching prefix from environment."""
    keys = []
    i = 1
    while key := os.getenv(f"{prefix}_{i}"):
        keys.append(key)
        i += 1
    return keys

# Each key is an "endpoint" with its own rate limit
openai_keys = load_api_keys("OPENAI_API_KEY")
# → ["sk-key1", "sk-key2", "sk-key3"]

# Create a client per key
from openai import OpenAI
key_clients = {key: OpenAI(api_key=key) for key in openai_keys}

# Total capacity = num_keys × per_key_rpm_limit
# e.g., 3 keys × 3,500 rpm = 10,500 effective rpm
```

---

## 5.8 Load Balancing Strategy Decision Guide

| Traffic Pattern | Recommended Strategy |
|---|---|
| Low, unpredictable | Round-robin across keys |
| One key has 2× the limit | Weighted (2:1 ratio) |
| Latency is critical | Least-latency |
| Approaching rate limits | Rate-limit-aware |
| Multi-provider + high availability | Load balance + circuit breaker |

---

## 📌 Key Takeaways

1. **Round-robin** = zero config, even distribution — good starting point
2. **Weighted** = use when keys/providers have different capacities
3. **Least-latency** = best for real-time applications needing minimal delay
4. **Rate-limit-aware** = most production-accurate; tracks actual usage per minute
5. **Combine with fallback**: load balancer selects endpoint, fallback handles failures
6. **Multi-key setup**: multiply your effective rate limit without changing providers
7. **Always monitor per-endpoint health**: log latency, error rate, capacity used
