# 04 — Fallback Strategies

> *A model call will eventually fail. Design your system to handle it gracefully.*

---

## 4.1 Why Fallbacks Are Non-Negotiable

In production:
- **OpenAI rate limits**: HTTP 429 during traffic spikes
- **Provider outages**: HTTP 500/503 (all providers have these)
- **Timeouts**: Request takes longer than your SLA allows
- **Context exceeded**: Input too long for the selected model
- **Budget exhaustion**: Monthly quota hit mid-operation

Without fallbacks, any of these causes **user-visible failures**. With fallbacks, they become invisible recovery events.

---

## 4.2 The Fallback Chain

The simplest fallback pattern — try a list of models in order:

```python
FALLBACK_CHAIN = [
    "gpt-4o",                    # Primary: best quality
    "claude-3-5-sonnet-20241022", # Fallback 1: alternative provider
    "gemini-1.5-pro",             # Fallback 2: third provider
    "gpt-4o-mini",                # Fallback 3: cheap same-provider
]

for model in FALLBACK_CHAIN:
    try:
        response = call_model(model, messages)
        return response   # First success wins
    except Exception as e:
        log(f"Model {model} failed: {e}")
        continue

raise AllModelsFailed("Every model in the fallback chain failed")
```

---

## 4.3 Retry with Exponential Backoff

For **rate limit errors (429)**, the right response is to wait and retry — not to immediately failover:

```python
import time
import random

def retry_with_backoff(
    fn,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_codes: set = {429, 500, 502, 503, 504}
):
    """
    Retry a function with exponential backoff.
    Formula: delay = min(base_delay * 2^attempt + jitter, max_delay)
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            status_code = getattr(e, "status_code", None)

            if attempt == max_retries:
                raise  # Out of retries — propagate

            if status_code and status_code not in retryable_codes:
                raise  # Non-retriable error (e.g. 400 bad request) — fail fast

            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay += random.uniform(0, delay * 0.1)  # ±10% jitter

            print(f"⚠️ Attempt {attempt+1} failed ({status_code}). Retrying in {delay:.1f}s...")
            time.sleep(delay)
```

**Jitter** is important for preventing **retry storms** — if 1000 clients all hit rate limits at once and retry at the same interval, they'll all hit rate limits again simultaneously.

---

## 4.4 Error Classification

Not all errors should be handled the same way:

```python
from enum import Enum

class ErrorClass(Enum):
    RETRYABLE   = "retryable"    # Wait and retry same model
    FAILOVER    = "failover"     # Switch to fallback model
    FATAL       = "fatal"        # Don't retry, surface error

def classify_error(error: Exception) -> ErrorClass:
    code = getattr(error, "status_code", None)
    msg  = str(error).lower()

    # Rate limits → retry same provider after backoff
    if code == 429:
        return ErrorClass.RETRYABLE

    # Provider outage → failover to different provider
    if code in {500, 502, 503, 504}:
        return ErrorClass.FAILOVER

    # Timeout → could be either, try failover
    if "timeout" in msg:
        return ErrorClass.FAILOVER

    # Context length exceeded → switch to larger-context model
    if "context" in msg and "length" in msg:
        return ErrorClass.FAILOVER

    # Content filter hit → failover to model with lighter filter
    if "content" in msg and "filter" in msg:
        return ErrorClass.FAILOVER

    # 400 bad request, 401 auth error → fatal (don't retry)
    if code in {400, 401, 403}:
        return ErrorClass.FATAL

    # Unknown → try a single failover
    return ErrorClass.FAILOVER
```

---

## 4.5 Circuit Breaker Pattern

Exponential backoff handles temporary spikes. But if a provider is **degraded for minutes**, you don't want to keep hitting it — you want to **open the circuit** and reroute all traffic until it recovers:

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class CircuitState(Enum):
    CLOSED   = "closed"    # Normal — requests flow through
    OPEN     = "open"      # Provider failing — requests blocked
    HALF_OPEN = "half_open" # Testing if provider has recovered

@dataclass
class CircuitBreaker:
    provider: str
    failure_threshold: int = 5       # Failures before opening circuit
    recovery_timeout: int = 60       # Seconds before trying again (half-open)
    success_threshold: int = 2       # Successes needed to close (re-enable)

    # State
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: datetime | None = None

    def call_allowed(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if datetime.now() > self.last_failure_time + timedelta(seconds=self.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                return True  # Allow one probe request
            return False
        return True  # HALF_OPEN: allow probe

    def record_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                print(f"✅ Circuit CLOSED for {self.provider} — recovered")

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"🚨 Circuit OPEN for {self.provider} — {self.failure_count} failures")
```

---

## 4.6 Full Fallback Chain with Circuit Breakers

```python
class FallbackRouter:
    """
    Manages a fallback chain with per-provider circuit breakers,
    retry logic, and error classification.
    """

    def __init__(self, fallback_chain: list[str]):
        self.chain = fallback_chain
        self.circuit_breakers = {
            model: CircuitBreaker(provider=model)
            for model in fallback_chain
        }

    def complete(self, messages: list[dict], **kwargs) -> dict:
        last_error = None

        for model in self.chain:
            breaker = self.circuit_breakers[model]

            if not breaker.call_allowed():
                print(f"⚡ Circuit OPEN for {model} — skipping")
                continue

            try:
                result = self._call_with_retry(model, messages, **kwargs)
                breaker.record_success()
                result["model_used"] = model
                result["fallback_used"] = model != self.chain[0]
                return result

            except Exception as e:
                error_class = classify_error(e)
                breaker.record_failure()
                last_error = e
                print(f"❌ {model} failed ({error_class.value}): {e}")

                if error_class == ErrorClass.FATAL:
                    raise  # Don't try fallbacks for fatal errors

        raise Exception(f"All models failed. Last error: {last_error}")

    def _call_with_retry(self, model: str, messages: list[dict], **kwargs) -> dict:
        # Retry logic with exponential backoff
        for attempt in range(3):
            try:
                return call_model(model, messages, **kwargs)
            except Exception as e:
                if classify_error(e) != ErrorClass.RETRYABLE or attempt == 2:
                    raise
                delay = min(1.0 * (2 ** attempt), 30.0)
                time.sleep(delay)
```

---

## 4.7 Cache Fallback — Last Resort

When all live calls fail, return a cached response:

```python
from functools import lru_cache
import hashlib, json

class CachedFallback:
    def __init__(self, main_router):
        self.router = main_router
        self._cache: dict[str, str] = {}

    def _cache_key(self, messages: list[dict]) -> str:
        content = json.dumps(messages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def complete(self, messages: list[dict], **kwargs) -> dict:
        key = self._cache_key(messages)
        try:
            result = self.router.complete(messages, **kwargs)
            self._cache[key] = result["content"]  # Store on success
            return result
        except Exception as e:
            cached = self._cache.get(key)
            if cached:
                return {"content": cached, "from_cache": True, "model_used": "cache"}
            raise  # No cache available — surface the error
```

---

## 4.8 Fallback Strategy Decision Matrix

```
Error Type              | First Action    | If That Fails
──────────────────────────────────────────────────────────
429 Rate Limit          | Retry + backoff | Switch provider
500/502/503 Outage      | Switch provider | Try third provider
Timeout                 | Switch to fast  | Use cached response
Context Too Long        | Larger model    | Compress + retry
Content Filter          | Alt. provider   | Return safe error
401 Auth Error          | FATAL — fix key | (don't retry)
400 Bad Request         | FATAL — fix req | (don't retry)
All providers down      | Cached response | Return graceful error
```

---

## 📌 Key Takeaways

1. **Classify errors first**: retryable vs. failover vs. fatal — handle differently
2. **Exponential backoff + jitter** for rate limits — prevents retry storms
3. **Linear fallback chain**: primary → fallback1 → fallback2 → default
4. **Circuit breaker**: stop hammering degraded providers; let them recover
5. **Cache fallback**: last resort when all live calls fail
6. **Never silence errors completely**: always log what failed and why
7. **Test your fallbacks**: regularly inject failures to verify recovery works
