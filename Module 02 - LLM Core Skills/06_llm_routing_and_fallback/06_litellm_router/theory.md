# 06 — LiteLLM Router

> *One interface for every model — routing, fallback, and load balancing built in.*

---

## 6.1 What Is LiteLLM?

LiteLLM is a Python library that provides:
- **Unified API** — call OpenAI, Anthropic, Google, Cohere, and 100+ models with the same interface
- **Built-in Router** — load balancing, fallback, and retries with zero boilerplate
- **Cost tracking** — per-model cost computation across providers
- **Proxy server** — self-hosted OpenAI-compatible API gateway

```bash
pip install litellm
```

---

## 6.2 Unified Completion API

```python
import litellm

# OpenAI
response = litellm.completion(model="gpt-4o-mini", messages=[...])

# Anthropic — same interface
response = litellm.completion(model="claude-3-5-sonnet-20241022", messages=[...])

# Google Gemini — same interface  
response = litellm.completion(model="gemini/gemini-1.5-flash", messages=[...])

# Groq (hosted Llama)
response = litellm.completion(model="groq/llama3-8b-8192", messages=[...])

# Access response the same way regardless of provider
print(response.choices[0].message.content)
print(response.usage.total_tokens)
```

LiteLLM normalizes the response format — your code doesn't need to change per provider.

---

## 6.3 LiteLLM Router — Core Configuration

The `Router` class manages a **model list** with deployment configs:

```python
from litellm import Router

router = Router(
    model_list=[
        # Primary: GPT-4o
        {
            "model_name": "smart",            # Alias used by caller
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        },
        # Same alias, different provider → automatic fallback
        {
            "model_name": "smart",
            "litellm_params": {
                "model": "claude-3-5-sonnet-20241022",
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            }
        },
        # Budget alias
        {
            "model_name": "fast",
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        },
    ],
    # Routing strategy
    routing_strategy="least-busy",   # or "simple-shuffle", "latency-based-routing"
    num_retries=3,
    retry_after=5,         # Wait 5s between retries
    fallback_models=["fast"],        # Fallback group if "smart" fails
)

# Usage — same as litellm.completion()
response = router.completion(model="smart", messages=[...])
```

---

## 6.4 Routing Strategies

LiteLLM Router supports multiple built-in strategies:

```python
# 1. Simple shuffle (default) — random selection among deployments
router = Router(model_list=[...], routing_strategy="simple-shuffle")

# 2. Least busy — route to deployment with fewest in-flight requests
router = Router(model_list=[...], routing_strategy="least-busy")

# 3. Latency-based — route to deployment with lowest recent latency
router = Router(model_list=[...], routing_strategy="latency-based-routing")

# 4. Usage-based (requires Redis) — route based on TPM/RPM usage
router = Router(
    model_list=[...],
    routing_strategy="usage-based-routing",
    redis_host="localhost",
    redis_port=6379
)
```

---

## 6.5 Multi-Key Load Balancing

Configure multiple API keys as separate deployments under the same alias:

```python
router = Router(
    model_list=[
        # Key 1 — 3,500 RPM limit
        {
            "model_name": "gpt4mini",
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_KEY_1"),
                "rpm": 3500,   # Tell router the limit
                "tpm": 200000,
            }
        },
        # Key 2 — 3,500 RPM limit
        {
            "model_name": "gpt4mini",
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_KEY_2"),
                "rpm": 3500,
                "tpm": 200000,
            }
        },
        # Key 3 — 3,500 RPM limit
        {
            "model_name": "gpt4mini",
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_KEY_3"),
                "rpm": 3500,
                "tpm": 200000,
            }
        },
    ],
    routing_strategy="usage-based-routing",  # Respects rpm/tpm limits
)

# Now your effective limit is 3 × 3,500 = 10,500 RPM
response = router.completion(model="gpt4mini", messages=[...])
```

---

## 6.6 Fallbacks

Configure what model to try if the primary fails:

```python
router = Router(
    model_list=[
        {"model_name": "primary",  "litellm_params": {"model": "gpt-4o",                "api_key": "..."}},
        {"model_name": "fallback1","litellm_params": {"model": "claude-3-5-sonnet-20241022", "api_key": "..."}},
        {"model_name": "fallback2","litellm_params": {"model": "gpt-4o-mini",            "api_key": "..."}},
    ],
    fallbacks=[
        {"primary": ["fallback1", "fallback2"]}   # Try these if primary fails
    ],
    context_window_fallbacks=[
        {"primary": ["fallback1"]}                # If context exceeded, use fallback1
    ],
    num_retries=2,
    timeout=30,
)
```

---

## 6.7 Cost Tracking

```python
import litellm

# Enable cost tracking
litellm.success_callback = ["langfuse"]   # Or custom callback
litellm.set_verbose = False

# After each call, get cost
response = litellm.completion(model="gpt-4o-mini", messages=[...])
cost = litellm.completion_cost(completion_response=response)
print(f"Cost: ${cost:.6f}")

# Total spend tracking
from litellm import get_max_budget
litellm.max_budget = 1.00   # $1 budget limit
litellm.budget_duration = "1d"  # Reset daily
```

---

## 6.8 LiteLLM Proxy — OpenAI-Compatible Gateway

Run LiteLLM as a self-hosted proxy that any OpenAI SDK can point to:

```bash
# Install
pip install 'litellm[proxy]'

# Config file: litellm_config.yaml
model_list:
  - model_name: gpt-4o-alias
    litellm_params:
      model: gpt-4o
      api_key: sk-...
  - model_name: claude-alias
    litellm_params:
      model: claude-3-5-sonnet-20241022
      api_key: sk-ant-...

router_settings:
  routing_strategy: least-busy
  num_retries: 3

# Start proxy
litellm --config litellm_config.yaml --port 4000
```

```python
# Point any OpenAI SDK to your proxy
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",
    api_key="any-string-works-here"
)

response = client.chat.completions.create(
    model="gpt-4o-alias",   # Maps to your configured deployment
    messages=[{"role": "user", "content": "Hello, proxy!"}]
)
```

---

## 6.9 When to Use LiteLLM vs. Custom Router

| Use LiteLLM when | Build custom router when |
|---|---|
| Need multi-provider support fast | Routing logic is highly specific to your domain |
| Want built-in cost tracking | Need semantic routing or ML-based routing |
| Team wants OpenAI SDK compatibility | You need full control over retry behavior |
| Running an LLM gateway for a team | Embedding LiteLLM is too heavy for your use case |
| Prototyping quickly | You need minimal dependencies |

---

## 📌 Key Takeaways

1. **LiteLLM unifies 100+ models** under a single `.completion()` interface
2. **Router model_list**: each entry is a deployment (model + key + limits)
3. **Same alias, multiple deployments** = automatic load balancing and fallback
4. **Routing strategies**: `simple-shuffle`, `least-busy`, `latency-based-routing`, `usage-based-routing`
5. **`fallbacks` config** = declarative fallback chains, no manual try/except
6. **LiteLLM Proxy** = deploy as an OpenAI-compatible gateway for your whole team
7. **Best for**: rapid multi-provider support, team gateways, cost tracking
