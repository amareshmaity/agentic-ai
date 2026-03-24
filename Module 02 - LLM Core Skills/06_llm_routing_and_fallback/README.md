# 🔀 LLM Routing and Fallback

> *Never let a single model be your single point of failure — route intelligently, fall back gracefully.*

---

## 📌 Why Routing and Fallback Matter

Production AI systems encounter:
- **Model outages** — OpenAI or Anthropic APIs go down (yes, it happens)
- **Rate limit exhaustion** — 429 Too Many Requests during traffic spikes
- **Budget overruns** — one expensive model handling tasks a cheap model could do
- **Latency spikes** — flagship model too slow for real-time use cases
- **Quality variation** — different tasks need different models

Routing and fallback solve all of these systematically.

---

## 📂 Folder Structure

```
06_llm_routing_and_fallback/
│
├── README.md                                   ← You are here
│
├── 01_what_is_routing/
│   ├── theory.md                               ← Routing concepts, taxonomy, why it matters
│   └── examples.ipynb                          ← Simple router demo, routing decisions
│
├── 02_rule_based_routing/
│   ├── theory.md                               ← Keyword, token-count, cost-ceiling rules
│   └── examples.ipynb                          ← Rule engine implementation
│
├── 03_semantic_routing/
│   ├── theory.md                               ← Embedding-based intent routing
│   └── examples.ipynb                          ← Classifier-based model routing
│
├── 04_fallback_strategies/
│   ├── theory.md                               ← Retry, exponential backoff, circuit breaker
│   └── examples.ipynb                          ← Fallback chain with error handling
│
├── 05_load_balancing/
│   ├── theory.md                               ← Round-robin, weighted, least-latency
│   └── examples.ipynb                          ← Load balancer across multiple providers
│
├── 06_litellm_router/
│   ├── theory.md                               ← LiteLLM Router — built-in routing+fallback
│   └── examples.ipynb                          ← LiteLLM Router configuration and usage
│
└── 07_production_router/
    ├── theory.md                               ← Full production router combining all strategies
    └── examples.ipynb                          ← ProductionRouter class with monitoring
```

---

## 📚 Topics Covered

| # | Topic | Core Question Answered |
|---|---|---|
| 1 | `01_what_is_routing` | What is LLM routing and why does it matter? |
| 2 | `02_rule_based_routing` | How do I route based on simple rules? |
| 3 | `03_semantic_routing` | How do I route based on what the user *means*? |
| 4 | `04_fallback_strategies` | What do I do when a model call fails? |
| 5 | `05_load_balancing` | How do I spread load across multiple providers? |
| 6 | `06_litellm_router` | How do I use LiteLLM's built-in routing? |
| 7 | `07_production_router` | How do I build a production-grade router? |

---

## 🔑 Core Architecture

```
User Request
    │
    ▼
┌────────────────────────────────────┐
│           Router                   │
│                                    │
│  1. Classify request               │
│  2. Apply routing rules            │
│  3. Select model + provider        │
│  4. Execute with retry/fallback    │
│  5. Return normalized response     │
└────────────────────────────────────┘
    │               │
    ▼               ▼
Primary Model   Fallback Chain
(fast, cheap)   [model2 → model3 → error]
```

---

## 🔧 Setup

```bash
pip install openai anthropic litellm tiktoken tenacity pydantic python-dotenv rich
```

```env
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key
```
