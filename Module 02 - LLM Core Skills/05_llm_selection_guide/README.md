# 🧭 LLM Selection Guide

> *Choosing the wrong model is the most expensive mistake in production AI — here's how to choose right.*

---

## 📌 Why Model Selection Matters

Not all LLMs are equal. The "best" model depends on your task:

```
Coding task?         → GPT-4o, Claude 3.5 Sonnet, or DeepSeek
Cheap bulk ops?      → GPT-4o-mini, Claude 3 Haiku, Gemini Flash
Long document?       → Claude 3.5 (200k), Gemini 1.5 Pro (2M tokens)
Private/on-premise?  → Llama 3.1 70B via Ollama
Multimodal?          → GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet
Fast real-time?      → Groq (70B at 750 tokens/sec), Gemini Flash
```

Picking wrong costs you: money, latency, quality, and reliability.

---

## 📂 Folder Structure

```
05_llm_selection_guide/
│
├── README.md                                   ← You are here
│
├── 01_model_landscape/
│   ├── theory.md                               ← Major LLM families, context windows, capabilities
│   └── examples.ipynb                          ← API connectivity test + capability matrix
│
├── 02_capability_comparison/
│   ├── theory.md                               ← Reasoning, coding, instruction-following, multimodal
│   └── examples.ipynb                          ← Benchmark same prompts across models
│
├── 03_cost_and_latency_analysis/
│   ├── theory.md                               ← Pricing, speed, throughput, total cost of ownership
│   └── examples.ipynb                          ← Cost calculator, latency benchmarker
│
├── 04_task_based_selection/
│   ├── theory.md                               ← Which model type for which task category
│   └── examples.ipynb                          ← Task-to-model routing decision tree
│
├── 05_provider_api_comparison/
│   ├── theory.md                               ← OpenAI vs Anthropic vs Google vs Groq APIs
│   └── examples.ipynb                          ← Same task, 3 APIs, side-by-side
│
├── 06_open_source_models/
│   ├── theory.md                               ← Llama, Mixtral, Mistral, local vs cloud
│   └── examples.ipynb                          ← Ollama + OpenAI-compatible API usage
│
└── 07_selection_framework/
    ├── theory.md                               ← Decision scorecard, production checklist
    └── examples.ipynb                          ← Automated model selection based on task profile
```

---

## 📚 Topics Covered

| # | Topic | Core Question Answered |
|---|---|---|
| 1 | `01_model_landscape` | What models exist and what are their specs? |
| 2 | `02_capability_comparison` | How do they differ in reasoning, coding, instruction following? |
| 3 | `03_cost_and_latency_analysis` | What does each model cost and how fast is it? |
| 4 | `04_task_based_selection` | Which model should I use for *this* task? |
| 5 | `05_provider_api_comparison` | How different are the APIs to use? |
| 6 | `06_open_source_models` | When and how to use open-source / self-hosted models? |
| 7 | `07_selection_framework` | How do I make the final decision systematically? |

---

## ⏱️ Estimated Time

| Activity | Time |
|---|---|
| Reading all 7 theory files | 2–3 hours |
| Running all 7 notebooks | 2–3 hours |
| **Total** | **~5 hours** |

---

## 🔧 Setup

```bash
pip install openai anthropic google-generativeai litellm tiktoken pydantic python-dotenv rich
```

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

---

## 🔑 The Selection Decision Tree

```
Is cost the primary constraint?
  YES → gpt-4o-mini / claude-3-haiku / gemini-flash
  
  NO → Is context > 64k tokens?
         YES → claude-3-5-sonnet (200k) or gemini-1.5-pro (2M)
         
         NO → Is it primarily coding?
                YES → gpt-4o or claude-3-5-sonnet
                
                NO → Is it multimodal (images/audio)?
                       YES → gpt-4o or gemini-1.5-pro
                       
                       NO → Is privacy/on-premise required?
                              YES → llama-3.1-70b via Ollama/vLLM
                              NO  → gpt-4o or claude-3-5-sonnet (roughly equivalent)
```
