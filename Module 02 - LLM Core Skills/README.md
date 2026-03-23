# 📖 Module 02: LLM Core Skills for Agents

> **Master LLM interactions the way agents actually use them — before you touch any framework.**

---

## 🎯 Module Goal

By the end of this module you will be able to:

- Write production-quality system prompts and agentic few-shot examples
- Implement native function calling across OpenAI, Anthropic, and Google APIs
- Enforce structured outputs using Pydantic + the `instructor` library
- Manage context windows with sliding windows, summarization, and token budgeting
- Select the right LLM for each task type using a cost/quality/latency scorecard
- Build a multi-model fallback pipeline using LiteLLM
- Handle streaming responses token-by-token in agent loops

---

## 📂 Folder Structure

```
Module 02 - LLM Core Skills/
│
├── README.md                              ← You are here
│
├── 01_prompt_engineering_for_agents/
│   ├── theory.md                          ← System prompts, persona, constraints, few-shot
│   └── examples.ipynb                     ← Code: prompting patterns for agentic use
│
├── 02_function_calling_tool_use/
│   ├── theory.md                          ← How function calling works at protocol level
│   └── examples.ipynb                     ← Code: OpenAI + Anthropic + Gemini side-by-side
│
├── 03_structured_outputs/
│   ├── theory.md                          ← JSON mode, Pydantic, instructor library
│   └── examples.ipynb                     ← Code: enforce structured agent outputs
│
├── 04_context_window_management/
│   ├── theory.md                          ← Token budgeting, chunking, compression
│   └── examples.ipynb                     ← Code: sliding window + summarization context manager
│
├── 05_llm_selection_guide/
│   ├── theory.md                          ← GPT-4o, Claude, Gemini, Llama — decision framework
│   └── examples.ipynb                     ← Code: benchmark 3 LLMs on the same agentic task
│
├── 06_llm_routing_and_fallback/
│   ├── theory.md                          ← LiteLLM, PortKey, cost caps, model fallback
│   └── examples.ipynb                     ← Code: LiteLLM router with fallback + cost limits
│
├── 07_streaming_responses/
│   ├── theory.md                          ← SSE, WebSockets, partial token handling in agents
│   └── examples.ipynb                     ← Code: streaming agent response, token-by-token output
│
└── exercises/
    └── exercises.md                       ← Practice problems + mini-project
```

---

## 📚 Topics Covered

| # | Topic | Theory | Practical |
|---|---|---|---|
| 1 | Prompt Engineering for Agents | `01_prompt_engineering_for_agents/theory.md` | `examples.ipynb` |
| 2 | Function Calling & Tool Use | `02_function_calling_tool_use/theory.md` | `examples.ipynb` |
| 3 | Structured Outputs | `03_structured_outputs/theory.md` | `examples.ipynb` |
| 4 | Context Window Management | `04_context_window_management/theory.md` | `examples.ipynb` |
| 5 | LLM Selection Guide | `05_llm_selection_guide/theory.md` | `examples.ipynb` |
| 6 | LLM Routing & Fallback | `06_llm_routing_and_fallback/theory.md` | `examples.ipynb` |
| 7 | Streaming Responses | `07_streaming_responses/theory.md` | `examples.ipynb` |

---

## 🧠 Why This Module Matters

Every agentic framework (LangChain, CrewAI, AutoGen, Agno) is ultimately a wrapper around:

1. **LLM API calls** — you need to understand what's happening underneath
2. **Function/tool calling** — the mechanism that gives agents their "hands"
3. **Structured outputs** — agents must return *parseable*, *reliable* data
4. **Context management** — agents run many turns; you must manage the window deliberately

**Skipping this module = building on a shaky foundation.** Framework magic hides these details until something breaks in production — and it will.

---

## ⏱️ Estimated Time

| Activity | Time |
|---|---|
| Reading all theory files | 4–5 hours |
| Running all notebooks | 3–4 hours |
| Exercises | 2–3 hours |
| **Total** | **~10 hours** |

---

## 🔧 Setup

```bash
pip install openai anthropic google-generativeai instructor tiktoken pydantic litellm python-dotenv rich
```

Create a `.env` file in this folder:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

> **Note**: Most practical examples run on `gpt-4o-mini` to minimize cost. Multi-model comparison notebooks require all three API keys.

---

## 📡 APIs & Libraries Covered

| Library | Purpose |
|---|---|
| `openai` | OpenAI GPT-4o / GPT-4o-mini API — function calling, streaming, structured output |
| `anthropic` | Claude 3.5 Sonnet / Haiku — tool use, streaming |
| `google-generativeai` | Gemini 1.5 Pro / Flash — function calling |
| `instructor` | Drop-in Pydantic structured output enforcement for any LLM |
| `tiktoken` | OpenAI tokenizer — count tokens before sending |
| `pydantic` | Data validation and schema definition for structured outputs |
| `litellm` | Unified API across 100+ LLMs, model fallback, cost tracking |
| `rich` | Beautiful terminal output for agent traces |

---

## 🔑 Key Concepts Preview

### Function Calling — What Actually Happens
```
User asks a question
    → LLM receives: system prompt + messages + tool schemas
    → LLM outputs: {"tool_call": {"name": "web_search", "args": {"query": "..."}}}
    → Framework executes tool → returns observation
    → LLM receives observation → generates final answer
```

### Structured Output — Why It Matters for Agents
```python
# Without structure: agent returns free text → parsing is fragile
response = "The company was founded in 2010 by John Smith"

# With Pydantic + instructor: agent returns validated Python objects → always parseable
class CompanyInfo(BaseModel):
    founded_year: int
    founder: str
    description: str

info: CompanyInfo = client.chat.completions.create(...)  # always returns CompanyInfo
```

### Context Window — The Agent's Critical Resource
```
GPT-4o:          128k tokens  (~96,000 words)
Claude 3.5:      200k tokens  (~150,000 words)
Gemini 1.5 Pro:  1M tokens    (~750,000 words)

Cost:   Input tokens × price/token
        Every agent loop step adds tokens — budget carefully
```

### LiteLLM — One API for All Models
```python
import litellm

# Same code, any model
response = litellm.completion(model="gpt-4o-mini", messages=[...])
response = litellm.completion(model="claude-3-haiku-20240307", messages=[...])
response = litellm.completion(model="gemini/gemini-1.5-flash", messages=[...])
```

---

## 🏆 Module Completion Checklist

- [ ] Read all 7 theory `.md` files
- [ ] Ran all 7 Jupyter notebooks
- [ ] Built a multi-model function-calling pipeline (OpenAI + at least one other)
- [ ] Used `instructor` + Pydantic to extract structured data from unstructured text
- [ ] Implemented a sliding-window context manager for long agent runs
- [ ] Configured LiteLLM with at least 2 models and a fallback rule
- [ ] Streamed a response token-by-token and handled it in a loop
- [ ] Completed the exercises

---

## ⬅️ Previous Module

[Module 01 — Agentic AI Foundations](../Module%2001%20-%20Agentic%20AI%20Foundations/)

## ➡️ Next Module

[Module 03 — Building Agents with LangChain & LangGraph](../Module%2003%20-%20LangChain%20and%20LangGraph/)
