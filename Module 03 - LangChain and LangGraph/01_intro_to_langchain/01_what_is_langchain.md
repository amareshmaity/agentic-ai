# What is LangChain?

> *LangChain is an open-source framework that makes it easy to build applications powered by Large Language Models (LLMs). It acts as the connective tissue between LLMs and the real world — data, tools, memory, and external systems.*

---

## 🤔 The Problem LangChain Solves

LLMs like GPT-4 or Claude are incredibly powerful, but they have serious limitations when used **raw**:

| Limitation | Problem | LangChain Solution |
|---|---|---|
| **Knowledge Cutoff** | LLM doesn't know events after training date | Connect to live data via tools/retrieval |
| **No Private Data** | LLM can't access your PDFs, DBs, emails | Document loaders + RAG pipelines |
| **No Actions** | LLM can only generate text, can't do things | Tool use / function calling abstraction |
| **No Memory** | Each API call is stateless | Conversation memory modules |
| **No Coordination** | Complex tasks need multiple LLM calls | Chains and agents |
| **Vendor Lock-in** | Different APIs for every LLM | Unified interface across all providers |

---

## 📖 A Simple Analogy

Think of building an LLM application like building a car:

```
Without LangChain (raw API):
    You have an engine (LLM) but need to custom-build:
    - The steering wheel (prompt management)
    - The gearbox (output parsing)
    - The fuel system (memory + context)
    - GPS navigation (retrieval / RAG)
    - Controls (tool calling)
    → Weeks of boilerplate code

With LangChain:
    You get all those components off the shelf.
    Just assemble → drive.
```

---

## 🏗️ What LangChain Actually Does

```
              ┌─────────────────────────────────┐
              │         Your Application         │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │           LangChain              │
              │                                  │
              │  Prompts │ Chains │ Agents        │
              │  Memory  │ Tools  │ RAG            │
              └──┬───────────────────────┬───────┘
                 │                       │
    ┌────────────▼──────┐   ┌────────────▼──────────┐
    │     LLM APIs       │   │   External Systems     │
    │  OpenAI / Claude  │   │  Databases / APIs      │
    │  Gemini / Llama   │   │  Web Search / Files    │
    └───────────────────┘   └───────────────────────┘
```

---

## 📅 Brief History

| Year | Milestone |
|---|---|
| **Oct 2022** | Harrison Chase creates LangChain as a side project |
| **Jan 2023** | Open-sourced on GitHub — explodes in popularity |
| **Apr 2023** | LangChain raises $10M seed round |
| **Jun 2023** | LangSmith (observability) released |
| **Jan 2024** | LangGraph (stateful agents) released |
| **2024+** | Becomes the dominant agent framework in production |

> LangChain grew from 0 to 60,000+ GitHub stars in under a year — the fastest growing ML framework at the time.

---

## 🎯 What LangChain is Best For

✅ **Great fit:**
- Building RAG pipelines (document Q&A, chatbots)
- Creating tool-calling agents
- Rapid prototyping of LLM applications
- Multi-step LLM workflows
- Applications that need to switch between LLM providers

❌ **Not the best fit:**
- Simple, single LLM calls (use the SDK directly)
- Highly custom model training or fine-tuning
- Real-time inference at extreme scale (consider vLLM + custom infra)

---

## 🔑 Core Value Propositions

### 1. Modularity
Every component (model, prompt, parser, memory, tool) is a **plug-and-play module**. Swap GPT-4 for Claude with one line change.

### 2. Composability
Components are designed to **chain together**. The output of one step becomes the input of the next.

```python
chain = prompt | model | output_parser
```

### 3. Standardization
One unified `.invoke()` / `.stream()` / `.batch()` interface across **all** components and **all** providers.

```python
# Same interface for any LLM
openai_llm.invoke("Hello")
claude_llm.invoke("Hello")
gemini_llm.invoke("Hello")
```

### 4. Ecosystem
1000+ pre-built integrations: OpenAI, Anthropic, Google, HuggingFace, Chroma, Pinecone, Postgres, Redis, Slack, Gmail, and more.

---

## 📦 LangChain Package Structure

LangChain is split into focused packages:

| Package | Purpose | Install |
|---|---|---|
| `langchain-core` | Base abstractions, interfaces, LCEL | `pip install langchain-core` |
| `langchain` | Chains, agents, memory — the main framework | `pip install langchain` |
| `langchain-openai` | OpenAI LLM + embeddings | `pip install langchain-openai` |
| `langchain-anthropic` | Claude integration | `pip install langchain-anthropic` |
| `langchain-google-genai` | Gemini integration | `pip install langchain-google-genai` |
| `langchain-community` | 1000+ community integrations | `pip install langchain-community` |

> **Why split?** Smaller installs, faster updates, clearer dependency management. You only install what you need.

---

## 🚀 Installation & Setup

```bash
# Core packages you'll use throughout this module
pip install langchain langchain-openai langchain-anthropic \
            langchain-community langchain-core \
            python-dotenv
```

```python
# .env file — create in your project root
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
LANGCHAIN_API_KEY=...       # For LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=my-project
```

```python
# Load env variables in Python
from dotenv import load_dotenv
load_dotenv()

# Verify setup
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
print(llm.invoke("Hello LangChain!").content)
```

---

## ✅ Key Takeaways

- LangChain is a **framework**, not an LLM — it wraps and orchestrates LLMs
- It solves the **glue code problem**: all the boilerplate between LLMs and real applications
- Built around **composability** — components snap together like LEGO blocks
- Split into **focused packages** to keep installs lean
- Powers a huge ecosystem: LangGraph, LangSmith, LangFlow, LangServe

---

## ➡️ Next
[LangChain Architecture →](./02_architecture.md)
