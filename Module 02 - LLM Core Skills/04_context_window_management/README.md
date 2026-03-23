# 🪟 Context Window Management

> *Every agent has a limited memory. Context window management is the art and science of making the most of it.*

---

## 📌 Why Context Window Management Is Critical for Agents

An agent doesn't run once — it runs in a loop, accumulating messages over many steps. Without careful management:

- **Token costs explode** — every message is re-sent on every API call
- **Context fills up** — older important information gets dropped off the end
- **Model quality degrades** — "lost in the middle" problem — LLMs are worse at retrieving info buried in long contexts
- **Agent loops fail** — hitting the context limit mid-task crashes the pipeline

**Context window management is what separates agents that can handle 5-step tasks from agents that can handle 100-step tasks.**

---

## 📂 Folder Structure

```
04_context_window_management/
│
├── README.md                                   ← You are here
│
├── 01_understanding_token_limits/
│   ├── theory.md                               ← Token basics, counting, model limits, cost math
│   └── examples.ipynb                          ← Counting tokens with tiktoken, cost calculator
│
├── 02_sliding_window_strategy/
│   ├── theory.md                               ← Fixed window, sliding window, keep-N patterns
│   └── examples.ipynb                          ← Sliding window context manager implementation
│
├── 03_summarization_compression/
│   ├── theory.md                               ← Compressing history via LLM summarization
│   └── examples.ipynb                          ← Rolling summary, progressive compression
│
├── 04_retrieval_augmented_context/
│   ├── theory.md                               ← RAG for context: embed, store, retrieve relevant chunks
│   └── examples.ipynb                          ← Vector similarity retrieval for context injection
│
├── 05_token_budgeting_and_allocation/
│   ├── theory.md                               ← Budget math, system/user/tool/completion budgets
│   └── examples.ipynb                          ← Token budget enforcer, dynamic allocation
│
├── 06_long_document_handling/
│   ├── theory.md                               ← Chunking strategies, overlap, map-reduce over docs
│   └── examples.ipynb                          ← Process a long document through a limited context window
│
└── 07_production_context_manager/
    ├── theory.md                               ← End-to-end production context manager architecture
    └── examples.ipynb                          ← Full context manager class for long-running agents
```

---

## 📚 Topics Covered

| # | Subfolder | Core Concept |
|---|---|---|
| 1 | `01_understanding_token_limits` | Token counting, model limits, cost math, tiktoken |
| 2 | `02_sliding_window_strategy` | Keep-N, sliding window, system prompt pinning |
| 3 | `03_summarization_compression` | Rolling LLM summary, progressive compression |
| 4 | `04_retrieval_augmented_context` | Embed history, retrieve relevant, inject into context |
| 5 | `05_token_budgeting_and_allocation` | Budget math, per-section limits, enforcement |
| 6 | `06_long_document_handling` | Chunking, overlap, map-reduce, document QA |
| 7 | `07_production_context_manager` | Full production context manager: all strategies combined |

---

## ⏱️ Estimated Time

| Activity | Time |
|---|---|
| Reading all 7 theory files | 3–4 hours |
| Running all 7 notebooks | 3–4 hours |
| **Total** | **~7 hours** |

---

## 🔧 Setup

```bash
pip install openai tiktoken numpy python-dotenv rich
```

```env
OPENAI_API_KEY=your_key
```

---

## 🔗 Prerequisites

- ✅ Module 02 → `01_prompt_engineering_for_agents`
- ✅ Module 02 → `03_structured_outputs`
- Basic understanding of how the messages array works in LLM APIs
