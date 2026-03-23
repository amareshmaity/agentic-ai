# 🔧 Function Calling & Tool Use

> *The mechanism that gives agents their "hands" — how LLMs interact with the real world.*

---

## 📌 Why Function Calling Is the Core of Agentic AI

Without function calling, an LLM is a text-in → text-out black box.  
With function calling, the LLM becomes an **action-taking agent** that can:
- Search the web in real time
- Query databases
- Call any REST API
- Execute code
- Send emails, create tickets, update records

**Function calling is the single most important technical primitive in agentic AI.**

---

## 📂 Folder Structure

```
02_function_calling_tool_use/
│
├── README.md                                ← You are here
│
├── 01_what_is_function_calling/
│   ├── theory.md                            ← What FC is & how it works at protocol level
│   └── examples.ipynb                       ← Code: first function call, anatomy walkthrough
│
├── 02_openai_function_calling/
│   ├── theory.md                            ← OpenAI API: tools array, tool_choice, parsing
│   └── examples.ipynb                       ← Code: full OpenAI FC pipeline
│
├── 03_tool_schema_design/
│   ├── theory.md                            ← Writing great tool schemas: names, descriptions, params
│   └── examples.ipynb                       ← Code: weak vs strong schemas, schema validation
│
├── 04_parallel_and_multi_tool_calls/
│   ├── theory.md                            ← Parallel calls, chained calls, sequential vs concurrent
│   └── examples.ipynb                       ← Code: parallel tool execution, fan-out patterns
│
├── 05_function_calling_across_providers/
│   ├── theory.md                            ← OpenAI vs Anthropic vs Gemini FC differences
│   └── examples.ipynb                       ← Code: same agent on 3 providers side-by-side
│
├── 06_tool_error_handling/
│   ├── theory.md                            ← Error types, retry patterns, graceful degradation
│   └── examples.ipynb                       ← Code: robust tool execution with full error handling
│
└── 07_building_agent_tool_loop/
    ├── theory.md                            ← End-to-end agentic tool loop architecture
    └── examples.ipynb                       ← Code: production-ready agent loop from scratch
```

---

## 📚 Topics Covered

| # | Subfolder | Core Concept |
|---|---|---|
| 1 | `01_what_is_function_calling` | FC protocol internals, how the LLM signals a tool call |
| 2 | `02_openai_function_calling` | Full OpenAI API: tools, tool_choice, streaming FC |
| 3 | `03_tool_schema_design` | Writing precise JSON schemas, description engineering |
| 4 | `04_parallel_and_multi_tool_calls` | Parallel calls, batching, dependency handling |
| 5 | `05_function_calling_across_providers` | OpenAI vs Anthropic vs Gemini — differences & LiteLLM |
| 6 | `06_tool_error_handling` | Error classification, retry logic, graceful degradation |
| 7 | `07_building_agent_tool_loop` | Complete production agent loop with all patterns combined |

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
pip install openai anthropic google-generativeai litellm python-dotenv rich
```

```env
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key
```
