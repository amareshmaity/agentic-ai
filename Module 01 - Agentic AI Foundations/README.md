# 📖 Module 01: Agentic AI Foundations & Mental Models

> **Build the right mental model before writing a single line of code.**

---

## 🎯 Module Goal

By the end of this module you will be able to:
- Precisely define what Agentic AI is and how it differs from traditional AI
- Explain the Perception → Reasoning → Action → Memory (PRAM) loop
- Identify and design all major agent types
- Recognize and apply the 4 core agentic design patterns
- Read and summarize key research papers that founded the field

---

## 📂 Folder Structure

```
Module 01 - Agentic AI Foundations/
│
├── README.md                          ← You are here
│
├── 01_what_is_agentic_ai/
│   ├── theory.md                      ← Deep conceptual explanation
│   └── examples.ipynb                 ← Code: basic vs agentic AI comparison
│
├── 02_types_of_agents/
│   ├── theory.md                      ← All agent types explained in depth
│   └── examples.ipynb                 ← Code: implement each agent type
│
├── 03_agent_anatomy_pram_loop/
│   ├── theory.md                      ← Anatomy of an agent, PRAM breakdown
│   └── examples.ipynb                 ← Code: build a minimal PRAM agent loop
│
├── 04_agentic_design_patterns/
│   ├── theory.md                      ← 4 patterns: Reflection, Tool Use, Planning, Multi-Agent
│   └── examples.ipynb                 ← Code: implement each design pattern
│
├── 05_autonomy_spectrum/
│   ├── theory.md                      ← Autonomy levels, when to automate vs HITL
│   └── examples.ipynb                 ← Code: autonomy level classifier + demo
│
├── 06_key_research_papers/
│   ├── theory.md                      ← How to read ML papers effectively
│   └── paper_summaries.md             ← Deep summaries of ReAct, Reflexion, Toolformer, etc.
│
└── exercises/
    └── exercises.md                   ← Practice problems + mini-projects
```

---

## 📚 Topics Covered

| # | Topic | Theory | Practical |
|---|---|---|---|
| 1 | What is Agentic AI? | `01_what_is_agentic_ai/theory.md` | `examples.ipynb` |
| 2 | Types of AI Agents | `02_types_of_agents/theory.md` | `examples.ipynb` |
| 3 | Agent Anatomy & PRAM Loop | `03_agent_anatomy_pram_loop/theory.md` | `examples.ipynb` |
| 4 | Agentic Design Patterns | `04_agentic_design_patterns/theory.md` | `examples.ipynb` |
| 5 | Autonomy Spectrum | `05_autonomy_spectrum/theory.md` | `examples.ipynb` |
| 6 | Key Research Papers | `06_key_research_papers/theory.md` | `paper_summaries.md` |

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
pip install openai python-dotenv rich
```

Create a `.env` file in this folder:
```
OPENAI_API_KEY=your_key_here
```

---

## ➡️ Next Module

[Module 02 — LLM Core Skills for Agents](../Module%2002%20-%20LLM%20Core%20Skills/)
