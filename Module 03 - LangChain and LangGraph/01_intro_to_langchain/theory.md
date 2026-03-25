# 01 — Introduction to LangChain

> **What is LangChain? Why does it exist? How is it structured?**
> This section builds the complete mental model before you write a single line of LangChain code.

---

## 📚 Subtopics in This Section

| File | Topic |
|---|---|
| [`01_what_is_langchain.md`](./01_what_is_langchain.md) | What is LangChain, why it was built, problem it solves |
| [`02_architecture.md`](./02_architecture.md) | LangChain's layered architecture and design philosophy |
| [`03_core_components.md`](./03_core_components.md) | Deep dive into every LangChain component |
| [`04_ecosystem.md`](./04_ecosystem.md) | LangSmith, LangGraph, LangFlow, LangServe, Hub |
| [`examples.ipynb`](./examples.ipynb) | Hands-on: first chains, component exploration |

---

## 🎯 Learning Objectives

By the end of this section you will be able to:

- Explain what LangChain is and why it was created
- Describe all major LangChain components and how they connect
- Understand the difference between LangChain core packages
- Set up a LangChain development environment
- Build your first LangChain chain from scratch

---

## ⚡ Quick Summary

```
LangChain = Framework to build LLM-powered applications

Without LangChain:     With LangChain:
────────────────       ─────────────────────────────────
Raw API calls    →     Composable, modular components
Manual parsing   →     Output parsers + structured output
No memory        →     Conversation memory built-in
No tools         →     Tool/function calling abstraction
No RAG           →     Document loaders + vector stores
Everything DIY   →     1000+ integrations out of the box
```

---

## ⬅️ Previous
[Module 03 README](../README.md)

## ➡️ Next Subtopic
[02 — Models & Prompts](../02_models_and_prompts/theory.md)
