# 04 — Chains & Runnables (LCEL)

> **LCEL — LangChain Expression Language — is how you compose components into pipelines using the `|` pipe operator. The most important skill for building LangChain apps.**

---

## 📚 Subtopics in This Section

| File | Topic |
|---|---|
| [`01_what_is_lcel.md`](./01_what_is_lcel.md) | What is LCEL, the pipe `\|` operator, why LCEL over manual chains |
| [`02_runnable_interface.md`](./02_runnable_interface.md) | Runnable protocol — invoke, stream, batch, async |
| [`03_runnable_parallel.md`](./03_runnable_parallel.md) | RunnableParallel — run multiple chains simultaneously |
| [`04_runnable_passthrough.md`](./04_runnable_passthrough.md) | RunnablePassthrough, RunnableLambda, RunnablePick |
| [`05_chain_patterns.md`](./05_chain_patterns.md) | Sequential, branching, conditional, fallback chain patterns |
| [`examples.ipynb`](./examples.ipynb) | Hands-on: build all chain types step-by-step |

---

## 🎯 Learning Objectives

By the end of this section you will be able to:

- Compose any LangChain components using the `|` pipe operator
- Use `.invoke()`, `.stream()`, `.batch()`, `.ainvoke()` on any chain
- Build parallel chains with `RunnableParallel`
- Pass inputs through with `RunnablePassthrough`
- Apply custom transformation functions with `RunnableLambda`
- Implement advanced patterns: branching, fallbacks, conditional routing

---

## ⚡ Quick Summary

```
LCEL: Everything is a Runnable, compose with |

prompt | llm | parser                   ← Basic sequential chain
{"a": chain1, "b": chain2}             ← Parallel (RunnableParallel)
{"context": retriever, "q": RunnablePassthrough()}  ← Pass-through
chain.with_fallbacks([backup_chain])   ← Automatic fallback
chain.with_retry(stop_after_attempt=3) ← Auto-retry
```

---

## ⬅️ Previous
[03 — Structured Output & Parsers](../03_structured_output_and_parsers/theory.md)

## ➡️ Next Subtopic
[05 — Document Loaders & Text Splitters](../05_document_loaders_and_text_splitters/theory.md)
