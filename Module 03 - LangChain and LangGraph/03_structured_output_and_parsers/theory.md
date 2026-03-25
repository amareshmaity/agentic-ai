# 03 — Structured Output & Parsers

> **Structured outputs transform raw LLM text into reliable, typed Python objects — the foundation of production-grade agents.**

---

## 📚 Subtopics in This Section

| File | Topic |
|---|---|
| [`01_why_structured_output.md`](./01_why_structured_output.md) | Problem with unstructured output, why structure matters |
| [`02_str_output_parser.md`](./02_str_output_parser.md) | StrOutputParser — simplest, most common parser |
| [`03_json_output_parser.md`](./03_json_output_parser.md) | JSONOutputParser, JSON mode, streaming JSON |
| [`04_pydantic_output_parser.md`](./04_pydantic_output_parser.md) | PydanticOutputParser — fully typed, validated output |
| [`05_with_structured_output.md`](./05_with_structured_output.md) | `.with_structured_output()` — modern, preferred approach |
| [`examples.ipynb`](./examples.ipynb) | Hands-on: all parsers + real extraction pipelines |

---

## 🎯 Learning Objectives

By the end of this section you will be able to:

- Explain why structured output is critical for agents
- Use `StrOutputParser` for simple string extraction
- Parse LLM output into JSON dicts with `JsonOutputParser`
- Enforce typed Pydantic schemas with `PydanticOutputParser`
- Use the modern `.with_structured_output()` API
- Build reliable data extraction pipelines

---

## ⚡ Quick Summary

```
LLM raw output → "The company was founded in 2010 by John Smith."
                          (fragile — hard to use programmatically)

Structured output → CompanyInfo(founder="John Smith", year=2010)
                          (reliable — Pydantic model, always correct type)

Parser options:
  StrOutputParser        → AIMessage → str      (simplest)
  JsonOutputParser       → str → dict            (flexible)
  PydanticOutputParser   → str → PydanticModel   (typed + validated)
  .with_structured_output() → modern native approach (preferred)
```

---

## ⬅️ Previous
[02 — Models & Prompts](../02_models_and_prompts/theory.md)

## ➡️ Next Subtopic
[04 — Chains & Runnables](../04_chains_and_runnables/theory.md)
