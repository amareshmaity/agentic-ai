# 02 — Models & Prompts

> **Deep dive into LangChain's model and prompt abstractions — understanding these is the foundation for everything else.**

---

## 📚 Subtopics in This Section

| File | Topic |
|---|---|
| [`01_llm_vs_chatmodel.md`](./01_llm_vs_chatmodel.md) | LLM vs ChatModel — differences, when to use each |
| [`02_model_providers.md`](./02_model_providers.md) | OpenAI, Anthropic, Google, Ollama — setup & comparison |
| [`03_prompt_template.md`](./03_prompt_template.md) | PromptTemplate — string templates with variables |
| [`04_chat_prompt_template.md`](./04_chat_prompt_template.md) | ChatPromptTemplate — multi-turn messages, system/human/ai |
| [`05_advanced_prompting.md`](./05_advanced_prompting.md) | Few-shot, partial variables, MessagesPlaceholder |
| [`examples.ipynb`](./examples.ipynb) | Hands-on: models, prompts, provider switching |

---

## 🎯 Learning Objectives

By the end of this section you will be able to:

- Explain the difference between LLM and ChatModel interfaces
- Connect LangChain to OpenAI, Anthropic, Google, and local (Ollama) models
- Build PromptTemplates and ChatPromptTemplates with variables
- Use few-shot prompting for consistent agent behavior
- Swap LLM providers with zero chain code change

---

## ⚡ Quick Summary

```
LLM (legacy):        text in → text out
ChatModel (modern):  [messages] in → AIMessage out
                     ↑ Use this — supports tools, streaming,
                       system prompts, multimodal

PromptTemplate:      "Answer {question}" → string
ChatPromptTemplate:  [(system, ...), (human, {q})] → messages
                     ↑ Use this — maps to ChatModel input
```

---

## ⬅️ Previous
[01 — Intro to LangChain](../01_intro_to_langchain/theory.md)

## ➡️ Next Subtopic
[03 — Structured Output & Parsers](../03_structured_output_and_parsers/theory.md)
