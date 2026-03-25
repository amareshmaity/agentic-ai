# 07 — RAG with LangChain

> **RAG (Retrieval-Augmented Generation) is the most important agentic pattern. Ground LLM answers in real, up-to-date knowledge — not just training data.**

---

## 📚 Subtopics in This Section

| File | Topic |
|---|---|
| [`01_rag_fundamentals.md`](./01_rag_fundamentals.md) | What is RAG, why it matters, naive vs advanced RAG |
| [`02_rag_pipeline.md`](./02_rag_pipeline.md) | Full 6-step pipeline: load → split → embed → store → retrieve → generate |
| [`03_retrieval_chain.md`](./03_retrieval_chain.md) | `create_retrieval_chain`, `create_stuff_documents_chain` |
| [`04_conversational_rag.md`](./04_conversational_rag.md) | Chat history, question contextualization, conversational chain |
| [`05_chatbot_project.md`](./05_chatbot_project.md) | Complete RAG chatbot: PDF knowledge base + memory + streaming |
| [`examples.ipynb`](./examples.ipynb) | Hands-on: build progressively from simple RAG to full chatbot |

---

## 🎯 Learning Objectives

By the end of this section you will be able to:

- Explain what RAG is and why it solves LLM hallucination
- Build a complete 6-step RAG pipeline from scratch
- Use `create_retrieval_chain` for production-ready Q&A
- Add conversation history to make RAG stateful
- Build a complete PDF chatbot with streaming

---

## 📊 RAG Pipeline — Overview

![RAG Pipeline](./images/rag_pipeline.png)

---

## 💬 Conversational RAG — Architecture

![Conversational RAG](./images/conversational_rag.png)

---

## ⚡ Quick Summary

```
Simple RAG (one-shot Q&A):
  docs = loader.load()
  chunks = splitter.split_documents(docs)
  vectorstore = Chroma.from_documents(chunks, embeddings)
  retriever = vectorstore.as_retriever()
  chain = create_retrieval_chain(retriever, document_chain)
  result = chain.invoke({"input": "your question"})

Conversational RAG (with memory):
  + contextualize_q_chain: reformulates question using chat history
  + history_aware_retriever: uses contextualized question for retrieval
  + chat_history: list of HumanMessage/AIMessage kept between turns
```

---

## ⬅️ Previous
[06 — Vector Stores & Retrievers](../06_vector_stores_and_retrievers/theory.md)

## ➡️ Next Subtopic
[08 — Tools & Tool Calling](../08_tools_and_tool_calling/theory.md)
