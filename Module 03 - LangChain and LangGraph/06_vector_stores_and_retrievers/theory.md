# 06 — Vector Stores & Retrievers

> **The heart of RAG. Embeddings encode meaning as numbers; vector stores find the most semantically similar content to any query in milliseconds.**

---

## 📚 Subtopics in This Section

| File | Topic |
|---|---|
| [`01_embeddings.md`](./01_embeddings.md) | What are embeddings, OpenAI, HuggingFace, local embedding models |
| [`02_chroma.md`](./02_chroma.md) | ChromaDB — in-memory, persistent storage, metadata filtering |
| [`03_faiss.md`](./03_faiss.md) | FAISS — fast local vector search, save/load index |
| [`04_similarity_search.md`](./04_similarity_search.md) | Similarity search, MMR, scores, filtered search |
| [`05_retrievers.md`](./05_retrievers.md) | Retriever interface, BM25, EnsembleRetriever, MultiQueryRetriever |
| [`examples.ipynb`](./examples.ipynb) | Hands-on: build a full vector store + retrieval pipeline |

---

## 🎯 Learning Objectives

By the end of this section you will be able to:

- Explain what embeddings are and why they enable semantic search
- Generate embeddings using OpenAI and local models
- Build and query a vector store with Chroma and FAISS
- Use similarity search and MMR for diverse, relevant retrievals
- Create Retriever objects and plug them into RAG chains

---

## ⚡ Quick Summary

```
Text         →   Embedding (vector)      →  Vector Store  →  Retriever
"What is RAG?" → [0.23, -0.71, 0.05...] →  Chroma/FAISS  →  .invoke(query)

Similarity search: finds vectors closest to query vector
MMR:               like similarity, but penalizes redundant results

Key classes:
    OpenAIEmbeddings()          → cloud embeddings
    OllamaEmbeddings()          → local embeddings
    Chroma.from_documents(...)  → in-memory or persistent
    FAISS.from_documents(...)   → fast in-memory (no server)
    vectorstore.as_retriever()  → make any vector store a Retriever
```

---

## ⬅️ Previous
[05 — Document Loaders & Text Splitters](../05_document_loaders_and_text_splitters/theory.md)

## ➡️ Next Subtopic
[07 — Memory & Conversation History](../07_memory_and_conversation_history/theory.md)
