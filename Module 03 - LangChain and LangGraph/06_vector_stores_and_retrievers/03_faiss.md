# FAISS

> *FAISS (Facebook AI Similarity Search) is a highly optimized in-memory vector library — lightning-fast retrieval with no server, ideal for development and batch processing.*

---

## 🤔 What is FAISS?

FAISS is a library by Meta AI for efficient **approximate nearest neighbor** search in high-dimensional vector spaces. In LangChain, it's wrapped as a vector store.

**Key characteristic:** FAISS runs **entirely in memory** (RAM) — no database server, no disk I/O during search. This makes it extremely fast.

```
Chroma:  stores vectors in a database file (persistent by default)
FAISS:   stores vectors in RAM (must explicitly save/load to persist)
```

---

## 📦 Installation

```bash
pip install faiss-cpu         # CPU version (most common)
# pip install faiss-gpu       # GPU version (requires CUDA)
```

---

## 1️⃣ Create FAISS from Documents

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docs = [
    Document(page_content="LangChain is a framework for LLM applications.", metadata={"source": "intro", "page": 1}),
    Document(page_content="LangGraph adds stateful agent orchestration.", metadata={"source": "intro", "page": 2}),
    Document(page_content="FAISS is fast for in-memory vector search.", metadata={"source": "tools", "page": 1}),
    Document(page_content="Chroma provides persistent vector storage.", metadata={"source": "tools", "page": 2}),
    Document(page_content="Retrievers connect vector stores to LangChain chains.", metadata={"source": "retrieval", "page": 1}),
]

vectorstore = FAISS.from_documents(docs, embedding=embeddings)
print("FAISS index created!")
print(f"Total vectors: {vectorstore.index.ntotal}")
```

### From Raw Texts

```python
texts = [
    "LangChain is a framework for LLM applications.",
    "FAISS enables fast similarity search.",
    "Embeddings convert text to vectors.",
]
metadatas = [
    {"source": "doc1"},
    {"source": "doc2"},
    {"source": "doc3"},
]

vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
```

---

## 2️⃣ Similarity Search

```python
# Basic search — returns List[Document]
results = vectorstore.similarity_search(
    query="How does vector search work?",
    k=3
)
for doc in results:
    print(f"- {doc.page_content}")

# With scores — returns List[Tuple[Document, float]]
results_with_scores = vectorstore.similarity_search_with_score(
    query="stateful agent",
    k=2
)
for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content[:80]}")
    # FAISS returns L2 distance — lower = more similar
```

---

## 3️⃣ Save and Load FAISS Index

Unlike Chroma (which persists automatically), FAISS must be explicitly saved and loaded:

```python
# Save to disk
vectorstore.save_local("./faiss_index")
# Creates: ./faiss_index/index.faiss + ./faiss_index/index.pkl

# Load from disk
loaded_vs = FAISS.load_local(
    folder_path="./faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True   # Required since v0.2
)

print(f"Loaded {loaded_vs.index.ntotal} vectors")
result = loaded_vs.similarity_search("LangChain")
print(result[0].page_content)
```

---

## 4️⃣ Merging FAISS Indexes

A powerful FAISS feature — combine multiple indexes:

```python
# Build separate indexes for different document sources
docs_a = FAISS.from_documents(pdf_docs,  embedding=embeddings)
docs_b = FAISS.from_documents(web_docs,  embedding=embeddings)
docs_c = FAISS.from_documents(csv_docs,  embedding=embeddings)

# Merge all into one searchable index
docs_a.merge_from(docs_b)
docs_a.merge_from(docs_c)

print(f"Combined: {docs_a.index.ntotal} total vectors")
results = docs_a.similarity_search("your query", k=5)
```

---

## 5️⃣ As a Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

docs = retriever.invoke("What is FAISS?")
for doc in docs:
    print(f"- {doc.page_content[:80]}")
```

---

## 6️⃣ Adding New Documents

```python
# Add documents to an existing FAISS index
new_docs = [
    Document(page_content="SemanticChunker splits text by meaning, not character count."),
    Document(page_content="RunnableParallel runs multiple chains simultaneously."),
]
new_ids = vectorstore.add_documents(new_docs)
print(f"Added {len(new_ids)} documents. Total: {vectorstore.index.ntotal}")
```

---

## 📊 FAISS vs Chroma — When to Use Which

| Feature | FAISS | Chroma |
|---|---|---|
| **Storage** | In-memory (RAM) | On-disk (file) |
| **Speed** | ⚡ Fastest | Fast |
| **Persistence** | Manual save/load | Automatic |
| **Metadata filtering** | ❌ Limited | ✅ Rich |
| **Server needed** | ❌ None | ❌ None |
| **Scale** | Medium | Medium-Large |
| **Best for** | Prototyping, batch, speed | Production, filtering |

**Use FAISS when:**
- Speed is the top priority
- You're doing batch processing or one-off analysis
- Simple similarity search without complex filtering
- Building quick prototypes

**Use Chroma when:**
- You need metadata filtering
- Persistence and data management matter
- Building a RAG system that evolves over time

---

## ✅ Key Takeaways

- `FAISS.from_documents(docs, embeddings)` — fast index creation
- Fastest vector search of all LangChain vector stores (in-memory)
- **Must explicitly save** with `.save_local()` to persist across sessions
- Supports **merging multiple indexes** — great for combining datasets
- No metadata filtering — if you need filters, use Chroma instead
- `vectorstore.as_retriever()` makes FAISS pluggable into any LCEL chain

---

## ➡️ Next
[Similarity Search & MMR →](./04_similarity_search.md)
