# ChromaDB

> *Chroma is the most popular vector store for LangChain development — easy setup, persistent storage, and rich metadata filtering. No server required.*

---

## 🤔 What is ChromaDB?

ChromaDB is an **open-source vector database** that stores document embeddings and lets you search them by semantic similarity. It runs in-memory (for prototyping) or on disk (persistent, no server needed).

```
Documents + Embeddings → Chroma → similarity_search(query) → relevant Documents
```

---

## 📦 Installation

```bash
pip install chromadb langchain-chroma
```

---

## 1️⃣ In-Memory Chroma (Prototyping)

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create from documents
docs = [
    Document(page_content="LangChain is a framework for LLM applications.", metadata={"source": "intro", "topic": "langchain"}),
    Document(page_content="LangGraph adds stateful graph-based orchestration.", metadata={"source": "intro", "topic": "langgraph"}),
    Document(page_content="LangSmith provides tracing and monitoring.", metadata={"source": "intro", "topic": "langsmith"}),
    Document(page_content="FAISS is a library for efficient similarity search.", metadata={"source": "tools", "topic": "faiss"}),
    Document(page_content="Chroma is an open-source vector database.", metadata={"source": "tools", "topic": "chroma"}),
]

# Automatically embeds and stores
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="my-collection",   # Optional name for the collection
)

print("Vector store created!")
print(f"Document count: {vectorstore._collection.count()}")
```

---

## 2️⃣ Persistent Chroma (Saved to Disk)

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Persist to disk — survives process restarts
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",     # ← Saves here
    collection_name="langchain-docs"
)
print("Saved to ./chroma_db")

# --- Later, in a new session ---
# Load existing database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="langchain-docs"
)
print(f"Loaded {vectorstore._collection.count()} documents")
```

---

## 3️⃣ Similarity Search

```python
# Basic similarity search — returns top-k most similar documents
results = vectorstore.similarity_search(
    query="How do I add monitoring to my app?",
    k=3                   # Return top 3 results
)

for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"  Content:  {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")
```

### With Similarity Scores

```python
# Returns (Document, score) tuples — lower score = more similar (distance)
results_with_scores = vectorstore.similarity_search_with_score(
    query="How do I monitor my LangChain app?",
    k=3
)

for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content[:80]}")
    # Score = L2 distance (lower = more similar)
    # To convert to cosine similarity: similarity = 1 - score/2
```

---

## 4️⃣ Metadata Filtering

Filter search results by metadata fields — only search a subset of your documents:

```python
# Filter by exact match
results = vectorstore.similarity_search(
    query="graph agents",
    filter={"topic": "langgraph"}   # Only search docs with topic=langgraph
)

# Filter with comparison operators
results = vectorstore.similarity_search(
    query="frameworks",
    filter={
        "source": {"$eq": "intro"},           # Equals
        # "$ne": not equal
        # "$gt", "$gte", "$lt", "$lte": comparisons
        # "$in": value in list
        # "$nin": value not in list
    }
)

# Combined filter with logical operators
results = vectorstore.similarity_search(
    query="frameworks",
    filter={
        "$or": [
            {"topic": {"$eq": "langchain"}},
            {"topic": {"$eq": "langgraph"}},
        ]
    }
)
```

---

## 5️⃣ Adding Documents After Creation

```python
# Add more documents to an existing vector store
new_docs = [
    Document(page_content="LCEL is the LangChain Expression Language.", metadata={"topic": "lcel"}),
    Document(page_content="Retrievers fetch relevant documents from vector stores.", metadata={"topic": "retrievers"}),
]

vectorstore.add_documents(new_docs)
print(f"New count: {vectorstore._collection.count()}")

# Add raw texts (with auto-generated IDs)
vectorstore.add_texts(
    texts=["More text 1...", "More text 2..."],
    metadatas=[{"source": "batch1"}, {"source": "batch1"}]
)
```

---

## 6️⃣ Delete Documents

```python
# Delete by metadata filter
vectorstore.delete(where={"topic": "faiss"})

# Delete by document IDs
ids = vectorstore.get(where={"topic": "chroma"})["ids"]
vectorstore.delete(ids=ids)
```

---

## 7️⃣ Convert to a Retriever

```python
# As a Runnable Retriever — plugs into any LCEL chain
retriever = vectorstore.as_retriever(
    search_type="similarity",           # or "mmr", "similarity_score_threshold"
    search_kwargs={"k": 4}              # Return top 4 documents
)

# Use directly
docs = retriever.invoke("What is LangGraph?")

# Or in a RAG chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_messages([
        ("system", "Answer using only this context:\n{context}"),
        ("human",  "{question}")
    ])
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

answer = rag_chain.invoke("What does LangSmith do?")
print(answer)
```

---

## 🔧 Chroma Collections

Chroma organizes documents into **collections** — like tables in a database:

```python
import chromadb

# Direct Chroma client
client = chromadb.PersistentClient(path="./chroma_db")

# List all collections
print(client.list_collections())

# Get existing collection
collection = client.get_or_create_collection("my-docs")

# Collection stats
print(f"Count: {collection.count()}")
```

---

## ✅ Key Takeaways

- `Chroma.from_documents(docs, embeddings)` — create and index in one call
- Set `persist_directory` to save the database; reload with `Chroma(persist_directory=...)`
- `similarity_search(query, k=N)` returns the N most relevant documents
- `similarity_search_with_score` returns `(Document, score)` tuples
- **Metadata filtering** (`filter={...}`) restricts search to a document subset
- `vectorstore.as_retriever()` makes Chroma pluggable into any LCEL chain

---

## ➡️ Next
[FAISS →](./03_faiss.md)
