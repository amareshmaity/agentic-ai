# Similarity Search & MMR

> *How you retrieve documents is just as important as what you store. Similarity search finds the closest; MMR finds the most relevant AND diverse set.*

---

## 1️⃣ Similarity Search (Default)

Standard similarity search returns the `k` documents with the **highest similarity** to the query vector.

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Basic similarity search
results = vectorstore.similarity_search(query="LangChain agents", k=4)
for doc in results:
    print(f"- {doc.page_content[:80]}")
```

### With Relevance Scores

```python
# OpenAI cosine similarity → score closer to 1.0 = more similar
results = vectorstore.similarity_search_with_relevance_scores(
    query="LangChain agents",
    k=4,
    score_threshold=0.7   # Only return docs with similarity ≥ 0.7
)

for doc, score in results:
    bar = "█" * int(score * 20)
    print(f"{score:.3f} {bar} | {doc.page_content[:60]}")
```

### Score Threshold Retriever

```python
# Only retrieve if confidence is high enough
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.75,    # Minimum similarity
        "k": 5
    }
)
results = retriever.invoke("stateful agents in LangChain")
print(f"Confident matches: {len(results)}")
```

---

## 🔴 The Problem With Plain Similarity Search

When you retrieve the top-k similar documents, you often get **redundant** results:

```
Query: "What are the best practices for building agents?"

Top 5 results (plain similarity):
  1. "Best practices for building agents include..."       ← ✅ relevant
  2. "When building agents, best practices are..."        ← ❌ almost identical to #1
  3. "Key best practices for agent development..."        ← ❌ redundant again
  4. "Building agents requires following these..."        ← ❌ still redundant
  5. "Agent building tools and frameworks..."             ← ✅ different angle
```

Results 2, 3, 4 are near-duplicates — they waste the LLM's context window and don't add new information.

---

## 2️⃣ MMR — Maximal Marginal Relevance

**MMR** balances two goals:
1. **Relevance** — documents should be similar to the query
2. **Diversity** — documents should NOT be too similar to each other

```
MMR formula:
  Select doc that maximizes:
    λ × similarity(doc, query) - (1-λ) × max_similarity(doc, selected_docs)
    ↑ stays relevant              ↑ penalizes if too similar to already selected

  λ (lambda_mult):
    λ = 1.0 → pure similarity search (no diversity)
    λ = 0.0 → maximum diversity (ignores query!)
    λ = 0.5 → balanced (good default)
```

### Using MMR

```python
# MMR via similarity_search
results_mmr = vectorstore.max_marginal_relevance_search(
    query="best practices for agents",
    k=4,                  # Number to return
    fetch_k=10,           # Fetch 10 candidates, then MMR selects best 4
    lambda_mult=0.5       # 0.0 = max diversity, 1.0 = max relevance
)

# MMR via retriever
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20,       # Fetch more candidates for better MMR selection
        "lambda_mult": 0.5
    }
)

results = retriever_mmr.invoke("best practices for agents")
```

### Comparing Similarity vs MMR

```python
# Same query, two different retrieval strategies
query = "LangChain tools"

sim_results = vectorstore.similarity_search(query, k=4)
mmr_results = vectorstore.max_marginal_relevance_search(
    query, k=4, fetch_k=10, lambda_mult=0.5
)

print("=== Similarity Search ===")
for doc in sim_results:
    print(f"  - {doc.page_content[:70]}")

print("\n=== MMR (diverse) ===")
for doc in mmr_results:
    print(f"  - {doc.page_content[:70]}")

# MMR results cover more ground — less redundant
```

---

## 3️⃣ Search Type Comparison

| Search Type | Method | Best For |
|---|---|---|
| **Similarity** | `.similarity_search()` | General retrieval, most used |
| **Similarity with score** | `.similarity_search_with_score()` | When you need confidence scores |
| **Score threshold** | `search_type="similarity_score_threshold"` | High-precision retrieval |
| **MMR** | `.max_marginal_relevance_search()` | Diverse, non-redundant results |

---

## 4️⃣ Filtering with Search

Combine semantic search with metadata filters for precise retrieval:

```python
# Chroma: filter by metadata
from langchain_chroma import Chroma

chroma_store = Chroma.from_documents(docs, embeddings)

# Only search within a specific category
results = chroma_store.similarity_search(
    query="best practices",
    k=3,
    filter={"category": "agents"}    # Only search agent-related docs
)

# Multiple conditions
results = chroma_store.similarity_search(
    query="installation setup",
    k=3,
    filter={
        "$and": [
            {"source": {"$eq": "docs"}},
            {"version": {"$gte": "0.2"}}
        ]
    }
)
```

---

## 5️⃣ Embedding-Based Search (Direct)

Search using a pre-computed vector instead of text:

```python
# Compute query vector yourself
query_vector = embeddings.embed_query("What is LCEL?")

# Search by vector
results = vectorstore.similarity_search_by_vector(
    embedding=query_vector,
    k=4
)

# Useful when you have pre-computed embeddings or
# want to reuse the same vector for multiple searches
```

---

## ✅ Key Takeaways

- **Default similarity search** finds the k closest vectors → may be redundant
- **MMR** finds k documents that are relevant AND diverse — better for LLM context
- `lambda_mult=0.5` is a good MMR default — adjust: higher = more relevant, lower = more diverse
- `fetch_k` should be 3-5× larger than k for MMR to have enough candidates to pick from
- `score_threshold` returns only high-confidence matches — use when quality > quantity
- **Always use MMR** when showing diverse results or when redundancy is a problem

---

## ➡️ Next
[Retriever Types →](./05_retrievers.md)
