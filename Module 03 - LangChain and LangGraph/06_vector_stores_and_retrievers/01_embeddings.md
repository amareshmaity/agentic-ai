# Embeddings

> *Embeddings convert text into vectors of numbers that capture semantic meaning. Similar texts → similar vectors. This is the foundation of semantic search.*

---

## 🤔 What Are Embeddings?

An embedding is a list of floating-point numbers that represents the **meaning** of a piece of text.

```
"The dog is playing in the park"  →  [0.23, -0.71, 0.05, 0.88, ..., -0.33]  (1536 numbers)
"A puppy runs across the garden"  →  [0.25, -0.69, 0.07, 0.85, ..., -0.31]  ← VERY SIMILAR!
"The stock market crashed today"  →  [-0.44, 0.12, -0.66, 0.01, ..., 0.77]  ← very different
```

**Similarity = closeness in vector space.** Texts about the same topic cluster together.

### Why This Enables Semantic Search

```
Keyword search (old):  "dog" != "puppy"  →  miss
Semantic search (new): embed("dog") ≈ embed("puppy") by cosine similarity → match!

Query: "What is Python?"
Keyword: must contain "Python"
Semantic: finds docs about "programming language", "snake" context → correct one
```

---

## 📐 Cosine Similarity

The standard way to measure how similar two vectors are:

```
score = cos(θ) = (A · B) / (|A| × |B|)

score = 1.0   → identical meaning
score = 0.0   → unrelated
score = -1.0  → opposite meaning (rare in practice)
```

---

## 1️⃣ OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

# Standard setup
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",   # Dimension: 1536
    # or "text-embedding-3-large"     # Dimension: 3072 (better, more expensive)
    # or "text-embedding-ada-002"     # Legacy (1536)
)

# Embed a single query
query_vector = embeddings.embed_query("What is LangChain?")
print(f"Dimensions: {len(query_vector)}")     # 1536
print(f"Type:       {type(query_vector)}")    # list
print(f"Sample:     {query_vector[:5]}")      # [0.23, -0.71, ...]

# Embed multiple documents (batch)
texts = [
    "LangChain is a framework for building LLM applications.",
    "LangGraph extends LangChain with stateful graph-based agents.",
    "LangSmith provides observability for LangChain apps.",
]
doc_vectors = embeddings.embed_documents(texts)
print(f"Documents embedded: {len(doc_vectors)}")    # 3
print(f"Each dimension:     {len(doc_vectors[0])}")  # 1536
```

---

## 2️⃣ OpenAI Model Comparison

| Model | Dimensions | Max Tokens | Performance | Cost |
|---|---|---|---|---|
| `text-embedding-ada-002` | 1536 | 8191 | Good | $ |
| `text-embedding-3-small` | 1536 | 8191 | Better | $ |
| `text-embedding-3-large` | 3072 | 8191 | Best | $$ |

```python
# text-embedding-3-small: great default — as good as ada-002, cheaper
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# text-embedding-3-large: when retrieval quality is critical
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Reduce dimensions (3-large supports Matryoshka representation)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=256   # Reduce from 3072 → 256 for faster/cheaper storage
)
```

---

## 3️⃣ Local Embeddings — Ollama (Free, Private)

```python
from langchain_ollama import OllamaEmbeddings

# Pull model first: ollama pull nomic-embed-text
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",     # 768 dims, 8192 token window
    # or "mxbai-embed-large"      # 1024 dims, higher quality
    base_url="http://localhost:11434"
)

vector = embeddings.embed_query("What is LangChain?")
print(f"Dimensions: {len(vector)}")  # 768

# 100% local — no API key, no cost, full privacy
```

---

## 4️⃣ HuggingFace Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings

# Download model locally (first run) — no API key needed
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # 384 dims, 256 tokens, fast
    # or "BAAI/bge-large-en-v1.5"                         # 1024 dims, slower, better
    model_kwargs={"device": "cpu"},   # or "cuda" if GPU available
    encode_kwargs={"normalize_embeddings": True}
)

vector = embeddings.embed_query("What is LangChain?")
print(f"Dimensions: {len(vector)}")  # 384
```

---

## 5️⃣ Measuring Similarity

```python
import numpy as np
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def cosine_similarity(v1, v2):
    a, b = np.array(v1), np.array(v2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

texts = [
    "LangChain is a Python framework for building AI applications",
    "LangChain helps developers build LLM-powered software",       # similar
    "Python is a general-purpose programming language",             # partial match
    "The Eiffel Tower is located in Paris, France",                # unrelated
]
vectors = embeddings.embed_documents(texts)

query = "What is LangChain used for?"
query_vec = embeddings.embed_query(query)

print(f"Query: '{query}'\n")
for text, vec in zip(texts, vectors):
    score = cosine_similarity(query_vec, vec)
    bar   = "█" * int(score * 20)
    print(f"  {score:.3f} {bar}  {text[:55]}...")
```

---

## 📊 Embedding Provider Comparison

| Provider | Model | Dims | Cost | Privacy | Speed |
|---|---|---|---|---|---|
| **OpenAI** | text-embedding-3-small | 1536 | $$ | Cloud | Fast |
| **OpenAI** | text-embedding-3-large | 3072 | $$$ | Cloud | Fast |
| **Ollama** | nomic-embed-text | 768 | Free | ✅ Local | Medium |
| **HuggingFace** | all-MiniLM-L6-v2 | 384 | Free | ✅ Local | Fast |
| **HuggingFace** | BAAI/bge-large-en | 1024 | Free | ✅ Local | Medium |

**Recommendation:** Use `text-embedding-3-small` (OpenAI) for production, `nomic-embed-text` (Ollama) for local/private workloads.

---

## ✅ Key Takeaways

- Embeddings convert text to vectors where **similar meaning → close vectors**
- `embed_query(text)` → single vector (for search queries)
- `embed_documents([...])` → list of vectors (for indexing)
- Always use the **same embedding model** for indexing AND querying
- OpenAI `text-embedding-3-small` is the best cloud default
- `nomic-embed-text` via Ollama is the best local free option

---

## ➡️ Next
[ChromaDB →](./02_chroma.md)
