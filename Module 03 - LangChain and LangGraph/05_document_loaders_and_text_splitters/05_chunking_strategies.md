# Chunking Strategies

> *Choosing the right chunk size and overlap can make the difference between a RAG system that works well and one that fails silently. Here's a systematic approach.*

---

## 🎯 Why Chunking Strategy Matters

```
Chunk too large:
    → LLM gets too much text → context diluted → noisy answers
    → More tokens per query → higher cost
    → Vector search matches are less precise

Chunk too small:
    → Each chunk lacks context → LLM can't answer fully
    → A single sentence loses meaning without surrounding text
    → More chunks to store and search → slower

The goal: chunks that are semantically complete units
    → Large enough to contain a full idea
    → Small enough to be specific and focused
```

---

## 📐 Choosing Chunk Size

### General Rules of Thumb

| Content Type | chunk_size | chunk_overlap |
|---|---|---|
| Research papers / academic | 1500–2000 | 200–400 |
| Legal documents | 1000–1500 | 200–300 |
| News articles / blog posts | 800–1200 | 150–200 |
| Product documentation | 500–1000 | 100–200 |
| Q&A / FAQ | 200–500 | 50–100 |
| Code files | 1500–3000 | 300–500 |
| Short social media posts | 200–400 | 50 |

### The Embedding Model Window

Your embedding model also has a token limit:

| Embedding Model | Max Tokens |
|---|---|
| `text-embedding-3-small` (OpenAI) | 8191 |
| `text-embedding-3-large` (OpenAI) | 8191 |
| `nomic-embed-text` (Ollama) | 8192 |
| `all-MiniLM-L6-v2` (HuggingFace) | 256 |

> Keep `chunk_size` well under the embedding model's limit. At ~4 chars/token, `chunk_size=1000` ≈ 250 tokens — safely within all models.

---

## 🧪 Evaluating Your Chunking

### Method 1: Visual Check

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Visual inspection
print(f"Total chunks: {len(chunks)}")
print(f"Min size: {min(len(c.page_content) for c in chunks)}")
print(f"Max size: {max(len(c.page_content) for c in chunks)}")
print(f"Avg size: {sum(len(c.page_content) for c in chunks) // len(chunks)}")

# Sample random chunks
import random
for i in random.sample(range(len(chunks)), min(3, len(chunks))):
    print(f"\n--- Chunk {i} ---")
    print(chunks[i].page_content)
    print(f"Length: {len(chunks[i].page_content)} chars")
```

### Method 2: Overlap Verification

```python
# Verify overlap is working correctly
chunk1 = chunks[0].page_content
chunk2 = chunks[1].page_content

# Find actual overlap
overlap_text = ""
for size in range(min(300, len(chunk1)), 0, -1):
    if chunk1[-size:] in chunk2:
        overlap_text = chunk1[-size:]
        break

print(f"Configured overlap: {splitter._chunk_overlap} chars")
print(f"Actual overlap:     {len(overlap_text)} chars")
print(f"Overlap text: '{overlap_text[:80]}...'")
```

---

## 🔁 Comparing Chunk Sizes — Experiment

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "... your long document ..."

configs = [
    {"chunk_size": 200,  "chunk_overlap": 20},
    {"chunk_size": 500,  "chunk_overlap": 50},
    {"chunk_size": 1000, "chunk_overlap": 200},
    {"chunk_size": 2000, "chunk_overlap": 400},
]

print(f"{'chunk_size':>12} {'overlap':>10} {'chunks':>8} {'avg_len':>8}")
print("-" * 45)
for cfg in configs:
    s = RecursiveCharacterTextSplitter(**cfg)
    c = s.create_documents([text])
    avg = sum(len(x.page_content) for x in c) // max(len(c), 1)
    print(f"{cfg['chunk_size']:>12} {cfg['chunk_overlap']:>10} {len(c):>8} {avg:>8}")
```

---

## 2️⃣ Semantic Chunking — Split by Meaning (Advanced)

Instead of splitting by character count, split when the **topic changes**.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# SemanticChunker uses embeddings to find natural topic boundaries
splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",    # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=95,            # Split when similarity drops below 95th percentile
)

chunks = splitter.create_documents([long_text])
# Chunks vary in size but are semantically coherent
```

**Trade-off:**
- ✅ Semantically coherent chunks — better retrieval quality
- ❌ Requires embedding call per sentence — slower and costs tokens
- Use for high-quality production RAG; use RecursiveCharacterTextSplitter for speed

---

## 3️⃣ Parent Document Retriever Pattern

Small chunks for precise retrieval, but return parent context to the LLM:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Small chunks → precise matching
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# Large chunks → sent to LLM (more context)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
store       = InMemoryStore()  # Stores parent documents

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(docs)

# When querying:
# 1. Finds the most relevant small chunk (precise match)
# 2. Returns the full PARENT chunk (rich context for LLM)
retrieved_docs = retriever.invoke("your query")
```

---

## 📋 Chunking Strategy Decision Tree

```
Is your document markdown/HTML?
    YES → MarkdownHeaderTextSplitter / HTMLHeaderTextSplitter
    NO  ↓

Is precise token control critical?
    YES → TokenTextSplitter
    NO  ↓

Is retrieval quality the top priority (cost not a constraint)?
    YES → SemanticChunker
    NO  ↓

→ RecursiveCharacterTextSplitter (default)
    └─ Start with: chunk_size=1000, chunk_overlap=200
    └─ Adjust based on: document type, avg paragraph length, embedding model
```

---

## 🔍 Production Checklist

Before shipping your chunking pipeline:

```
✅ Inspect 5-10 random chunks manually — do they make sense standalone?
✅ Check overlap is working (adjacent chunks share content)
✅ Verify chunk sizes are under embedding model token limit
✅ No chunks that are only whitespace or metadata
✅ Metadata (source, page) preserved through splitting
✅ Test with queries that span chunk boundaries — does retrieval still work?
✅ Measure retrieval quality: are top-k chunks actually relevant?
```

---

## ✅ Key Takeaways

- **Start with `chunk_size=1000, chunk_overlap=200`** — works for most cases
- Smaller chunks = more precise retrieval but less context
- Larger chunks = more context but less precise retrieval
- Use **SemanticChunker** for highest quality (costs more)
- Use **ParentDocumentRetriever** to get the best of both worlds
- Always manually inspect chunks before building a production RAG system

---

## ⬅️ Previous
[Text Splitters](./04_text_splitters.md)

## ➡️ Next
[Hands-on Examples →](./examples.ipynb)
