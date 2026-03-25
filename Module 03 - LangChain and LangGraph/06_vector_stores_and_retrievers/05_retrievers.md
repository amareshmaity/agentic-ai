# Retriever Types

> *The Retriever interface standardizes how documents are fetched for RAG. LangChain provides many retriever types — from simple vector search to advanced multi-query strategies.*

---

## 🔌 The Retriever Interface

A `Retriever` is a **Runnable** that accepts a string query and returns `List[Document]`:

```python
# Every retriever has this interface
docs = retriever.invoke("your query")           # List[Document]
docs = await retriever.ainvoke("your query")    # Async
docs_list = retriever.batch(["q1", "q2", "q3"]) # Parallel

# Plug into any LCEL chain
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
```

---

## 1️⃣ VectorStore Retriever (Most Common)

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

# Similarity search retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# MMR retriever
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
)

# Score threshold retriever — only return confident matches
retriever_threshold = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 5}
)

# Use any retriever the same way
results = retriever.invoke("How do I use LangGraph?")
```

---

## 2️⃣ MultiQueryRetriever — Better Coverage

Generates **multiple rephrased versions** of your query, retrieves docs for each, and returns the union (deduplicated). Overcomes limitations of a single query phrasing.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

# For query "LangChain agents", it auto-generates multiple queries like:
# "How do LangChain agents work?"
# "What is the agent framework in LangChain?"
# "LangChain ReAct and tool-calling agents"
# → retrieves for ALL → merges → deduplicates
results = retriever.invoke("LangChain agents")
print(f"Unique docs retrieved: {len(results)}")

# Enable logging to see generated queries
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
```

---

## 3️⃣ ContextualCompressionRetriever — Compress Results

Retrieves documents then **compresses** each to only the relevant portion:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

# Compressor: extracts only the relevant portion of each doc
compressor = LLMChainExtractor.from_llm(
    ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

# Retriever with compression
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
)

# Returns compressed excerpts — less noise for the LLM
compressed_docs = compression_retriever.invoke("How does LCEL work?")
for doc in compressed_docs:
    print(f"- {doc.page_content}")   # Only the relevant portion
```

---

## 4️⃣ BM25Retriever — Keyword Search

BM25 is a classic **keyword-based** retrieval algorithm — not semantic, but very good at exact keyword matching:

```python
from langchain_community.retrievers import BM25Retriever

# Create from documents
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4   # Return top 4

results = bm25_retriever.invoke("FAISS similarity search")   # Exact keyword match!
for doc in results:
    print(doc.page_content)
```

**BM25 excels at:**
- Exact keyword matches ("product code XYZ-123")
- Named entities, IDs, codes (words semantic embeddings may miss)
- Short, specific queries that depend on exact terminology

---

## 5️⃣ EnsembleRetriever — Hybrid Search (Best of Both)

Combines **semantic** (vector) and **keyword** (BM25) search with Reciprocal Rank Fusion — consistently outperforms either alone:

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Keyword retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4

# Semantic retriever
faiss_vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
faiss_retriever   = faiss_vectorstore.as_retriever(search_kwargs={"k": 4})

# Ensemble: combine both with weights
ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]     # 40% keyword, 60% semantic
)

# Best of both: finds exact keyword matches AND semantic meaning
results = ensemble.invoke("FAISS similarity search performance")
print(f"Retrieved: {len(results)} unique documents")
```

---

## 6️⃣ Self-Query Retriever — Natural Language Filters

Converts natural language queries into structured metadata filters automatically:

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

# Define what metadata fields exist
metadata_field_info = [
    AttributeInfo(name="source",   description="Document source file", type="string"),
    AttributeInfo(name="topic",    description="Main topic of the document", type="string"),
    AttributeInfo(name="year",     description="Publication year", type="integer"),
    AttributeInfo(name="level",    description="Difficulty: beginner, intermediate, advanced", type="string"),
]

retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    vectorstore=chroma_store,
    document_contents="Technical documentation about LangChain",
    metadata_field_info=metadata_field_info,
)

# Natural language queries with implicit filters
results = retriever.invoke("beginner friendly LangChain tutorials from 2024")
# Auto-translates to: semantic search + filter(level=beginner, year=2024)

results = retriever.invoke("advanced LangGraph examples")
# Auto-translates to: semantic search + filter(topic=langgraph, level=advanced)
```

---

## 📊 Retriever Comparison

| Retriever | Type | Best For |
|---|---|---|
| `VectorStoreRetriever` | Semantic | General purpose RAG |
| `MultiQueryRetriever` | Semantic × N | Complex queries, better recall |
| `ContextualCompressionRetriever` | Semantic + compress | Long docs, precise answers |
| `BM25Retriever` | Keyword | Exact terms, IDs, codes |
| `EnsembleRetriever` | Hybrid | ✅ Best overall (recommended) |
| `SelfQueryRetriever` | Semantic + filter | When metadata filtering matters |

---

## ✅ Key Takeaways

- All retrievers share the same `.invoke(query)` → `List[Document]` interface
- **For production**: use `EnsembleRetriever` (hybrid BM25 + semantic) — best results
- **MultiQueryRetriever** improves recall by generating query variations
- **ContextualCompressionRetriever** reduces noise by compressing retrieved docs
- **SelfQueryRetriever** enables natural language metadata filtering
- All retrievers are Runnables — drop them into any LCEL chain

---

## ⬅️ Previous
[Similarity Search & MMR](./04_similarity_search.md)

## ➡️ Next
[Hands-on Examples →](./examples.ipynb)
