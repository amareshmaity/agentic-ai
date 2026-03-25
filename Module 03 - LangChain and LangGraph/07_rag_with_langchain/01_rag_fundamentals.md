# RAG Fundamentals

> *RAG lets LLMs answer questions about YOUR data, not just their training data. It's the most practical pattern in production AI systems today.*

---

## 🤔 What is RAG?

**RAG** = **R**etrieval-**A**ugmented **G**eneration

Instead of relying purely on what the LLM memorized during training, RAG:
1. **Retrieves** relevant information from your knowledge base at query time
2. **Augments** the LLM prompt with this retrieved context
3. **Generates** an answer grounded in real, specific documents

```
Without RAG:
  User: "What is our company's refund policy?"
  LLM: "I don't have information about your specific company." ❌
  OR: Makes something up (hallucination) ❌

With RAG:
  User: "What is our company's refund policy?"
  System: Retrieves policy doc → injects into prompt
  LLM: "Based on our policy document: refunds are processed within 5-7 business days..." ✅
```

---

## 🔴 Problems RAG Solves

| Problem | How RAG Fixes It |
|---|---|
| **Knowledge cutoff** | LLM only knows pre-training data → RAG fetches current docs |
| **Hallucination** | LLM invents facts → RAG grounds answers in real text |
| **Private data** | LLM doesn't know your data → RAG accesses your knowledge base |
| **Context window** | Can't fit 500 pages → RAG retrieves only the 3-5 relevant chunks |
| **Source attribution** | Can't cite sources → RAG tracks document metadata |

---

## 📐 Naive RAG vs Advanced RAG

```
Naive RAG (basic):
  Query → similarity_search(k=4) → stuff docs into prompt → LLM

Problems:
  - All retrieved docs may be similar/redundant
  - No conversation history → multi-turn breaks
  - No quality check on retrieved docs
  - Single query strategy may fail to capture intent

Advanced RAG (production):
  Query + chat history → contextualize → hybrid search (BM25 + semantic)
  → rerank results → compress to relevant parts → grade relevance
  → conditional: retrieve more or answer → LLM with citations
```

---

## 🏗️ The 6-Step RAG Pipeline

```
INDEXING (one-time setup):
  1. LOAD    → Read source documents (PDF, web, CSV, etc.)
  2. SPLIT   → Chunk into overlapping pieces (~1000 chars)
  3. EMBED   → Convert chunks to vectors (text-embedding-3-small)
  4. STORE   → Index vectors in a vector database (Chroma/FAISS)

QUERYING (per user request):
  5. RETRIEVE → semantic search for top-k relevant chunks
  6. GENERATE → LLM answers using retrieved context
```

---

## 💡 When NOT to Use RAG

RAG is not always the right tool:

| Situation | Better Alternative |
|---|---|
| Knowledge fits in context window | Just include the text directly |
| Need exact lookup (by ID/key) | Traditional database |
| Purely creative tasks | Standard LLM prompt |
| Real-time data (stocks, weather) | API/tool calling |
| Learning/reasoning about concepts | Fine-tuning |

---

## 📊 RAG Metrics — How to Evaluate

| Metric | What It Measures |
|---|---|
| **Context Precision** | Are retrieved docs actually relevant? |
| **Context Recall** | Did we retrieve ALL the relevant docs? |
| **Answer Faithfulness** | Is the answer grounded in the context? |
| **Answer Relevance** | Does the answer address the question? |

```python
# With RAGAS evaluation library:
# pip install ragas
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
```

---

## ✅ Key Takeaways

- RAG = Retrieve relevant context at query time → inject into LLM prompt
- Solves: hallucination, knowledge cutoff, private data, source attribution
- 6 steps: Load → Split → Embed → Store → Retrieve → Generate
- Indexing (steps 1-4) is done once; querying (steps 5-6) runs per request
- "Naive RAG" works; "Advanced RAG" adds reranking, history, quality checks

---

## ➡️ Next
[Full RAG Pipeline →](./02_rag_pipeline.md)
