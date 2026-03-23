# 04 — Retrieval-Augmented Context

> *Instead of keeping everything in context, embed conversation history and retrieve only the relevant chunks when needed.*

---

## 4.1 The RAG Approach to Context Management

Traditional context management: **keep everything** (or drop/summarize).  
RAG-based context management: **embed everything, retrieve selectively**.

```
Traditional (all recent history):
  [system] [turn1] [turn2] [turn3] ... [turn50]
                                       ^^^^^^^^^^
                                       Always growing, hits limit

RAG-based:
  [system] [retrieved: turn7 about topic X] [retrieved: turn23 about topic Y] [turn50]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            Only 2-3 relevant past turns, regardless of total history length
            Token usage stays constant, not linear
```

---

## 4.2 How It Works

```
1. Store:    Every message → encode to embedding vector → save to vector store
2. Retrieve: New user query → encode → find top-K similar past messages by cosine similarity
3. Inject:   Insert retrieved messages into context as "background knowledge"
4. Call LLM: Full context = system + retrieved relevant history + recent turns + current query
```

---

## 4.3 Embedding Messages

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed_text(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Get embedding vector for a text string."""
    response = client.embeddings.create(
        model=model,
        input=text.replace("\n", " ")  # Embeddings work better without newlines
    )
    return response.data[0].embedding

# Embedding dimensions
# text-embedding-3-small: 1536 dimensions
# text-embedding-3-large: 3072 dimensions

sample = "What tools can I use for context window management in LangChain?"
embedding = embed_text(sample)
print(f"Text: {sample!r}")
print(f"Embedding dims: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

---

## 4.4 Simple In-Memory Vector Store

```python
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MessageRecord:
    role: str
    content: str
    embedding: list[float]
    timestamp: int       # Turn number
    metadata: dict = field(default_factory=dict)

class SimpleVectorStore:
    """
    Minimal in-memory vector store for conversation history.
    Uses cosine similarity for retrieval.
    For production, use ChromaDB, Pinecone, Qdrant, etc.
    """
    
    def __init__(self, embed_model: str = "text-embedding-3-small"):
        self.records: list[MessageRecord] = []
        self.embed_model = embed_model
        self._turn = 0
    
    def add(self, role: str, content: str, metadata: dict = None):
        """Embed and store a message."""
        embedding = embed_text(content, self.embed_model)
        self.records.append(MessageRecord(
            role=role,
            content=content,
            embedding=embedding,
            timestamp=self._turn,
            metadata=metadata or {}
        ))
        self._turn += 1
    
    def search(self, query: str, top_k: int = 3, min_similarity: float = 0.7) -> list[MessageRecord]:
        """Find the top-K most similar messages to the query."""
        if not self.records:
            return []
        
        query_embedding = embed_text(query, self.embed_model)
        query_vec = np.array(query_embedding)
        
        scores = []
        for record in self.records:
            rec_vec = np.array(record.embedding)
            # Cosine similarity
            sim = float(np.dot(query_vec, rec_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(rec_vec)))
            scores.append((sim, record))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return top_k results above similarity threshold
        return [rec for sim, rec in scores[:top_k] if sim >= min_similarity]
    
    def get_recent(self, n: int) -> list[MessageRecord]:
        """Get the N most recent records."""
        return self.records[-n:]
```

---

## 4.5 RAG Context Manager — Full Implementation

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

class RAGContextManager:
    """
    Combines:
    - Recent history (raw, last N turns)
    - Relevant history (retrieved by semantic similarity)
    - Always-on system prompt
    """
    
    def __init__(
        self,
        system_prompt: str,
        recent_turns: int = 5,        # Always include last N raw turns
        retrieved_turns: int = 3,     # Include top-K retrieved turns
        min_similarity: float = 0.70,
        model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small"
    ):
        self.system_prompt = system_prompt
        self.recent_turns = recent_turns
        self.retrieved_turns = retrieved_turns
        self.min_similarity = min_similarity
        self.model = model
        self.vector_store = SimpleVectorStore(embed_model)
        self._all_messages: list[dict] = []  # Raw chronological history
    
    def add(self, role: str, content: str):
        """Add message to both raw history and vector store."""
        msg = {"role": role, "content": content}
        self._all_messages.append(msg)
        # Only embed user messages (they carry the semantic search value)
        if role in ("user", "assistant"):
            self.vector_store.add(role, content)
    
    def get_messages(self, current_query: str) -> list[dict]:
        """
        Build context: system + retrieved relevant + recent raw.
        """
        msgs = [{"role": "system", "content": self.system_prompt}]
        
        # Retrieve semantically relevant past messages
        if len(self._all_messages) > self.recent_turns * 2:
            retrieved = self.vector_store.search(
                current_query,
                top_k=self.retrieved_turns,
                min_similarity=self.min_similarity
            )
            
            if retrieved:
                # Add retrieved history as bracketed context
                retrieved_text = "\n".join(
                    f"{r.role.upper()}: {r.content}" for r in retrieved
                )
                msgs.append({
                    "role": "user",
                    "content": f"[Relevant context from earlier in our conversation]\n{retrieved_text}"
                })
                msgs.append({
                    "role": "assistant",
                    "content": "I recall this relevant context from our earlier discussion."
                })
        
        # Add recent raw history (last N turns)
        recent = self._all_messages[-(self.recent_turns * 2):]
        msgs.extend(recent)
        
        return msgs
    
    def stats(self) -> dict:
        return {
            "total_messages_stored": len(self._all_messages),
            "embeddings_stored": len(self.vector_store.records)
        }
```

---

## 4.6 RAG vs Sliding Window vs Summarization

| | Sliding Window | Summarization | RAG Context |
|---|---|---|---|
| **Token usage** | Grows linearly, then drops | Bounded + summary | Near-constant |
| **Recall quality** | Only recent | Good for key facts | High — relevant retrieval |
| **Setup cost** | None | LLM API call to summarize | Embedding API call per message |
| **Latency** | None | +1 LLM call | +1 embedding call |
| **Best for** | Short sessions | Medium sessions | Long sessions, knowledge-heavy |
| **Requires vector DB** | ❌ | ❌ | ✅ (or in-memory) |

---

## 4.7 When to Use RAG Context Management

✅ **Use RAG context when:**
- Agent session spans hours or days
- History has heterogeneous topics (user jumps between subjects)
- You need to recall specific facts from many turns ago
- History is large but queries are specific

❌ **Don't use RAG context when:**
- Sessions are short (< 20 turns)
- All of history is relevant to the current query (coding session, sequential task)
- Latency is critical (embedding adds delay)
- Embedding API cost matters for high volume

---

## 📌 Key Takeaways

1. **RAG context = embed history → retrieve relevant → inject into context**
2. **Cosine similarity** is the standard similarity metric for text embeddings
3. **`text-embedding-3-small`** is the best cost/quality tradeoff for OpenAI embeddings
4. **Combine RAG + recent window** — retrieve relevant old context AND keep latest turns raw
5. **Use `min_similarity`** threshold (0.7+) to avoid injecting irrelevant old context
6. **Production**: use ChromaDB, Qdrant, or Pinecone instead of in-memory store
7. **Embedding cost**: ~$0.02 per 1M tokens — very cheap compared to LLM inference
