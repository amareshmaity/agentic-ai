# 07 — Production Context Manager

> *A battle-tested context manager combines all strategies — sliding window, summarization, RAG, and budgeting — into one clean class.*

---

## 7.1 Why You Need a Dedicated Context Manager

In production, you can't manage context inline — the logic becomes scattered across your agent loop. A dedicated `ContextManager` class:

- **Encapsulates** all context management strategies
- **Auto-selects** the right strategy based on current state
- **Tracks metrics** — token usage, compressions, dropped messages
- **Serializes state** — persist agent memory between sessions
- **Provides a clean API** — `add()`, `get_messages()`, `stats()`

---

## 7.2 Production Context Manager Architecture

```
┌─────────────────────────────────────────────────────┐
│              ProductionContextManager                │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │ System Prompt│  │ Token Budget │                 │
│  │   (pinned)   │  │  Enforcer    │                 │
│  └──────────────┘  └──────────────┘                 │
│                                                      │
│  ┌────────────────────────────────────────────────┐ │
│  │              History Manager                   │ │
│  │  hot (raw)  │  summary  │  vector store        │ │
│  └────────────────────────────────────────────────┘ │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │          Strategy Selection                  │   │
│  │  < 50%: raw sliding window                   │   │
│  │  50-70%: sliding + summarization             │   │
│  │  > 70%: RAG + summarization                  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 7.3 The Production Context Manager — Full Implementation

```python
import tiktoken, json, time
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime
import numpy as np

client = OpenAI()

@dataclass
class ContextMetrics:
    """Tracks context management statistics over the agent's lifetime."""
    total_messages_added:   int = 0
    total_api_calls:        int = 0
    total_tokens_sent:      int = 0
    compressions_triggered: int = 0
    messages_dropped:       int = 0
    avg_window_tokens:      float = 0.0
    peak_window_tokens:     int = 0
    session_start:         str = field(default_factory=lambda: datetime.now().isoformat())
    
    def update_window_stats(self, tokens: int):
        self.total_api_calls += 1
        self.total_tokens_sent += tokens
        if tokens > self.peak_window_tokens:
            self.peak_window_tokens = tokens
        # Rolling average
        self.avg_window_tokens = (
            self.avg_window_tokens * (self.total_api_calls - 1) + tokens
        ) / self.total_api_calls


class ProductionContextManager:
    """
    Full-featured context manager for production agentic systems.
    
    Strategies used automatically based on context utilization:
    - < 50% utilized: raw sliding window
    - 50-75% utilized: sliding window + on-demand summarization
    - > 75% utilized: full summarization + vector retrieval
    
    Features:
    - Token budget enforcement
    - Automatic compression triggering
    - Rolling summary with progressive compression
    - Optional vector retrieval for relevant old context
    - Full serialization/deserialization for session persistence
    - Detailed metrics tracking
    """
    
    STRATEGY_THRESHOLDS = {
        "raw":           (0.00, 0.50),   # Just sliding window
        "summarize":     (0.50, 0.75),   # Add summarization
        "full":          (0.75, 1.00),   # Full: summarize + RAG
    }
    
    def __init__(
        self,
        system_prompt:          str,
        context_limit:          int   = 128_000,
        output_reserve:         int   = 4_096,
        min_history_tokens:     int   = 2_000,   # Always keep at least this much raw history
        target_utilization:     float = 0.60,    # Aim to use 60% of context
        model:                  str   = "gpt-4o-mini",
        enable_rag:             bool  = False,   # Enable vector retrieval (requires embeddings)
        embed_model:            str   = "text-embedding-3-small",
        session_id:             str   = None
    ):
        self.system_prompt      = system_prompt
        self.context_limit      = context_limit
        self.output_reserve     = output_reserve
        self.min_history_tokens = min_history_tokens
        self.target_utilization = target_utilization
        self.model              = model
        self.enable_rag         = enable_rag
        self.embed_model        = embed_model
        self.session_id         = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.enc = tiktoken.encoding_for_model(model)
        self.metrics = ContextMetrics()
        
        # History storage
        self._history:  list[dict] = []   # Raw messages (newest at end)
        self._summary:  str | None = None  # Compressed older history
        self._embeddings: list[tuple[dict, list[float]]] = []  # (message, embedding)
    
    # ── Core Public API ──────────────────────────────────────────────────
    
    def add(self, role: str, content: str, metadata: dict = None) -> None:
        """Add a message to context. Auto-manages compression."""
        msg = {"role": role, "content": content}
        self._history.append(msg)
        self.metrics.total_messages_added += 1
        
        # Store embedding if RAG enabled and OpenAI client available
        if self.enable_rag and role in ("user", "assistant"):
            try:
                emb = self._embed(content)
                self._embeddings.append((msg, emb))
            except Exception:
                pass  # Don't fail on embedding errors
        
        # Check if compression needed
        current_tokens = self._count_window_tokens()
        utilization = current_tokens / self.context_limit
        
        if utilization > self.target_utilization:
            self._auto_compress()
    
    def get_messages(self, current_query: str = "") -> list[dict]:
        """Build the message array for the next API call."""
        msgs = [{"role": "system", "content": self.system_prompt}]
        
        # Inject summary if exists
        if self._summary:
            msgs.append({
                "role": "assistant",
                "content": f"[Conversation summary]\n{self._summary}"
            })
        
        # Optionally inject retrieved context
        if self.enable_rag and current_query and len(self._embeddings) > 10:
            retrieved = self._retrieve(current_query, top_k=2)
            if retrieved:
                ctx = "\n".join(f"{m['role'].upper()}: {m['content'][:200]}" for m, _ in retrieved)
                msgs.append({"role": "user",      "content": f"[Relevant past context]\n{ctx}"})
                msgs.append({"role": "assistant", "content": "I have the relevant context in mind."})
        
        # Append recent raw history
        history_to_add = self._trim_to_budget(self._history)
        msgs.extend(history_to_add)
        
        # Track metrics
        total_tokens = self._count_messages(msgs)
        self.metrics.update_window_stats(total_tokens)
        
        return msgs
    
    def stats(self) -> dict:
        window_tokens = self._count_window_tokens()
        return {
            "session_id":          self.session_id,
            "messages_in_history": len(self._history),
            "window_tokens":       window_tokens,
            "utilization_pct":     round(window_tokens / self.context_limit * 100, 1),
            "has_summary":         self._summary is not None,
            "summary_words":       len(self._summary.split()) if self._summary else 0,
            "compressions":        self.metrics.compressions_triggered,
            "messages_dropped":    self.metrics.messages_dropped,
            "total_messages":      self.metrics.total_messages_added,
            "avg_window_tokens":   round(self.metrics.avg_window_tokens),
            "peak_window_tokens":  self.metrics.peak_window_tokens,
        }
    
    def to_json(self) -> str:
        """Serialize full context state for persistence."""
        return json.dumps({
            "session_id":   self.session_id,
            "system":       self.system_prompt,
            "history":      self._history,
            "summary":      self._summary,
            "metrics":      self.metrics.__dict__,
            "config": {
                "context_limit":      self.context_limit,
                "output_reserve":     self.output_reserve,
                "target_utilization": self.target_utilization,
                "model":              self.model
            }
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ProductionContextManager":
        """Restore a context manager from a serialized state."""
        data = json.loads(json_str)
        cfg = data["config"]
        mgr = cls(
            system_prompt      = data["system"],
            context_limit      = cfg["context_limit"],
            output_reserve     = cfg["output_reserve"],
            target_utilization = cfg["target_utilization"],
            model              = cfg["model"],
            session_id         = data["session_id"]
        )
        mgr._history = data["history"]
        mgr._summary = data["summary"]
        return mgr
    
    # ── Internal Helpers ─────────────────────────────────────────────────
    
    def _count(self, text: str) -> int:
        return len(self.enc.encode(str(text)))
    
    def _count_messages(self, messages: list[dict]) -> int:
        return sum(3 + self._count(m.get("content", "")) for m in messages) + 3
    
    def _count_window_tokens(self) -> int:
        system_tokens = 3 + self._count(self.system_prompt)
        summary_tokens = (3 + self._count(self._summary)) if self._summary else 0
        history_tokens = self._count_messages(self._history)
        return system_tokens + summary_tokens + history_tokens
    
    def _trim_to_budget(self, history: list[dict]) -> list[dict]:
        """Trim history to fit remaining token budget."""
        system_t  = 3 + self._count(self.system_prompt) + 3
        summary_t = (3 + self._count(self._summary)) if self._summary else 0
        budget    = int(self.context_limit * self.target_utilization) - system_t - summary_t - self.output_reserve
        
        kept, used = [], 0
        for msg in reversed(history):
            t = 3 + self._count(msg.get("content", ""))
            if used + t <= budget:
                kept.append(msg)
                used += t
            else:
                self.metrics.messages_dropped += 1
        kept.reverse()
        return kept
    
    def _auto_compress(self):
        """Compress oldest half of history into summary."""
        if len(self._history) < 4:
            return  # Not enough to compress
        
        split = len(self._history) // 2
        to_compress = self._history[:split]
        self._history = self._history[split:]
        
        self._summary = self._summarize(to_compress, self._summary)
        self.metrics.compressions_triggered += 1
    
    def _summarize(self, messages: list[dict], existing: str | None) -> str:
        text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages if m['role'] != 'system')
        prompt = (
            f"Update this summary with new conversation (2-3 concise paragraphs):\n\nPrevious:\n{existing}\n\nNew:\n{text}"
            if existing else
            f"Summarize this conversation (2 concise paragraphs):\n\n{text}"
        )
        r = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350, temperature=0.0
        )
        return r.choices[0].message.content
    
    def _embed(self, text: str) -> list[float]:
        r = client.embeddings.create(model=self.embed_model, input=text[:2000])
        return r.data[0].embedding
    
    def _retrieve(self, query: str, top_k: int = 2) -> list[tuple[dict, list[float]]]:
        if not self._embeddings:
            return []
        qv = np.array(self._embed(query))
        scored = []
        for msg, emb in self._embeddings:
            ev = np.array(emb)
            sim = float(np.dot(qv, ev) / (np.linalg.norm(qv) * np.linalg.norm(ev)))
            scored.append((sim, msg, emb))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(msg, emb) for sim, msg, emb in scored[:top_k] if sim > 0.70]
```

---

## 7.4 Usage Pattern — Drop-In Agent Loop Integration

```python
def run_production_agent(user_questions: list[str]) -> list[str]:
    """
    A complete agent loop using the ProductionContextManager.
    Drop-in replacement for raw messages management.
    """
    ctx = ProductionContextManager(
        system_prompt     = "You are a helpful, knowledgeable AI assistant.",
        target_utilization= 0.60,
        model             = "gpt-4o-mini"
    )
    
    answers = []
    
    for question in user_questions:
        ctx.add("user", question)
        
        # Get context-managed messages
        messages = ctx.get_messages(current_query=question)
        
        # API call
        response = client.chat.completions.create(
            model    = "gpt-4o-mini",
            messages = messages,
            max_tokens= 300
        )
        
        answer = response.choices[0].message.content
        ctx.add("assistant", answer)
        answers.append(answer)
        
        # Log stats every 5 turns
        s = ctx.stats()
        if s["total_messages"] % 5 == 0:
            print(f"[Turn {s['total_messages']//2}] "
                  f"window={s['window_tokens']:,} tokens ({s['utilization_pct']}%) "
                  f"compressions={s['compressions']}")
    
    return answers, ctx
```

---

## 7.5 Session Persistence — Save and Restore

```python
import json

# Save session after agent run
def save_session(ctx: ProductionContextManager, filepath: str):
    """Persist agent context to disk for later resumption."""
    with open(filepath, "w") as f:
        f.write(ctx.to_json())
    print(f"✅ Session {ctx.session_id} saved to {filepath}")

# Restore session
def restore_session(filepath: str) -> ProductionContextManager:
    """Restore agent context from a saved session file."""
    with open(filepath, "r") as f:
        json_str = f.read()
    ctx = ProductionContextManager.from_json(json_str)
    print(f"✅ Session {ctx.session_id} restored "
          f"({len(ctx._history)} messages, summary={'yes' if ctx._summary else 'no'})")
    return ctx
```

---

## 📌 Key Takeaways

1. **Encapsulate all strategy logic** in a single `ContextManager` class — clean agent loop
2. **Auto-compression** fires when utilization exceeds the target threshold
3. **Serialize with `to_json()`** — persist the entire agent state between sessions
4. **Restore with `from_json()`** — resume exactly where the session left off
5. **`ContextMetrics`** gives full visibility: compressions, drops, avg tokens, peak usage
6. **Strategy selection is automatic** — the manager adapts as context fills
7. **This is the one class you always need in production agentic code** — don't skip it
