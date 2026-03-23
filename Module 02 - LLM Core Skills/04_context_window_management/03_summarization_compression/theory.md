# 03 — Summarization Compression

> *When you can't drop context, compress it — use the LLM itself to summarize older conversation history.*

---

## 3.1 Why Summarization Instead of Dropping?

The sliding window drops older messages. This works well for short tasks, but fails when:

- **The task spans hours** — the agent needs to remember goals set at the start
- **Important context is always in the "old" messages** — initial task specification, user preferences
- **The conversation contains cumulative progress** — each step builds on all prior steps

Summarization compresses older messages into a compact summary, preserving the *gist* while freeing up tokens.

```
Without summarization (token-limited):
  [system] [turn1] [turn2] ... [turn30] [turn31] [turn32]
                   ^^^^^^^^ dropped when window fills ^^^

With rolling summarization:
  [system] [SUMMARY of turns 1-20] [turn21] [turn22] ... [turn32]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
           ~200 tokens instead of 3,000+ tokens,
           but key facts preserved
```

---

## 3.2 The Rolling Summary Pattern

The most common pattern: maintain a running summary of older history. Expand the summary when history exceeds a threshold:

```python
from openai import OpenAI
import tiktoken

client = OpenAI()

def compress_history_to_summary(
    messages_to_compress: list[dict],
    existing_summary: str | None = None,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Ask the LLM to summarize a list of messages into a compact paragraph.
    If existing_summary is provided, incorporate it into the new summary.
    """
    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in messages_to_compress
        if m['role'] != 'system'  # Don't summarize the system prompt
    )
    
    if existing_summary:
        prompt = f"""You are summarizing an ongoing conversation.

Previous summary:
{existing_summary}

New conversation to add to the summary:
{conversation_text}

Write an updated summary that captures all key points, decisions, findings, and facts from BOTH the previous summary and the new conversation. Be concise but complete. Maximum 3 paragraphs."""
    else:
        prompt = f"""Summarize this conversation, capturing key points, decisions, findings, and facts. Be concise but complete. Maximum 2 paragraphs.

{conversation_text}"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.0
    )
    
    return response.choices[0].message.content
```

---

## 3.3 RollingSummaryContext — Full Implementation

```python
import tiktoken
from openai import OpenAI

client = OpenAI()

class RollingSummaryContext:
    """
    Context manager that automatically compresses old history via LLM summarization.
    
    Strategy:
    - Keep recent history within `recent_window_tokens`
    - When history exceeds `compress_threshold_tokens`, compress the oldest half
    - Always keep system prompt + summary + recent history
    """
    
    def __init__(
        self,
        system_prompt: str,
        model: str = "gpt-4o-mini",
        recent_window_tokens: int = 2000,   # Keep this many tokens of recent history "raw"
        compress_threshold: int = 3000,     # Trigger compression when history exceeds this
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.recent_window_tokens = recent_window_tokens
        self.compress_threshold = compress_threshold
        self.enc = tiktoken.encoding_for_model(model)
        
        self._summary: str | None = None       # Compressed summary of older history
        self._recent: list[dict] = []           # Recent raw messages
        self._compression_count = 0
    
    def add(self, role: str, content: str):
        """Add a new message and compress if needed."""
        self._recent.append({"role": role, "content": content})
        
        # Check if we should compress
        if self._count_recent_tokens() > self.compress_threshold:
            self._compress()
    
    def _count_recent_tokens(self) -> int:
        total = 0
        for m in self._recent:
            total += 3 + len(self.enc.encode(str(m.get("content", ""))))
        return total
    
    def _compress(self):
        """Compress the oldest half of recent messages into the summary."""
        split = len(self._recent) // 2
        to_compress = self._recent[:split]
        self._recent = self._recent[split:]   # Keep the newer half
        
        # Compress old messages into summary
        self._summary = compress_history_to_summary(
            to_compress,
            existing_summary=self._summary,
            model=self.model
        )
        self._compression_count += 1
        print(f"  🗜️  Compressed {len(to_compress)} messages → summary (compression #{self._compression_count})")
    
    def get_messages(self) -> list[dict]:
        """Build the messages array: system + [summary] + recent."""
        msgs = [{"role": "system", "content": self.system_prompt}]
        
        if self._summary:
            # Inject summary as an assistant message (common pattern)
            msgs.append({
                "role": "assistant",
                "content": f"[Summary of earlier conversation]\n{self._summary}"
            })
        
        msgs.extend(self._recent)
        return msgs
    
    def stats(self) -> dict:
        msgs = self.get_messages()
        total_tokens = sum(
            3 + len(self.enc.encode(str(m.get("content", ""))))
            for m in msgs
        ) + 3
        return {
            "messages_in_window": len(msgs),
            "total_tokens": total_tokens,
            "has_summary": self._summary is not None,
            "summary_words": len(self._summary.split()) if self._summary else 0,
            "compressions_done": self._compression_count
        }
```

---

## 3.4 Progressive Compression — Multiple Levels

For very long agent runs, use a hierarchical approach:

```python
class HierarchicalSummaryContext:
    """
    Three-level compression for extremely long agent runs.
    
    Level 1 (Hot):    Last 20 raw messages  — full detail
    Level 2 (Warm):   Summary of last 100   — compressed
    Level 3 (Cold):   Summary of all prior  — highly compressed
    """
    
    def __init__(self, system_prompt: str, model: str = "gpt-4o-mini"):
        self.system_prompt = system_prompt
        self.model = model
        self.hot:    list[dict] = []     # Raw recent messages
        self.warm_summary: str = ""      # Compressed mid-term
        self.cold_summary: str = ""      # Highly compressed long-term
        
        self.HOT_MAX  = 20   # messages
        self.WARM_MAX = 100  # messages before upgrading to cold
        self._warm_count = 0
    
    def add(self, role: str, content: str):
        self.hot.append({"role": role, "content": content})
        
        if len(self.hot) > self.HOT_MAX:
            # Overflow hot → warm
            overflow = self.hot[:-self.HOT_MAX]
            self.hot = self.hot[-self.HOT_MAX:]
            self.warm_summary = self._summarize(overflow, self.warm_summary, "brief")
            self._warm_count += len(overflow)
            
            if self._warm_count > self.WARM_MAX:
                # Compress warm → cold
                self.cold_summary = self._summarize([], self.cold_summary + "\n\n" + self.warm_summary, "very brief")
                self.warm_summary = ""
                self._warm_count = 0
    
    def _summarize(self, msgs: list[dict], existing: str, style: str) -> str:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs if m['role'] != 'system')
        prompt_parts = [f"Summarize ({style}):"]
        if existing:
            prompt_parts.append(f"Previous: {existing}")
        if text:
            prompt_parts.append(f"New: {text}")
        
        r = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "\n\n".join(prompt_parts)}],
            max_tokens=250 if style == "brief" else 100
        )
        return r.choices[0].message.content
    
    def get_messages(self) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        if self.cold_summary:
            msgs.append({"role": "user", "content": f"[Long-term summary]\n{self.cold_summary}"})
            msgs.append({"role": "assistant", "content": "Understood. I have context from previous work."})
        if self.warm_summary:
            msgs.append({"role": "user", "content": f"[Recent summary]\n{self.warm_summary}"})
            msgs.append({"role": "assistant", "content": "Got it, continuing from recent context."})
        msgs.extend(self.hot)
        return msgs
```

---

## 3.5 When to Trigger Compression — Rules

```python
def should_compress(
    current_tokens: int,
    context_limit: int,
    trigger_at: float = 0.6  # Trigger at 60% of limit
) -> bool:
    """Compress before hitting the limit — not after."""
    return current_tokens / context_limit >= trigger_at

# Practical thresholds for common models
COMPRESSION_TRIGGERS = {
    "gpt-4o-mini":     {"limit": 128_000, "trigger_tokens": 76_800},   # 60%
    "claude-3-haiku":  {"limit": 200_000, "trigger_tokens": 120_000},
    "gemini-1.5-flash":{"limit": 1_000_000, "trigger_tokens": 600_000},
}
```

---

## 3.6 Summary Message Format Patterns

```python
# Pattern 1: Summary as assistant message (simple)
summary_message = {
    "role": "assistant",
    "content": f"[Context summary]\n{summary_text}"
}

# Pattern 2: Summary as user+assistant exchange (cleaner for some models)
summary_pair = [
    {"role": "user",      "content": "[Please recall our conversation history]"},
    {"role": "assistant", "content": summary_text}
]

# Pattern 3: Summary injected into system prompt (always visible)
# ⚠️ Rebuilds system prompt each time — careful with token budget
updated_system = f"""{original_system}

## Conversation History Summary
{summary_text}"""
```

---

## 📌 Key Takeaways

1. **Sliding window drops; summarization compresses** — choose based on how much history matters
2. **Rolling summary** accumulates key facts while freeing old tokens
3. **Trigger compression at 60% of context limit** — don't wait until near the ceiling
4. **Hierarchical summaries** for very long runs: hot (raw) → warm → cold
5. **Summary accuracy degrades** over many compressions — monitor and test
6. **LLM summarization costs tokens** — factor this into your agent's per-turn budget
7. **Always keep the system prompt outside** the summarization loop
