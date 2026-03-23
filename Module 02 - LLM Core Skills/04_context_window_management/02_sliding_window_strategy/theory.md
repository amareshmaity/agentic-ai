# 02 — Sliding Window Strategy

> *The simplest, most reliable context management pattern — always keep the most recent N messages.*

---

## 2.1 The Core Problem: Growing Context

Every turn in an agent loop adds messages. Without any management:

```
Turn 1:   [system, user1, assistant1]                    → 350 tokens
Turn 2:   [system, user1, assistant1, user2, assistant2] → 650 tokens
Turn 10:  Full conversation history growing every turn   → 3,500 tokens
Turn 50:  Entire half-year history                       → 17,500 tokens
Turn 100: Potentially hits context limit                 → CRASH
```

We need strategies to **keep token usage bounded** regardless of how many turns take place.

---

## 2.2 Strategy 1 — Keep-N (Truncate Oldest)

The simplest approach: always keep only the last `N` messages:

```python
def keep_n_messages(
    messages: list[dict],
    n: int,
    always_keep_system: bool = True
) -> list[dict]:
    """
    Keep only the last N messages. Optionally always preserve the system message.
    
    Args:
        messages: Full conversation history
        n: Maximum number of messages to keep (excluding system)
        always_keep_system: If True, preserve the system message regardless
    
    Returns:
        Trimmed messages list
    """
    if not messages:
        return messages
    
    # Separate system message from the rest
    if always_keep_system and messages[0]["role"] == "system":
        system_msg = [messages[0]]
        history = messages[1:]
    else:
        system_msg = []
        history = messages
    
    # Keep only the last N messages from history
    trimmed_history = history[-n:] if len(history) > n else history
    
    return system_msg + trimmed_history
```

**Trade-off**: Simple and predictable, but **older context is permanently lost**. The agent can't remember what happened 20 turns ago.

---

## 2.3 Strategy 2 — Token-Aware Sliding Window

A more precise approach: instead of counting messages, count tokens and slide the window by token budget:

```python
import tiktoken

def sliding_window_by_tokens(
    messages: list[dict],
    max_tokens: int = 4000,
    model: str = "gpt-4o-mini",
    always_keep_system: bool = True
) -> list[dict]:
    """
    Trim messages so that total token count stays under max_tokens.
    Removes oldest messages first (except system prompt).
    
    Args:
        messages: Full conversation history
        max_tokens: Maximum tokens budget for the messages array
        model: Model name (for tokenizer)
        always_keep_system: Always preserve system message
    
    Returns:
        Trimmed messages fitting within token budget
    """
    enc = tiktoken.encoding_for_model(model)
    
    def count_msg_tokens(msg: dict) -> int:
        return 3 + len(enc.encode(str(msg.get("content", ""))))
    
    if not messages:
        return messages
    
    # Pin system message
    if always_keep_system and messages[0]["role"] == "system":
        system_msg = messages[0]
        history = list(messages[1:])
        used_tokens = count_msg_tokens(system_msg) + 3  # +3 for priming
    else:
        system_msg = None
        history = list(messages)
        used_tokens = 3
    
    # Greedily add messages from newest to oldest
    kept = []
    for msg in reversed(history):
        msg_tokens = count_msg_tokens(msg)
        if used_tokens + msg_tokens <= max_tokens:
            kept.append(msg)
            used_tokens += msg_tokens
        else:
            break  # Window full — stop adding older messages
    
    kept.reverse()  # Restore chronological order
    
    if system_msg:
        return [system_msg] + kept
    return kept
```

---

## 2.4 Strategy 3 — Paired Turn Window

A refinement of Keep-N: always keep whole turns (user + assistant pairs) to avoid orphaned half-conversations:

```python
def keep_n_turns(
    messages: list[dict],
    n_turns: int,
    always_keep_system: bool = True
) -> list[dict]:
    """
    Keep the last N full turns (user + assistant pairs).
    Prevents orphaned messages (e.g., an assistant message without its user prompt).
    """
    if not messages:
        return messages
    
    # Separate system message
    if always_keep_system and messages[0]["role"] == "system":
        system_msg = [messages[0]]
        history = messages[1:]
    else:
        system_msg = []
        history = messages
    
    # Group into turns: each turn = (user_msg, assistant_msg)
    turns = []
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i+1]["role"] == "assistant":
            turns.append((history[i], history[i+1]))
            i += 2
        else:
            i += 1  # Skip unpaired messages
    
    # Keep last N turns
    recent_turns = turns[-n_turns:]
    
    # Flatten back to message list
    flat = []
    for user_msg, asst_msg in recent_turns:
        flat.extend([user_msg, asst_msg])
    
    return system_msg + flat
```

---

## 2.5 Strategy 4 — Pinned + Sliding

For agents with long important system prompts or critical early instructions:

```python
def pinned_plus_sliding(
    messages: list[dict],
    max_tokens: int,
    pin_count: int = 3,   # Always keep first N non-system messages
    model: str = "gpt-4o-mini"
) -> list[dict]:
    """
    Always keep: system message + first pin_count messages + recent history.
    Useful for: keeping initial task context + recent conversation.
    
    Structure: [system, pin0, pin1, pin2, ..., recent_n-2, recent_n-1, recent_n]
    """
    enc = tiktoken.encoding_for_model(model)
    
    def count_tokens(msg):
        return 3 + len(enc.encode(str(msg.get("content", ""))))
    
    if not messages:
        return messages
    
    result = []
    used_tokens = 3  # Priming
    
    # 1. Always include system message
    if messages[0]["role"] == "system":
        result.append(messages[0])
        used_tokens += count_tokens(messages[0])
        history = messages[1:]
    else:
        history = messages
    
    # 2. Always include first pin_count messages
    pinned = history[:pin_count]
    for msg in pinned:
        result.append(msg)
        used_tokens += count_tokens(msg)
    
    remaining = history[pin_count:]
    
    # 3. Fill remaining budget with newest messages
    tail = []
    for msg in reversed(remaining):
        t = count_tokens(msg)
        if used_tokens + t <= max_tokens:
            tail.append(msg)
            used_tokens += t
        else:
            break
    
    tail.reverse()
    return result + tail
```

---

## 2.6 System Prompt Protection — Never Truncate Instructions

Your system prompt is the agent's operating manual. Always protect it:

```python
class ContextWindow:
    """A context window that always protects the system prompt."""
    
    def __init__(
        self,
        system_prompt: str,
        max_history_tokens: int = 4000,
        model: str = "gpt-4o-mini"
    ):
        self.system_prompt = system_prompt
        self.max_history_tokens = max_history_tokens
        self.model = model
        self.enc = tiktoken.encoding_for_model(model)
        self._history: list[dict] = []  # Only non-system messages
    
    def add(self, message: dict):
        """Add a message to history."""
        self._history.append(message)
    
    def get_messages(self) -> list[dict]:
        """Get sliding window of messages — system always first."""
        system_msg = {"role": "system", "content": self.system_prompt}
        return [system_msg] + self._trim_history()
    
    def _trim_history(self) -> list[dict]:
        """Trim history to fit within token budget."""
        def count(msg):
            return 3 + len(self.enc.encode(str(msg.get("content", ""))))
        
        kept = []
        used = 0
        for msg in reversed(self._history):
            t = count(msg)
            if used + t <= self.max_history_tokens:
                kept.append(msg)
                used += t
            else:
                break
        kept.reverse()
        return kept
    
    @property
    def history_length(self) -> int:
        return len(self._history)
    
    @property
    def current_token_count(self) -> int:
        import tiktoken
        msgs = self.get_messages()
        enc = tiktoken.encoding_for_model(self.model)
        return sum(3 + len(enc.encode(str(m.get("content", "")))) for m in msgs) + 3
```

---

## 2.7 Choosing the Right Strategy

| Strategy | When to Use | Trade-off |
|---|---|---|
| **Keep-N messages** | Simple chatbots, short tasks | Fast, simple — older context lost |
| **Token-aware sliding** | Production agents | Precise budget — older context lost |
| **Paired turns** | Conversation agents | Clean pairs — older context lost |
| **Pinned + sliding** | Tasks with important intro context | Preserves task description + recency |
| **Summarization** | Long multi-hour runs (see topic 03) | Compresses history — some detail lost |
| **RAG context** | Knowledge-heavy agents (see topic 04) | Smart retrieval — requires embedding |

---

## 📌 Key Takeaways

1. **Always protect the system prompt** — it lives outside the sliding window
2. **Token-aware is better than message-count** — messages vary wildly in length
3. **Keep paired turns** — avoid orphaned messages (assistant without its user prompt)
4. **Pinned messages** — preserve critical early context while sliding the tail
5. **Start managing at ~50% of limit** — don't wait until near the limit to act
6. **Log what you dropped** — useful for debugging when agents "forget" older context
