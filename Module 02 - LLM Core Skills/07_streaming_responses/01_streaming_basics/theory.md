# 01 — Streaming Basics

> *Understanding what happens between "send request" and "receive response."*

---

## 1.1 The Problem With Waiting

A typical GPT-4o call generating 500 tokens takes **3–8 seconds**. Without streaming:

```
User sends: "Explain quantum entanglement"
─────────────────────────────────────────
0ms   → Request sent
3,000ms → [SILENCE]
7,000ms → Full response arrives: "Quantum entanglement is..."
```

The user sees nothing for 7 seconds. Your agent loop blocks for 7 seconds before it can react.

**With streaming**, the first token arrives in ~200ms, and the user sees a live typewriter effect:

```
0ms   → Request sent
200ms → "Quantum"
250ms → " entanglement"
300ms → " is"
...
7,000ms → "...and this is why Einstein called it 'spooky action at a distance.'"
```

---

## 1.2 How Streaming Works Under the Hood

LLM APIs use **Server-Sent Events (SSE)** — a simple HTTP/1.1 pattern where the server keeps the connection open and pushes newline-delimited JSON chunks:

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Transfer-Encoding: chunked
Connection: keep-alive

data: {"id":"chatcmpl-abc","choices":[{"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"Quantum"}}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":" entanglement"}}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":" is"}}]}

data: [DONE]
```

Each line starting with `data:` is one **chunk**. The Python SDK reads these and yields them one by one.

---

## 1.3 Key Streaming Vocabulary

| Term | Meaning |
|---|---|
| **Chunk** | One SSE event containing a partial response delta |
| **Delta** | The incremental new content in a chunk (`choices[0].delta`) |
| **TTFT** | Time-To-First-Token — latency before user sees first character |
| **Throughput** | Tokens per second after the first token |
| **[DONE]** | Special SSE event indicating the stream is finished |
| **Finish reason** | Why the stream ended: `stop`, `tool_calls`, `length`, `content_filter` |

---

## 1.4 What Each Chunk Contains

```python
# A typical chunk from OpenAI streaming:
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion.chunk",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "delta": {
                "role": "assistant",   # Only in first chunk
                "content": "Quantum"   # The new text (can be empty)
            },
            "finish_reason": None      # None until the last chunk
        }
    ]
}

# Last chunk:
{
    "choices": [{"delta": {}, "finish_reason": "stop"}]
}
```

---

## 1.5 Basic Streaming Pattern

```python
from openai import OpenAI

client = OpenAI()

def stream_response(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Stream a response and return the full accumulated text."""
    full_text = ""

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta

        # Extract text content
        if delta.content:
            full_text += delta.content
            print(delta.content, end="", flush=True)  # Live display

        # Check if stream is finished
        finish = chunk.choices[0].finish_reason
        if finish:
            print()  # Newline after stream ends
            print(f"\n[Stream ended: finish_reason='{finish}']")

    return full_text
```

Three things to always handle:
1. **`delta.content`** — the new text (may be `None` in tool-call chunks)
2. **`flush=True`** — force Python to print immediately, not buffer
3. **`finish_reason`** — tells you why streaming stopped

---

## 1.6 Streaming vs. Non-Streaming: When to Use Each

| Use streaming when | Use non-streaming when |
|---|---|
| User-facing output (chat, agent thoughts) | Batch processing / offline tasks |
| You want low perceived latency | You need the full response before proceeding |
| Agent loop can process partial output | Simple scripts / automation |
| Building web apps / APIs | Unit tests (easier with full response) |

---

## 1.7 Key Metrics to Track

```python
import time

def stream_with_metrics(prompt: str) -> dict:
    """Stream response and track TTFT + throughput."""
    start = time.time()
    first_token_time = None
    token_count = 0
    full_text = ""

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1
            full_text += content

    total_time = time.time() - start
    ttft = (first_token_time - start) * 1000 if first_token_time else None
    throughput = token_count / total_time if total_time > 0 else 0

    return {
        "ttft_ms": round(ttft, 1) if ttft else None,
        "total_ms": round(total_time * 1000, 1),
        "tokens": token_count,
        "throughput_tps": round(throughput, 1),
        "text": full_text,
    }
```

---

## 📌 Key Takeaways

1. **Streaming = Server-Sent Events** — HTTP connection stays open, server pushes JSON chunks
2. **Each chunk has `delta.content`** — accumulate these to build the full response
3. **TTFT < 300ms** is typical — users see text almost instantly
4. **Always handle `finish_reason`** — tells you why streaming stopped (`stop`, `tool_calls`, `length`)
5. **`flush=True`** — without this, buffered output ruins the streaming effect in terminals
6. **Non-streaming mode** still useful for batch jobs, testing, and cases needing the full response first
