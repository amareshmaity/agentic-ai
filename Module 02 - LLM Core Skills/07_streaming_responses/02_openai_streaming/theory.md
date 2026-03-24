# 02 — OpenAI Streaming

> *The complete OpenAI streaming API — text, tool calls, metadata, and context managers.*

---

## 2.1 The OpenAI Streaming API Modes

OpenAI supports three streaming modes:

```python
# Mode 1: Standard streaming (most common)
stream = client.chat.completions.create(model=..., messages=..., stream=True)
for chunk in stream: ...

# Mode 2: Context manager (automatic cleanup)
with client.chat.completions.stream(model=..., messages=...) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Mode 3: Async streaming (for async apps)
stream = await async_client.chat.completions.create(..., stream=True)
async for chunk in stream: ...
```

---

## 2.2 Chunk Anatomy — Full Detail

```python
# Full structure of a streaming chunk:
chunk = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion.chunk",
    "created": 1702345678,
    "model": "gpt-4o-mini-2024-07-18",
    "system_fingerprint": "fp_abc",
    "choices": [
        {
            "index": 0,
            "delta": {
                # Text content delta — most common
                "role": "assistant",   # only in first chunk
                "content": "The",     # None when tool_calls present

                # Tool call delta — when model wants to call a function
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_abc",      # Only in first chunk for this tool call
                        "type": "function",    # Only in first chunk
                        "function": {
                            "name": "search",  # Only in first chunk for this function
                            "arguments": '{"q'  # Accumulate across ALL chunks
                        }
                    }
                ]
            },
            "logprobs": None,
            "finish_reason": None  # "stop" | "tool_calls" | "length" | "content_filter" | None
        }
    ],
    "usage": None  # Only populated in final chunk if stream_options.include_usage=True
}
```

---

## 2.3 Streaming with Usage Statistics

By default, streaming doesn't return token counts. Opt in with `stream_options`:

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    stream=True,
    stream_options={"include_usage": True}  # ← Request usage stats
)

usage = None
full_text = ""

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        full_text += chunk.choices[0].delta.content
    if chunk.usage:  # Populated only in final chunk
        usage = chunk.usage

print(f"Tokens: input={usage.prompt_tokens}, output={usage.completion_tokens}")
print(f"Cost: ${usage.prompt_tokens * 0.15e-6 + usage.completion_tokens * 0.60e-6:.6f}")
```

---

## 2.4 Streaming Tool Calls

When a model decides to call a function, `finish_reason` is `"tool_calls"` and the `content` is empty. Instead, `tool_calls` deltas accumulate across multiple chunks:

```python
def stream_with_tool_call_detection(messages, tools):
    """Stream and detect when model wants to call a tool."""
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        stream=True
    )

    full_text = ""
    tool_calls_accumulator = {}  # {index: {id, name, arguments}}

    for chunk in stream:
        delta = chunk.choices[0].delta
        finish = chunk.choices[0].finish_reason

        # Accumulate text content
        if delta.content:
            full_text += delta.content
            print(delta.content, end="", flush=True)

        # Accumulate tool call deltas
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_accumulator:
                    tool_calls_accumulator[idx] = {"id": "", "name": "", "arguments": ""}

                if tc_delta.id:
                    tool_calls_accumulator[idx]["id"] += tc_delta.id
                if tc_delta.function.name:
                    tool_calls_accumulator[idx]["name"] += tc_delta.function.name
                if tc_delta.function.arguments:
                    tool_calls_accumulator[idx]["arguments"] += tc_delta.function.arguments

        # Stream finished
        if finish == "tool_calls":
            print(f"\n[Tool call detected: {tool_calls_accumulator}]")
            return {"type": "tool_calls", "tool_calls": tool_calls_accumulator}
        elif finish == "stop":
            return {"type": "text", "content": full_text}
```

---

## 2.5 The `.stream()` Context Manager (OpenAI SDK ≥ 1.x)

The context manager approach provides convenience methods:

```python
with client.chat.completions.stream(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain streaming in 3 sentences."}]
) as stream:
    # Option 1: text_stream — yields only non-empty content strings
    for text in stream.text_stream:
        print(text, end="", flush=True)

    # Get the final complete message after streaming
    final = stream.get_final_completion()
    print(f"\nFinish reason: {final.choices[0].finish_reason}")
    print(f"Total tokens:  {final.usage.total_tokens}")
```

The context manager:
- Automatically handles connection cleanup
- Provides `text_stream` (skips empty/tool-call chunks)
- `get_final_completion()` gives the assembled full response

---

## 2.6 Streaming with `max_tokens` and Handling `"length"` Finish Reason

```python
def safe_stream(prompt: str, max_tokens: int = 100) -> dict:
    """Stream with a token budget. Handle truncation gracefully."""
    full_text = ""
    finish_reason = None

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            full_text += content
            print(content, end="", flush=True)
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    print()
    if finish_reason == "length":
        print(f"\n⚠️  Response truncated at {max_tokens} tokens. Consider increasing max_tokens.")

    return {"text": full_text, "truncated": finish_reason == "length"}
```

---

## 2.7 Streaming + Structured Output Pattern

Stream text and parse it when complete (for structured cases):

```python
import json
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str

def stream_then_parse(text_to_summarize: str) -> Summary:
    """Stream the response, then parse JSON once complete."""

    SYSTEM = """You analyze text. Always respond with valid JSON matching:
{"title": str, "key_points": [str], "sentiment": "positive"|"negative"|"neutral"}"""

    full_text = ""
    with client.chat.completions.stream(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": f"Analyze: {text_to_summarize}"}
        ],
        response_format={"type": "json_object"}
    ) as stream:
        for text in stream.text_stream:
            full_text += text
            print(text, end="", flush=True)  # Show partial JSON building up
    print()

    data = json.loads(full_text)
    return Summary(**data)
```

---

## 📌 Key Takeaways

1. **`stream=True`** is the one-line toggle for streaming
2. **`delta.content`** is the text fragment — accumulate across all chunks
3. **`delta.tool_calls`** contains partial JSON — must accumulate `arguments` across chunks
4. **`finish_reason`**: `"stop"` = normal end, `"tool_calls"` = tool needed, `"length"` = truncated
5. **`stream_options={"include_usage": True}`** — opt-in for token counts during streaming
6. **`.stream()` context manager** = cleaner API with `text_stream` and `get_final_completion()`
7. **Parse structured output after streaming** — accumulate full text, then `json.loads()`
