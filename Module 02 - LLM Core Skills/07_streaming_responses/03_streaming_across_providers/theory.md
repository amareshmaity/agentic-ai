# 03 — Streaming Across Providers

> *Anthropic, Google Gemini, and LiteLLM — one streaming interface for all.*

---

## 3.1 Provider Streaming APIs — Side by Side

Each provider has slightly different streaming syntax:

```python
# ── OpenAI ──────────────────────────────────────────────────────────────
from openai import OpenAI
client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4o-mini", messages=[...], stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)


# ── Anthropic ────────────────────────────────────────────────────────────
import anthropic
client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-3-5-haiku-20241022",
    max_tokens=500,
    messages=[...]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)


# ── Google Gemini ────────────────────────────────────────────────────────
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

for chunk in model.generate_content("...", stream=True):
    print(chunk.text, end="", flush=True)


# ── LiteLLM (unified) ───────────────────────────────────────────────────
import litellm

stream = litellm.completion(
    model="claude-3-5-haiku-20241022",  # or gpt-4o-mini or gemini/gemini-1.5-flash
    messages=[...],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## 3.2 Anthropic Streaming in Depth

Anthropic's streaming has a richer event system:

```python
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-3-5-haiku-20241022",
    max_tokens=500,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Explain streaming in 3 sentences."}]
) as stream:
    # Event types in Anthropic's streaming:
    for event in stream:
        if event.type == "content_block_start":
            pass  # New content block started
        elif event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                print(event.delta.text, end="", flush=True)
        elif event.type == "content_block_stop":
            pass  # Block finished
        elif event.type == "message_stop":
            pass  # Entire message finished

    # Or use the convenience text_stream:
    # for text in stream.text_stream: print(text, end="", flush=True)

    # Get final usage stats
    final = stream.get_final_message()
    print(f"\n\nTokens: input={final.usage.input_tokens}, output={final.usage.output_tokens}")
```

### Anthropic Event Types

| Event | Description |
|---|---|
| `message_start` | Stream begins; includes model metadata |
| `content_block_start` | New block starts (text or tool use) |
| `content_block_delta` | Incremental text (`text_delta`) or tool JSON (`input_json_delta`) |
| `content_block_stop` | Block completed |
| `message_delta` | Final message metadata (stop_reason, usage) |
| `message_stop` | Stream complete |

---

## 3.3 Google Gemini Streaming

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="You are a helpful assistant."
)

# ── Simple streaming ──────────────────────────────────────────────────────
for chunk in model.generate_content("Explain LLM streaming in 2 sentences.", stream=True):
    print(chunk.text, end="", flush=True)

# ── With full response on completion ─────────────────────────────────────
response = model.generate_content("...", stream=True)
response.resolve()  # Wait for full response
print(response.text)
print(f"Tokens: {response.usage_metadata}")
```

---

## 3.4 LiteLLM — Unified Streaming

LiteLLM normalizes all provider APIs to the same OpenAI format:

```python
import litellm

def unified_stream(prompt: str, model: str) -> str:
    """Stream any model using the same code."""
    full_text = ""
    stream = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    for chunk in stream:
        # Same access pattern regardless of provider
        content = chunk.choices[0].delta.content or ""
        full_text += content
        print(content, end="", flush=True)
    print()
    return full_text


# All these work with identical code:
unified_stream("What is 2+2?", "gpt-4o-mini")
unified_stream("What is 2+2?", "claude-3-5-haiku-20241022")
unified_stream("What is 2+2?", "gemini/gemini-1.5-flash")
```

---

## 3.5 Cross-Provider Streaming Differences

| Feature | OpenAI | Anthropic | Gemini | LiteLLM |
|---|---|---|---|---|
| Stream toggle | `stream=True` | `client.messages.stream()` | `stream=True` | `stream=True` |
| Content access | `chunk.choices[0].delta.content` | `event.delta.text` | `chunk.text` | `chunk.choices[0].delta.content` |
| Usage stats | `stream_options={"include_usage": True}` | `stream.get_final_message().usage` | `response.usage_metadata` | Varies |
| Finish reason | `chunk.choices[0].finish_reason` | `final.stop_reason` | Implicit | `chunk.choices[0].finish_reason` |
| Tool call streams | `delta.tool_calls[i].function.arguments` | `input_json_delta` | Partial function support | Normalized |

---

## 3.6 Provider Selection for Streaming

```python
STREAMING_SPEED_RANKING = {
    # Approx TTFT and throughput from public benchmarks (2024)
    "gemini-2.0-flash":          {"ttft_ms": 200,  "tps": 200},
    "claude-3-5-haiku-20241022": {"ttft_ms": 300,  "tps": 120},
    "gpt-4o-mini":               {"ttft_ms": 400,  "tps": 100},
    "gpt-4o":                    {"ttft_ms": 500,  "tps": 60},
    "claude-3-5-sonnet-20241022":{"ttft_ms": 600,  "tps": 70},
}
```

For real-time streaming UX:
- **Best TTFT**: Gemini 2.0 Flash, Claude 3.5 Haiku
- **Best throughput**: Gemini models
- **Best streaming tool call support**: OpenAI (most mature)
- **Unified API**: LiteLLM (recommended for multi-provider)

---

## 📌 Key Takeaways

1. **Every major provider supports streaming** — syntax differs slightly
2. **LiteLLM unifies all providers** — `chunk.choices[0].delta.content` for any model
3. **Anthropic uses event types** — richer than OpenAI's chunk model
4. **Gemini** — simplest API: `for chunk in model.generate_content(..., stream=True): chunk.text`
5. **For tool call streaming**: OpenAI has the best support; Anthropic via `input_json_delta`
6. **Gemini Flash** has the highest throughput for streaming-heavy use cases
