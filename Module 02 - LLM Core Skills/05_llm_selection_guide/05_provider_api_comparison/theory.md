# 05 — Provider API Comparison

> *OpenAI, Anthropic, Google, and Groq all have different APIs, SDK patterns, and quirks — know them before you build.*

---

## 5.1 Why API Differences Matter

Even when two models have similar quality, their APIs differ in:
- Authentication patterns and SDK setup
- Message format (roles, content types, tool schemas)
- Response formats and streaming protocols
- Rate limit structures and error handling
- Unique features (caching, extended thinking, grounding)

Knowing these differences upfront saves debugging time in production.

---

## 5.2 OpenAI API — The Standard

OpenAI's API has become the de-facto standard that others emulate:

```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Standard chat completion
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": "What is Python?"},
    ],
    max_tokens=200,
    temperature=0.7,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)

# Access response
content = response.choices[0].message.content
finish_reason = response.choices[0].finish_reason  # "stop", "length", "tool_calls"

# Token usage (always log this)
print(response.usage.prompt_tokens)
print(response.usage.completion_tokens)
print(response.usage.total_tokens)
```

### OpenAI Unique Features:
- **Structured outputs**: `.parse()` with Pydantic models (strict JSON schema enforcement)
- **Assistants API**: stateful threads, built-in file handling
- **Batch API**: 50% cost reduction for async bulk processing
- **Realtime API**: WebSocket streaming for voice/audio
- **o1/o1-mini**: Extended thinking (no temperature, no streaming initially)

---

## 5.3 Anthropic (Claude) API — Key Differences

```python
import anthropic

ac = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Key differences from OpenAI:
# 1. System prompt is a SEPARATE parameter, NOT a message
# 2. max_tokens is REQUIRED (no default)
# 3. Content can be a list of content blocks (for multimodal)

response = ac.messages.create(
    model="claude-3-5-sonnet-20241022",
    system="You are a helpful assistant.",    # ← Separate parameter
    messages=[
        {"role": "user", "content": "What is Python?"}
    ],
    max_tokens=1024,     # ← Required, no default
    temperature=0.7,
)

# Access response
content = response.content[0].text     # ← .content[0].text, not .choices[0].message.content
stop_reason = response.stop_reason     # "end_turn", "max_tokens", "tool_use"

# Token usage
print(response.usage.input_tokens)
print(response.usage.output_tokens)
```

### Anthropic Unique Features:
- **Prompt caching**: Mark content blocks for caching: `{"cache_control": {"type": "ephemeral"}}`
- **Extended thinking**: `thinking={"type": "enabled", "budget_tokens": 8000}` for deep reasoning
- **Vision**: Pass images as base64 or URL in content blocks
- **Computer use**: Tool for direct computer interaction (beta)
- **Tool use**: `tool_choice={"type": "tool", "name": "specific_tool"}` for forced tool calling

---

## 5.4 Google Gemini API

```python
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Key differences from OpenAI:
# 1. Uses GenerativeModel class
# 2. Chat vs single-turn uses different methods
# 3. System instruction is a model-level parameter

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="You are a helpful assistant."  # ← At model level
)

# Single turn
response = model.generate_content("What is Python?")
print(response.text)

# Multi-turn chat
chat = model.start_chat()
response = chat.send_message("What is Python?")
response2 = chat.send_message("Give me an example.")
print(response2.text)

# JSON mode
json_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={"response_mime_type": "application/json"}
)
```

### Gemini Unique Features:
- **Largest context**: 1M-2M tokens (process entire codebases)
- **Grounding with Google Search**: Real-time web search integration
- **File API**: Upload files (PDFs, audio, video) for multimodal processing
- **Function calling**: Similar to OpenAI but uses `Tool` objects
- **Streaming**: `model.generate_content(prompt, stream=True)` for chunked responses

---

## 5.5 Groq API — Speed-First

Groq runs open-source models on custom LPU hardware — extremely fast inference:

```python
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Same OpenAI-compatible API (easy migration)
response = client.chat.completions.create(
    model="llama-3.1-70b-versatile",   # Groq runs Llama, Mixtral, Gemma
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is Python?"}
    ],
    max_tokens=200,
    temperature=0.7,
)

print(response.choices[0].message.content)

# Groq provides actual speed metrics
print(response.usage.queue_time)       # Wait time before processing (ms)
print(response.usage.prompt_time)      # Time to process prompt (ms)
print(response.usage.completion_time)  # Time to generate output (ms)
```

### Groq Strengths:
- **Fastest inference**: 300-800 tokens/sec vs ~100 for standard cloud
- **OpenAI-compatible**: minimal code changes to migrate
- **Models**: Llama 3.1, Mixtral, Gemma 2 — all open source
- **Free tier**: generous for prototyping

---

## 5.6 LiteLLM — Unified Interface

Use LiteLLM to abstract away provider differences:

```python
import litellm

# Same code, any model — LiteLLM handles translation
response = litellm.completion(
    model="gpt-4o-mini",          # Just change this line for any provider
    # model="claude-3-5-sonnet-20241022",
    # model="gemini/gemini-1.5-flash",
    # model="groq/llama3-70b-8192",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user",   "content": "What is Python?"}
    ],
    max_tokens=100
)

content = response.choices[0].message.content  # Always same format
cost = litellm.completion_cost(response)        # Cost in USD for any provider
print(f"Response: {content}")
print(f"Cost: ${cost:.6f}")
```

---

## 5.7 API Comparison Matrix

| Feature | OpenAI | Anthropic | Google | Groq |
|---|---|---|---|---|
| System prompt | In messages | Separate param | Model-level | In messages |
| max_tokens | Optional | **Required** | Optional | Optional |
| Token usage field | `usage.prompt_tokens` | `usage.input_tokens` | — | `usage.prompt_time` |
| Structured output | `.parse()` native | Prompt-based | JSON mime type | OpenAI compatible |
| Streaming | ✅ | ✅ | ✅ | ✅ (fastest) |
| Prompt caching | Auto (50% off) | Manual (90% off) | Auto | ❌ |
| Largest context | 128k | 200k | 2M | 128k |
| Open source models | ❌ | ❌ | ❌ | ✅ |
| OpenAI compatible | ✅ (is the standard) | ❌ | ❌ | ✅ |

---

## 📌 Key Takeaways

1. **OpenAI** = the standard; most tutorials and frameworks target it
2. **Anthropic** = system prompt is separate, `max_tokens` required, best prompt caching
3. **Google Gemini** = biggest context (2M), system at model level, JSON mime type
4. **Groq** = fastest inference, OpenAI-compatible, open-source models only
5. **LiteLLM** = abstract away all differences with one unified SDK
6. **Log token usage from every provider** — field names differ (prompt_tokens vs input_tokens)
7. **Test streaming from each provider** — streaming implementation also varies
