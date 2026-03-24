# 🌊 Streaming Responses

> *Don't wait for the whole response — process tokens as they arrive.*

---

## 📌 Why Streaming Matters for Agents

Without streaming, every LLM call looks like this:
- **User sends request** → waits in silence for 5–15 seconds → full response arrives
- **Agent loops wait** on each step before proceeding

With streaming:
- **Tokens arrive immediately** — users see text appear word-by-word
- **Agent loops can pre-process** partial output (e.g., detect tool calls as they stream)
- **Time-to-first-token (TTFT)** drops to ~100-300ms vs. 5+ seconds

Streaming is **essential for production agents** — it's the difference between an app that feels alive vs. one that feels broken.

---

## 📂 Folder Structure

```
07_streaming_responses/
│
├── README.md                                    ← You are here
│
├── 01_streaming_basics/
│   ├── theory.md                                ← SSE, how streaming works under the hood
│   └── examples.ipynb                           ← First streaming call, token iteration
│
├── 02_openai_streaming/
│   ├── theory.md                                ← OpenAI stream API, chunk structure
│   └── examples.ipynb                           ← Stream text, tool calls, structured output
│
├── 03_streaming_across_providers/
│   ├── theory.md                                ← Anthropic, Google Gemini streaming
│   └── examples.ipynb                           ← Unified streaming with LiteLLM
│
├── 04_streaming_in_agent_loops/
│   ├── theory.md                                ← Streaming while executing tools, multi-turn
│   └── examples.ipynb                           ← Agent that streams and calls tools mid-response
│
├── 05_streaming_tool_calls/
│   ├── theory.md                                ← Partial tool call accumulation, JSON assembly
│   └── examples.ipynb                           ← Detect + execute tool calls during stream
│
├── 06_async_streaming/
│   ├── theory.md                                ← AsyncOpenAI, asyncio, websocket patterns
│   └── examples.ipynb                           ← Async streaming for concurrent agent tasks
│
└── 07_production_streaming/
    ├── theory.md                                ← FastAPI SSE, backpressure, error recovery
    └── examples.ipynb                           ← StreamingRouter — production class
```

---

## 📚 Topics Covered

| # | Topic | Core Question Answered |
|---|---|---|
| 1 | `01_streaming_basics` | What is streaming and how does it work? |
| 2 | `02_openai_streaming` | How do I stream with the OpenAI API? |
| 3 | `03_streaming_across_providers` | How do I stream across Anthropic, Google, and others? |
| 4 | `04_streaming_in_agent_loops` | How do I integrate streaming into an agent's reasoning loop? |
| 5 | `05_streaming_tool_calls` | How do I handle tool calls that arrive mid-stream? |
| 6 | `06_async_streaming` | How do I stream multiple responses concurrently? |
| 7 | `07_production_streaming` | How do I expose streaming via a FastAPI endpoint? |

---

## 🔑 Core Architecture

```
LLM API (OpenAI / Anthropic / Gemini)
    │
    │  HTTP/2  Server-Sent Events (SSE)
    │  ─────────────────────────────────
    │  data: {"delta": {"content": "The"}}
    │  data: {"delta": {"content": " answer"}}
    │  data: {"delta": {"content": " is..."}}
    │  data: [DONE]
    │
    ▼
Stream Handler
    ├── Accumulate text chunks → display to user
    ├── Watch for tool_call deltas → assemble JSON → execute tool
    ├── Detect stop reason → finish or continue loop
    └── Handle errors mid-stream → retry / fallback
```

---

## 🔧 Setup

```bash
pip install openai anthropic litellm python-dotenv rich fastapi uvicorn
```

```env
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

---

## ⚡ Quick Preview

```python
# Minimal streaming example
from openai import OpenAI

client = OpenAI()
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Count to 5 slowly"}],
    stream=True   # ← This is all it takes
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```
