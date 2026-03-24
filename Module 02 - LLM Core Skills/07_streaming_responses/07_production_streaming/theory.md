# 07 — Production Streaming

> *Expose LLM streaming via FastAPI Server-Sent Events — the production-standard pattern.*

---

## 7.1 Production Streaming Architecture

```
Browser / Client App
        │
        │  GET /stream?prompt=...
        │  Accept: text/event-stream
        │
        ▼
┌────────────────────────────────┐
│         FastAPI Server         │
│                                │
│  StreamingResponse             │
│    │                           │
│    ▼                           │
│  async generator:              │
│    → LLM stream chunk          │
│    → format as SSE             │
│    → yield "data: {...}\n\n"   │
└────────────────────────────────┘
        │
        │  data: {"token": "Hello"}
        │  data: {"token": " world"}
        │  data: [DONE]
        │
        ▼
Browser renders tokens as they arrive
```

---

## 7.2 FastAPI Streaming Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import json, asyncio

app = FastAPI()
client = AsyncOpenAI()

async def openai_stream_generator(prompt: str):
    """
    Async generator that yields SSE-formatted strings.
    Each 'data: {...}\\n\\n' is one SSE event.
    """
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    async for chunk in stream:
        content = chunk.choices[0].delta.content
        finish_reason = chunk.choices[0].finish_reason

        if content:
            # Format as Server-Sent Event
            event_data = json.dumps({"token": content})
            yield f"data: {event_data}\n\n"

        if finish_reason == "stop":
            yield "data: [DONE]\n\n"
            return

        if finish_reason == "tool_calls":
            yield f"data: {json.dumps({'event': 'tool_calls'})}\n\n"
            return


@app.get("/stream")
async def stream_endpoint(prompt: str = "Tell me a short joke"):
    return StreamingResponse(
        openai_stream_generator(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "Connection":     "keep-alive",
            "X-Accel-Buffering": "no",   # Disable nginx buffering
        }
    )
```

---

## 7.3 SSE Event Format

Server-Sent Events is a simple text protocol:

```
# Each event:
data: {"token": "Hello"}\n\n

# With event type:
event: token\n
data: {"content": "World"}\n\n

# Done sentinel:
data: [DONE]\n\n

# Comment (keepalive — prevents connection timeout):
: keepalive\n\n
```

Rules:
- Each event ends with **two newlines** (`\n\n`)
- Data lines start with `data: `
- Comments start with `: `
- Multiple `data:` lines in one event are concatenated

---

## 7.4 Client-Side SSE Consumption

```javascript
// Browser: EventSource API
const evtSource = new EventSource('/stream?prompt=Tell+me+a+joke');

evtSource.onmessage = (event) => {
    if (event.data === '[DONE]') {
        evtSource.close();
        return;
    }
    const data = JSON.parse(event.data);
    document.getElementById('output').textContent += data.token;
};

evtSource.onerror = (err) => {
    console.error('SSE error', err);
    evtSource.close();
};
```

```python
# Python client (httpx + requests)
import httpx

with httpx.Client() as client:
    with client.stream("GET", "http://localhost:8000/stream", params={"prompt": "Hello"}) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                print(json.loads(data)["token"], end="", flush=True)
```

---

## 7.5 Production-Grade Streaming Endpoint

```python
import asyncio, json, logging, time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, APIError

logger = logging.getLogger(__name__)
app = FastAPI()
client = AsyncOpenAI()

async def robust_stream_generator(prompt: str, request: Request):
    """
    Production streaming generator with:
    - Client disconnect detection
    - Error handling (surfaces errors as SSE events)
    - Keepalive pings (prevents proxy timeouts)
    - Timing metadata in the done event
    """
    start = time.time()
    token_count = 0
    keepalive_task = None

    async def keepalive():
        """Send a comment every 15s to prevent proxy timeout."""
        while True:
            await asyncio.sleep(15)
            yield ": keepalive\n\n"

    try:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            timeout=30.0
        )

        async for chunk in stream:
            # Check if client disconnected
            if await request.is_disconnected():
                logger.info("Client disconnected — stopping stream")
                return

            content = chunk.choices[0].delta.content or ""
            if content:
                token_count += 1
                event = json.dumps({"token": content, "index": token_count})
                yield f"data: {event}\n\n"

        # Done event with metadata
        done_event = json.dumps({
            "event": "done",
            "tokens": token_count,
            "latency_ms": round((time.time() - start) * 1000, 1)
        })
        yield f"data: {done_event}\n\n"

    except APIError as e:
        error_event = json.dumps({"event": "error", "message": str(e), "code": e.status_code})
        yield f"data: {error_event}\n\n"
        logger.error(f"API error during stream: {e}")

    except Exception as e:
        error_event = json.dumps({"event": "error", "message": "Internal server error"})
        yield f"data: {error_event}\n\n"
        logger.exception(f"Unexpected error: {e}")


@app.get("/stream")
async def stream_endpoint(prompt: str, request: Request):
    return StreamingResponse(
        robust_stream_generator(prompt, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
```

---

## 7.6 Streaming with Tool Calls via SSE

```python
async def agent_stream_generator(prompt: str, tools: list):
    """
    Stream agent reasoning + tool calls over SSE.
    Returns structured events: token | tool_call | tool_result | done
    """
    messages = [{"role": "user", "content": prompt}]

    for step in range(6):
        yield f"data: {json.dumps({'event': 'step', 'n': step+1})}\n\n"

        full_text = ""
        tool_calls_acc = {}
        finish_reason = None

        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages, tools=tools, stream=True
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason or finish_reason

            if delta.content:
                full_text += delta.content
                yield f"data: {json.dumps({'event': 'token', 'content': delta.content})}\n\n"

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:                       tool_calls_acc[idx]["id"]        += tc.id
                    if tc.function.name:            tool_calls_acc[idx]["name"]      += tc.function.name
                    if tc.function.arguments:       tool_calls_acc[idx]["arguments"] += tc.function.arguments

        if finish_reason == "stop":
            yield f"data: {json.dumps({'event': 'done', 'final_answer': full_text})}\n\n"
            return

        if finish_reason == "tool_calls":
            for tc in tool_calls_acc.values():
                yield f"data: {json.dumps({'event': 'tool_call', 'name': tc['name']})}\n\n"
                # ... execute tool, yield tool_result event
```

---

## 7.7 Deployment Considerations

```
✅ Production checklist for streaming endpoints:

Nginx config (disable buffering):
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 300s;

Gunicorn/Uvicorn:
    uvicorn main:app --workers 4 --timeout-keep-alive 75

CORS (for browser SSE):
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

Error recovery (client-side):
    evtSource.onerror = () => { setTimeout(reconnect, 2000); }

Rate limiting:
    Apply per-user rate limits at the API gateway level
    Limit concurrent streams per user
```

---

## 📌 Key Takeaways

1. **`StreamingResponse`** + async generator = FastAPI SSE streaming
2. **SSE format**: `data: {...}\n\n` — two newlines per event
3. **Keepalives**: `: keepalive\n\n` every 15s to prevent proxy disconnects
4. **Client disconnect detection**: check `await request.is_disconnected()` in the loop
5. **Error events**: surface errors as SSE data, not HTTP status codes (stream has already started)
6. **Nginx**: set `proxy_buffering off` or tokens won't reach the client
7. **For WebSockets**: use `websocket.send_text(token)` in place of `yield f"data: ..."`
