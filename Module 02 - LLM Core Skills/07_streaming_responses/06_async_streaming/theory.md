# 06 — Async Streaming

> *Stream multiple LLM responses concurrently with Python asyncio.*

---

## 6.1 Why Async Streaming?

Synchronous streaming processes one request at a time:

```
Sync: Request A [████████████] Request B [████████████]
Time: 0s                      8s                       16s
```

Async streaming overlaps I/O wait time:

```
Async: Request A [████████████]
       Request B   [████████████]
       Request C     [████████████]
Time:  0s                        10s  (total, not 24s)
```

Use async when:
- **Multi-agent systems** — multiple sub-agents running simultaneously
- **Batch processing with live output** — stream 10+ queries in parallel
- **Web APIs** — serve many users concurrently with `FastAPI` + `asyncio`
- **Agent fan-out** — one orchestrator spawns multiple parallel agents

---

## 6.2 AsyncOpenAI Client

The `AsyncOpenAI` client is the async equivalent of `OpenAI`:

```python
from openai import AsyncOpenAI
import asyncio

async_client = AsyncOpenAI()

async def async_stream(prompt: str) -> str:
    """Async streaming — identical to sync but with async/await."""
    full_text = ""

    stream = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    async for chunk in stream:   # ← async for instead of for
        content = chunk.choices[0].delta.content or ""
        full_text += content
        print(content, end="", flush=True)

    print()
    return full_text

# Run it
result = asyncio.run(async_stream("What is asyncio?"))
```

---

## 6.3 Concurrent Streaming with asyncio.gather

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def stream_one(prompt: str, label: str) -> dict:
    """Stream a single prompt, prefixing output with its label."""
    import time
    start = time.perf_counter()
    full_text = ""

    stream = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True, max_tokens=100
    )

    async for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        full_text += content

    latency = (time.perf_counter() - start) * 1000
    return {"label": label, "text": full_text, "ms": round(latency, 1)}


async def parallel_streams(prompts: list[tuple[str, str]]) -> list[dict]:
    """Run all prompts concurrently, return all results."""
    tasks = [stream_one(prompt, label) for prompt, label in prompts]
    return await asyncio.gather(*tasks)


# Run 4 queries in parallel
prompts = [
    ("Explain streaming in one sentence.", "streaming"),
    ("What is asyncio in one sentence?",    "asyncio"),
    ("What is Python in one sentence?",     "python"),
    ("What is an LLM in one sentence?",     "llm"),
]

results = asyncio.run(parallel_streams(prompts))
for r in results:
    print(f"{r['label']}: {r['ms']}ms | {r['text'][:80]}")
```

---

## 6.4 Async Streaming Agent Loop

```python
async def async_streaming_agent(user_message: str, tools: list, executor: dict) -> str:
    """Async version of the streaming agent loop."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant with tools."},
        {"role": "user",   "content": user_message}
    ]

    for step in range(6):
        full_text = ""
        tool_calls_acc = {}
        finish_reason = None

        stream = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages, tools=tools, stream=True
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason or finish_reason

            if delta.content:
                full_text += delta.content
                print(delta.content, end="", flush=True)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:                       tool_calls_acc[idx]["id"]        += tc.id
                    if tc.function.name:            tool_calls_acc[idx]["name"]      += tc.function.name
                    if tc.function.arguments:       tool_calls_acc[idx]["arguments"] += tc.function.arguments

        print()
        if finish_reason == "stop":
            messages.append({"role": "assistant", "content": full_text})
            return full_text

        if finish_reason == "tool_calls":
            tc_list = [{
                "id": tc["id"], "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]}
            } for tc in tool_calls_acc.values()]

            messages.append({"role": "assistant", "content": full_text or None, "tool_calls": tc_list})

            # Execute tools concurrently if there are multiple
            async def execute_tool(tc):
                fn = executor.get(tc["name"])
                args = json.loads(tc["arguments"] or "{}")
                return tc["id"], str(fn(**args)) if fn else "tool not found"

            tool_results = await asyncio.gather(*[
                execute_tool(tc) for tc in tool_calls_acc.values()
            ])

            for call_id, result in tool_results:
                messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

    return "Max steps reached."
```

---

## 6.5 Streaming with Timeouts

```python
import asyncio
from openai import AsyncOpenAI

async def stream_with_timeout(prompt: str, timeout_s: float = 10.0) -> str:
    """Stream with a hard timeout — cancel if it takes too long."""
    async_client = AsyncOpenAI()

    async def _stream():
        text = ""
        stream = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        async for chunk in stream:
            c = chunk.choices[0].delta.content or ""
            text += c
            print(c, end="", flush=True)
        print()
        return text

    try:
        return await asyncio.wait_for(_stream(), timeout=timeout_s)
    except asyncio.TimeoutError:
        print(f"\n⚠️ Timeout after {timeout_s}s — partial response returned")
        return ""
```

---

## 6.6 Queue-Based Streaming (Producer-Consumer)

For web APIs that need to push tokens to a websocket or SSE endpoint:

```python
import asyncio, queue

async def stream_to_queue(prompt: str, q: asyncio.Queue):
    """Producer: stream tokens into a queue."""
    stream = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    async for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        if content:
            await q.put(content)  # Push to queue
    await q.put(None)  # Sentinel — signals end of stream


async def consumer(q: asyncio.Queue, websocket=None):
    """Consumer: read from queue and send to client."""
    while True:
        token = await q.get()
        if token is None:
            break  # Stream ended
        if websocket:
            await websocket.send_text(token)  # Push to WebSocket
        else:
            print(token, end="", flush=True)  # Demo: print it
    print()
```

---

## 📌 Key Takeaways

1. **`AsyncOpenAI`** = drop-in async replacement for `OpenAI`
2. **`async for chunk in stream`** replaces `for chunk in stream`
3. **`asyncio.gather()`** = run multiple streams concurrently — critical for multi-agent workloads
4. **Multiple parallel streams** are much faster than sequential for independent tasks
5. **`asyncio.wait_for()`** = hard timeout on any async operation
6. **Queue pattern** = decouple streaming from consumption (ideal for WebSocket/SSE servers)
7. **Use async agents** when your framework (FastAPI, aiohttp) is already async
