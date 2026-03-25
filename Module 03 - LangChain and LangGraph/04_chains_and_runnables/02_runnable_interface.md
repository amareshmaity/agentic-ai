# The Runnable Interface

> *Every component in LangChain implements the Runnable interface — one set of methods that works for all chains, models, prompts, parsers, and retrievers.*

---

## 🔌 The Universal Interface

All LangChain components — models, prompts, parsers, retrievers, chains — share the exact same Runnable interface:

```python
component.invoke(input)             # Single synchronous call
component.stream(input)             # Generator, yields chunks
component.batch([i1, i2, i3])      # Parallel processing of multiple inputs
component.ainvoke(input)            # Async single call
component.astream(input)            # Async streaming generator
component.abatch([i1, i2, i3])     # Async parallel processing
```

This is what makes LCEL composition work — **every step in the chain uses the same contract**.

---

## 1️⃣ `.invoke()` — Single Synchronous Call

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("What is {topic}?")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# Basic invoke — blocks until complete
result = chain.invoke({"topic": "LangChain"})
print(type(result))   # str
print(result)         # "LangChain is a framework..."

# With configuration
result = chain.invoke(
    {"topic": "LangGraph"},
    config={
        "run_name": "my-run",                    # Name in LangSmith
        "tags": ["production", "user-123"],       # Filter in LangSmith
        "metadata": {"user_id": "123"},           # Attach metadata
        "max_concurrency": 4,                     # For batch calls
    }
)
```

---

## 2️⃣ `.stream()` — Token-by-Token Streaming

```python
# stream() yields chunks — for models it's one token at a time
for chunk in chain.stream({"topic": "LCEL"}):
    print(chunk, end="", flush=True)  # Print without newline
print()

# What each chunk looks like for different components:
# StrOutputParser → str fragment: "Lang", "Chain", " is", " a", ...
# PydanticParser  → partial object: {"name": None} → {"name": "Alice"}
# JsonOutputParser → partial dict building up
```

### Streaming in Practice (Web Backend)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/chat")
async def chat(question: str):
    async def generate():
        async for chunk in chain.astream({"topic": question}):
            yield chunk  # Send each token to client

    return StreamingResponse(generate(), media_type="text/plain")
```

---

## 3️⃣ `.batch()` — Parallel Processing

```python
# Process multiple inputs in parallel (default: max_concurrency=5)
inputs = [
    {"topic": "LangChain"},
    {"topic": "LangGraph"},
    {"topic": "LangSmith"},
    {"topic": "LangFlow"},
]

# All 4 processed concurrently!
results = chain.batch(inputs)
for topic, result in zip(["LangChain", "LangGraph", "LangSmith", "LangFlow"], results):
    print(f"{topic}: {result[:50]}...")

# Control concurrency
results = chain.batch(inputs, config={"max_concurrency": 2})  # Max 2 at a time
```

### Batch vs Sequential Speed

```python
import time

inputs = [{"topic": f"concept_{i}"} for i in range(10)]

# Sequential (slow)
start = time.time()
sequential = [chain.invoke(inp) for inp in inputs]
print(f"Sequential: {time.time() - start:.2f}s")

# Batch/parallel (fast)
start = time.time()
parallel = chain.batch(inputs)
print(f"Parallel:   {time.time() - start:.2f}s")
# Parallel is often 3-5x faster!
```

---

## 4️⃣ `.ainvoke()`, `.astream()`, `.abatch()` — Async

```python
import asyncio

async def main():
    # ainvoke — single async call
    result = await chain.ainvoke({"topic": "async LangChain"})
    print(result)

    # astream — async token streaming
    async for chunk in chain.astream({"topic": "streaming"}):
        print(chunk, end="", flush=True)
    print()

    # abatch — async parallel processing
    results = await chain.abatch([
        {"topic": "Python"},
        {"topic": "LangChain"},
        {"topic": "AI"},
    ])
    print(f"Processed {len(results)} items async")

asyncio.run(main())
```

> **When to use async?** Use async (`ainvoke`, `astream`, `abatch`) when:
> - Building web APIs (FastAPI, Django async views)
> - Handling many concurrent users
> - Chaining I/O-bound operations like searches, DB calls

---

## 5️⃣ Schema Inspection

Every Runnable exposes its input and output schema:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# What does this chain accept?
print(chain.input_schema.schema())
# {'properties': {'question': {'title': 'Question', 'type': 'string'}},
#  'required': ['question'], 'title': 'PromptInput', 'type': 'object'}

# What does it return?
print(chain.output_schema.schema())
# {'title': 'StrOutputParserOutput', 'type': 'string'}
```

---

## 6️⃣ Chain Visualization

```python
# Visualize the chain as ASCII art
chain.get_graph().print_ascii()

# Output:
#        +-----------------------+
#        | ChatPromptTemplate    |
#        +-----------------------+
#                   *
#                   *
#        +-----------------------+
#        | ChatOpenAI            |
#        +-----------------------+
#                   *
#                   *
#        +-----------------------+
#        | StrOutputParser       |
#        +-----------------------+
```

---

## 7️⃣ `.with_retry()` — Automatic Retry

```python
# Auto-retry on failure (e.g., rate limit errors)
resilient_chain = chain.with_retry(
    retry_if_exception_type=(Exception,),  # Retry on any exception
    stop_after_attempt=3,                  # Max 3 attempts
    wait_exponential_jitter=True,          # Jitter between retries
)

result = resilient_chain.invoke({"topic": "AI"})
```

---

## 8️⃣ `.with_fallbacks()` — Fallback Chain

```python
# If primary chain fails, try backup chains in order
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

primary_chain   = ChatPromptTemplate.from_template("{topic}") | ChatOpenAI(model="gpt-4o")
fallback_chain1 = ChatPromptTemplate.from_template("{topic}") | ChatAnthropic(model="claude-3-haiku-20240307")
fallback_chain2 = ChatPromptTemplate.from_template("{topic}") | ChatOpenAI(model="gpt-4o-mini")

chain_with_fallback = primary_chain.with_fallbacks(
    [fallback_chain1, fallback_chain2]
)

# Tries primary_chain first → if fails, tries fallback_chain1 → then fallback_chain2
result = chain_with_fallback.invoke({"topic": "What is LangChain?"})
```

---

## 9️⃣ `.bind()` — Pre-bind Arguments

```python
# Bind static arguments to a component
llm = ChatOpenAI(model="gpt-4o-mini")

# Bind stop sequence — chain will always stop at these tokens
llm_with_stop = llm.bind(stop=["END", "STOP"])

# Bind tools
llm_with_tools = llm.bind_tools([my_tool_1, my_tool_2])

# Use in chain as normal
chain = prompt | llm_with_stop | StrOutputParser()
```

---

## ✅ Key Takeaways

| Method | Use When |
|---|---|
| `.invoke()` | Single synchronous call — most common |
| `.stream()` | Streaming tokens to UI |
| `.batch()` | Processing multiple inputs — 3-5x faster |
| `.ainvoke()` | Async web APIs (FastAPI) |
| `.astream()` | Async streaming (WebSocket, SSE) |
| `.with_retry()` | Brittle APIs, rate limits |
| `.with_fallbacks()` | High availability, multi-provider |
| `.bind()` | Pre-configure model args |

---

## ➡️ Next
[RunnableParallel →](./03_runnable_parallel.md)
