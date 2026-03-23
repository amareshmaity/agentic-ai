# 04 — Parallel & Multi-Tool Calls

> *How to run multiple tools simultaneously — cutting agent latency from minutes to seconds.*

---

## 4.1 What Are Parallel Tool Calls?

By default, an LLM agent runs tools **sequentially**:

```
Step 1: Call web_search("company A")     → wait 2s
Step 2: Call web_search("company B")     → wait 2s
Step 3: Call web_search("company C")     → wait 2s
Total: 6 seconds
```

With **parallel tool calls**, the LLM requests multiple tools in a single response, and you execute them **concurrently**:

```
Step 1: LLM requests all 3 searches simultaneously
Step 2: Execute all 3 in parallel               → wait 2s (longest)
Step 3: Inject all 3 results back at once
Total: 2 seconds  ← 3× faster
```

OpenAI enabled parallel tool calls by default since GPT-4 Turbo.

---

## 4.2 How Parallel Tool Calls Work

When `parallel_tool_calls=True` (the default), the LLM may return **multiple tool calls** in a single response:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=TOOLS,
    parallel_tool_calls=True   # True by default
)

msg = response.choices[0].message
print(f"Number of tool calls: {len(msg.tool_calls)}")

# If parallel: msg.tool_calls contains 2+ items
# [
#   ToolCall(id="call_001", function=Function(name="web_search", arguments='{"query": "Company A"}')),
#   ToolCall(id="call_002", function=Function(name="web_search", arguments='{"query": "Company B"}')),
#   ToolCall(id="call_003", function=Function(name="web_search", arguments='{"query": "Company C"}'))
# ]
```

---

## 4.3 Executing Parallel Calls Concurrently in Python

```python
import asyncio
import json
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def execute_tool_async(tool_call, tool_map: dict) -> dict:
    """Execute a single tool call asynchronously."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    # Execute the tool (use async version if available)
    if asyncio.iscoroutinefunction(tool_map[name]):
        result = await tool_map[name](**args)
    else:
        # Run sync functions in thread pool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: tool_map[name](**args)
        )
    
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result)
    }

async def run_parallel_tools(msg, tool_map: dict) -> list[dict]:
    """Execute all tool calls in msg in parallel, return all results."""
    if not msg.tool_calls:
        return []
    
    # Fire all tool calls concurrently
    tasks = [execute_tool_async(tc, tool_map) for tc in msg.tool_calls]
    results = await asyncio.gather(*tasks)
    return list(results)

# Full async agent loop
async def agent_loop_async(user_input: str) -> str:
    messages = [
        {"role": "system", "content": "You are a research agent. Use tools in parallel when researching multiple topics."},
        {"role": "user",   "content": user_input}
    ]
    
    for step in range(5):
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            parallel_tool_calls=True
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        if response.choices[0].finish_reason == "tool_calls":
            print(f"Step {step+1}: Executing {len(msg.tool_calls)} tool(s) in parallel")
            tool_results = await run_parallel_tools(msg, TOOL_MAP)
            messages.extend(tool_results)
        else:
            return msg.content
    
    return "Max steps reached"
```

---

## 4.4 The Order of Tool Results Matters

When injecting parallel tool results back, each result must reference its `tool_call_id`. The order doesn't need to be sequential — the LLM matches by ID:

```python
# ✅ Correct: each result has matching tool_call_id
messages.append({
    "role": "tool",
    "tool_call_id": "call_001",  # ← matches the specific call
    "content": "Company A: Revenue $500M"
})
messages.append({
    "role": "tool",
    "tool_call_id": "call_003",  # ← out of order is fine
    "content": "Company C: Revenue $200M"
})
messages.append({
    "role": "tool",
    "tool_call_id": "call_002",
    "content": "Company B: Revenue $350M"
})
```

**Rule**: Every `tool_call_id` in the assistant message must have exactly one matching tool result before the next LLM call.

---

## 4.5 When to Use vs Avoid Parallel Calls

### Use Parallel Calls When
```
✅ Independent lookups: "Compare prices of X, Y, and Z"
✅ Multi-source research: "Search web + query database + check cache"
✅ Fan-out: "Get data for 5 different companies at once"
✅ Parallel enrichment: "Look up user profile, order history, and preferences simultaneously"
```

### Avoid Parallel Calls When
```
❌ Dependent operations: search result is needed to build the next query
❌ Write operations: two simultaneous writes to the same record = race condition
❌ Rate-limited APIs: parallel calls hit rate limits faster
❌ Stateful sequences: step B requires the result of step A
```

---

## 4.6 Chained Tool Calls — Sequential by Design

Some tasks require the output of one tool to feed the next. This is **sequential (chained)** calling:

```
Step 1: search_web("latest AI papers 2025")
   ↓ result: list of paper URLs
Step 2: read_url(url_1) + read_url(url_2)  ← parallel! both URLs at once
   ↓ results: full paper content for both
Step 3: summarize(paper_1_content + paper_2_content)
   ↓ result: combined summary
```

```python
async def chained_agent(goal: str) -> str:
    """Example of mixed sequential + parallel tool calls."""
    
    # Step 1: Sequential — must search first to get URLs
    urls = await search_web(goal)
    
    # Step 2: Parallel — read all URLs simultaneously
    contents = await asyncio.gather(*[read_url(url) for url in urls[:3]])
    
    # Step 3: Sequential — summarize all content together
    summary = await summarize("\n\n".join(contents))
    
    return summary
```

---

## 4.7 Disabling Parallel Tool Calls

Sometimes you want to force sequential tool calls (e.g., to avoid race conditions or to simplify debugging):

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=TOOLS,
    parallel_tool_calls=False  # Force sequential, one at a time
)
```

---

## 4.8 Fan-Out / Fan-In Pattern — Scaling Research

The fan-out/fan-in pattern scales research tasks across many parallel agents:

```
                        USER GOAL
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
              Decompose into        ...
              parallel tasks
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    Task A      Task B      Task C    ← All run in parallel (fan-out)
    (search)   (database)  (API call)
        │          │          │
        └──────────┴──────────┘
                   │
                Fan-in: merge results
                   │
             Final synthesis
```

```python
async def fan_out_research(topics: list[str]) -> dict:
    """Research multiple topics in parallel and aggregate."""
    
    async def research_one(topic: str) -> dict:
        result = await search_web(f"{topic} latest 2025")
        return {"topic": topic, "result": result}
    
    # Fan-out: all topics researched simultaneously
    results = await asyncio.gather(*[research_one(t) for t in topics])
    
    # Fan-in: aggregate
    combined = {r["topic"]: r["result"] for r in results}
    return combined

# Usage
topics = ["LangChain", "CrewAI", "AutoGen", "Agno", "LangFlow"]
all_data = await fan_out_research(topics)
```

---

## 📌 Key Takeaways

1. **Parallel tool calls** = LLM returns multiple tool calls in one response → execute simultaneously
2. **3–5× speed improvement** for independent lookups vs sequential
3. **`tool_call_id` must match** — every call must have exactly one result injected before next LLM call
4. **asyncio.gather()** = Python's mechanism for true parallel execution
5. **Sequential for dependent tasks, parallel for independent** — mix as needed
6. **Disable with `parallel_tool_calls=False`** when write safety or ordering matters
7. **Fan-out/Fan-in** is the pattern for scaling research to many parallel tasks
