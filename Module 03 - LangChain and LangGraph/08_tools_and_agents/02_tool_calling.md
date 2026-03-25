# Tool Calling

> *Tool calling is not magic — it's the LLM returning a JSON object that says "please call this function with these arguments." Your code does the actual running.*

---

## 🤔 How Tool Calling Works

Tool calling is a **two-step process**:

```
Step 1 — LLM decides to use a tool:
  User input → LLM → "I need to call get_weather(city='Paris')"
  LLM outputs a structured ToolCall object (not a text answer)

Step 2 — Your code runs the tool:
  ToolCall → execute function → get result → feed back to LLM
  LLM → final natural language answer
```

The LLM never runs code directly. It only outputs a **decision**. Your application loop executes the actual function.

---

## 🔗 Binding Tools to an LLM

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 22°C, sunny"

@tool
def get_time(timezone: str) -> str:
    """Get the current time in a given timezone."""
    return f"Current time in {timezone}: 14:35"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Bind tools to the LLM — now it knows these tools exist
llm_with_tools = llm.bind_tools([get_weather, get_time])
```

---

## 📤 Understanding Tool Call Responses

```python
from langchain_core.messages import HumanMessage

# Ask something that requires a tool
response = llm_with_tools.invoke([
    HumanMessage(content="What's the weather in Tokyo?")
])

print(type(response))          # AIMessage
print(response.content)        # "" (empty — LLM is returning a tool call, not text)
print(response.tool_calls)
# [
#   {
#     'name': 'get_weather',
#     'args': {'city': 'Tokyo'},
#     'id': 'call_abc123'
#   }
# ]
```

> When a tool call is made, `response.content` is empty and `response.tool_calls` contains the decisions.

---

## 🔄 The Full Tool Calling Loop (Manual)

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

tools = [get_weather, get_time]
tools_by_name = {t.name: t for t in tools}

messages = [HumanMessage(content="What's the weather in Tokyo and Paris?")]

# ── Round 1: LLM decides ──────────────────────────────────
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

print(f"Tool calls: {ai_msg.tool_calls}")
# [{'name': 'get_weather', 'args': {'city': 'Tokyo'}, 'id': 'call_1'},
#  {'name': 'get_weather', 'args': {'city': 'Paris'}, 'id': 'call_2'}]

# ── Round 2: Execute tools ────────────────────────────────
for tool_call in ai_msg.tool_calls:
    tool = tools_by_name[tool_call["name"]]
    result = tool.invoke(tool_call["args"])
    
    messages.append(
        ToolMessage(
            content=result,
            tool_call_id=tool_call["id"]   # Links result back to the tool call
        )
    )

# ── Round 3: LLM synthesizes final answer ─────────────────
final = llm_with_tools.invoke(messages)
print(final.content)
# "The weather in Tokyo is 22°C and sunny. In Paris, it is also 22°C and sunny."
```

---

## 🛠️ Tool Choice Control

You can control how the LLM selects tools:

```python
# Auto (default) — LLM decides whether to use tools
llm_auto = llm.bind_tools(tools, tool_choice="auto")

# Required — LLM MUST use a tool (at least one)
llm_required = llm.bind_tools(tools, tool_choice="required")

# None — LLM must NOT use any tool (just text)
llm_none = llm.bind_tools(tools, tool_choice="none")

# Specific — Force a specific tool
llm_specific = llm.bind_tools(tools, tool_choice="get_weather")
```

---

## ⚡ Parallel Tool Calling

Modern LLMs can call multiple tools in a single response:

```python
response = llm_with_tools.invoke([
    HumanMessage("What's the weather in London, Paris, and Tokyo?")
])

# LLM returns ALL three tool calls at once — not one at a time
print(len(response.tool_calls))  # 3

# Execute them all (could parallelize with asyncio)
for call in response.tool_calls:
    tool = tools_by_name[call["name"]]
    result = tool.invoke(call["args"])
    print(f"{call['args']['city']}: {result}")
```

---

## 🔍 Checking if a Response Has Tool Calls

```python
response = llm_with_tools.invoke(messages)

if response.tool_calls:
    # LLM wants to use tools
    print(f"Tool calls: {len(response.tool_calls)}")
    for call in response.tool_calls:
        print(f"  → {call['name']}({call['args']})")
else:
    # LLM answered directly without tools
    print(f"Direct answer: {response.content}")
```

---

## 🧱 Tool Calling with LCEL

LCEL makes tool binding clean and chainable:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    ("human", "{input}")
])

# Simple chain — returns AIMessage with potential tool calls
chain = prompt | llm_with_tools

# Invoke
result = chain.invoke({"input": "What's the weather in New York?"})
print(result.tool_calls)
```

---

## 📊 Tool Calling vs Function Calling

| Term | Meaning |
|---|---|
| **Function calling** | OpenAI API term (the underlying API feature) |
| **Tool calling** | LangChain term (abstracted over all providers) |
| **Tool use** | Anthropic term (same concept) |

LangChain normalizes all these into the same `tool_calls` interface — your code works the same regardless of provider (OpenAI, Anthropic, Google, etc.).

---

## ✅ Key Takeaways

- `llm.bind_tools(tools)` → tells the LLM which tools exist
- LLM responds with `tool_calls` list (not text) when it wants to use a tool
- Your code executes the tools and returns `ToolMessage` results
- The LLM then synthesizes the final answer using all tool results
- Use `tool_choice="required"` to force tool use, `"none"` to disable
- Parallel tool calling = one LLM call → multiple tool decisions at once

---

## ➡️ Next
[ReAct Pattern →](./03_react_pattern.md)
