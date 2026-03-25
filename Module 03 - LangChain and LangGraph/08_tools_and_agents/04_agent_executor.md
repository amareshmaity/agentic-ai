# AgentExecutor

> *AgentExecutor is LangChain's classic agent runner — it manages the ReAct loop, tool execution, error handling, and iteration limits so you don't have to.*

---

## 🤔 What is AgentExecutor?

`AgentExecutor` is the **runtime** for LangChain agents. It:

1. Runs the **ReAct loop** automatically
2. Executes **tool calls** returned by the LLM
3. Feeds **tool results** back into the conversation
4. Handles **errors**, **retries**, and **max iteration** limits
5. Returns the **final answer**

```
You write:       AgentExecutor handles:
──────────       ────────────────────────
Define tools  →  Looping until done
Define LLM    →  Calling each tool
Invoke once   →  Managing message history
              →  Error handling
              →  Iteration limits
              →  Returning final output
```

---

## 🏗️ Building an AgentExecutor

### Step 1 — Define Your Tools

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the internet for current, real-time information.
    Use for: recent events, live data, specific facts.
    """
    # In production, use Tavily or SerpAPI
    return f"Search results for '{query}': [relevant articles and data]"

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use for math calculations."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"

@tool
def get_weather(city: str) -> str:
    """Get current weather conditions for a city."""
    return f"Weather in {city}: 22°C, partly cloudy, humidity 65%"

tools = [search_web, calculator, get_weather]
```

---

### Step 2 — Create the Agent (LLM + Prompt + Tools)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# AgentExecutor works best with this prompt structure
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools. "
               "Think step by step before using any tool."),
    MessagesPlaceholder("chat_history", optional=True),  # For memory
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),             # Tool calling happens here
])

# Create the agent (LLM + tools + prompt, no loop yet)
agent = create_tool_calling_agent(llm, tools, prompt)
```

---

### Step 3 — Wrap in AgentExecutor (Adds the Loop)

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # Print reasoning steps
    max_iterations=10,     # Stop after 10 tool calls max
    handle_parsing_errors=True,  # Don't crash on malformed tool calls
    return_intermediate_steps=True,  # Return full trace
)

# Run it!
result = agent_executor.invoke({
    "input": "What is the weather in Paris, and what is 15% of 200?"
})

print(result["output"])
```

---

## 📊 What `verbose=True` Shows

When you enable verbose mode, you see every step:

```
> Entering new AgentExecutor chain...

Invoking: `get_weather` with `{'city': 'Paris'}`

Weather in Paris: 18°C, cloudy, humidity 70%

Invoking: `calculator` with `{'expression': '0.15 * 200'}`

30.0

The weather in Paris is 18°C and cloudy.
15% of 200 is 30.

> Finished chain.
```

---

## 🔁 AgentExecutor Configuration Options

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    
    # ── Control ────────────────────────────────────
    max_iterations=15,             # Max tool calls before stopping
    max_execution_time=30.0,       # Timeout in seconds
    early_stopping_method="force", # "force" = return current answer; "generate" = ask LLM to wrap up
    
    # ── Error Handling ──────────────────────────────
    handle_parsing_errors=True,    # Recover from malformed tool call JSON
    
    # ── Output ─────────────────────────────────────
    verbose=True,                  # Print step-by-step
    return_intermediate_steps=True,# Include tool calls in output dict
)
```

---

## 💾 Adding Memory to AgentExecutor

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# AgentExecutor with input/output tracking
agent_executor_base = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Wrap with message history
agent_with_memory = RunnableWithMessageHistory(
    agent_executor_base,
    get_session_history=lambda session_id: InMemoryChatMessageHistory(),
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Session 1 — multi-turn conversation
config = {"configurable": {"session_id": "user-123"}}

resp1 = agent_with_memory.invoke({"input": "My name is Alex."}, config=config)
print(resp1["output"])  # "Nice to meet you, Alex!"

resp2 = agent_with_memory.invoke({"input": "What is my name?"}, config=config)
print(resp2["output"])  # "Your name is Alex."
```

---

## 📤 Reading Intermediate Steps

```python
result = agent_executor.invoke({"input": "What's 5 + 3 and then double it?"})

# result["intermediate_steps"] = list of (AgentAction, tool_output) tuples
for action, observation in result["intermediate_steps"]:
    print(f"Tool: {action.tool}")
    print(f"Input: {action.tool_input}")
    print(f"Output: {observation}")
    print("---")

print(f"Final answer: {result['output']}")
```

---

## ⚡ Streaming with AgentExecutor

```python
# Stream intermediate steps as they happen
for event in agent_executor.stream({"input": "What's the weather in Tokyo?"}):
    if "actions" in event:
        for action in event["actions"]:
            print(f"Calling: {action.tool}({action.tool_input})")
    elif "steps" in event:
        for step in event["steps"]:
            print(f"Result: {step.observation}")
    elif "output" in event:
        print(f"Answer: {event['output']}")
```

---

## 🆚 AgentExecutor vs `create_react_agent` (LangGraph)

| Feature | AgentExecutor | `create_react_agent` (LangGraph) |
|---|---|---|
| **State persistence** | In-memory only | SQLite / Postgres checkpointing |
| **Human-in-the-Loop** | ❌ | ✅ Native interrupt |
| **Parallel branches** | ❌ | ✅ |
| **Streaming** | Token-level only | Token + Node-level |
| **Debugging** | `verbose=True` | LangSmith node traces |
| **Best for** | Learning, prototypes | Production systems |

> AgentExecutor is a great learning tool. For production multi-step agents, migrate to LangGraph.

---

## ✅ Key Takeaways

- `AgentExecutor` = runtime that manages the ReAct loop automatically
- Three parts: tools → `create_tool_calling_agent` → `AgentExecutor`
- `agent_scratchpad` in the prompt is where tool call memory accumulates
- Use `verbose=True` to see every thought and tool call during development
- Use `return_intermediate_steps=True` to inspect the full reasoning trace
- Use `RunnableWithMessageHistory` to add conversation memory

---

## ➡️ Next
[End-to-End Agent →](./05_end_to_end_agent.md)
