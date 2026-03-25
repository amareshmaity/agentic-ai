# The ReAct Pattern

> *ReAct = Reason + Act. The agent thinks out loud before acting — this chain-of-thought reasoning is what separates a smart agent from a dumb tool-caller.*

---

## 🤔 What is ReAct?

**ReAct** (Reasoning + Acting) is an agent prompting strategy where the LLM alternates between:

1. **Thought** → "Let me think about what I need to do..."
2. **Action** → call a tool with specific arguments
3. **Observation** → see the tool result
4. **Repeat** → until the question is fully answered

```
User: "Who is the CEO of the company that makes the iPhone?"

Thought: I need to find out who makes the iPhone, then find their CEO.
Action: search_web("who makes the iPhone")
Observation: "Apple Inc. makes the iPhone."

Thought: Now I know it's Apple. Let me find their CEO.
Action: search_web("Apple Inc CEO 2024")
Observation: "Tim Cook is the CEO of Apple Inc."

Thought: I have all the information needed.
Final Answer: The iPhone is made by Apple Inc., and its CEO is Tim Cook.
```

---

## 🔄 The ReAct Loop

```
┌─────────────────────────────────────────────┐
│                                             │
│   User Question                             │
│        │                                   │
│        ▼                                   │
│   [THOUGHT] — LLM reasons about what       │
│               to do next                   │
│        │                                   │
│        ▼                                   │
│   [ACTION] — LLM picks a tool + args       │
│        │                                   │
│        ▼                                   │
│   [OBSERVATION] — Tool result injected     │
│        │                                   │
│        └──────── loop ──────────────────── │
│                                   │        │
│                    ┌──────────────┘        │
│                    │ enough info?          │
│                    ▼                       │
│             [FINAL ANSWER]                 │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 📝 The ReAct Prompt Structure

The system prompt for a ReAct agent tells the LLM to follow this exact format:

```
You are an assistant that can use tools.

To answer questions, use this format:

Thought: [think step by step about what to do]
Action: [tool_name]
Action Input: [tool arguments as JSON]
Observation: [tool result — filled in by the system]
... (repeat Thought/Action/Observation as needed)
Thought: I now have all the information needed.
Final Answer: [your complete answer to the user]
```

---

## 🏗️ Building a ReAct Agent from Scratch

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

@tool
def search_web(query: str) -> str:
    """Search the internet for current information."""
    # Simulate a web search
    results = {
        "who makes iPhone": "Apple Inc. makes the iPhone.",
        "Apple CEO": "Tim Cook is the CEO of Apple Inc.",
    }
    for key, value in results.items():
        if key.lower() in query.lower():
            return value
    return f"Search results for '{query}': No results found."

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [search_web, calculate]
tools_map = {t.name: t for t in tools}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def run_react_agent(user_question: str, max_steps: int = 10):
    """Run the ReAct loop."""
    messages = [HumanMessage(content=user_question)]
    
    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            # LLM gave a final answer — no more tool calls
            print(f"\n✅ Final Answer (step {step+1}):")
            print(response.content)
            return response.content
        
        # Execute all tool calls
        print(f"\n🔄 Step {step+1}: {len(response.tool_calls)} tool call(s)")
        for call in response.tool_calls:
            print(f"  → {call['name']}({call['args']})")
            tool = tools_map[call["name"]]
            result = tool.invoke(call["args"])
            print(f"  ← {result}")
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    
    return "Max steps reached"

# Run it
run_react_agent("What is 25 multiplied by the number of letters in 'Python'?")
```

---

## 🧪 ReAct Trace Example

Here's what the message list looks like after a multi-step ReAct run:

```
messages = [
    HumanMessage("What is 25 * len('Python')?"),
    
    AIMessage(tool_calls=[
        {"name": "calculate", "args": {"expression": "len('Python')"}, "id": "call_1"}
    ]),
    
    ToolMessage(content="6", tool_call_id="call_1"),
    
    AIMessage(tool_calls=[
        {"name": "calculate", "args": {"expression": "25 * 6"}, "id": "call_2"}
    ]),
    
    ToolMessage(content="150", tool_call_id="call_2"),
    
    AIMessage(content="25 multiplied by the number of letters in 'Python' (6) is 150.")
]
```

Each loop iteration adds 2 messages: the AI's tool call decision + the tool's result.

---

## ⚙️ ReAct with `create_react_agent` (LangChain)

LangChain provides a prebuilt helper that creates a ReAct agent graph using LangGraph under the hood:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a helpful assistant. Think step by step."
)

# Run the agent
result = agent.invoke({
    "messages": [HumanMessage(content="What is 15% of 200?")]
})

# Get the final answer
print(result["messages"][-1].content)
```

---

## ⚡ ReAct vs Simple Tool Calling

| | Simple Tool Calling | ReAct |
|---|---|---|
| **Reasoning** | No explicit thought step | Thinks before each action |
| **Multi-step** | Requires manual loops | Built into the pattern |
| **Transparency** | Black box decisions | Visible reasoning chain |
| **Error recovery** | Fails silently | Can reason about failures |
| **LangChain API** | `llm.bind_tools()` | `create_react_agent()` or `AgentExecutor` |

---

## ✅ Key Takeaways

- ReAct = **Reason** then **Act** — the LLM thinks before calling a tool
- The loop: Thought → Action → Observation → repeat → Final Answer
- Each iteration adds 2 messages to the conversation: tool call + tool result
- The key difference from simple tool calling: **explicit reasoning** at each step
- `create_react_agent` (LangGraph) is the modern way; `AgentExecutor` is the classic way

---

## ➡️ Next
[AgentExecutor →](./04_agent_executor.md)
