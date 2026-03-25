# 08 — Tools & Agents

> **Tools transform LLMs from text generators into action-taking agents. This section covers everything from defining tools, to the ReAct reasoning loop, to deploying a full production-ready agent.**

---

## 📚 Subtopics in This Section

| File | Topic |
|---|---|
| [`01_tool_schemas.md`](./01_tool_schemas.md) | What is a tool, anatomy of a schema, `@tool` decorator, Pydantic schemas |
| [`02_tool_calling.md`](./02_tool_calling.md) | `bind_tools`, ToolCall/ToolMessage, the manual tool loop, parallel calling |
| [`03_react_pattern.md`](./03_react_pattern.md) | Reason + Act loop, Thought→Action→Observation, multi-step reasoning |
| [`04_agent_executor.md`](./04_agent_executor.md) | `create_tool_calling_agent`, `AgentExecutor`, memory, streaming |
| [`05_end_to_end_agent.md`](./05_end_to_end_agent.md) | Complete research assistant: web search + calculator + weather + memory |
| [`examples.ipynb`](./examples.ipynb) | Hands-on: build tools, run ReAct loop, deploy full agent |

---

## 🎯 Learning Objectives

By the end of this section you will be able to:

- Define tools with `@tool` and Pydantic schemas
- Understand what the LLM actually sees (JSON function schema)
- Trace the full tool calling flow: bind → call → execute → synthesize
- Explain the ReAct (Reason + Act) pattern and why it works
- Build a complete agent using `create_tool_calling_agent` + `AgentExecutor`
- Add conversation memory using `RunnableWithMessageHistory`
- Deploy a multi-tool research assistant end-to-end

---

## ⚡ Quick Summary

```
TOOLS:
  @tool
  def get_weather(city: str) -> str:
      """Get current weather for a city."""
      ...

  tools = [get_weather, calculator, search_web]

BIND TO LLM:
  llm_with_tools = llm.bind_tools(tools)

REACT LOOP (manual):
  while True:
      response = llm_with_tools.invoke(messages)
      if not response.tool_calls: break     ← final answer
      for call in response.tool_calls:
          result = tools_map[call["name"]].invoke(call["args"])
          messages.append(ToolMessage(result, call["id"]))

AGENT EXECUTOR (automated loop):
  agent = create_tool_calling_agent(llm, tools, prompt)
  executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
  result = executor.invoke({"input": "What's the weather in Paris?"})

WITH MEMORY:
  agent_with_memory = RunnableWithMessageHistory(executor, get_session_history,
      input_messages_key="input", history_messages_key="chat_history")
```

---

## 🔑 Core Classes

| Class | Import | Purpose |
|---|---|---|
| `@tool` | `langchain_core.tools` | Define a tool from a function |
| `StructuredTool` | `langchain_core.tools` | Wrap an existing function as a tool |
| `TavilySearchResults` | `langchain_community.tools` | Web search tool |
| `create_tool_calling_agent` | `langchain.agents` | Create agent (no loop) |
| `AgentExecutor` | `langchain.agents` | Run the agent loop |
| `RunnableWithMessageHistory` | `langchain_core.runnables.history` | Add session memory |
| `ToolMessage` | `langchain_core.messages` | Tool result message |

---

## 🔄 How It All Fits Together

```
User Question
      │
      ▼
 AgentExecutor
      │
      │  ┌──────────────────────────────────────────────┐
      │  │              ReAct Loop                      │
      │  │                                              │
      │  │  1. LLM reasons + picks tool(s)              │
      │  │        ↓                                     │
      │  │  2. AgentExecutor calls the tool(s)          │
      │  │        ↓                                     │
      │  │  3. Results fed back to LLM                  │
      │  │        ↓                                     │
      │  │  4. Repeat until: no tool calls              │
      │  └──────────────────────────────────────────────┘
      │
      ▼
 Final Answer
```

---

## ⬅️ Previous
[07 — RAG with LangChain](../07_rag_with_langchain/theory.md)

## ➡️ Next
[09 — LangGraph Core Concepts](../09_langgraph_core_concepts/theory.md)
