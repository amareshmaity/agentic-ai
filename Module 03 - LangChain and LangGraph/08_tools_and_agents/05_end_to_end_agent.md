# End-to-End Agent Project

> *Build a complete AI research assistant in one file — web search + calculations + weather, all orchestrated by a ReAct agent with conversation memory.*

---

## 🎯 Project Overview

We'll build a **Personal Research Assistant** agent that can:

- 🔍 Search the web for current information
- 🧮 Perform calculations
- 🌤️ Check weather
- 📝 Summarize findings
- 💬 Remember the conversation history

---

## 📦 Setup

```python
# pip install langchain langchain-openai langchain-community tavily-python python-dotenv

import os
from dotenv import load_dotenv
load_dotenv()

# Required environment variables:
# OPENAI_API_KEY=your_key
# TAVILY_API_KEY=your_key  (for real web search — free tier at app.tavily.com)
```

---

## 🔧 Step 1 — Define the Tools

```python
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from datetime import datetime

# ── Tool 1: Web Search (Tavily) ────────────────────────────────────────────
search = TavilySearchResults(
    max_results=3,
    description="Search the internet for current facts, news, and information. "
                "Use this for anything that requires up-to-date knowledge."
)

# ── Tool 2: Calculator ─────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """Evaluate a Python mathematical expression.
    
    Use for: arithmetic, percentages, unit conversions.
    Examples: "25 * 4", "100 / 7", "2 ** 10", "15 / 100 * 200"
    
    Args:
        expression: A valid Python math expression (no variables, no functions)
    """
    try:
        # Restrict to math-only evaluation for safety
        allowed = {k: v for k, v in __builtins__.items()
                   if k in ('abs', 'round', 'max', 'min', 'sum', 'len', 'pow')}
        result = eval(expression, {"__builtins__": allowed})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}. Please use a valid Python math expression."

# ── Tool 3: Weather ───────────────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city.
    
    Args:
        city: Name of the city (e.g., 'Tokyo', 'New York', 'London')
    """
    # In production, call OpenWeatherMap API or similar
    import random
    conditions = ["sunny", "partly cloudy", "overcast", "light rain"]
    temp = random.randint(10, 35)
    condition = random.choice(conditions)
    return f"Weather in {city}: {temp}°C, {condition}, humidity {random.randint(40, 80)}%"

# ── Tool 4: Current Date/Time ─────────────────────────────────────────────
@tool
def get_current_time() -> str:
    """Get the current date and time. Use when the user asks about today's date or time."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"

tools = [search, calculator, get_weather, get_current_time]
print(f"✅ Loaded {len(tools)} tools: {[t.name for t in tools]}")
```

---

## 🤖 Step 2 — Create the Agent

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

# ── LLM ───────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── System Prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a smart research assistant with access to tools.

Your capabilities:
- Search the internet for current information
- Perform mathematical calculations  
- Check weather for any city
- Tell the current date and time

Guidelines:
1. Think step by step before acting
2. Use tools when needed — don't guess facts
3. If a question requires multiple tools, use them in logical order
4. Summarize your findings clearly and concisely
5. Always cite where you got information from

Today's context: You are helping a user with research, calculations, and real-time data.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# ── Agent (no execution yet) ──────────────────────────────────────────────
agent = create_tool_calling_agent(llm, tools, prompt)

# ── AgentExecutor (adds the loop) ────────────────────────────────────────
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
    return_intermediate_steps=False,
)

print("✅ Agent ready!")
```

---

## 💾 Step 3 — Add Conversation Memory

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Store chat histories per session
session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# Wrap the agent executor with memory
agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

print("✅ Memory enabled!")
```

---

## 🚀 Step 4 — Run the Agent

```python
def chat(user_input: str, session_id: str = "default") -> str:
    """Send a message and get a response."""
    config = {"configurable": {"session_id": session_id}}
    result = agent_with_memory.invoke({"input": user_input}, config=config)
    return result["output"]

# ── Test Multi-Step Reasoning ─────────────────────────────────────────────
print("=" * 60)
print("TEST 1: Multi-step question")
print("=" * 60)
response = chat("What is today's date, and what's the weather in Tokyo?")
print(f"\nAssistant: {response}\n")

# ── Test Calculation ──────────────────────────────────────────────────────
print("=" * 60)
print("TEST 2: Math calculation")
print("=" * 60)
response = chat("If I invest $5000 at 7% annual return, how much will I have after 10 years?")
print(f"\nAssistant: {response}\n")

# ── Test Web Search ───────────────────────────────────────────────────────
print("=" * 60)
print("TEST 3: Web search")
print("=" * 60)
response = chat("What are the latest developments in quantum computing?")
print(f"\nAssistant: {response}\n")

# ── Test Memory (Follow-up) ───────────────────────────────────────────────
print("=" * 60)
print("TEST 4: Memory follow-up")
print("=" * 60)
response = chat("Based on that investment calculation you just did, what if the rate was 10% instead?")
print(f"\nAssistant: {response}\n")
```

---

## 🖥️ Step 5 — Interactive REPL

```python
def run_interactive():
    """Run an interactive chat session with the agent."""
    print("\n" + "=" * 60)
    print("🤖 Research Assistant Agent")
    print("=" * 60)
    print("Capabilities: web search, calculator, weather, date/time")
    print("Type 'exit' to quit, 'clear' to start a new session")
    print("=" * 60 + "\n")
    
    session_id = "interactive-session"
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "clear":
            session_store.pop(session_id, None)
            print("✅ Conversation cleared. Starting fresh.\n")
            continue
        
        print("\nAssistant: ", end="", flush=True)
        response = chat(user_input, session_id=session_id)
        print(response + "\n")

# run_interactive()  # Uncomment to run interactively
```

---

## 📊 Full Architecture Diagram

```
User Input
    │
    ▼
RunnableWithMessageHistory  ─── loads past conversation ───►  InMemoryChatMessageHistory
    │                                                              (per session_id)
    ▼
AgentExecutor
    │
    ├── Iteration 1:
    │       LLM (gpt-4o-mini + tools) ──► tool_calls [weather, calculator]
    │       Execute tools              ──► ToolMessages with results
    │
    ├── Iteration 2:
    │       LLM sees tool results ──► Final answer (no more tool_calls)
    │
    ▼
Final Output ──────────────────────────────────────────► User
```

---

## 🔧 Production Hardening

For a real deployment, add these improvements:

```python
# 1. Real web search with proper error handling
from langchain_community.tools.tavily_search import TavilySearchResults
search_tool = TavilySearchResults(
    max_results=5,
    include_answer=True,    # Get a direct answer, not just URLs
    include_raw_content=False,
)

# 2. Add tool timeout protection
from langchain_core.tools import tool
import asyncio

@tool
async def safe_search(query: str) -> str:
    """Search with timeout protection."""
    try:
        async with asyncio.timeout(10.0):
            return await search_tool.ainvoke(query)
    except asyncio.TimeoutError:
        return "Search timed out. Please try a more specific query."

# 3. Track costs per session
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = agent_executor.invoke({"input": "What is quantum computing?"})
    print(f"Tokens: {cb.total_tokens} | Cost: ${cb.total_cost:.4f}")

# 4. Log every interaction
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

def chat_with_logging(user_input: str, session_id: str) -> str:
    logger.info(f"Session {session_id}: {user_input}")
    response = chat(user_input, session_id)
    logger.info(f"Response: {response[:100]}...")
    return response
```

---

## ✅ Key Takeaways

- **4-step build**: Define tools → Create agent → Add memory → Run
- `create_tool_calling_agent` = LLM + tools + prompt (no execution)
- `AgentExecutor` = wraps agent, manages the loop, handles errors
- `RunnableWithMessageHistory` = adds per-session conversation memory
- The `agent_scratchpad` placeholder is critical — it holds intermediate tool results
- For production: use Tavily for real search, add timeouts, log costs

---

## ⬅️ Previous
[AgentExecutor ←](./04_agent_executor.md)

## ➡️ Next Section
[09 — LangGraph Core Concepts →](../09_langgraph_core_concepts/theory.md)
