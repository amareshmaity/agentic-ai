# LLM vs ChatModel

> *LangChain supports two types of model interfaces. Understanding the difference is critical — they accept different inputs and return different outputs.*

---

## 🔀 The Two Model Types

| Feature | LLM | ChatModel |
|---|---|---|
| **Input** | Single string | List of messages |
| **Output** | String | `AIMessage` object |
| **System prompt** | Baked into input string | Dedicated `SystemMessage` |
| **Tool calling** | ❌ Not supported | ✅ Native support |
| **Streaming** | Basic | ✅ Full support |
| **Multimodal** | ❌ | ✅ (vision, audio) |
| **Status** | Legacy | ✅ Modern standard |
| **Use this?** | ❌ Avoid | ✅ Always use |

---

## 📦 LLM Interface (Legacy — Know It, Don't Use It)

The old interface. Input = string. Output = string.

```python
from langchain_openai import OpenAI  # LLM, NOT ChatOpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")

# Input: plain string
response = llm.invoke("What is LangChain?")
print(type(response))  # str
print(response)        # "LangChain is a framework..."

# Equivalent older API call:
# openai.Completion.create(model="...", prompt="What is LangChain?")
```

**Why it exists:** Legacy support for older completion-style models like `gpt-3.5-turbo-instruct`. Modern models are all chat-based.

---

## 💬 ChatModel Interface (Modern — Always Use This)

The current standard. Input = list of typed messages. Output = `AIMessage`.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Input method 1: string (auto-converts to HumanMessage)
response = llm.invoke("What is LangChain?")

# Input method 2: typed message objects (recommended)
from langchain_core.messages import SystemMessage, HumanMessage
response = llm.invoke([
    SystemMessage(content="You are a concise technical assistant."),
    HumanMessage(content="What is LangChain?")
])

# Input method 3: tuple shorthand
response = llm.invoke([
    ("system", "You are a concise technical assistant."),
    ("human",  "What is LangChain?")
])

# Output: AIMessage (NOT a string)
print(type(response))              # <class 'AIMessage'>
print(response.content)            # "LangChain is..."
print(response.type)               # "ai"
print(response.usage_metadata)     # {'input_tokens': 22, 'output_tokens': 45}
print(response.response_metadata)  # {'model_name': 'gpt-4o-mini', ...}
```

---

## 📨 The Message Types

| Class | Role | Content |
|---|---|---|
| `SystemMessage` | Instructions to the LLM | Persona, constraints, format rules |
| `HumanMessage` | User's input | The question or request |
| `AIMessage` | LLM's response | The answer; may contain `tool_calls` |
| `ToolMessage` | Result of a tool call | Returned to LLM after tool execution |
| `FunctionMessage` | Legacy tool result | Deprecated — use `ToolMessage` |

```python
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

# Multi-turn conversation
messages = [
    SystemMessage(content="You are a helpful Python tutor."),
    HumanMessage(content="What is a list comprehension?"),
    AIMessage(content="A list comprehension creates a list from an expression: [x*2 for x in range(5)]"),
    HumanMessage(content="Can you give me a more complex example?")
]

response = llm.invoke(messages)
print(response.content)
```

---

## ⚙️ Key ChatModel Parameters

```python
ChatOpenAI(
    model="gpt-4o-mini",     # Model version
    temperature=0,            # 0=deterministic, 1=creative, 2=chaotic
    max_tokens=500,           # Max tokens in response
    timeout=30,               # HTTP timeout (seconds)
    max_retries=2,            # Auto-retry on failure
    streaming=False,          # Return all at once or token-by-token
    api_key="sk-...",         # Can also use OPENAI_API_KEY env var
)
```

### Temperature Guide

```
temperature=0.0  → Deterministic. Always the same answer. Use for: structured output, factual Q&A
temperature=0.3  → Slightly varied. Good for: coding, analysis
temperature=0.7  → Balanced. Good for: general assistants  
temperature=1.0  → Creative. Good for: writing, ideation
temperature=2.0  → Maximum chaos. Rarely useful
```

---

## 🔄 The Unified Runnable Interface

Both LLM and ChatModel implement the same `Runnable` interface:

```python
# All of these work on ANY ChatModel
response   = llm.invoke(input)           # Single, blocking call
streamed   = llm.stream(input)           # Generator of chunks
batched    = llm.batch([input1, input2]) # Parallel processing
async_resp = await llm.ainvoke(input)    # Async single call
async_str  = llm.astream(input)         # Async streaming

# Runtime configuration
response = llm.invoke(
    input,
    config={
        "tags": ["production"],          # Tag for LangSmith filtering
        "metadata": {"user_id": "123"},  # Metadata for LangSmith
        "run_name": "my-run",            # Custom run name in LangSmith
    }
)
```

---

## 🧠 AIMessage in Detail

```python
response = llm.invoke("What is 2+2?")

# Content
print(response.content)          # "2 + 2 = 4"

# Metadata
print(response.id)               # Unique message ID
print(response.type)             # "ai"

# Token usage
print(response.usage_metadata)
# {'input_tokens': 12, 'output_tokens': 8, 'total_tokens': 20}

# Model info
print(response.response_metadata)
# {'model_name': 'gpt-4o-mini', 'finish_reason': 'stop', ...}

# Tool calls (when model decides to call a tool)
print(response.tool_calls)
# [] if no tools, otherwise:
# [{'name': 'search', 'args': {'query': '...'}, 'id': 'call_...'}]
```

---

## 📊 When to Use Which

| Scenario | Use |
|---|---|
| Any modern application | `ChatOpenAI`, `ChatAnthropic`, etc. |
| Older completion-style API | `OpenAI` (LLM) — rare |
| Tool calling / function calling | ChatModel only |
| Streaming to frontend | ChatModel only |
| System prompts / personas | ChatModel (use `SystemMessage`) |
| Multimodal (images + text) | ChatModel only |

---

## ✅ Key Takeaways

- **Always use ChatModel** (`ChatOpenAI`, `ChatAnthropic`, etc.) — LLM is legacy
- ChatModels take **list of typed messages** → return **`AIMessage`**
- All models share the same `.invoke()`, `.stream()`, `.batch()` interface
- `AIMessage` contains content, token usage, tool calls, and metadata
- `temperature=0` for deterministic output; higher for creativity

---

## ➡️ Next
[Model Providers →](./02_model_providers.md)
