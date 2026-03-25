# StrOutputParser

> *The simplest output parser — extracts the string content from an AIMessage. The default choice for any chain that returns plain text.*

---

## 🤔 What is StrOutputParser?

`StrOutputParser` converts the `AIMessage` object returned by a ChatModel into a plain Python `str`.

```
ChatModel returns:  AIMessage(content="LangChain is a framework...")
                              ↓ StrOutputParser
Chain returns:      "LangChain is a framework..."   ← plain string
```

**Without it:**
```python
chain = prompt | llm
result = chain.invoke({"question": "What is LangChain?"})
print(type(result))     # <class 'AIMessage'>
print(result)           # AIMessage(content='LangChain is...', ...)
print(result.content)   # You'd need .content every time
```

**With it:**
```python
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "What is LangChain?"})
print(type(result))     # <class 'str'>
print(result)           # "LangChain is a framework..."
```

---

## 📦 Basic Usage

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human",  "{question}")
])
parser = StrOutputParser()

# Chain: prompt → llm → parser
chain = prompt | llm | parser

result = chain.invoke({"question": "What is Python?"})
print(type(result))   # <class 'str'>
print(result)         # "Python is a high-level programming language..."
```

---

## 🔧 StrOutputParser Properties

```python
parser = StrOutputParser()

# It's a Runnable — supports all standard methods
result   = parser.invoke(AIMessage(content="Hello!"))   # "Hello!"
streamed = parser.stream(...)                            # Token-by-token str chunks
batched  = parser.batch([...])                          # List of strings

# Input schema
print(parser.input_schema)   # Expects AIMessage or BaseMessage
print(parser.output_schema)  # Returns str
```

---

## 💬 Streaming with StrOutputParser

The most important feature — StrOutputParser enables **token-by-token streaming** through the entire chain:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Write a poem about {topic}.")
    | ChatOpenAI(model="gpt-4o-mini", streaming=True)
    | StrOutputParser()
)

# Stream token by token
print("Streaming: ", end="")
for chunk in chain.stream({"topic": "LangChain"}):
    print(chunk, end="", flush=True)  # Each chunk is a string fragment
print()  # newline
```

---

## 🔗 Async Streaming (for Web Apps)

```python
import asyncio

async def stream_response(topic: str):
    async for chunk in chain.astream({"topic": topic}):
        print(chunk, end="", flush=True)
        # In a FastAPI/WebSocket: await websocket.send_text(chunk)

asyncio.run(stream_response("artificial intelligence"))
```

---

## 📊 When to Use StrOutputParser

✅ **Use StrOutputParser when:**
- Your chain returns text that will be displayed to users
- You need streaming
- You pipe output to another chain that expects string input
- Simple summarization, Q&A, generation tasks

❌ **Don't use StrOutputParser when:**
- You need structured data (name, age, category, etc.)
- The output needs to be stored in a database
- Another piece of code needs to use specific fields from the output
- → Use `PydanticOutputParser` or `.with_structured_output()` instead

---

## 🔄 StrOutputParser vs Directly Accessing .content

| Approach | Code | Best For |
|---|---|---|
| `StrOutputParser` in chain | `chain = prompt \| llm \| StrOutputParser()` | ✅ Always — makes chain return str |
| Manual `.content` access | `result = chain.invoke(...)` then `result.content` | Legacy code, accessing other fields |

Always prefer `StrOutputParser` in a chain over manually accessing `.content` — it makes your chain's output type explicit and enables streaming.

---

## ✅ Key Takeaways

- `StrOutputParser` is the most commonly used parser — add it to every text-output chain
- Converts `AIMessage` → `str`, nothing else
- Critical for enabling **token-by-token streaming** through a chain
- Use it whenever your output is plain text — don't access `.content` manually
- When you need structured data, graduate to `JsonOutputParser` or `PydanticOutputParser`

---

## ➡️ Next
[JsonOutputParser →](./03_json_output_parser.md)
