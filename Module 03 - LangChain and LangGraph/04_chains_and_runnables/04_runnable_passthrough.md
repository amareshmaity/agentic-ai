# RunnablePassthrough, RunnableLambda & RunnablePick

> *These utilities let you shape and route data between chain steps — essential for building real pipelines where you need to pass inputs through, transform data, or select specific fields.*

---

## 1️⃣ RunnablePassthrough — Pass Input Unchanged

`RunnablePassthrough` passes its input through to the output unchanged. Essential when you need to keep the original input alongside transformed data.

```python
from langchain_core.runnables import RunnablePassthrough

passthrough = RunnablePassthrough()

# Input = output (identity function)
result = passthrough.invoke({"question": "What is LangChain?"})
print(result)  # {"question": "What is LangChain?"}
```

### The Classic RAG Pattern

This is the most important use of `RunnablePassthrough`:

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# In a RAG chain, you need BOTH:
# 1. The retrieved context (from retriever)
# 2. The original question (to put in the prompt)

rag_input = RunnableParallel({
    "context":  retriever,           # retriever.invoke(question) → docs
    "question": RunnablePassthrough() # passes question through unchanged
})

# rag_input.invoke("What is LangGraph?")
# → {"context": [Document(...)], "question": "What is LangGraph?"}
```

### Shorthand in Dict Context

Inside a dict (which is a `RunnableParallel`), `RunnablePassthrough()` passes the input through for that key:

```python
chain = (
    {
        "context":  retriever,
        "question": RunnablePassthrough()  # ← passes original question through
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

chain.invoke("What is LangGraph?")
# → invokes retriever("What is LangGraph?")
# → passes "What is LangGraph?" straight to "question" key
```

### `.assign()` — Add Fields to Existing Dict

```python
# Add a new key to the dict without changing existing ones
chain = RunnablePassthrough.assign(
    context=lambda x: retriever.invoke(x["question"]),
    upper_q=lambda x: x["question"].upper()
)

result = chain.invoke({"question": "What is LangChain?"})
# → {"question": "What is LangChain?",   ← original preserved
#    "context": [...docs...],              ← added
#    "upper_q": "WHAT IS LANGCHAIN?"}     ← added
```

---

## 2️⃣ RunnableLambda — Custom Functions as Runnables

`RunnableLambda` wraps any Python function as a Runnable, enabling it to be composed in LCEL chains.

```python
from langchain_core.runnables import RunnableLambda

# Basic usage
double = RunnableLambda(lambda x: x * 2)
print(double.invoke(5))   # 10

# String transformation
upper = RunnableLambda(str.upper)
print(upper.invoke("hello"))  # "HELLO"
```

### Transforming Between Chain Steps

The most common use — reshape data between steps:

```python
from langchain_core.runnables import RunnableLambda

# Retriever returns List[Document], but prompt needs str
def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RunnableLambda(lambda x: {
        "context":  format_docs(x["context"]),  # List[Doc] → str
        "question": x["question"]
      })
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

### Using `|` With Regular Functions (Shorthand)

When you pipe a dict to a function, it auto-wraps in `RunnableLambda`:

```python
# These are equivalent:
chain_explicit  = prev_chain | RunnableLambda(lambda x: {"key": x})
chain_shorthand = prev_chain | (lambda x: {"key": x})  # ← auto-wrapped
```

### Async Lambda

```python
import asyncio

async def async_lookup(x: dict) -> dict:
    # e.g., async DB call
    await asyncio.sleep(0.1)
    return {**x, "extra": "data"}

chain = RunnableLambda(async_lookup)
result = await chain.ainvoke({"key": "value"})
```

---

## 3️⃣ RunnablePick — Select Fields from Dict

Pick specific fields from a dict output:

```python
from langchain_core.runnables import RunnablePick

# Select only specific keys
pick = RunnablePick("answer")
result = pick.invoke({"question": "What?", "answer": "42", "context": "..."})
print(result)  # "42"

# Pick multiple keys
pick_multi = RunnablePick(["answer", "question"])
result = pick_multi.invoke({"question": "What?", "answer": "42", "context": "...", "timestamp": 123})
print(result)  # {"answer": "42", "question": "What?"}
```

---

## 4️⃣ `itemgetter` — Quick Field Selection

Python's `itemgetter` from `operator` module works as a Runnable lambda shorthand:

```python
from operator import itemgetter
from langchain_core.runnables import RunnableParallel

# Use itemgetter to select specific fields from a dict
chain = RunnableParallel({
    "question": itemgetter("question"),  # ← picks "question" key
    "context":  itemgetter("context"),   # ← picks "context" key
}) | rag_prompt | llm | StrOutputParser()
```

---

## 5️⃣ Practical Patterns

### Pattern 1: Format Retrieved Docs

```python
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs)
    | rag_prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke({"question": "What is LangGraph?"})
```

### Pattern 2: Add Metadata to Output

```python
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime

# Add timestamp and question to the output dict
chain_with_meta = (
    {"answer": main_chain, "question": RunnablePassthrough()}
    | RunnablePassthrough.assign(
        timestamp=lambda _: datetime.now().isoformat(),
        word_count=lambda x: len(x["answer"].split())
    )
)

result = chain_with_meta.invoke({"question": "What is LangChain?"})
print(result["answer"])     # The answer
print(result["timestamp"])  # When it was generated
print(result["word_count"]) # How many words
```

### Pattern 3: Conditional Processing

```python
from langchain_core.runnables import RunnableLambda

def route_by_lang(x: dict):
    if x.get("language") == "french":
        return french_chain
    elif x.get("language") == "spanish":
        return spanish_chain
    else:
        return english_chain

router = RunnableLambda(route_by_lang)
# router.invoke({"language": "french", ...}) → uses french_chain
```

---

## ✅ Key Takeaways

| Component | Purpose |
|---|---|
| `RunnablePassthrough()` | Pass input through unchanged (identity) |
| `RunnablePassthrough.assign(key=fn)` | Add new fields to existing dict |
| `RunnableLambda(fn)` | Wrap any Python function as a Runnable |
| `RunnablePick("key")` | Select specific field(s) from dict |
| `itemgetter("key")` | Shorthand for picking fields (works as Runnable) |

---

## ➡️ Next
[Chain Patterns →](./05_chain_patterns.md)
