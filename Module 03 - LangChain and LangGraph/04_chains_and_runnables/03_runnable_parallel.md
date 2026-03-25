# RunnableParallel

> *RunnableParallel runs multiple chains simultaneously and merges their outputs into a single dict — the key primitive for RAG and multi-perspective analysis.*

---

## 🤔 What is RunnableParallel?

`RunnableParallel` takes a dictionary of named Runnables and runs them **all at the same time**, returning a dict with their results.

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel({
    "key1": chain_A,
    "key2": chain_B,
    "key3": chain_C,
})

result = parallel.invoke(input)
# → {"key1": result_of_A, "key2": result_of_B, "key3": result_of_C}
# All 3 run in parallel!
```

---

## 📦 Basic Usage — Two Equivalent Syntaxes

### Syntax 1: Explicit `RunnableParallel`

```python
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

pros_chain = (
    ChatPromptTemplate.from_template("List 3 pros of {topic}")
    | llm | StrOutputParser()
)
cons_chain = (
    ChatPromptTemplate.from_template("List 3 cons of {topic}")
    | llm | StrOutputParser()
)

parallel = RunnableParallel({"pros": pros_chain, "cons": cons_chain})
result = parallel.invoke({"topic": "LangChain"})

print("Pros:", result["pros"])
print("Cons:", result["cons"])
```

### Syntax 2: Dict shorthand (most common)

```python
# When used inline in a chain, a dict = RunnableParallel automatically
parallel = {"pros": pros_chain, "cons": cons_chain}

# Or in a chain:
chain = {"pros": pros_chain, "cons": cons_chain} | combine_prompt | llm | StrOutputParser()
```

---

## 🔗 RunnableParallel in a RAG Chain

The most common use of `RunnableParallel` — fetch context AND pass the question through simultaneously:

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set up retriever
vectorstore = FAISS.from_texts(
    ["LangChain builds LLM apps", "LangGraph adds stateful agents"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

# RAG chain — retrieve and pass question in parallel
setup = RunnableParallel({
    "context":  retriever,                  # → retrieved docs
    "question": RunnablePassthrough()       # → original question string
})

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only this context:\n{context}"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    setup
    | {"context": lambda x: format_docs(x["context"]), "question": lambda x: x["question"]}
    | rag_prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

result = chain.invoke("What is LangGraph?")
print(result)
```

---

## 💡 Common Pattern: Multi-Perspective Analysis

```python
# Get multiple perspectives on the same topic — all in parallel!
analysis_chain = RunnableParallel({
    "summary": (
        ChatPromptTemplate.from_template("Summarize {topic} in 2 sentences.")
        | llm | StrOutputParser()
    ),
    "use_cases": (
        ChatPromptTemplate.from_template("List 3 use cases for {topic}.")
        | llm | StrOutputParser()
    ),
    "challenges": (
        ChatPromptTemplate.from_template("What are the main challenges with {topic}?")
        | llm | StrOutputParser()
    ),
    "alternatives": (
        ChatPromptTemplate.from_template("What are alternatives to {topic}?")
        | llm | StrOutputParser()
    ),
})

result = analysis_chain.invoke({"topic": "LangChain"})

for key, value in result.items():
    print(f"\n📌 {key.upper()}:")
    print(value)
```

---

## 🔗 Chaining After RunnableParallel

After parallel execution, pipe the combined dict into the next step:

```python
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Run two chains in parallel
parallel = RunnableParallel({
    "pros": pros_chain,
    "cons": cons_chain,
})

# Step 2: Combine outputs with a synthesis prompt
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a balanced analyst."),
    ("human", """Given these perspectives on {topic}:

PROS:
{pros}

CONS:
{cons}

Write a balanced 2-sentence verdict.""")
])

# Full chain: parallel → combine
full_chain = {
    "topic": RunnablePassthrough(),
    **{k: v for k, v in {"pros": pros_chain, "cons": cons_chain}.items()}
} | synthesis_prompt | llm | StrOutputParser()

# Hmm — simpler approach:
import operator
from langchain_core.runnables import RunnableParallel, RunnableLambda

gather = RunnableParallel({"pros": pros_chain, "cons": cons_chain})

def make_synthesis_input(gathered: dict, topic: str) -> dict:
    return {"topic": topic, "pros": gathered["pros"], "cons": gathered["cons"]}

# Chain it
result = (gather | RunnableLambda(lambda x: {
    "pros": x["pros"],
    "cons": x["cons"],
    "topic": "LangChain"
}) | synthesis_prompt | llm | StrOutputParser()).invoke({"topic": "LangChain"})
```

---

## ⚡ Performance: Parallel vs Sequential

```python
import time

# Sequential — each runs after the previous
start = time.time()
pros    = pros_chain.invoke({"topic": "LangChain"})
cons    = cons_chain.invoke({"topic": "LangChain"})
print(f"Sequential: {time.time() - start:.2f}s")

# Parallel — all run at the same time
start = time.time()
result = RunnableParallel({"pros": pros_chain, "cons": cons_chain}).invoke({"topic": "LangChain"})
print(f"Parallel:   {time.time() - start:.2f}s")
# Parallel ~= time of slowest single chain (not sum of all!)
```

---

## ✅ Key Takeaways

- `RunnableParallel({"key1": chain1, "key2": chain2})` runs chains simultaneously
- Dict shorthand `{"key1": chain1, "key2": chain2}` auto-creates `RunnableParallel` in LCEL
- Essential for RAG: fetch context and pass question through in parallel
- All branches receive the **same input** — output is a **merged dict**
- Performance: parallel time ≈ slowest single branch (not sum)

---

## ➡️ Next
[RunnablePassthrough & RunnableLambda →](./04_runnable_passthrough.md)
