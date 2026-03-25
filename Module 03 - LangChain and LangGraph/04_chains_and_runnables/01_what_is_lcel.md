# What is LCEL?

> *LangChain Expression Language (LCEL) is a declarative way to compose Runnables using the `|` pipe operator — giving you streaming, async, parallelism, and observability for free.*

---

## 🤔 What is LCEL?

LCEL stands for **LangChain Expression Language**. It's the standard way to build chains in modern LangChain using the `|` operator to connect components.

```python
# This is a chain built with LCEL
chain = prompt | llm | parser

# It reads like a pipeline:
# user input → prompt fills in → llm processes → parser cleans up → output
```

Every component in LangChain is a **Runnable**. LCEL chains them together so they execute in sequence, with the output of each step flowing into the next.

---

## 🔴 Before LCEL: The Old Way

```python
# Old-style chains (LangChain < 0.1) — verbose, less flexible
from langchain.chains import LLMChain, SequentialChain

chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)

overall = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["topic"],
    output_variables=["final_answer"]
)
result = overall({"topic": "AI"})
```

**Problems with the old approach:**
- ❌ Verbose — lots of boilerplate
- ❌ No streaming by default
- ❌ No async by default
- ❌ Hard to compose and extend
- ❌ Different APIs for different chain types

---

## ✅ With LCEL: The New Way

```python
# LCEL — clean, composable, powerful
chain = prompt1 | llm | parser1 | prompt2 | llm | parser2

result = chain.invoke({"topic": "AI"})  # Single call!
```

**What you get for free with LCEL:**
- ✅ Streaming — `chain.stream(input)`
- ✅ Async — `await chain.ainvoke(input)`
- ✅ Batching — `chain.batch([i1, i2, i3])`
- ✅ LangSmith tracing — automatic
- ✅ Schema inspection — `chain.input_schema`, `chain.output_schema`
- ✅ Graph visualization — `chain.get_graph().print_ascii()`
- ✅ Retry/fallback — `chain.with_retry()`, `chain.with_fallbacks()`

---

## 🔗 The Pipe `|` Operator

The `|` operator chains Runnables together. When you call `.invoke()` on the chain, data flows left to right:

```python
chain = A | B | C

chain.invoke(x)
# Equivalent to:
# step1 = A.invoke(x)
# step2 = B.invoke(step1)
# step3 = C.invoke(step2)
# return step3
```

### A Complete Example

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Components (each is a Runnable)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant."),
    ("human",  "Explain {topic} in one sentence.")
])
llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# Build the chain with |
chain = prompt | llm | parser

# Use it
result = chain.invoke({"topic": "LCEL"})
print(result)
# "LCEL (LangChain Expression Language) is a declarative syntax..."
```

---

## 🔄 Input/Output Types Must Match

LCEL chains pass output from one step as input to the next. The types must be compatible:

```
ChatPromptTemplate.invoke(dict)        → ChatPromptValue
ChatOpenAI.invoke(ChatPromptValue)     → AIMessage
StrOutputParser.invoke(AIMessage)      → str

So: prompt | llm | parser
    dict → ChatPromptValue → AIMessage → str   ✅ Types match!
```

If types don't match, use `RunnableLambda` to transform between steps (covered in `04_runnable_passthrough.md`).

---

## 🏗️ Multi-Step Chains

```python
# Multi-step chain: first summarize, then translate
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following in 2 sentences."),
    ("human",  "{text}")
])

translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate to {language}. Output only the translation."),
    ("human",  "{text}")
])

# Step 1: summarize
summarize_chain = summarize_prompt | llm | StrOutputParser()

# Step 2: translate — needs {"text": ..., "language": ...}
# Use RunnableLambda to reshape the output
from langchain_core.runnables import RunnableLambda

full_chain = (
    summarize_chain
    | RunnableLambda(lambda summary: {"text": summary, "language": "Spanish"})
    | translate_prompt
    | llm
    | StrOutputParser()
)

result = full_chain.invoke({"text": "LangChain is a framework for building LLM applications that provides modular components..."})
print(result)  # Spanish translation of the summary
```

---

## 📊 LCEL vs Manual Python

| Feature | Manual Python | LCEL |
|---|---|---|
| Code length | Long | Concise |
| Streaming | Custom implementation | ✅ Free |
| Async | Custom implementation | ✅ Free |
| Parallel branches | Threads/asyncio | ✅ `RunnableParallel` |
| LangSmith tracing | Manual | ✅ Automatic |
| Input/output schema | Not typed | ✅ `input_schema` |
| Retry on failure | Try/except | ✅ `.with_retry()` |
| Fallback chain | If/else | ✅ `.with_fallbacks()` |

---

## ✅ Key Takeaways

- **LCEL** = compose Runnables with `|` into executable pipelines
- Every component (prompt, LLM, parser, retriever) is a `Runnable`
- Chains created with LCEL get streaming, async, batching for **free**
- Old `LLMChain` / `SequentialChain` patterns are legacy — use LCEL
- Data flows left-to-right through `|`, output of each step = input of next

---

## ➡️ Next
[The Runnable Interface →](./02_runnable_interface.md)
