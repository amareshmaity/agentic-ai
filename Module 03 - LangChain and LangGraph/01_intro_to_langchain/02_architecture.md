# LangChain Architecture

> *Understanding LangChain's layered architecture is what separates someone who "uses LangChain" from someone who truly understands it.*

---

## 🏗️ The Three-Layer Architecture

LangChain is organized into three conceptual layers:

```
┌───────────────────────────────────────────────────────┐
│                   LAYER 3: APPLICATIONS                │
│          Chains · Agents · RAG Pipelines               │
│          (the things users build)                      │
├───────────────────────────────────────────────────────┤
│                   LAYER 2: COMPONENTS                  │
│   Models · Prompts · Parsers · Memory · Tools          │
│   Document Loaders · Text Splitters · Vector Stores    │
│         (the reusable building blocks)                 │
├───────────────────────────────────────────────────────┤
│                   LAYER 1: CORE                        │
│   Runnables · LCEL · Base Interfaces · Schemas         │
│   (the foundation everything is built on)              │
└───────────────────────────────────────────────────────┘
```

---

## 🔩 Layer 1: LangChain Core (The Foundation)

### What is `langchain-core`?
The lowest level. Defines **base classes and interfaces** that all other components implement.

Key abstractions:
- **`Runnable`** — the universal interface (`.invoke()`, `.stream()`, `.batch()`, `.ainvoke()`)
- **`BaseLanguageModel`** — base class all LLMs implement
- **`BasePromptTemplate`** — base class all prompts implement
- **`BaseOutputParser`** — base class all parsers implement
- **`BaseRetriever`** — base class all retrievers implement

### The Runnable Protocol — The Heart of LangChain

Every component in LangChain implements the `Runnable` interface:

```python
from langchain_core.runnables import Runnable

# Every component supports these methods:
component.invoke(input)           # Single synchronous call
component.stream(input)           # Stream output token by token
component.batch([input1, input2]) # Process multiple inputs in parallel
component.ainvoke(input)          # Async version of invoke
component.astream(input)          # Async streaming

# Every component can be inspected:
component.input_schema            # What input type it expects
component.output_schema           # What output type it returns
component.get_graph()             # Visualize the computation graph
```

> This is what makes **LCEL pipe composition** work — every component is a Runnable, so they can be chained with `|`.

---

## 🔩 Layer 2: Components (The Building Blocks)

The reusable components. Each is a Runnable. Each can be composed with others.

```
Models         → the LLM (brain)
Prompts        → structured LLM input
Output Parsers → structured LLM output
Document Loaders → data ingestion
Text Splitters  → chunk documents
Vector Stores   → store embeddings
Retrievers     → search embeddings
Memory         → conversation history
Tools          → functions agents can call
```

See [`03_core_components.md`](./03_core_components.md) for full detail on each.

---

## 🔩 Layer 3: Chains & Agents (The Application Layer)

Built from Layer 2 components. This is where application logic lives.

**Chains** — deterministic sequences:
```
Prompt → LLM → Parser      (simple chain)
Prompt → LLM → Retriever → LLM → Parser  (RAG chain)
```

**Agents** — LLM decides the next step:
```
User Input → LLM → "I need to search the web"
           → Tool: web_search("query")
           → Observation: "results..."
           → LLM → "Based on results, the answer is..."
           → Final Answer
```

---

## 🔗 LCEL: LangChain Expression Language

The **pipe `|` operator** is the syntax for composing Runnables.

```python
# Chain = sequence of Runnables connected with |
chain = prompt | llm | output_parser

# This is equivalent to:
def chain(input):
    step1 = prompt.invoke(input)
    step2 = llm.invoke(step1)
    step3 = output_parser.invoke(step2)
    return step3

# But with LCEL you also get:
chain.stream(input)        # Streaming through entire chain
chain.batch([i1, i2, i3]) # Parallel batch processing
chain.ainvoke(input)       # Async execution
chain.get_graph()          # Visual graph of the pipeline
```

### Why LCEL Over Traditional Chains?

| Feature | Traditional Chains | LCEL |
|---|---|---|
| Streaming | ❌ Not built-in | ✅ Automatic |
| Async | ❌ Manual | ✅ Built-in |
| Batch | ❌ Loop manually | ✅ Parallel batch |
| Composability | Limited | ✅ Fully composable |
| Introspection | ❌ Limited | ✅ `get_graph()`, `input_schema` |
| LangSmith tracing | Limited | ✅ Full traces |

---

## 🔄 Data Flow Through a LangChain Chain

Let's trace a complete request through a simple RAG chain:

```
User: "What is LangGraph?"
         │
         ▼
   [PromptTemplate]
   Formats input:
   "Answer based on context: {context}
    Question: {question}"
         │
         ▼
     [Retriever]
   Searches vector DB:
   → Returns 3 relevant chunks about LangGraph
         │
         ▼
   [ChatOpenAI]
   LLM receives: formatted prompt + retrieved context
   LLM outputs: AIMessage("LangGraph is a library...")
         │
         ▼
  [StrOutputParser]
  Extracts: "LangGraph is a library..."
         │
         ▼
   Final Answer → User
```

---

## 📦 Package Dependency Map

```
langchain-core          ← no dependencies on other LC packages
    ↑
langchain               ← depends on langchain-core
    ↑
langchain-openai        ← depends on langchain-core
langchain-anthropic     ← depends on langchain-core
langchain-community     ← depends on langchain-core + langchain
    ↑
Your Application        ← imports from all of the above
```

> `langchain-core` is the stable foundation. Model-specific packages (`langchain-openai`, etc.) can update independently without breaking your chains.

---

## 🔧 Schema and Type Safety

LangChain uses **Pydantic** throughout for input/output validation:

```python
from langchain_core.messages import (
    HumanMessage,    # User's message
    AIMessage,       # LLM's response
    SystemMessage,   # System instruction
    ToolMessage,     # Result of a tool call
)

# Messages are typed Pydantic objects
msg = HumanMessage(content="Hello!")
print(msg.type)     # "human"
print(msg.content)  # "Hello!"

# LLM output is also typed
response = llm.invoke([msg])
print(type(response))        # <class 'AIMessage'>
print(response.content)      # "Hi there!"
print(response.usage_metadata)  # token counts
```

---

## 🧩 Architecture in Practice: A Complete Example

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Layer 1: Core interface (Runnable) is used by all
# Layer 2: Components
model  = ChatOpenAI(model="gpt-4o-mini")         # Model component
prompt = ChatPromptTemplate.from_messages([        # Prompt component
    ("system", "You are a helpful AI assistant."),
    ("human", "{question}")
])
parser = StrOutputParser()                         # Parser component

# Layer 3: Application — compose with LCEL
chain = prompt | model | parser                    # Chain!

# All Runnable methods work on the full chain
result   = chain.invoke({"question": "What is LangChain?"})
streamed = chain.stream({"question": "What is LangChain?"})
batched  = chain.batch([{"question": "Q1?"}, {"question": "Q2?"}])
```

---

## ✅ Key Takeaways

- LangChain has **3 layers**: Core → Components → Applications
- **Everything is a `Runnable`** — the universal interface that enables composition
- **LCEL with `|`** is how you compose Runnables into chains
- **Split packages** mean lean installs and independent updates
- **Pydantic throughout** gives type safety at every step
- LangChain sits **between your app and LLMs/tools**, handling all the glue

---

## ⬅️ Previous
[What is LangChain?](./01_what_is_langchain.md)

## ➡️ Next
[Core Components →](./03_core_components.md)
