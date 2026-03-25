# 📖 Module 03: Building Agents with LangChain & LangGraph

> **The most widely used agent framework in industry — master it end-to-end.**
> Covers LangChain completely first, then LangGraph — so you build deep intuition before moving to the graph model.

---

## 🎯 Module Goal

By the end of this module you will be able to:

**LangChain:**
- Understand LangChain's full architecture: Models, Prompts, Chains, Runnables, LCEL
- Build structured output pipelines with Output Parsers and Pydantic
- Load, split, embed and retrieve documents for RAG systems
- Build production-grade tool-calling agents using `AgentExecutor`

**LangGraph:**
- Design stateful, graph-based workflows with Nodes, Edges, and Conditional logic
- Implement Sequential, Parallel, Conditional, and Iterative workflow patterns
- Persist agent state across sessions (SQLite → Postgres)
- Add real-time streaming at token and node level
- Integrate Tools, RAG, and MCP inside LangGraph
- Implement Human-in-the-Loop (HITL) approval gates and Subgraphs
- Build agents with Short-Term and Long-Term Memory
- Apply Advanced RAG: Corrective RAG (CRAG) and Self-RAG
- Ship a full Blog Writing Agent project end-to-end

---

## 📂 Folder Structure

```
Module 03 - LangChain and LangGraph/
│
├── README.md                                         ← You are here
│
│  ══════════════════════════════════════
│  🔵 PART 1 — LANGCHAIN
│  ══════════════════════════════════════
│
├── 01_intro_to_langchain/
│   ├── theory.md                                     ← What is LangChain, architecture, components overview
│   └── examples.ipynb                                ← First LangChain chain, explore all components
│
├── 02_models_and_prompts/
│   ├── theory.md                                     ← LLM vs ChatModel, PromptTemplate vs ChatPromptTemplate
│   └── examples.ipynb                                ← Model comparisons, prompt templates in action
│
├── 03_structured_output_and_parsers/
│   ├── theory.md                                     ← Structured output, JSON mode, Output Parsers, Pydantic
│   └── examples.ipynb                                ← StrOutputParser, JSONOutputParser, PydanticOutputParser
│
├── 04_chains_and_runnables/
│   ├── theory.md                                     ← LCEL, pipe | operator, Runnables, RunnableParallel
│   └── examples.ipynb                                ← Build chains with LCEL, parallel chains, branching
│
├── 05_document_loaders_and_text_splitters/
│   ├── theory.md                                     ← Document loaders (PDF, web, CSV), text splitters, chunking
│   └── examples.ipynb                                ← Load documents, chunk strategies, compare splitters
│
├── 06_vector_stores_and_retrievers/
│   ├── theory.md                                     ← Embeddings, vector stores (Chroma, FAISS), retrievers
│   └── examples.ipynb                                ← Build a vector store, similarity search, MMR retrieval
│
├── 07_rag_with_langchain/
│   ├── theory.md                                     ← RAG architecture, naive RAG, retrieval chain, Q&A chain
│   └── examples.ipynb                                ← Full RAG pipeline + YouTube Chatbot project
│
├── 08_tools_and_agents/
│   ├── theory.md                                     ← Tools, tool schemas, tool calling, AgentExecutor, ReAct
│   └── examples.ipynb                                ← Custom tools + end-to-end AI agent with AgentExecutor
│
│  ══════════════════════════════════════
│  🟣 PART 2 — LANGGRAPH
│  ══════════════════════════════════════
│
├── 09_langgraph_core_concepts/
│   ├── theory.md                                     ← Why LangGraph, Agentic AI, StateGraph, Nodes, Edges
│   └── examples.ipynb                                ← First LangGraph, LangChain vs LangGraph comparison
│
├── 10_workflow_patterns/
│   ├── theory.md                                     ← Sequential, Parallel, Conditional, Iterative workflows
│   └── examples.ipynb                                ← Build all 4 workflow patterns with StateGraph
│
├── 11_chatbot_and_persistence/
│   ├── theory.md                                     ← Stateful chatbot, checkpointers, SQLite, time travel
│   └── examples.ipynb                                ← Chatbot with Streamlit UI + SQLite persistence
│
├── 12_streaming_and_observability/
│   ├── theory.md                                     ← stream_mode types, LangSmith, tracing, observability
│   └── examples.ipynb                                ← Token + node streaming, LangSmith integration
│
├── 13_tools_and_rag_in_langgraph/
│   ├── theory.md                                     ← Tools in LangGraph, MCP client, RAG as a graph node
│   └── examples.ipynb                                ← Tool-calling graph, MCP client, RAG in LangGraph
│
├── 14_hitl_and_subgraphs/
│   ├── theory.md                                     ← interrupt_before/after, approval gates, nested subgraphs
│   └── examples.ipynb                                ← HITL approval gate + composable subgraph architecture
│
├── 15_memory_in_langgraph/
│   ├── theory.md                                     ← How LLMs "remember", short-term vs long-term memory
│   └── examples.ipynb                                ← Short-term (in-context) + Long-term (persistent) memory
│
├── 16_langgraph_supervisor/
│   ├── theory.md                                     ← Supervisor+Worker pattern, routing decisions, agent handoffs
│   └── examples.ipynb                                ← Supervisor agent coordinating specialist sub-agents
│
├── 17_project_blog_writing_agent/
│   ├── theory.md                                     ← Agent design: plan → research → write → review
│   └── examples.ipynb                                ← Full blog writing agent: LangGraph end-to-end project
│
└── exercises/
    └── exercises.md                                  ← Practice problems + mini-projects for both parts
```

---

## 📚 Topics Covered

### 🔵 Part 1 — LangChain

| # | Topic | Core Concepts |
|---|---|---|
| 01 | Intro to LangChain | Architecture, components, ecosystem overview |
| 02 | Models & Prompts | LLM vs ChatModel, PromptTemplate, ChatPromptTemplate |
| 03 | Structured Output & Parsers | JSON mode, Pydantic, StrOutputParser, PydanticOutputParser |
| 04 | Chains & Runnables | LCEL, pipe `\|` operator, RunnableParallel, RunnablePassthrough |
| 05 | Document Loaders & Text Splitters | PDF/web loaders, RecursiveCharacterTextSplitter, chunk strategies |
| 06 | Vector Stores & Retrievers | Embeddings, Chroma, FAISS, similarity search, MMR |
| 07 | RAG with LangChain | RAG pipeline, retrieval chain, Q&A chain, chatbot project |
| 08 | Tools & Agents | Tool schemas, tool calling, ReAct, AgentExecutor, end-to-end agent |

### 🟣 Part 2 — LangGraph

| # | Topic | Core Concepts |
|---|---|---|
| 09 | LangGraph Core Concepts | Agentic AI, StateGraph, Nodes, Edges, TypedDict state |
| 10 | Workflow Patterns | Sequential, Parallel, Conditional, Iterative workflows |
| 11 | Chatbot & Persistence | Stateful chatbot, checkpointers, SQLite, time travel |
| 12 | Streaming & Observability | stream_mode, LangSmith tracing, observability dashboard |
| 13 | Tools & RAG in LangGraph | Tools node, MCP client, RAG as graph workflow |
| 14 | HITL & Subgraphs | interrupt_before/after, approval gates, nested subgraphs |
| 15 | Memory in LangGraph | Short-term (in-context) + Long-term (persistent store) memory |
| 16 | LangGraph Supervisor | Supervisor+Worker, multi-agent routing, handoffs between agents |
| 17 | Project: Blog Writing Agent | Plan → Research → Write → Review — full LangGraph project |

---

## 🧠 Why LangChain First, Then LangGraph?

```
LangChain teaches you:          LangGraph adds:
─────────────────────           ──────────────────────────
✅ How LLMs are called          ✅ State that persists across steps
✅ How prompts are built        ✅ Branching & conditional logic
✅ How chains compose           ✅ Loops & iterative workflows
✅ How tools are defined        ✅ Human-in-the-Loop control
✅ How RAG works                ✅ Multi-agent orchestration
✅ How agents think             ✅ Production-grade persistence
```

> LangGraph was built *by the LangChain team* specifically because AgentExecutor wasn't enough for production. Once you understand LangChain, LangGraph's design decisions will click immediately.

---

## 🗺️ LangGraph Mental Model

```
                    ┌──────────────────────────────────┐
                    │           StateGraph              │
                    │                                  │
        START ─────►│  [Node A] ──► [Node B] ──► END  │
                    │       │                    ▲     │
                    │       └── conditional ─────┘     │
                    │              edge                │
                    └──────────────────────────────────┘
                         State saved at every step via
                      Checkpointer → SQLite / Postgres
```

---

## 🔑 Key Concept Previews

### LCEL — The Pipe Operator

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = ChatPromptTemplate.from_template("Answer: {question}") \
      | ChatOpenAI(model="gpt-4o-mini") \
      | StrOutputParser()

chain.invoke({"question": "What is LangChain?"})
```

### RAG Pipeline

```python
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

vectorstore = Chroma.from_documents(docs, embedding=embeddings)
retriever   = vectorstore.as_retriever()
qa_chain    = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa_chain.invoke({"query": "Explain LangGraph checkpointing"})
```

### LangGraph — Stateful Agent

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    messages: list

def call_llm(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_conditional_edges("llm", should_continue)
graph.set_entry_point("llm")
app = graph.compile()
```

### HITL — Human Approval Gate

```python
# Pause BEFORE a critical node
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_action"]
)

# Stream until pause
for event in app.stream(inputs, config):
    print(event)

# Human reviews → resume
for event in app.stream(None, config):   # None = resume
    print(event)
```

### Corrective RAG (CRAG)

```python
# CRAG adds an evaluation step after retrieval
retrieve → grade_documents → (good? use them : web_search) → generate
```

---

## 🆚 AgentExecutor vs LangGraph

| Feature | AgentExecutor | LangGraph |
|---|---|---|
| State persistence | ❌ In-memory only | ✅ SQLite / Postgres |
| Human-in-the-Loop | ❌ Not supported | ✅ Native interrupt |
| Conditional branching | Limited | ✅ Full DAG support |
| Streaming | Token only | ✅ Token + Node + Event |
| Debugging | Basic verbose | ✅ LangSmith node traces |
| Multi-agent | Manual | ✅ Supervisor pattern |
| **Use for** | Learning / prototypes | ✅ **Production** |

---

## 🏗️ Module Project: Blog Writing Agent

```
User Topic
    │
    ▼
[Plan]  → decide structure + subtopics
    │
    ▼
[Research] → web search per subtopic (Tavily)
    │
    ▼
[Write] → generate blog sections
    │
    ▼
[Review] → score quality, loop back if score < 7
    │
    ▼
[Publish] → final markdown blog post
```

**Stack**: LangGraph · Tavily · OpenAI · LangSmith · Streamlit

---

## ⏱️ Estimated Time

| Activity | Time |
|---|---|
| Part 1 — LangChain (8 theory files + notebooks) | 1 week |
| Part 2 — LangGraph (9 theory files + notebooks) | 2 weeks |
| Module project | 3–4 days |
| Exercises | 1–2 days |
| **Total** | **~3 weeks** |

---

## 🔧 Setup

```bash
pip install langchain langchain-openai langchain-anthropic langchain-community \
            langgraph langsmith tavily-python \
            chromadb faiss-cpu \
            psycopg2-binary redis \
            fastapi uvicorn streamlit \
            python-dotenv pydantic rich jupyterlab
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
TAVILY_API_KEY=your_tavily_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=module-03-langchain-langgraph
```

---

## 📡 Libraries Covered

| Library | Purpose |
|---|---|
| `langchain` | Core framework — chains, prompts, agents, memory |
| `langchain-openai` | OpenAI GPT-4o integration |
| `langchain-community` | 1000+ community tools, loaders, vector stores |
| `langgraph` | Stateful graph-based agent workflows |
| `langsmith` | Tracing, evaluation, cost + latency monitoring |
| `tavily-python` | Web search API for LLM agents |
| `chromadb` / `faiss-cpu` | Local vector stores for RAG |
| `streamlit` | UI for chatbot demos |
| `psycopg2-binary` | Postgres checkpointing |

---

## 🏆 Module Completion Checklist

**Part 1 — LangChain**
- [ ] Built a basic chain using LCEL pipe operator
- [ ] Used ChatModel with PromptTemplate and output parsers
- [ ] Extracted structured data using PydanticOutputParser
- [ ] Built a parallel chain with RunnableParallel
- [ ] Loaded a PDF and split it into chunks
- [ ] Created a Chroma vector store and ran similarity search
- [ ] Built a full RAG Q&A pipeline
- [ ] Defined custom tools and built a ReAct agent with AgentExecutor

**Part 2 — LangGraph**
- [ ] Built first StateGraph with nodes and edges
- [ ] Implemented all 4 workflow patterns (Sequential, Parallel, Conditional, Iterative)
- [ ] Built a persistent chatbot with SQLite checkpointer
- [ ] Streamed token-level output from a LangGraph agent
- [ ] Set up LangSmith tracing and viewed node-level traces
- [ ] Integrated tools and RAG inside a LangGraph workflow
- [ ] Implemented a HITL approval gate
- [ ] Built a nested subgraph architecture
- [ ] Implemented short-term and long-term memory
- [ ] Completed the Blog Writing Agent project end-to-end

---

## ⬅️ Previous Module

[Module 02 — LLM Core Skills](../Module%2002%20-%20LLM%20Core%20Skills/)

## ➡️ Next Module

[Module 04 — Visual Agent Workflows with LangFlow](../Module%2004%20-%20LangFlow/)
