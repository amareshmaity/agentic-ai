# 🤖 Mastering Agentic AI — Industry-Standard Complete Course

> **Theory-first, practically-driven. Learn Agentic AI the way the industry actually builds it.**
> Covers: LangChain · LangFlow · Agno · CrewAI · AutoGen · Production Deployment

---

## 📌 Prerequisites

| Area | Expected Level |
|---|---|
| Python | Advanced (async, OOP, type hints, decorators) |
| ML / Deep Learning | Comfortable with model training & inference |
| LLMs & GenAI | RAG, embeddings, vector DBs, LLM API usage |
| Software Engineering | REST APIs, Git, Docker basics |

---

## 🗺️ Course Roadmap at a Glance

```
Module 01 → Agentic AI Foundations & Mental Models
Module 02 → LLM Core Skills for Agents
Module 03 → Building Agents with LangChain & LangGraph
Module 04 → Visual Agent Workflows with LangFlow
Module 05 → High-Performance Agents with Agno
Module 06 → Multi-Agent Teams with CrewAI
Module 07 → Conversational Multi-Agent Systems with AutoGen
Module 08 → Memory, Knowledge & Advanced RAG
Module 09 → Planning, Reasoning & Decision-Making  ← DEEPENED
Module 10 → Multi-Agent Orchestration & Patterns
Module 11 → Agent Communication Protocols & Interoperability  ← NEW
Module 12 → Multimodal & Embodied Agents  ← NEW
Module 13 → Self-Improving & Learning Agents  ← NEW
Module 14 → Evaluation, Safety, Guardrails & Alignment  ← DEEPENED
Module 15 → Production Deployment & MLOps for Agents
Module 16 → Capstone Projects (Industry-Grade)
```

---

## 📖 Module 01: Agentic AI Foundations & Mental Models

> *Build the right mental model before writing a single line of code.*

### Theory
- What is Agentic AI? → Autonomy, goal-directedness, proactiveness
- Agentic vs Traditional AI pipelines — key architectural differences
- The **Perception → Reasoning → Action → Memory** loop (PRAM)
- Types of AI Agents:
  - Simple Reflex → Model-Based → Goal-Based → Utility-Based → Learning Agents
  - Hierarchical agents, cooperative agents, adversarial agents
- Agent cognitive architecture: LLM as the brain
- **Agentic design patterns** (Andrew Ng): Reflection, Tool Use, Planning, Multi-Agent
- The "degree of autonomy" spectrum — when to automate vs keep humans in loop
- Key research milestones: ReAct, Toolformer, Reflexion, Generative Agents, Voyager, SWE-agent

### Practical
- [ ] Analyze real-world agent traces (LangSmith / LangFuse dashboards)
- [ ] Trace a ReAct agent step-by-step on paper before coding
- [ ] Identify agent patterns in existing AI products (ChatGPT, Perplexity, Devin, Cursor)

### Key Papers to Read
| Paper | Core Idea |
|---|---|
| ReAct (2022) | Interleave reasoning with actions |
| Reflexion (2023) | Self-reflection for error correction |
| Generative Agents (2023) | Believable human-like agent behavior (memory + planning) |
| Toolformer (2023) | LLMs that learn to use tools autonomously |
| Voyager (2023) | Lifelong learning agent in Minecraft — skill library + curriculum |
| SWE-agent (2024) | Autonomous software engineering agent |

---

## 📖 Module 02: LLM Core Skills for Agents

> *The foundation every agent is built on — master LLM interactions for agentic use.*

### Theory
- **Prompt engineering for agents**: system prompts, persona, constraints, few-shot
- **Function Calling / Tool Use** — how it works at the protocol level
- **Structured output**: JSON mode, Pydantic models, `instructor` library
- Context window management — token budgeting, chunking strategies
- **LLM selection guide**: GPT-4o, Gemini 1.5 Pro, Claude 3.5, Llama 3, Mistral — when to use what
- Cost vs quality vs latency tradeoffs for agent workloads
- **LLM routing & fallback** — LiteLLM, PortKey gateway patterns
- Streaming LLM responses — SSE, WebSockets, partial token handling

### Practical
- [ ] Build a robust function-calling pipeline (OpenAI + Gemini + Claude side-by-side)
- [ ] Use `instructor` + Pydantic to enforce structured agent outputs
- [ ] Build a sliding-window + summarization context manager for long agent runs
- [ ] Configure LiteLLM with model fallback + cost limits
- [ ] Compare 3 LLMs on the same agentic task — document tradeoffs in a scorecard

### Tools & Libraries
```
openai  anthropic  google-generativeai  instructor  tiktoken  pydantic  litellm
```

---

## 📖 Module 03: Building Agents with LangChain & LangGraph

> *The most widely used agent framework in industry — master it end-to-end.*

### Theory
- LangChain architecture: Chains, Runnables, LCEL (LangChain Expression Language)
- **AgentExecutor** — tool routing, observation parsing, loop control
- **LangGraph** — stateful, graph-based workflows
  - Nodes, Edges, Conditional Edges, State schema
  - **Checkpointing & persistence**: SQLite → Redis → Postgres
  - **Human-in-the-Loop (HITL)**: interrupt, approve, modify, retry
  - Subgraphs — composable multi-level agent architectures
  - Streaming: token-level, node-level, full event stream
- Memory in LangChain: ConversationBuffer, Summary, VectorStore-backed
- **LangGraph Supervisor** — built-in multi-agent supervisor+worker pattern

### Practical
- [ ] Build a ReAct agent with AgentExecutor + 5 custom tools
- [ ] Migrate to LangGraph — stateful research agent with conditional branching
- [ ] Add Postgres checkpointer — resume interrupted agent runs
- [ ] Implement HITL approval gate — agent pauses for human review before critical action
- [ ] Build LangGraph Supervisor pattern — coordinator spawning specialized sub-agents
- [ ] Deploy as FastAPI REST API + WebSocket streaming endpoint
- [ ] Monitor full traces in LangSmith — latency, token cost, error attribution

### Project: **AI Research Assistant** (LangChain + LangGraph)
> Web search → Read articles → Synthesize → Cite sources → Human review gate → PDF report

```
Stack: LangGraph · Tavily · LangSmith · FastAPI · Redis · Postgres
```

---

## 📖 Module 04: Visual Agent Workflows with LangFlow

> *Low-code agent design — critical for rapid prototyping & non-technical collaboration.*

### Theory
- LangFlow architecture — flow-based programming for LLM apps
- Components: Agents, Chains, Tools, Prompts, Memory, Embeddings, VectorStores
- Flow design patterns: sequential, conditional branching, loops
- **API export** — turn visual flows into production REST endpoints
- LangFlow + LangSmith integration for observability
- When to use LangFlow vs code-first (and when NOT to use it)

### Practical
- [ ] Build a complete Hybrid RAG pipeline visually in LangFlow
- [ ] Create a multi-tool agent with conditional routing (if intent = X → route to agent Y)
- [ ] Export flow as a REST API and call it from a Python client
- [ ] Build a customer support bot — LangFlow + vector DB + CRM tool
- [ ] Add LangSmith tracing to a LangFlow deployment

### Project: **Customer Support Bot** (LangFlow)
> FAQ retrieval → CRM lookup → Ticket creation → Escalation routing — deployed as an API

```
Stack: LangFlow · ChromaDB · OpenAI · FastAPI
```

---

## 📖 Module 05: High-Performance Agents with Agno

> *The fastest, most lightweight multi-modal agent framework — built for production scale.*

### Theory
- Agno architecture: `Agent`, `Team`, `Tool`, `Model`, `Storage`, `Knowledge` abstractions
- Why Agno? — 10,000× faster instantiation than LangChain, native async, minimal overhead
- **Multimodal agents natively**: text, image, audio, video, PDF, tables
- **Agent Teams**: `coordinate` and `route` modes
- Built-in memory: session memory, persistent storage (PostgreSQL, SQLite)
- Built-in knowledge: PDF, CSV, JSON, web crawl, vector search
- Monitoring with Agno Platform

### Practical
- [ ] Build a basic Agno agent with web search, calculator, and file I/O tools
- [ ] Build a multimodal agent: analyze uploaded images + answer follow-up questions
- [ ] Create an Agno Team — coordinator + 3 specialist agents (researcher, analyst, writer)
- [ ] Implement long-term memory with PostgreSQL + pgvector via Agno storage
- [ ] Benchmark Agno vs LangChain: task latency, memory, cold start time
- [ ] Expose Agno agent as an async FastAPI endpoint

### Project: **Multimodal Data Analysis Agent** (Agno)
> Upload PDF/image/CSV → Extract insights → Generate charts + written report

```
Stack: Agno · PostgreSQL · pgvector · OpenAI · Gemini · FastAPI
```

---

## 📖 Module 06: Role-Based Multi-Agent Teams with CrewAI

> *The go-to framework for structured multi-agent collaboration with role-based design.*

### Theory
- CrewAI architecture: `Crew`, `Agent`, `Task`, `Process`, `Tool`, `Flow`
- **Role-based agent design**: persona, backstory, goal, constraints
- Process types: `Sequential` → `Hierarchical` → `Parallel`
- Task dependencies, conditional tasks, async tasks
- **Memory in CrewAI**: Short-term, Long-term, Entity, Contextual
- **Knowledge Sources**: PDF, CSV, JSON, web scraping — built-in chunking + retrieval
- **CrewAI Flows** — event-driven orchestration with `@start`, `@listen`, `@router`
- Manager LLM — separate, more powerful model for delegation decisions

### Practical
- [ ] Build a Sequential Crew: Researcher → Writer → SEO Editor → Publisher
- [ ] Build a Hierarchical Crew with a Manager agent delegating based on task type
- [ ] Implement custom tools: web scraper, SQL query, REST API caller
- [ ] Use Flows for a conditional multi-step workflow (if score > 8 → publish, else → revise)
- [ ] Add Entity memory — crew remembers entities (companies, people) across tasks
- [ ] Test + evaluate crew output quality using LangSmith + custom rubrics

### Project: **AI Content Marketing Crew** (CrewAI)
> Topic brief → SEO research → Blog draft → Social media posts → Editor review → Publish

```
Stack: CrewAI · OpenAI · Serper API · LangSmith · FastAPI
```

---

## 📖 Module 07: Conversational Multi-Agent Systems with AutoGen

> *Microsoft's framework for multi-agent conversation — powerful for reasoning & coding tasks.*

### Theory
- AutoGen architecture: `ConversableAgent`, `AssistantAgent`, `UserProxyAgent`
- Agent conversation patterns: **two-agent**, **group chat**, **nested chat**
- Speaker selection strategies: `auto`, `round_robin`, `random`, custom LLM-based
- **Code execution**: sandboxed Python/shell via Docker executor
- **Teachable agents** — learning & persisting facts from user feedback
- AutoGen Studio — visual multi-agent conversation builder
- **AutoGen 0.4 / AG2** — event-driven, actor-model redesign
- Combining AutoGen with external tools & LangChain chains

### Practical
- [ ] Build a two-agent loop: Coder ↔ Code Reviewer with iterative refinement
- [ ] Build a 4-agent group chat: PM → Developer → QA → DevOps with custom speaker logic
- [ ] Implement nested conversations — outer agent spawns an inner specialized agent crew
- [ ] Use AutoGen with local LLMs via Ollama + LiteLLM proxy
- [ ] Build a teachable agent that remembers user preferences across sessions
- [ ] Create a self-debugging code pipeline: generate → run → read error → fix → repeat

### Project: **AI Software Engineering Team** (AutoGen)
> GitHub issue → Architecture plan → Code generation → Test writing → Bug fixing → Docs

```
Stack: AutoGen · Docker sandbox · GPT-4o · GitHub API · LangFuse
```

---

## 📖 Module 08: Memory, Knowledge & Advanced RAG

> *Give your agents the right information at the right time.*

### Theory — Memory Taxonomy
- **Short-Term Memory (STM)**: in-context conversation buffer, sliding window, summarization
- **Long-Term Memory (LTM)**: vector stores, relational DBs, key-value stores
- **Episodic Memory**: log of past agent interactions → retrieval by similarity
- **Semantic Memory**: factual knowledge base → RAG, knowledge graphs
- **Procedural Memory**: learned skills, workflows, reusable tools
- Memory compression techniques: extractive summarization, LLM-based compression

### Theory — Advanced RAG Patterns
- Naive RAG → Advanced RAG → Modular RAG → Agentic RAG
- **Hybrid Search**: dense (embeddings) + sparse (BM25) + re-ranking (cross-encoder)
- **Agentic RAG**: agent decides *when*, *what*, and *how much* to retrieve
- **Self-RAG**: retrieve → generate → reflect → decide if retrieval was helpful
- **Corrective RAG (CRAG)**: retrieve → evaluate quality → correct if poor
- **Multi-hop RAG**: chain multiple retrievals for complex multi-part questions
- **GraphRAG**: knowledge graph + vector search for relationship-aware retrieval
- Embedding model selection: OpenAI, Cohere, BGE-M3, Nomic-embed

### Practical
- [ ] Build a Hybrid RAG pipeline: BM25 (Elasticsearch) + dense embedding + Cohere re-ranker
- [ ] Implement **Agentic RAG** — agent has a `retrieve` tool and decides when to use it
- [ ] Build **Self-RAG** — agent reflects on retrieved content quality before using it
- [ ] Build **GraphRAG** with Neo4j — extract entities/relations → store graph → query
- [ ] Implement episodic memory store — agent retrieves relevant past sessions
- [ ] Build a memory-augmented LangGraph agent with Postgres + pgvector

### Vector Databases Covered
```
ChromaDB (local dev) · Qdrant (scalable OSS) · Pinecone (managed) · pgvector (SQL-native)
Neo4j (graph) · Elasticsearch (hybrid BM25 + vector)
```

---

## 📖 Module 09: Planning, Reasoning & Decision-Making (Deep)

> *Advanced cognition — the gap between a chatbot and a true autonomous agent.*

### Theory — Reasoning Patterns
- **Chain-of-Thought (CoT)**: step-by-step verbal reasoning
- **Self-Consistency**: generate N reasoning paths → majority vote
- **Tree-of-Thought (ToT)**: explore multiple reasoning branches, prune with evaluation
- **Graph-of-Thought (GoT)**: non-linear, revisitable reasoning graph
- **Step-back prompting**: abstract the problem before solving
- **Analogical reasoning**: map known solutions to new problems
- **Causal & counterfactual reasoning**: "what if" analysis
- **Meta-reasoning**: reasoning about your own reasoning quality

### Theory — Planning Algorithms
- Classical AI planning: STRIPS, PDDL (conceptual grounding)
- **LLM-based planning**: translate goals → subtask list → execute
- **Plan-and-Execute**: plan upfront → execute → evaluate → re-plan on failure
- **Hierarchical Task Networks (HTN)**: decompose tasks recursively
- **Monte Carlo Tree Search (MCTS) for agents**: simulate branches → backpropagate scores
- Contingency planning: if X fails, do Y — building robust agent plans

### Theory — Decision-Making
- Confidence estimation & calibration — when is the agent "sure enough" to act?
- **Ask vs Act**: decision boundary — when to request clarification vs proceed
- Risk-aware decision-making — weighting potential harms
- **Exploration vs Exploitation** in agent loops
- World models — simulate outcomes internally before committing to actions

### Practical
- [ ] Implement **Tree-of-Thought** agent for multi-step logical problems
- [ ] Build a **Plan-and-Execute** agent with automatic re-planning on tool failure
- [ ] Implement **self-consistency**: run 5 reasoning chains → majority-vote final answer
- [ ] Build a **self-reflection loop**: execute → score output → identify errors → refine
- [ ] Implement an **MCTS-based decision tree** for a planning task
- [ ] Build a **confidence estimator** — agent asks clarifying question if score < threshold

---

## 📖 Module 10: Multi-Agent Orchestration & Patterns

> *Design systems where multiple agents collaborate reliably at scale.*

### Theory — Coordination Patterns
| Pattern | Description | When to Use |
|---|---|---|
| **Supervisor-Worker** | Manager delegates to specialists | Complex tasks with clear roles |
| **Sequential Pipeline** | Output of A → input of B | Ordered step-by-step workflows |
| **Parallel Fan-out/Fan-in** | Broadcast → collect → merge | Independent subtasks (MapReduce) |
| **Peer Debate** | Agents argue positions → synthesize | High-stakes decisions |
| **Mixture-of-Agents (MoA)** | Multiple LLMs vote/merge answers | Accuracy-critical outputs |
| **Assembly Line** | Each agent transforms artifact | Document processing pipelines |

### Theory — Orchestration Deep Dive
- DAG-based workflow design — dependency graphs for agent tasks
- State machines for complex multi-turn agent conversations
- **Event-driven orchestration** — agents react to events, not just sequential calls
- Handling partial failures — retry, skip, compensate, escalate strategies
- Backpressure & throttling in high-throughput multi-agent systems

### Practical
- [ ] Implement **Supervisor-Worker** pattern from scratch with LangGraph
- [ ] Build **Mixture-of-Agents** — 3 LLMs generate answers → aggregator merges best
- [ ] Create a **parallel fan-out** system — distribute research across 5 parallel agents
- [ ] Implement **Peer Debate** pattern — two agents argue, judge agent scores arguments
- [ ] Build an event-driven orchestration with Redis Streams as the agent message bus
- [ ] Implement retry + compensate logic for graceful multi-agent failure handling

---

## 📖 Module 11: Agent Communication Protocols & Interoperability *(NEW)*

> *The emerging standards that let agents talk to each other and to tools.*

### Theory
- **Model Context Protocol (MCP)** by Anthropic
  - Architecture: Host, Client, Server, Resources, Tools, Prompts
  - MCP transport: stdio, SSE
  - Building MCP servers (expose your own tools to any agent)
  - Building MCP clients (connect agents to any MCP tool server)
- **Agent-to-Agent (A2A) Protocol** by Google
  - Agent Cards — self-describing agent capabilities
  - Task lifecycle: submitted → working → completed/failed
  - Push notifications for async tasks
- **Agent Protocol** (open standard, LangChain-compatible)
- **OpenAI Agents SDK** — handoffs, routines, context variables
- Cross-framework interoperability — calling a CrewAI agent from LangGraph

### Practical
- [ ] Build an **MCP Server** — expose a custom tool (database query) as an MCP endpoint
- [ ] Connect Claude Desktop to your custom MCP server
- [ ] Build an **A2A Agent** — agent publishes its Agent Card, receives tasks via A2A protocol
- [ ] Implement cross-framework handoff: LangGraph supervisor → CrewAI specialist agent
- [ ] Build a **multi-framework pipeline**: Agno researcher → AutoGen coder → LangGraph reviewer

---

## 📖 Module 12: Multimodal & Embodied Agents *(NEW)*

> *Agents that see, hear, speak, browse, and interact with the world.*

### Theory
- **Vision-Language agents**: understanding images/screenshots + taking action
- **Audio / voice agents**: speech-to-text → agent reasoning → text-to-speech (real-time)
- **Computer-use agents**: agents that control a browser or desktop (mouse, keyboard)
- **Document intelligence agents**: PDF parsing, table extraction, form understanding
- **Video understanding**: frame extraction → temporal reasoning → action
- Multimodal memory — storing and retrieving non-text modalities
- Tool use for embodied actions: Playwright (browser), PyAutoGUI (desktop)

### Practical
- [ ] Build a **vision agent** — upload screenshot → agent identifies UI elements → takes actions
- [ ] Build a **real-time voice agent** — Whisper (STT) → LLM → ElevenLabs (TTS) pipeline
- [ ] Build a **browser-use agent** — Playwright + LLM to autonomously browse and extract data
- [ ] Build a **document intelligence agent** — upload PDF → extract tables + text → Q&A
- [ ] Create a multimodal memory store — store image embeddings + retrieve by semantic query

### Key Libraries
```
playwright  browser-use  PyAutoGUI  whisper  elevenlabs  pymupdf  unstructured
```

---

## 📖 Module 13: Self-Improving & Learning Agents *(NEW)*

> *Agents that get better over time — the frontier of agentic AI.*

### Theory
- **Reflexion pattern**: error → verbal self-critique → refined strategy → retry
- **Curriculum learning for agents**: progressively harder tasks → growing capability
- **Fine-tuning from agent trajectories**: collect (state, action, reward) → DPO / SFT
- **RLHF / RLAIF on agent outputs**: human or AI preference labels → policy improvement
- **Skill/tool library building**: agent discovers useful sub-routines → saves for reuse (Voyager pattern)
- **Self-play & adversarial improvement**: red-team agent vs blue-team agent
- Meta-learning for agents: learn to learn from few examples

### Practical
- [ ] Implement **Reflexion loop** — agent critiques its own failed attempts and re-plans
- [ ] Build a **skill library** — agent saves successful tool-call sequences as reusable skills
- [ ] Collect agent trajectories → fine-tune a small model with **DPO** to mimic good behavior
- [ ] Set up a **self-play loop** — one agent generates test cases, another solves them
- [ ] Implement curriculum: start with easy tasks, auto-scale difficulty based on success rate

---

## 📖 Module 14: Evaluation, Safety, Guardrails & Alignment (Deep)

> *Production agents must be reliable, safe, measurable, and aligned.*

### Theory — Evaluation
- **Evaluation dimensions**: task completion rate, step efficiency, tool selection accuracy, cost, latency
- Trajectory-level vs output-level evaluation
- Benchmark suites: **AgentBench**, **SWE-Bench**, **WebArena**, **GAIA**, **TAU-Bench**
- LLM-as-judge evaluation — scoring with a separate grader model
- A/B testing agent configurations in production
- Regression testing — no silent capability degradation across versions

### Theory — Failure Modes
- **Hallucination in agent context** — fabricating tool results, making up API responses
- **Infinite loops** — agent stuck in perception-action cycles
- **Context window overflow** — losing critical info in long runs
- **Tool misuse** — wrong tool selected, wrong parameters passed
- **Prompt injection** — malicious content in tool output hijacks agent
- **Cascading failures** — one bad agent poisons downstream agents
- Debugging techniques: replay, step-through tracing, counterfactual testing

### Theory — Guardrails & Safety
- Input/output guardrails: **NeMo Guardrails**, **Guardrails AI**, **LlamaGuard**
- Action-level permissions — whitelist/blacklist allowed tools and APIs
- Sandboxed execution — agent code runs in isolated containers
- Rate limiting, cost caps, token budgets
- **Prompt injection defense** — sanitization, instruction hierarchy, spotlighting
- **Red-teaming agentic systems** — systematic adversarial testing

### Theory — Alignment & Ethics
- Intent alignment — does the agent do what the *user actually wants*?
- Value alignment — does the agent's behavior reflect broader human values?
- **Constitutional AI** principles applied to agent design
- Transparency & explainability — can users understand *why* the agent acted?
- Accountability — logging, audit trails, reversible actions
- Bias detection in agent decisions

### Practical
- [ ] Set up **LangSmith evals** — automated regression test suite (50+ test cases)
- [ ] Use **LangFuse** for cost, latency, quality tracking dashboard
- [ ] Implement **LLM-as-judge** evaluator — grader model scores agent outputs 1–5
- [ ] Add **NeMo Guardrails** to a customer-facing agent (topic rails + fact-checking)
- [ ] Red-team your agent: test prompt injection via 10 attack vectors
- [ ] Implement full audit logging — every agent action stored with timestamp + reversibility flag
- [ ] Run your agent on **AgentBench** — compare vs baseline, document improvements

---

## 📖 Module 15: Production Deployment & MLOps for Agents

> *Take agents from prototype to scalable, monitored, production systems.*

### Theory
- **Production architecture patterns**: synchronous API, async task queue, event-driven
- Async agent execution: **Celery**, **RQ**, **ARQ** (async Python) + Redis/RabbitMQ
- Horizontal scaling with **Kubernetes** — agent worker pods
- **Caching strategies**: LLM response cache (GPTCache), tool result cache, semantic cache
- **LLM gateways**: LiteLLM, PortKey — routing, cost control, model fallback, load balancing
- Prompt & config version control: PromptLayer, Langfuse prompt management
- **CI/CD for agents**: push code → run eval suite → deploy if KPIs pass
- Secret & credential management: Vault, AWS Secrets Manager, environment isolation

### Practical
- [ ] Deploy a LangGraph agent as a **FastAPI** service in a **Docker** container
- [ ] Build an async agent task queue with **Celery + Redis** — fire-and-forget with result polling
- [ ] Configure **LiteLLM gateway** — fallback GPT-4o → Claude → Gemini on rate limits
- [ ] Set up **GitHub Actions CI/CD**: test → eval → docker build → deploy to cloud
- [ ] Deploy to **AWS ECS** (Fargate) and **GCP Cloud Run** — compare cost & scaling
- [ ] Set up full observability: **LangFuse** dashboard — traces, cost, errors, latency P95
- [ ] Implement semantic caching — identical/similar queries return cached results

### Deployment Stack
```
FastAPI · Docker · Kubernetes · Celery · Redis · LiteLLM
AWS ECS (Fargate) · GCP Cloud Run · Railway · GitHub Actions
LangFuse · LangSmith · Arize Phoenix
```

---

## 📖 Module 16: Capstone Projects (Industry-Grade)

> *Build 5 production-quality systems that belong in a professional portfolio.*

---

### 🏗️ Capstone 1: Autonomous Research & Report Agent
**Frameworks**: LangGraph + LangChain
> End-to-end: query → multi-hop web search → read papers → synthesize → cite sources → PDF report
- Agentic RAG + Hybrid Search + Self-RAG loop
- HITL review gate before final report generation
- Full LangSmith observability

```
Stack: LangGraph · Tavily · ArXiv API · Postgres · FastAPI · React frontend
```

---

### 🏗️ Capstone 2: AI Content Marketing Crew
**Frameworks**: CrewAI + LangFlow (visual pipeline)
> Topic brief → SEO research → Blog draft → Social posts → Quality review → Publish API
- Hierarchical crew with Manager LLM
- Entity memory — remembers brand voice and past topics
- Conditional Flows: if quality score < 7 → revise loop

```
Stack: CrewAI · Serper · OpenAI · LangFlow · WordPress API
```

---

### 🏗️ Capstone 3: AI Software Engineering Team
**Frameworks**: AutoGen + Agno
> GitHub Issue → Architecture plan → Code generation → Test writing → Debugging → PR creation
- Self-debugging loop: run → error → reflect → fix
- Docker sandboxed code execution
- Skill library: agent saves reusable code patterns

```
Stack: AutoGen · Docker sandbox · GitHub API · Agno · pytest · LangFuse
```

---

### 🏗️ Capstone 4: Enterprise Customer Support System
**Frameworks**: LangGraph + LangFlow
> Intent routing → FAQ retrieval → CRM lookup → Ticket creation → Human handoff → Analytics
- Multi-hop retrieval for complex queries
- NeMo Guardrails: topic rails + PII detection
- Full audit log for compliance

```
Stack: LangGraph · Qdrant · Zendesk API · LangFlow · NeMo Guardrails · LangFuse
```

---

### 🏗️ Capstone 5: Personal AI Chief of Staff *(Full-Stack Production)*
**Frameworks**: Agno + LangGraph (hybrid)
> Calendar + email + tasks + web search + multimodal input + long-term memory + proactive nudges
- Voice interface: Whisper → Agent → ElevenLabs TTS
- GraphRAG personal knowledge base
- Reflexion loop for self-improvement on failed tasks
- Full CI/CD: GitHub Actions → Docker → GCP Cloud Run → LangFuse monitoring

```
Stack: Agno · LangGraph · Google APIs · Notion · Redis · Neo4j
FastAPI · Next.js · Docker · GCP Cloud Run · GitHub Actions
```

---

## 📚 Framework Quick Reference

| Framework | Best For | Key Strength | Maturity |
|---|---|---|---|
| **LangChain/Graph** | Stateful, complex workflows | Largest ecosystem, most features | ⭐⭐⭐⭐⭐ |
| **LangFlow** | Visual prototyping, low-code | Speed, team collaboration | ⭐⭐⭐⭐ |
| **Agno** | High-performance, multimodal | Speed, simplicity, native async | ⭐⭐⭐⭐ |
| **CrewAI** | Role-based team collaboration | Role design, task delegation, memory | ⭐⭐⭐⭐⭐ |
| **AutoGen** | Conversational multi-agent | Code gen, group chat, teachable | ⭐⭐⭐⭐⭐ |

---

## 🛠️ Full Course Tech Stack

```
LLMs:          OpenAI GPT-4o · Anthropic Claude 3.5 · Google Gemini · Llama 3 (Ollama)
Frameworks:    LangChain · LangGraph · LangFlow · Agno · CrewAI · AutoGen
Memory & DBs:  PostgreSQL + pgvector · Qdrant · Pinecone · ChromaDB · Redis · Neo4j · Elasticsearch
RAG:           Hybrid (BM25 + dense) · Self-RAG · Agentic RAG · GraphRAG · CRAG
Protocols:     MCP (Anthropic) · A2A (Google) · Agent Protocol · OpenAI Agents SDK
Multimodal:    Whisper · ElevenLabs · Playwright · browser-use · PyMuPDF · Unstructured
Observability: LangSmith · LangFuse · Arize Phoenix · OpenTelemetry
Safety:        NeMo Guardrails · Guardrails AI · LlamaGuard
Deployment:    FastAPI · Docker · Kubernetes · Celery · LiteLLM · PortKey
Cloud:         AWS ECS / Lambda · GCP Cloud Run · Railway
CI/CD:         GitHub Actions + automated eval pipelines
```

---

## 📅 Suggested Learning Timeline

| Module | Topic | Duration |
|---|---|---|
| 01 | Foundations | 1 week |
| 02 | LLM Core Skills | 1 week |
| 03 | LangChain & LangGraph | 3 weeks |
| 04 | LangFlow | 1 week |
| 05 | Agno | 1.5 weeks |
| 06 | CrewAI | 2 weeks |
| 07 | AutoGen | 2 weeks |
| 08 | Memory & Advanced RAG | 2 weeks |
| 09 | Planning & Reasoning (Deep) | 2 weeks |
| 10 | Multi-Agent Orchestration | 1.5 weeks |
| 11 | Protocols & Interoperability | 1 week |
| 12 | Multimodal & Embodied Agents | 1.5 weeks |
| 13 | Self-Improving Agents | 1.5 weeks |
| 14 | Evaluation, Safety & Alignment | 2 weeks |
| 15 | Production Deployment | 2 weeks |
| 16 | Capstone Projects | 4 weeks |
| **Total** | **All Modules** | **~30 weeks** ✅ |

---

## ✅ Learning Principles

1. **Theory before code** — understand *why* before *how*; read the source paper for every major pattern
2. **Cross-framework exercises** — implement the same agent in 2 frameworks; compare architecture, code, and output
3. **Instrument from day 1** — use LangSmith/LangFuse from your very first agent, always trace
4. **Break your own agents** — red-team, inject bad inputs, force failures — resilience is a skill
5. **Benchmark everything** — task completion rate, cost per run, latency P95 — measure before and after changes
6. **Deploy early** — Dockerize from Module 3; don't let deployment be a surprise at the end
7. **Build your portfolio** — every capstone project should be on GitHub with a clear README and demo video

---

<p align="center">
  <b>🚀 Build AI that doesn't just answer questions — it takes action. 🚀</b>
</p>
