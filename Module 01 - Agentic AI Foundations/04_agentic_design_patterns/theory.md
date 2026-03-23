# 04 — Agentic Design Patterns

> *Andrew Ng's four foundational patterns — the vocabulary every AI engineer must know.*

---

## 4.1 Overview

In 2024, Andrew Ng identified **four core agentic design patterns** that underpin virtually all production agent systems. These patterns are not mutually exclusive — the most capable agents combine all four.

```
┌─────────────────────────────────────────────────────┐
│            FOUR AGENTIC DESIGN PATTERNS              │
├─────────────────┬───────────────────────────────────┤
│  1. REFLECTION  │  Agent reviews & improves output  │
│  2. TOOL USE    │  Agent uses external capabilities │
│  3. PLANNING    │  Agent decomposes & sequences tasks│
│  4. MULTI-AGENT │  Multiple agents collaborate       │
└─────────────────┴───────────────────────────────────┘
```

---

## 4.2 Pattern 1: Reflection

### What It Is
The agent **critiques and revises its own output** in an iterative loop until quality meets a threshold.

```
Generate Output → Evaluate Quality → Is it good enough?
       ▲              │                    │
       │              │                   YES → Return
       └──────────────┘ NO → Revise
```

### How It Works
1. Agent generates an initial response
2. A **critic** (same or separate LLM) evaluates the response
3. Critique is fed back as new context
4. Agent generates an improved version
5. Loop until quality score passes threshold or max iterations reached

### Implementation Patterns
- **Self-reflection**: same LLM generates and critiques
- **Cross-reflection**: separate critic LLM (more objective)
- **Human critique**: human provides feedback, agent revises

### Real-World Examples
- Code generation: write code → run tests → fix based on errors → repeat
- Essay writing: draft → peer review → rewrite
- Tool call: call API → parse unexpected output → retry differently

### Key Research: Reflexion (2023)
> Agents use verbal reinforcement learning — instead of gradient updates, they reflect in natural language and update a "memory" of what went wrong.

### Advantages
- Dramatically improves output quality
- No training required — pure prompting
- Works with any LLM

### Disadvantages
- Increases latency (multiple LLM calls)
- Risk of looping without convergence
- More expensive (2–5× more tokens)

---

## 4.3 Pattern 2: Tool Use

### What It Is
The agent is equipped with **tools** — functions it can call to interact with external systems, extending its capabilities far beyond text generation.

```
Agent ──► Tool Registry ──► Tool Function ──► Observation
                                                    │
                                                    ▼
                                             Agent processes
                                             observation and
                                             continues reasoning
```

### Tool Categories

| Category | Examples |
|---|---|
| **Search & Retrieval** | web_search, vector_db_query, SQL_query |
| **Code Execution** | python_repl, bash_shell, javascript_runner |
| **External APIs** | weather_api, stock_api, email_api, calendar |
| **File Operations** | read_file, write_file, list_directory |
| **Communication** | send_email, post_slack, create_ticket |
| **Data Processing** | data_analysis, chart_generation, pdf_reader |

### Tool Design Principles
1. **Single responsibility**: each tool does one thing well
2. **Clear schema**: function name, description, parameter types must be precise
3. **Error returns**: tools should return structured errors, not raise exceptions
4. **Idempotency**: safe to call multiple times with same params

### Tool Schema Example (OpenAI format)
```json
{
  "name": "web_search",
  "description": "Search the web for current information. Use this when you need up-to-date data not in your training.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query"
      },
      "num_results": {
        "type": "integer",
        "description": "Number of results to return (1-10)",
        "default": 5
      }
    },
    "required": ["query"]
  }
}
```

---

## 4.4 Pattern 3: Planning

### What It Is
The agent **decomposes a complex goal into a sequence of sub-tasks** and executes them in a logical order, adapting the plan when needed.

```
Goal → Plan → [Task1, Task2, Task3, ...] → Execute → Evaluate → Re-plan if needed
```

### Planning Approaches

**Upfront Planning (Plan-then-Execute)**
```
1. Agent receives goal
2. Generates full plan: [step1, step2, step3...]
3. Executes each step sequentially
4. Re-plans if a step fails
```

**Reactive Planning (ReAct style)**
```
1. Agent receives goal
2. Takes one action at a time
3. Observes result
4. Decides next action based on current state
```

**Hierarchical Planning**
```
High-level plan: [Research, Write, Review, Publish]
    ↓
Low-level sub-tasks per step:
  Research → [search_web, read_papers, summarize_findings]
  Write → [outline, draft_intro, draft_body, draft_conclusion]
```

### Plan Representation Formats
- Numbered list in the scratchpad
- JSON structure with dependencies
- Directed Acyclic Graph (DAG)
- PDDL (for formal planning)

### Re-planning Triggers
- Tool call returns an error
- Output quality check fails
- New information contradicts the plan
- Human feedback changes requirements

---

## 4.5 Pattern 4: Multi-Agent

### What It Is
**Multiple specialized agents collaborate** to accomplish tasks that would be too complex, too broad, or too inefficient for a single agent.

```
          ┌─────────────────────────────────┐
          │      ORCHESTRATOR AGENT         │
          │  (receives goal, delegates)     │
          └──────────────┬──────────────────┘
                 ┌───────┼───────┐
                 ▼       ▼       ▼
          🔍 Research  💻 Coder  📝 Writer
             Agent     Agent    Agent
```

### Why Multi-Agent?
1. **Context window limits**: one agent can't hold everything in mind
2. **Specialization**: a coding agent can have a different system prompt, tools, and model than a research agent
3. **Parallelism**: multiple agents work simultaneously on independent subtasks
4. **Quality**: separate critic agents provide more objective feedback

### Communication Patterns
| Pattern | Description |
|---|---|
| **Sequential** | Agent A → Agent B → Agent C (pipeline) |
| **Parallel** | All agents run simultaneously, results merged |
| **Hierarchical** | Manager delegates to workers, collects results |
| **Group Chat** | All agents discuss to reach consensus |

### Key Challenge: Coordination
- Who decides when handoff happens?
- How do agents share state?
- What happens when one agent fails?
- How do you prevent conflicting actions?

---

## 4.6 Combining All Four Patterns

Production-grade agents combine all patterns:

```
User Goal
    │
    ▼
[PLANNING] Decompose into 5 sub-tasks
    │
    ├──► Sub-task 1 → [TOOL USE] web_search → [REFLECTION] evaluate quality → retry if poor
    │
    ├──► Sub-task 2 → [MULTI-AGENT] spin up specialist agent (coder)
    │                     └──► [TOOL USE] run_code → [REFLECTION] fix errors
    │
    └──► Sub-task 3 → [TOOL USE] write_report → [REFLECTION] review + revise
```

---

## 4.7 Pattern Selection Guide

| If you need... | Use Pattern |
|---|---|
| Better output quality without training | **Reflection** |
| Access to external data or capabilities | **Tool Use** |
| To handle complex, multi-step goals | **Planning** |
| To parallelize or specialize | **Multi-Agent** |
| Production-grade quality | **All four combined** |

---

## 📌 Key Takeaways

1. These 4 patterns are the **fundamental vocabulary** of agentic AI
2. **Tool Use** is the most universally applied — almost every agent uses it
3. **Reflection** is the cheapest way to improve quality without fine-tuning
4. **Planning** separates capable agents from simple prompt-response loops
5. **Multi-Agent** is necessary for tasks beyond one agent's context or capability
