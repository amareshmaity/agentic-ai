# 02 — Types of AI Agents

> Based on Russell & Norvig's AI classification, extended for the LLM era.

---

## 2.1 The Agent Taxonomy

```
                        AI Agents
                           │
         ┌─────────────────┼─────────────────┐
    Simple Reflex     Model-Based         Goal-Based
                           │
                    ┌──────┴──────┐
               Utility-Based   Learning
                                   │
                            Hierarchical / Multi-Agent
```

---

## 2.2 Type 1: Simple Reflex Agent

### What It Is
Makes decisions based **solely on the current percept** (input). No memory of past states, no planning.

```
Percept → Condition-Action Rules → Action
```

### Mechanism
```
if condition_A: take action_X
elif condition_B: take action_Y
```

### Characteristics
- No internal state or memory
- Deterministic — same input always gives same output
- Fast and cheap to build
- Brittle — breaks on unseen inputs

### LLM-Era Example
A basic chatbot with a fixed system prompt and no memory. Each message is processed independently.

### When to Use
- Simple FAQ bots
- Fixed decision trees
- High-volume, low-complexity routing

### Limitation
Cannot handle partially observable environments. If the full context is not in the current input, it fails.

---

## 2.3 Type 2: Model-Based Reflex Agent

### What It Is
Maintains an **internal state** (model of the world) that it updates with each percept. Makes decisions based on both current input AND internal state.

```
Percept + Internal State → World Model Update → Action
```

### Characteristics
- Has memory of past states
- More robust than simple reflex
- Still rule-based at its core (no reasoning about goals)

### LLM-Era Example
A chatbot with a **conversation buffer** — it remembers the last N messages and uses them for context.

```python
history = []  # internal state
def model_based_agent(user_input):
    history.append({"role": "user", "content": user_input})
    response = llm.call(messages=history)
    history.append({"role": "assistant", "content": response})
    return response
```

### When to Use
- Multi-turn conversational assistants
- State machines that track workflow progress

---

## 2.4 Type 3: Goal-Based Agent

### What It Is
Has an **explicit goal** and reasons about what actions will achieve that goal. Performs **search and planning** to find action sequences.

```
Percept + Goal → Planning → Action Sequence → Execute
```

### Characteristics
- Deliberative reasoning — thinks before acting
- Plans multiple steps into the future
- Can handle novel situations not covered by fixed rules
- More flexible but computationally expensive

### LLM-Era Example
A ReAct agent that receives a goal ("Research the latest AI news and write a summary") and plans: web search → read → synthesize → write.

```
Goal: "Find the top 3 AI papers from this week and summarize them"

Plan:
  Step 1: Search "latest AI papers this week"
  Step 2: Read each paper abstract
  Step 3: Extract key contributions
  Step 4: Write a structured summary
```

### When to Use
- Research agents
- Task automation with variable objectives
- Any multi-step goal that requires planning

---

## 2.5 Type 4: Utility-Based Agent

### What It Is
Like goal-based agents, but instead of a binary "goal achieved / not achieved", uses a **utility function** to rank outcomes by desirability. Selects actions that **maximize expected utility**.

```
Percept + Utility Function → Expected Utility Calculation → Best Action
```

### Characteristics
- Handles tradeoffs: speed vs quality, cost vs accuracy
- Can reason under uncertainty
- More nuanced than "goal vs no-goal" binary

### LLM-Era Example
An agent that selects which LLM to call based on a utility function:

```python
def utility(model, task):
    cost_score = 1 / model.cost_per_token
    quality_score = model.benchmark_score[task.type]
    speed_score = 1 / model.avg_latency
    return 0.4 * quality_score + 0.4 * cost_score + 0.2 * speed_score

# Agent chooses the model with highest utility for this task
best_model = max(available_models, key=lambda m: utility(m, task))
```

### When to Use
- Multi-model routing systems
- Agents with cost/quality constraints
- Decision-making under uncertainty

---

## 2.6 Type 5: Learning Agent

### What It Is
Can **improve its performance over time** from experience. Has four components:
1. **Learning Element**: improves based on feedback
2. **Performance Element**: selects actions (the agent itself)
3. **Critic**: evaluates performance against a standard
4. **Problem Generator**: suggests actions that lead to more learning

```
Experience → Critic → Learning Element → Updated Policy → Better Actions
```

### Characteristics
- Gets better with use
- Can generalize to new situations
- Requires feedback signals
- Most complex to build correctly

### LLM-Era Examples
- **Reflexion agent**: self-reflects on failures → updates scratchpad → retries
- **RLHF-tuned model**: learns from human preference ratings
- **Voyager (Minecraft)**: builds a library of skills it acquired over many episodes
- **Fine-tuned agents**: DPO/SFT on successful trajectories

### When to Use
- Long-running autonomous agents
- Systems where data collection is feasible
- High-value tasks where improvement over time matters

---

## 2.7 Type 6: Hierarchical Agent

### What It Is
Multiple agents organized in a **hierarchy** where high-level agents manage and delegate to lower-level agents.

```
           ┌────────────────────┐
           │  Manager Agent     │  (high-level goal setting)
           └─────────┬──────────┘
              ┌──────┼──────┐
              ▼      ▼      ▼
         🔍 Research  💻 Coder  📝 Writer   (specialists)
```

### Characteristics
- Task decomposition at multiple levels
- Specialization at each level
- Scales to complex, long-horizon tasks
- Emergent coordination between levels

### LLM-Era Examples
- CrewAI hierarchical process (Manager LLM delegates tasks)
- LangGraph Supervisor + Worker sub-graphs
- AutoGen nested conversations

---

## 2.8 Comparison Table

| Agent Type | Has Memory | Plans Ahead | Learns | Uses Tools | Complexity |
|---|---|---|---|---|---|
| Simple Reflex | ❌ | ❌ | ❌ | ❌ | Very Low |
| Model-Based | ✅ | ❌ | ❌ | ❌ | Low |
| Goal-Based | ✅ | ✅ | ❌ | ✅ | Medium |
| Utility-Based | ✅ | ✅ | ❌ | ✅ | Medium |
| Learning | ✅ | ✅ | ✅ | ✅ | High |
| Hierarchical | ✅ | ✅ | ✅ | ✅ | Very High |

---

## 2.9 Choosing the Right Agent Type

```
Is the task well-defined with few input variations?
  YES → Simple Reflex or Model-Based (chatbot, FAQ bot)
  NO ↓

Does success require multi-step planning?
  NO → Model-Based with memory
  YES ↓

Are there tradeoffs to optimize (cost, quality, speed)?
  NO → Goal-Based Agent
  YES → Utility-Based Agent

Does the agent need to improve with experience?
  YES → Learning Agent (+ Reflexion, DPO, RLHF)

Is the task complex enough to require specialization?
  YES → Hierarchical / Multi-Agent System
```

---

## 📌 Key Takeaways

1. Most production agents are **Goal-Based** or **Utility-Based** — they plan and optimize
2. **Learning agents** are the frontier — Reflexion, fine-tuning on trajectories, RLHF
3. **Hierarchical agents** unlock complex tasks by decomposing + specializing
4. Understanding type determines your architecture choice before picking any framework
