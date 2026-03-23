# 04 — Chain-of-Thought Prompting

> *Make the LLM show its work — step-by-step reasoning dramatically improves accuracy on complex tasks.*

---

## 4.1 What Is Chain-of-Thought (CoT)?

**Chain-of-Thought prompting** encourages the LLM to produce **explicit intermediate reasoning steps** before reaching a final answer, rather than jumping directly to a conclusion.

It was introduced in the paper **"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"** (Wei et al., 2022, Google).

```
WITHOUT CoT:
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls. How many does he have?
A: 11

WITH CoT:
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls. How many does he have?
A: Roger starts with 5 balls. He buys 2 cans × 3 balls = 6 more balls.
   5 + 6 = 11. The answer is 11.
```

Both give the right answer here — but for harder problems, only CoT succeeds.

---

## 4.2 Why CoT Dramatically Improves Agent Performance

For agents, CoT is not just a reasoning trick — it's a **reliability mechanism**.

### Without CoT
```
User: "Should I invest in Company X?"
Agent action: web_search("Company X stock")
Agent final: "Yes, you should invest." ← no visible reasoning
```

Problems:
- We can't audit the decision
- The agent might be guessing
- No self-correction opportunity

### With CoT
```
User: "Should I invest in Company X?"
Agent thought: "I need to consider: (1) recent financials, (2) market trend, 
               (3) competitors. Let me search for each."
Action: web_search("Company X Q4 2024 earnings")
Observation: Revenue down 23% YoY
Agent thought: "Revenue is declining. I need to check why before concluding."
Action: web_search("Company X revenue decline reason 2024")
Observation: Lost major contract with key client
Agent thought: "Revenue decline + lost major client = significant risk. 
               I should also check if this is an industry trend."
...
Final answer: "Based on declining revenue (-23%), lost key client contract, 
              and high sector competition, this appears to be a high-risk 
              investment. I'd recommend caution..."
```

Benefits:
- ✅ Each step is auditable
- ✅ Agent self-corrects mid-sequence  
- ✅ More reliable conclusions from complex reasoning
- ✅ Easier to debug when the agent makes mistakes

---

## 4.3 Two Types of CoT

### Type 1: Few-Shot CoT (Original Paper)
Provide examples that include the reasoning chain:

```
Q: The cafeteria had 23 apples. If they used 20 to make lunch and 
   bought 6 more, how many apples do they have?
A: The cafeteria started with 23 apples and used 20, so they had 
   23 - 20 = 3 remaining. Then they bought 6 more: 3 + 6 = 9.
   The answer is 9.

Q: {new math question}
A: [model now generates step-by-step reasoning]
```

### Type 2: Zero-Shot CoT (Simpler, Often Sufficient)
Just add a phrase at the end of your instruction:

```
"Let's think step by step."      ← Most common
"Think through this carefully."
"Work through this step by step before answering."
"Break this down into steps."
"Reason through each part before giving a final answer."
```

**Remarkable finding**: The phrase *"Let's think step by step"* alone improves accuracy significantly on math and logic tasks — no examples needed.

---

## 4.4 CoT for Agent Reasoning: The Scratchpad

In agents, CoT takes the form of a **scratchpad** — a running internal monologue the agent maintains before taking each action.

### Scratchpad Pattern
```
[Goal]: Write a competitive analysis of GPT-4o vs Claude 3.5

[Scratchpad Begin]
Step 1: I need information on both models. Let me start with GPT-4o specs.
→ Action: web_search("GPT-4o capabilities benchmarks 2025")
→ Observation: [results]

Step 2: Now I have GPT-4o data. Need Claude 3.5 data for comparison.
→ Action: web_search("Claude 3.5 Sonnet benchmarks performance 2025")
→ Observation: [results]

Step 3: I have both datasets. Let me compare on: 1) reasoning 2) coding 
        3) cost 4) context window 5) multimodality.
→ No search needed — I can now synthesize.
[Scratchpad End]

[Final Answer]: Structured comparison...
```

### Implementing Scratchpad in System Prompt
```
## Reasoning Protocol

Before EVERY action, write your reasoning in this format:

Thought: [1-3 sentences explaining what you know, what you need, 
          and why you're taking the next action]
Action: [tool_name](args) OR FinalAnswer: [your response]

This scratchpad is your PRIVATE reasoning space — use it freely 
to plan, check your work, and decide next steps.
```

---

## 4.5 Structured CoT Formats

### Format 1: Thought → Action → Observation (ReAct)
```
Thought: I need the current Bitcoin price.
Action: web_search("Bitcoin BTC price USD today")
Observation: Bitcoin is trading at $67,342 USD
Thought: I now have the price. The user asked for it, so I'm done.
Final Answer: Bitcoin's current price is $67,342 USD.
```

### Format 2: Plan → Execute → Verify
```
Plan:
1. Search for the company's recent news
2. Check their Q4 financials
3. Look at analyst ratings
4. Synthesize into a recommendation

Execute Step 1: [action + observation]
Execute Step 2: [action + observation]
Execute Step 3: [action + observation]

Verify: Do I have enough data to give a reliable recommendation? Yes/No
[Final recommendation]
```

### Format 3: Pros/Cons Analysis CoT
```
Question: Should we use PostgreSQL or MongoDB for this agent's memory?

Analyzing PostgreSQL:
+ ACID compliant — reliable for transaction-based writes
+ pgvector extension for embedding storage
+ Strong consistency guarantees
- More complex schema management for flexible data
- Slower for high write throughput

Analyzing MongoDB:
+ Schema-flexible — good for varying agent output structures
+ High write throughput
- Eventually consistent by default
- No native vector support without Atlas Vector Search

For agent memory with mixed structured + vector data:
PostgreSQL + pgvector is better because: ACID compliance matters for 
agent state consistency, and pgvector gives native vector support.
```

---

## 4.6 Self-Consistency: CoT × N

**Self-consistency** (Wang et al., 2022) extends CoT by generating **multiple independent reasoning chains** and **majority-voting** on the final answer.

```
Run same question 5 times with temperature > 0:

Chain 1: [reasoning] → Answer: A
Chain 2: [reasoning] → Answer: A  
Chain 3: [reasoning] → Answer: B  ← outlier
Chain 4: [reasoning] → Answer: A
Chain 5: [reasoning] → Answer: A

Majority vote: A wins (4/5)
Final answer: A
```

**When to use self-consistency**:
- High-stakes decisions where you need confidence
- Mathematical or logical reasoning tasks
- When a single chain is producing inconsistent results

**Cost**: 5× more LLM calls → use selectively.

```python
from collections import Counter

def self_consistent_answer(question: str, n: int = 5) -> str:
    answers = []
    for _ in range(n):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Think step by step."},
                {"role": "user", "content": question}
            ],
            temperature=0.7  # Some variation per run
        )
        # Extract just the final answer line
        answer = extract_final_answer(response.choices[0].message.content)
        answers.append(answer)
    
    # Majority vote
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common
```

---

## 4.7 CoT Failure Modes and Fixes

| Failure Mode | Symptom | Fix |
|---|---|---|
| **Verbose reasoning, wrong conclusion** | Long scratchpad but wrong answer | Add "Verify your reasoning before answering" |
| **Reasoning loop** | Agent keeps thinking but never acts | Set max_thinking_tokens or add Thought limit |
| **Confident wrong CoT** | Step-by-step reasoning confidently wrong | Use self-consistency to detect outliers |
| **Reasoning drift** | Agent starts answering a different question | Rephrase goal at start of each Thought step |
| **Skipping reasoning under pressure** | High temp + long prompts → agent skips CoT | Enforce Thought format via structured output |
| **Circular reasoning** | Agent justifies itself with its own previous output | Add: "Do not reference your previous thoughts as evidence" |

---

## 4.8 When to Use vs Skip CoT

| Use CoT For | Skip CoT For |
|---|---|
| Multi-step reasoning tasks | Simple factual lookups |
| Math, logic, planning | Direct classification (A/B/C) |
| Decision making with tradeoffs | Short, simple Q&A |
| Complex tool selection logic | Creative generation (CoT disrupts flow) |
| Debugging why agent failed | High-throughput, cost-sensitive systems |

**Rule of thumb**: If a task requires more than 2 mental steps for a human → use CoT.

---

## 📌 Key Takeaways

1. **CoT = asking the model to show its work** before giving a final answer
2. **Zero-shot CoT**: just add "Let's think step by step" — deceptively powerful
3. **Few-shot CoT**: demonstrate reasoning chains in examples for complex tasks
4. **Scratchpad**: the agent-specific form of CoT — think before each action
5. **Self-consistency**: run N chains, take the majority vote — higher accuracy, higher cost
6. **CoT enables auditability**: you can see WHY the agent took an action
7. **Don't use CoT for simple tasks** — it adds tokens and latency without benefit

---

## 🔗 Key Papers
- [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903) - Wei et al. (2022)
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) - Wang et al. (2022)
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) - Kojima et al. (2022) - "Let's think step by step"
