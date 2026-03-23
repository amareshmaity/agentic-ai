# 🏋️ Module 01 — Exercises & Mini-Projects

> Complete these exercises after going through all theory + notebooks.

---

## Exercise 1: Conceptual Mapping (30 min)

### Part A — Identify the Agent Type
For each of the following products, identify which agent type (Simple Reflex, Model-Based, Goal-Based, Utility-Based, Learning, Hierarchical) best describes it and justify your answer in 2–3 sentences.

1. A spam filter that marks emails based on keyword rules
2. GitHub Copilot (autocomplete suggestions)
3. Devin — the autonomous software engineering agent
4. A customer support bot that remembers your past orders
5. An agent that chooses between GPT-4o, Claude, and Gemini based on task type and cost
6. Perplexity AI's real-time web search answering

### Part B — PRAM Loop Analysis
Pick any real-world AI product (not from the list above). Map it to the PRAM loop:
- **P**: What does it perceive? What are its input sources?
- **R**: How does it reason? What is the LLM's role?
- **A**: What actions can it take?
- **M**: What memory does it maintain?

---

## Exercise 2: Agentic vs Non-Agentic (20 min)

Below are 6 system descriptions. For each, rate it on the autonomy spectrum (Level 0–5) and explain which elements make it agentic or non-agentic.

1. A Python script that reads a CSV and sends a weekly email report
2. A chatbot that answers questions from a fixed FAQ document with no memory
3. An agent that monitors Twitter, detects trending topics, writes a blog post, and schedules it for publishing — with a Slack notification to the team before publish
4. A system that reads GitHub issues, proposes a fix, runs tests, and submits a PR if all tests pass
5. A voice assistant that takes a grocery list, orders from DoorDash, and confirms delivery time via SMS
6. An AI that reads a student's essay, provides feedback, and records the grade in a database

---

## Exercise 3: Design Pattern Recognition (30 min)

Read the following agent descriptions and identify which of the 4 design patterns (Reflection, Tool Use, Planning, Multi-Agent) are being used. Multiple patterns can apply.

1. *Agent writes code → runs it → reads error message → rewrites code → repeats until tests pass*
2. *A manager agent takes a user's research request, assigns web search to Agent A, academic paper search to Agent B, and synthesis to Agent C, then combines results*
3. *Before answering, the agent searches DuckDuckGo, Wikipedia, and a company knowledge base, then synthesizes all three sources*
4. *Agent generates a blog post → evaluator scores it 1–10 → if score < 7, agent revises with feedback → loop until score ≥ 7*
5. *Agent: (1) plans 5 steps to research a company, (2) executes each step sequentially, (3) if a step fails, re-plans from that point*

---

## Exercise 4: Paper Deep Dive (60 min)

Choose ONE paper from the list below and complete the analysis template:

**Papers**: ReAct | Reflexion | Generative Agents | Voyager | SWE-agent

### Analysis Template
```markdown
## Paper: [Title]
**My 1-sentence summary**:

**Core architectural insight**:

**Key figure/table** (describe what it shows):

**Strongest result**: (Which benchmark? What improvement?)

**Most important limitation**:

**How I would apply this** (specific use case in my work):

**One thing I would change**: (design critique)
```

---

## Exercise 5: Mini-Project — Build a Basic PRAM Agent (2–3 hours)

### Goal
Build a minimal agent from scratch (no frameworks) that demonstrates the PRAM loop.

### Requirements
- [ ] Agent takes a user goal as input
- [ ] Has at least 2 tools (e.g., `web_search_mock`, `calculator`)
- [ ] Runs a ReAct-style loop: Thought → Action → Observation
- [ ] Terminates when it reaches the goal OR after 5 iterations
- [ ] Prints each step clearly to show the loop in action
- [ ] Has a basic conversation history (Model-Based memory)

### Starter Template
```python
from openai import OpenAI
import json

client = OpenAI()
TOOLS = [...]  # Define your tools here

def run_agent(goal: str, max_steps: int = 5):
    messages = [
        {"role": "system", "content": "You are an agent..."},
        {"role": "user", "content": goal}
    ]
    
    for step in range(max_steps):
        # Reasoning: call LLM
        response = client.chat.completions.create(...)
        
        # Check if done
        # Execute action if tool call
        # Append observation
        # Print step
    
    return "Max steps reached"

run_agent("What is 15% of 847 and what is the capital of Japan?")
```

### Evaluation Criteria
- Does it correctly loop through PRAM?
- Does it successfully use both tools?
- Does it terminate cleanly?
- Is each step logged/printed?

---

## Exercise 6: Autonomy Audit (45 min)

You are the AI architect for a healthcare company. Design the autonomy level for each of these agent use cases, and justify your choice with specific risk analysis:

1. Summarizing patient notes for a doctor's review
2. Scheduling follow-up appointments based on doctor instructions
3. Sending prescription refill requests to pharmacies
4. Triaging emergency vs non-emergency patient calls
5. Generating personalized treatment plan suggestions

For each, also specify:
- What HITL mechanism would you use?
- What audit logging is required?
- What would need to happen before you'd increase autonomy by one level?

---

## 🏆 Module Completion Checklist

- [ ] Read all 6 theory `.md` files
- [ ] Ran all 5 Jupyter notebooks
- [ ] Completed Exercises 1–4 (written)
- [ ] Built the Mini-Project agent (Exercise 5)
- [ ] Completed the Autonomy Audit (Exercise 6)
- [ ] Can explain the PRAM loop without looking at notes
- [ ] Can name all 4 agentic design patterns and give a real example of each

> ✅ If you checked all boxes — you're ready for Module 02!
