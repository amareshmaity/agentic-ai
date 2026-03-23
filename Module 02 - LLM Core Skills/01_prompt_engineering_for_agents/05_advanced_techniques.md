# 05 — Advanced Prompting Techniques

> *Beyond CoT — frontier techniques that push the quality ceiling of LLM reasoning.*

---

## 5.1 Overview

These advanced techniques build on Chain-of-Thought and are used when:
- Standard prompting + CoT is still producing errors
- The task involves multi-dimensional reasoning, planning, or creative exploration
- You need higher confidence and consistency on critical decisions

| Technique | Best For | Added Cost |
|---|---|---|
| **Tree-of-Thought (ToT)** | Complex planning, creative problem-solving | High (branching) |
| **Step-Back Prompting** | Abstract reasoning, science, strategy | Low (1 extra call) |
| **Analogical Reasoning** | Novel problems with known analogues | Low |
| **Meta-Prompting** | Improving prompts with the model's help | Medium |
| **Self-Refinement** | Iterative quality improvement | Medium (N calls) |
| **Maieutic Prompting** | Verifying logical consistency | Medium |
| **Directional Stimulus** | Steering outputs toward specific styles | Low |

---

## 5.2 Tree-of-Thought (ToT)

**Paper**: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023)

### What It Is
Instead of a single linear reasoning chain, ToT **explores multiple reasoning branches simultaneously**, evaluates each branch, and **prunes** unpromising paths — like a human brainstorming multiple approaches before committing.

```
                    ROOT: [Problem]
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Approach A    Approach B    Approach C
      [evaluate]    [evaluate]    [evaluate]
      Score: 3/10   Score: 8/10   Score: 5/10
          ✗              │              ✗
                    Expand B
                   ┌──────┐
                Sub-B1  Sub-B2
                 7/10    9/10
                          │
                      Best Path → ANSWER
```

### When to Use ToT
- Creative writing with many possible directions
- Multi-step planning tasks (travel planning, project scheduling)
- Mathematics with multiple solution approaches
- Any task where the "correct path" isn't obvious upfront

### Implementing ToT with LLMs

```python
SYSTEM_PROMPT_TOT = """
You are solving problems using Tree-of-Thought reasoning.

Step 1 — EXPLORE: Generate 3 distinct approaches/solutions.
  Format: "Approach A: ..." "Approach B: ..." "Approach C: ..."

Step 2 — EVALUATE: Score each approach 1-10 with reasons.
  Format: "A: 7/10 because..." "B: 9/10 because..." "C: 4/10 because..."

Step 3 — SELECT: Pick the best approach and explain why.
  Format: "Best approach: B because..."

Step 4 — EXECUTE: Implement the selected approach in full detail.

Always follow all 4 steps.
"""

def tree_of_thought(problem: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TOT},
            {"role": "user", "content": problem}
        ],
        temperature=0.7  # Allow creative exploration
    )
    return response.choices[0].message.content
```

### Multi-LLM ToT (True Parallel Branching)
```python
import asyncio

async def explore_branch(problem: str, approach_hint: str) -> tuple[str, float]:
    """Explore one branch and return (reasoning, score)."""
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Problem: {problem}\nApproach: {approach_hint}\nSolve this step by step, then rate your solution confidence 0-10."}
        ]
    )
    content = response.choices[0].message.content
    score = extract_confidence_score(content)  # parse "Confidence: X/10"
    return content, score

async def tree_of_thought_parallel(problem: str) -> str:
    approaches = ["analytical approach", "creative approach", "systematic approach"]
    branches = await asyncio.gather(*[explore_branch(problem, a) for a in approaches])
    # Select highest confidence branch
    best_branch = max(branches, key=lambda x: x[1])
    return best_branch[0]
```

---

## 5.3 Step-Back Prompting

**Paper**: "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models" (Zheng et al., 2023)

### What It Is
Before answering a specific question, the model first answers a **more abstract, general version** of the question. This activates broader relevant knowledge before narrowing to the specific answer.

```
Specific Question: "Why did this patient's white blood cell count spike after surgery?"
                        ↓
Step Back:   "What are the general biological mechanisms of WBC elevation?"
             → Model generates broad principles: infection response, inflammation, stress...
                        ↓
Final Answer: Apply those principles to the specific case → much more accurate
```

### Prompt Pattern

```
For complex questions that require domain knowledge, first ask:
"Before answering, step back and answer: what general principles or 
knowledge are most relevant to this question?"

Then use that broad answer to ground your specific response.
```

### Implementation

```python
def step_back_prompt(question: str) -> str:
    # Step 1: Abstract step
    abstract_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Before answering the specific question below, first identify the general principles, concepts, or background knowledge most relevant to it. Just state these principles clearly.\n\nQuestion: {question}\n\nGeneral principles relevant to this question:"
        }]
    )
    principles = abstract_response.choices[0].message.content
    
    # Step 2: Specific answer grounded in principles
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Using these general principles:\n{principles}\n\nNow answer this specific question:\n{question}"
        }]
    )
    return final_response.choices[0].message.content
```

---

## 5.4 Analogical Reasoning

### What It Is
Frame an unfamiliar problem in terms of a **familiar analogue** to leverage the model's deep knowledge of well-understood domains.

### Pattern
```
"This problem is analogous to [WELL-KNOWN PROBLEM]. 
In [WELL-KNOWN PROBLEM], the solution was [SOLUTION].
Apply the same reasoning structure to [NEW PROBLEM]."
```

### Examples in Agent Prompting

```
# Example 1: Teaching an agent about rate limiting
"Designing an API rate limiter is analogous to a traffic light system.
In a traffic system: green=allowed, yellow=warning, red=stop.
Apply the same structure: requests_left > 20% = proceed, 
5-20% = slow down, < 5% = stop and wait."

# Example 2: Teaching multi-agent coordination
"Coordinating multiple agents is analogous to an orchestra.
In an orchestra: conductor sets tempo, each section plays its part,
they wait for cues, and the conductor synthesizes the result.
Design our agent system the same way: orchestrator = conductor,
specialist agents = sections, task completion = musical cues."
```

---

## 5.5 Meta-Prompting (Prompt Optimization with LLMs)

### What It Is
Use an LLM to **improve, critique, and rewrite your prompts**. The LLM acts as a prompt engineer.

### Pattern: Critique → Rewrite
```python
META_SYSTEM = """
You are an expert prompt engineer. When given a prompt, you:
1. Identify weaknesses (ambiguity, missing constraints, poor structure)
2. Suggest specific improvements
3. Rewrite the improved version

Format:
WEAKNESSES: [bullet list]
IMPROVEMENTS: [bullet list]
IMPROVED PROMPT:
[the rewritten prompt]
"""

def improve_prompt(original_prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": META_SYSTEM},
            {"role": "user", "content": f"Improve this prompt:\n\n{original_prompt}"}
        ]
    )
    return response.choices[0].message.content
```

### Automatic Prompt Optimization Loop

```python
def optimize_prompt(initial_prompt: str, test_cases: list[dict], max_iterations: int = 5) -> str:
    """
    Iteratively improve a prompt by:
    1. Running it on test cases
    2. Identifying failures
    3. Asking LLM to fix the prompt
    """
    current_prompt = initial_prompt
    
    for iteration in range(max_iterations):
        # Run test cases
        failures = []
        for case in test_cases:
            output = run_prompt(current_prompt, case["input"])
            if not evaluate_output(output, case["expected"]):
                failures.append({"input": case["input"], "got": output, "expected": case["expected"]})
        
        if not failures:
            print(f"✅ Perfect prompt achieved in {iteration} iterations!")
            break
        
        # Ask LLM to fix the prompt based on failures
        fix_request = f"""
        This prompt is failing on these test cases:
        {format_failures(failures)}
        
        Original prompt:
        {current_prompt}
        
        Rewrite the prompt to fix these failures without breaking other cases.
        Return ONLY the improved prompt text.
        """
        current_prompt = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": fix_request}]
        ).choices[0].message.content
    
    return current_prompt
```

---

## 5.6 Self-Refinement

**Paper**: "Self-Refine: Iterative Refinement with Self-Feedback" (Madaan et al., 2023)

### What It Is
The model generates an output, then **critiques its own output**, then **refines it based on the critique**. This can loop multiple times.

```
Generate → Critique → Refine → Critique → Refine → ... → Final
```

### Self-Refinement Pattern

```python
def self_refine(task: str, max_iterations: int = 3) -> str:
    # Step 1: Initial generation
    output = generate(task)
    
    for i in range(max_iterations):
        # Step 2: Critique
        critique = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""You are a harsh critic. Critique this output for the task: "{task}"
                
Output to critique:
{output}

Identify: specific errors, missing elements, quality issues.
Be specific, not generic. Point to exact lines.
If the output is already excellent, say "NO_CHANGES_NEEDED"."""
            }]
        ).choices[0].message.content
        
        if "NO_CHANGES_NEEDED" in critique:
            break
        
        # Step 3: Refine based on critique
        output = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Task: {task}
                
Current output:
{output}

Critique of current output:
{critique}

Rewrite the output addressing all critique points. Return only the improved output."""
            }]
        ).choices[0].message.content
    
    return output
```

---

## 5.7 Directional Stimulus Prompting (DSP)

### What It Is
Provide small **hints or hints** (stimuli) that steer the model toward a desired output style, format, or answer direction — without full instructions.

```
Instead of: "Write in a formal, academic, third-person style"
DSP uses:   A brief cue like "Academic analysis:" prepended to the output template

Instead of: "Include at least 5 specific examples"
DSP uses:   Starting the output with "Examples: 1. " forcing the model to continue
```

### Pattern
```python
# Force structured output by providing the BEGINNING of the desired format
prompt = f"""Analyze the following text:

{text}

## Analysis

**Main Themes:**
1."""  # ← The model CONTINUES this structure, giving you a 5-point list
```

---

## 5.8 Combining Advanced Techniques

In production, you often combine multiple techniques:

```
Step-Back (activate domain knowledge)
    +
Chain-of-Thought (step-by-step reasoning)
    +
Self-Refinement (quality check + iterate)
    +
Self-Consistency (run 3 times, majority vote)
```

```python
def advanced_agent_response(question: str) -> str:
    # 1. Step-back to activate relevant knowledge
    principles = step_back(question)
    
    # 2. Generate 3 CoT responses (self-consistency)
    responses = [cot_generate(question, principles) for _ in range(3)]
    
    # 3. Self-refine the majority vote winner
    best = majority_vote(responses)
    final = self_refine(best, task=question)
    
    return final
```

**Use judiciously**: Each addition multiplies your token cost. Use combinations only for high-stakes, low-frequency decisions.

---

## 📌 Key Takeaways

1. **Tree-of-Thought**: explore multiple paths before committing — for planning and creative tasks
2. **Step-Back**: abstract before specific — activate domain knowledge before answering
3. **Analogical Reasoning**: frame new problems as familiar ones — leverages deep training knowledge
4. **Meta-Prompting**: let the LLM improve your prompts — iterative prompt optimization
5. **Self-Refinement**: generate → critique → improve loop — no retraining needed for quality boost
6. **Combine techniques** for critical tasks — but each technique multiplies cost
7. **Benchmark before + after** each technique — not all help on every task type

---

## 🔗 Key Papers
- [Tree of Thoughts (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)
- [Take a Step Back (Zheng et al., 2023)](https://arxiv.org/abs/2310.06117)
- [Self-Refine (Madaan et al., 2023)](https://arxiv.org/abs/2303.17651)
- [Large Language Models as Optimizers - OPRO (Yang et al., 2023)](https://arxiv.org/abs/2309.03409)
