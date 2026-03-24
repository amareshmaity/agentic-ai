# 02 — Capability Comparison

> *Knowing benchmark scores is not enough — you need to understand what each model is actually good at and why.*

---

## 2.1 The Five Capability Dimensions

When comparing LLMs, evaluate them across five dimensions:

```
1. REASONING        Multi-step logic, math, planning, chain-of-thought
2. CODING           Writing, debugging, refactoring, explaining code
3. INSTRUCTION-FOLLOW Adhering exactly to complex, multi-constraint prompts
4. LONG-CONTEXT     Finding and using information from long documents
5. CREATIVITY       Writing quality, nuance, tone variation, style
```

---

## 2.2 Reasoning Capability

**What it means**: Can the model solve multi-step problems requiring logical deduction, mathematical operations, or structured planning?

### Models that excel at reasoning:
- `o1` / `o1-mini` — dedicated chain-of-thought reasoning; best on GPQA, MATH benchmarks
- `claude-3-5-sonnet` — excellent structured reasoning, very strong on agentic tasks
- `gpt-4o` — strong reasoning, especially for structured problems

### Reasoning failure modes:
```python
# Prompt that exposes weak reasoning
REASONING_TEST = """
A baker makes 3 batches of bread daily. Each batch takes 90 minutes.
She starts at 6:00 AM and takes a 30-minute lunch break after the 2nd batch.
What time does she finish her 3rd batch?

Show your work step by step.
"""
# Strong models: correctly compute 6:00 + 90 + 90 + 30 + 90 = 12:00
# Weak models: miss the lunch break or compute wrong
```

### When reasoning matters:
- Complex planning (multi-step agent tasks)
- Math/science applications
- Logical reasoning over structured data
- Writing and evaluating code that solves algorithmic problems

---

## 2.3 Coding Capability

**What it means**: Can the model write correct, idiomatic, production-quality code?

### Best coding models (2024–2025):
1. `claude-3-5-sonnet` — top on SWE-Bench, best at editing/refactoring
2. `gpt-4o` — excellent at generating new code from specifications
3. `deepseek-v3` — surprisingly strong at coding, very cheap
4. `o1` — best for hard algorithmic problems (competitive programming)

### Coding tasks and what matters:
```
Task                        What matters
─────────────────────────────────────────────────────────
Write new function          Correctness, language idioms
Debug existing code         Understanding context, error diagnosis  
Refactor code               Following conventions, minimal diff
Explain code                Clarity, accuracy, didactic quality
Code review                 Identifying bugs, style issues
Algorithm design            Reasoning + correctness
```

### SWE-Bench scores (approximate, early 2025):
```python
SWE_BENCH = {
    "claude-3-5-sonnet": 49.0,  # ← Leader
    "gpt-4o":            38.0,
    "gemini-1.5-pro":    30.0,
    "llama-3.1-405b":    24.0,
}
```

---

## 2.4 Instruction-Following Capability

**What it means**: Does the model precisely follow complex, multi-constraint system prompts?

This is **critical for agents** — a model that ignores instructions is unreliable in automated pipelines.

```python
# IFEval test pattern — strict format instructions
INSTRUCTION_TEST = """
Respond with ONLY a JSON object. No prose, no markdown fences.
The JSON must have exactly these fields:
- "answer": string
- "confidence": float between 0.0 and 1.0
- "citations": list of exactly 2 strings

Question: What is photosynthesis?
"""
# Strong instruction-followers: output JSON directly, correct structure, 2 citations
# Weak instruction-followers: add "Here is the JSON:" text, wrong field count
```

### Why it matters more for agents:
- Agents have complex system prompts with many rules
- Agentic tasks require specific output formats (JSON, structured objects)
- Non-compliant output breaks automation pipelines

### IFEval benchmark (approximate):
```python
IFEVAL_SCORES = {
    "gpt-4o":            87.5,  # Best at following instructions
    "claude-3-5-sonnet": 86.5,
    "gpt-4o-mini":       84.0,  # Surprisingly good
    "gemini-1.5-pro":    80.0,
    "llama-3.1-70b":     75.0,
}
```

---

## 2.5 Long-Context Capability

**What it means**: Can the model accurately find and use information from a very long document?

### The "Lost in the Middle" Problem

Research shows LLMs retrieve information better from the beginning and end of documents, worse from the middle:

```
Context position:  [0%...25%...50%...75%...100%]
Retrieval accuracy: HIGH  MED   LOW   MED   HIGH
```

### Models and long-context performance:
- `gemini-1.5-pro` (2M) — best long-context retrieval at scale
- `claude-3-5-sonnet` (200k) — excellent quality at 200k
- `gpt-4o` (128k) — solid but drops off in the middle

### RULER benchmark (long-context retrieval):
```python
# Simulated task: find specific facts hidden in long documents
RULER_SCORES = {
    "gemini-1.5-pro":    96.7,   # Best
    "claude-3-5-sonnet": 90.8,
    "gpt-4o":            85.4,
    "llama-3.1-70b":     79.0,
}
```

---

## 2.6 Speed and Latency

Latency matters as much as quality for real-time applications:

```python
# Approximate latency (time to first token, TTFT) — ms
TTFT_MS = {
    "gemini-2.0-flash":      50,   # Fastest
    "claude-3-5-haiku":      80,
    "gpt-4o-mini":          120,
    "gpt-4o":               200,
    "claude-3-5-sonnet":    300,
    "gemini-1.5-pro":       400,
    "o1":                 3000,   # Extended thinking — very slow
}

# Approximate throughput (tokens/second for output)
THROUGHPUT_TPS = {
    "gemini-2.0-flash": 300,
    "gpt-4o-mini":      200,
    "claude-3-5-haiku": 180,
    "gpt-4o":           120,
    "claude-3-5-sonnet": 90,
    "o1":                40,   # Thinking takes time
}
```

---

## 2.7 The Capability–Cost–Speed Triangle

You can optimize for at most **two** of these three factors:

```
         Quality
          /\
         /  \
        /    \
       /  ↑   \
      /  best  \
     ────────────
    /             \
   /               \
  /                 \
Speed ─────────────── Cost

"Fast + Cheap"  = gpt-4o-mini, gemini-flash   (less quality)
"Fast + Quality" = claude-3-5-sonnet streamed  (more expensive)
"Cheap + Quality" = claude-3-5-sonnet (batch)  (slow batch mode)
```

---

## 📌 Key Takeaways

1. **Reasoning**: o1 > claude-3-5-sonnet ≈ gpt-4o > mini/haiku models
2. **Coding**: claude-3-5-sonnet > gpt-4o > deepseek-v3 (SWE-Bench leader)
3. **Instruction-following**: gpt-4o ≈ claude-3-5-sonnet > others (critical for agents)
4. **Long-context**: gemini-1.5-pro > claude-3.5 > gpt-4o (diminishing for middle context)
5. **Speed**: gemini-flash > claude-haiku > gpt-4o-mini > flagships
6. **Always test on your task** — benchmarks are proxies, not guarantees
7. **The "quality" leader changes every 3-6 months** — maintain benchmark tracking
