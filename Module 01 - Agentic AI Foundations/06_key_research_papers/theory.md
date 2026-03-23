# 06 — Key Research Papers: How to Read ML Papers

> *Every major agentic AI pattern traces back to a research paper. Learn to read them.*

---

## 6.1 Why Read Papers?

As an AI engineer, reading papers gives you:
1. **The "why"** behind every pattern — understanding origin prevents misuse
2. **First-mover advantage** — new papers arrive months before framework support
3. **Vocabulary** — precise terminology for team communication
4. **Critical thinking** — you can evaluate when a technique actually applies

---

## 6.2 How to Read an ML Paper Efficiently

### Step 1: The Abstract Pass (5 minutes)
Read only the abstract + conclusion. Ask:
- What problem does this solve?
- What is the proposed approach?
- What are the key results?
- Is this relevant to my needs?

### Step 2: The Figure Pass (10 minutes)
Look at every figure and table without reading the text. Most papers communicate their core insight in 1–2 key figures.
- What is Figure 1 showing? (Usually the architecture or key comparison)
- What does the results table compare?

### Step 3: The Introduction + Conclusion (15 minutes)
Read introduction and conclusion fully. Skip related work section for now.
- What is the problem framing?
- What assumptions are made?
- What limitations are acknowledged?

### Step 4: The Method Deep Dive (30–60 minutes)
Read the core method section. Implement pseudocode in your head or on paper.
- How exactly does it work?
- What are the inputs and outputs?
- What are the key hyperparameters or design choices?

### Step 5: Experiments (20 minutes)
- What benchmarks are used? Are they relevant to your use case?
- What baselines are compared against?
- What ablations show which components matter most?

---

## 6.3 Key Questions to Ask About Any Agentic AI Paper

1. **What agent architecture does it assume?** (single agent, multi-agent, hierarchical)
2. **What LLM(s) were used?** (results may not generalize to other models)
3. **What benchmarks?** (WebArena, AgentBench, SWE-Bench — are they applicable?)
4. **What are the failure modes?** (every paper has them, sometimes hidden)
5. **How were the prompts designed?** (often the "secret sauce")
6. **What would it take to reproduce this?** (code available? compute requirements?)

---

## 6.4 Essential Paper Reading Order for Agentic AI

Read in this order — each builds on the previous:

```
1. Chain-of-Thought (Wei et al., 2022)
   └── Foundation: verbal step-by-step reasoning
       │
2. ReAct (Yao et al., 2022)
   └── Combines CoT reasoning WITH actions + tool use
       │
3. Toolformer (Schick et al., 2023)
   └── LLMs that learn WHEN and HOW to use tools (self-supervised)
       │
4. Reflexion (Shinn et al., 2023)
   └── Self-reflection and verbal RL for error correction
       │
5. Generative Agents (Park et al., 2023)
   └── Memory, reflection, planning for believable behavior
       │
6. Voyager (Wang et al., 2023)
   └── Lifelong learning: skill library + curriculum
       │
7. SWE-agent (Yang et al., 2024)
   └── Autonomous software engineering — real-world task performance
```

---

## 6.5 Where to Find Papers

| Source | URL | Best For |
|---|---|---|
| ArXiv | arxiv.org | Preprints (1–3 weeks before conferences) |
| Papers With Code | paperswithcode.com | Papers + reproducible code |
| Semantic Scholar | semanticscholar.org | Citation graphs + related work |
| Hugging Face Daily Papers | huggingface.co/papers | Curated most-discussed daily papers |
| Twitter/X | @_lewtun, @karpathy, @ylecun | Real-time commentary from researchers |

---

## 📌 Keep in Mind

- Papers optimize for benchmark scores — real-world performance often differs
- Prompt engineering is often the actual contribution, not the architecture
- Always check: what LLM? what prompt? what benchmark? before applying a paper's lessons
