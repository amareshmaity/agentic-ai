# 🎯 Prompt Engineering for Agents

> *The highest-leverage skill in agentic AI — the quality of your prompts determines the quality of your agents.*

---

## 📌 Why Prompt Engineering Is Different for Agents

In a simple chatbot, a bad prompt gives a bad answer. In an agent, a bad prompt causes:
- Wrong tool selection → wrong actions → compounding errors across 10+ steps
- Infinite loops because the agent can't determine when it's done
- Hallucinated tool arguments that crash your system
- Agents that ignore constraints and take unauthorized actions

**Prompting for agents is systems engineering, not just text writing.**

---

## 📂 Files in This Topic

| File | What It Covers |
|---|---|
| `01_prompting_basics.md` | Anatomy of a prompt, roles, token awareness |
| `02_system_prompts_for_agents.md` | System prompt design, persona, constraints, output format |
| `03_few_shot_prompting.md` | Zero-shot, few-shot, in-context learning, example selection |
| `04_chain_of_thought.md` | CoT, Zero-shot CoT, structured reasoning for agents |
| `05_advanced_techniques.md` | Self-consistency, step-back, analogical, meta-prompting |
| `06_agentic_prompt_patterns.md` | ReAct format, scratchpad, tool-use prompts, HITL prompting |
| `07_prompt_security_and_injection.md` | Injection attacks, jailbreaks, defense techniques |
| `examples.ipynb` | All practical code — every concept demonstrated |

---

## 🗺️ Learning Path

```
01 Basics          → understand the building blocks
02 System Prompts  → learn to design agent identities
03 Few-Shot        → teach by example, not instruction
04 Chain-of-Thought→ make agents reason step by step
05 Advanced        → push quality ceiling with frontier techniques
06 Agentic Patterns→ patterns specific to tool-using agent loops
07 Security        → protect agents from adversarial inputs
examples.ipynb     → practice everything end-to-end
```

---

## ⏱️ Estimated Time

| Activity | Time |
|---|---|
| Reading all 7 theory files | 3–4 hours |
| Running notebook | 2–3 hours |
| **Total** | **~6 hours** |

---

## 🔧 Setup

```bash
pip install openai python-dotenv rich tiktoken
```

```env
# .env
OPENAI_API_KEY=your_key
```
