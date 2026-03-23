# 05 — The Autonomy Spectrum

> *How much should an agent act on its own? The most important design decision you'll make.*

---

## 5.1 The Autonomy Spectrum

```
FULLY MANUAL ◄──────────────────────────────────────► FULLY AUTONOMOUS
       │                                                        │
 Human does                                             Agent does
 everything                                             everything
       │        │        │         │          │                 │
   Level 0   Level 1  Level 2   Level 3    Level 4          Level 5
   No AI    Suggest   Draft+    Auto with  Auto with         Full
            only      Review    notify     audit log        autonomy
```

No production system sits at Level 5 for critical tasks — the sweet spot depends on **task criticality** and **LLM reliability**.

---

## 5.2 The Six Autonomy Levels

### Level 0 — No Automation
Human does everything. AI is not involved.
- **Example**: Manual SQL queries for customer data
- **When**: AI adds no value, or regulatory requirements prohibit it

---

### Level 1 — AI Suggestions Only
AI generates suggestions. Human reviews **before** anything is shown or done.
- **Example**: Autocomplete in Gmail
- **When**: High-stakes, high-variability tasks where human judgment is irreplaceable

---

### Level 2 — AI Drafts + Human Review
AI generates a complete draft. Human reviews, edits, and approves before execution.
- **Example**: AI writes a blog post → human edits → human publishes
- **When**: Content generation, document summarization, report creation

---

### Level 3 — Automated with Notification
Agent acts autonomously but **notifies** humans about what it did.
- **Example**: Agent automatically tags and categorizes incoming emails, notifies user of summary
- **When**: Low-risk, reversible actions where notifications give adequate oversight

---

### Level 4 — Automated with Audit Log
Agent acts fully autonomously. All actions logged for periodic human review.
- **Example**: An agent that monitors a database and auto-responds to common support tickets
- **When**: Well-defined, tested, high-volume, low-risk tasks

---

### Level 5 — Full Autonomy
No human oversight. Agent makes and executes all decisions independently.
- **Example**: Automated trading systems (with hard limits and kill switches)
- **When**: Extremely well-defined, high-frequency, low-consequence actions only

---

## 5.3 The Risk-Autonomy Matrix

```
              LOW RISK                 HIGH RISK
HIGH        ┌──────────────────┬───────────────────┐
FREQUENCY   │ ✅ Automate fully │ ⚠️ Automate with  │
            │ Level 3–4        │ audit & alerts    │
            │                  │ Level 3           │
            ├──────────────────┼───────────────────┤
LOW         │ 🔄 Automate with │ 🚫 Keep human in  │
FREQUENCY   │ review           │ the loop          │
            │ Level 2          │ Level 1–2         │
            └──────────────────┴───────────────────┘
```

---

## 5.4 Human-in-the-Loop (HITL) Patterns

### Approval Gate
Agent pauses at a checkpoint and waits for human approval before proceeding.
```
Agent plans → ✋ PAUSE → Human reviews plan → ✅ Approve → Agent executes
```
**Use for**: Sending emails, financial transactions, code deployment

### Review-before-publish
Agent completes a full draft → human reviews → human triggers final action.
```
Agent drafts document → 📋 Show to human → Human edits → Human publishes
```
**Use for**: Content creation, reports, proposals

### Interrupt-on-uncertainty
Agent runs autonomously but interrupts when it encounters low-confidence situations.
```
Agent runs → Uncertainty detected → ⚠️ Alert human → Human resolves → Agent continues
```
**Use for**: Complex research tasks, customer interactions with edge cases

### Post-hoc Audit
Agent runs fully autonomously. All actions logged. Human reviews logs on a schedule.
```
Agent acts → 📝 Log action → Continue → Human reviews logs weekly
```
**Use for**: Low-risk, high-volume automations (tagging, categorization, routing)

---

## 5.5 When to Increase vs Decrease Autonomy

### Increase Autonomy When:
- Task completion rate > 95% in testing
- Actions are reversible or low-stakes
- Human review creates bottlenecks that outweigh risk
- Agent has been running reliably for 30+ days in production
- Cost of human review exceeds benefit

### Decrease Autonomy When:
- Task involves irreversible actions (send emails, execute payments, delete data)
- LLM hallucination rate is unacceptably high for this task
- Task involves regulated domains (healthcare, finance, legal)
- New agent deployment — always start conservatively
- Recent errors or incidents

---

## 5.6 The Autonomy Decision Framework

```
                    START HERE
                        │
                        ▼
            What is the consequence of a mistake?
                  │               │
            MINOR (reversible)  MAJOR (irreversible / high-cost)
                  │               │
                  ▼               ▼
         How reliable is the   → Start at Level 1–2
         agent on this task?     Increase only after
                  │               thorough testing
         >90% →  Level 3–4
         <90% →  Level 1–2
```

---

## 5.7 Autonomy in Framework Design

| Framework | HITL Support | Autonomy Controls |
|---|---|---|
| **LangGraph** | `interrupt_before`, `interrupt_after` nodes | Explicit checkpoints in graph |
| **CrewAI** | `human_input=True` on Agent | Task-level human input requests |
| **AutoGen** | `UserProxyAgent` as human proxy | Code execution approval gates |
| **Agno** | Custom confirm hooks | Response confirmation before action |

---

## 📌 Key Takeaways

1. **Never deploy at Level 5** for tasks with meaningful consequences — always include safety valves
2. **Start low, increase progressively** — deploy at Level 1–2, gather data, escalate autonomy conservatively
3. The **cost of human review** must be weighed against the **cost of mistakes** — this is a business decision
4. **HITL is not a failure** — it's an intentional design choice for safety-critical paths
5. Log everything — you can't debug or improve an agent you can't observe
