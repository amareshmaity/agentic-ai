# 04 — Task-Based Model Selection

> *The fastest path to the right model: match your task type to the model that dominates that category.*

---

## 4.1 Task Taxonomy for LLM Selection

Categorize your tasks, then pick accordingly:

```
Task Category              Best Default         Budget Option
──────────────────────────────────────────────────────────────
REASONING (hard logic)     o1                   o1-mini
REASONING (general)        claude-3-5-sonnet    gpt-4o-mini
CODING (generate)          claude-3-5-sonnet    gpt-4o-mini
CODING (hard algorithms)   o1                   gpt-4o
SUMMARIZATION              gpt-4o-mini          gemini-flash
EXTRACTION (structured)    gpt-4o-mini          claude-3-5-haiku
CLASSIFICATION             gpt-4o-mini          gemini-flash
LONG DOCUMENT QA           claude-3-5-sonnet    gemini-1.5-flash
TRANSLATION                gpt-4o               gpt-4o-mini
CREATIVE WRITING           claude-3-5-sonnet    gpt-4o
MULTIMODAL (vision)        gpt-4o               gemini-flash
REAL-TIME CHAT             claude-3-5-haiku     gemini-2.0-flash
HIGH-VOLUME BATCH          gpt-4o-mini          gemini-flash
ON-PREMISE/PRIVATE         llama-3.1-70b        llama-3.1-8b
```

---

## 4.2 Agent Task Types in Detail

### Type A: Planning and Reasoning Agents

The agent must decompose problems, create multi-step plans, reason about what to do next.

**Requirements**: High reasoning, strong instruction following, reliable tool use
**Recommended**: `claude-3-5-sonnet` or `gpt-4o`
**Why not mini**: Planning quality significantly drops; wrong action sequences are costly

```python
# Planning prompt — needs strong reasoning
PLANNING_PROMPT = """
You are a research agent. Break this goal into 5 specific steps.
Each step must have: step_number, description, tool (search/calculate/none), expected_output.
Return as JSON array.

Goal: Analyze the performance of top 3 Python web frameworks in 2024.
"""
```

### Type B: Data Extraction Agents

The agent reads documents and extracts structured information.

**Requirements**: Good instruction following, structured output, moderate reasoning
**Recommended**: `gpt-4o-mini` or `claude-3-5-haiku`
**Why not flagship**: Extraction is formulaic; cost savings are huge at scale

```python
# Extraction prompt — gpt-4o-mini excels, much cheaper
EXTRACTION_PROMPT = """
Extract from the following invoice. Return JSON only.
Fields: vendor(str), total_usd(float), date(str), items(list[{name,qty,unit_price}])
"""
```

### Type C: Conversational Agents

Long multi-turn dialogue, context-aware responses, natural interaction.

**Requirements**: Good conversational flow, context retention, personality consistency
**Recommended**: `claude-3-5-haiku` (fast, consistent) or `gpt-4o-mini`
**Latency matters**: Users expect < 2s response — avoid slow models here

### Type D: Code Generation Agents

Agents that write, debug, explain, or refactor code.

**Requirements**: Strong coding, ability to understand context, agentic editing
**Recommended**: `claude-3-5-sonnet` (best SWE-Bench) or `gpt-4o`
**Why it matters**: Code bugs compound; get it right the first time

### Type E: Document Intelligence Agents

Processes large collections of documents — contracts, reports, research papers.

**Requirements**: Large context, reliable information retrieval, good at following templates
**Recommended**: `claude-3-5-sonnet` (200k) for quality, `gemini-1.5-flash` (1M) for scale

---

## 4.3 Decision Flowchart — Step by Step

```
START
  │
  ▼
Is privacy/on-premise required?
  YES → llama-3.1-70b (Ollama/vLLM)        STOP
  NO  →
        │
        ▼
        Is context > 64k tokens?
          YES → claude-3-5-sonnet (200k) or gemini-1.5-pro (2M)   STOP
          NO  →
                │
                ▼
                Is cost the top priority?
                  YES → gpt-4o-mini / claude-3-5-haiku / gemini-flash   STOP
                  NO  →
                        │
                        ▼
                        Task type?
                        
                        HARD MATH/LOGIC  → o1 or o1-mini
                        CODING           → claude-3-5-sonnet
                        MULTIMODAL       → gpt-4o or gemini-1.5-pro
                        REAL-TIME CHAT   → claude-3-5-haiku
                        GENERAL QUALITY  → gpt-4o or claude-3-5-sonnet
```

---

## 4.4 Model Switching Strategies

### Strategy 1: Static Selection
Choose one model per agent type and stick with it.
```python
AGENT_MODELS = {
    "planner":   "gpt-4o",            # Uses flagship for planning
    "extractor": "gpt-4o-mini",       # Uses mini for bulk extraction
    "coder":     "claude-3-5-sonnet", # Uses Claude for coding
    "reviewer":  "gpt-4o-mini",       # Mini for cheap review pass
}
```

### Strategy 2: Quality-Based Escalation
Start cheap, automatically escalate when validation fails:
```python
ESCALATION_CHAIN = [
    "gpt-4o-mini",       # Try cheapest first
    "gpt-4o",            # Escalate if quality insufficient
    "claude-3-5-sonnet", # Final escalation
]
```

### Strategy 3: LiteLLM Router
Configure model priorities, fallbacks, and cost limits via LiteLLM:
```python
import litellm

router = litellm.Router(
    model_list=[
        {"model_name": "cheap", "litellm_params": {"model": "gpt-4o-mini"}},
        {"model_name": "quality", "litellm_params": {"model": "gpt-4o"}},
    ]
)
response = router.completion(model="cheap", messages=[...])
```

---

## 4.5 Multi-Model Agentic Architectures

Many production systems use **different models for different subtasks**:

```
┌─────────────────────────────────────────┐
│            User Request                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
        ┌──────────────────┐
        │  Router/Classifier│ ← gpt-4o-mini (cheap, fast)
        │  (what task?)     │
        └──────────┬───────┘
                   │
        ┌──────────┴──────────────┐
        │                         │
        ▼                         ▼
  ┌───────────┐           ┌───────────────┐
  │  Planner  │           │   Extractor   │
  │  gpt-4o   │           │  gpt-4o-mini  │
  └─────┬─────┘           └───────┬───────┘
        │                         │
        ▼                         ▼
  ┌───────────┐           ┌───────────────┐
  │   Coder   │           │   Summarizer  │
  │claude-3-5 │           │  gemini-flash │
  └───────────┘           └───────────────┘
```

---

## 📌 Key Takeaways

1. **Match task type to model strength** — don't use one model blindly for everything
2. **Use mini/haiku for extraction, classification, summarization** — flagship overkill
3. **Use flagship for planning, hard reasoning, complex coding** — quality pays off
4. **Multi-model architectures** — different models per subtask = quality + cost optimized
5. **Escalation patterns** — start cheap, trigger quality validation, escalate selectively
6. **Privacy constraint overrides all** — if data can't leave premises, go self-hosted
7. **Re-evaluate every 6 months** — model rankings change rapidly
