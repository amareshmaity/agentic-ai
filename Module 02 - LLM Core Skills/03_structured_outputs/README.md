# 📐 Structured Outputs

> *Teaching LLMs to speak your language — reliably, schema-perfectly, every time.*

---

## 📌 Why Structured Outputs Are the Foundation of Reliable Agents

When an LLM returns free text, your downstream code must parse, guess, and hope.  
When an LLM returns **structured output**, your code can trust every field, every type, every key.

Structured outputs are what transform LLMs from *chatbots* into *programmable system components*:
- Extract data from documents with guaranteed JSON schemas
- Build reliable pipelines where step N's output feeds step N+1 as typed objects
- Eliminate `try/except json.loads()` hacks from your entire codebase
- Make agents that always return machine-readable decisions

**Structured outputs are how you make LLMs production-ready.**

---

## 📂 Folder Structure

```
03_structured_outputs/
│
├── README.md                                    ← You are here
│
├── 01_what_are_structured_outputs/
│   ├── theory.md                                ← What they are, why they matter, the evolution
│   └── examples.ipynb                           ← Unstructured vs structured comparison
│
├── 02_json_mode/
│   ├── theory.md                                ← JSON mode across providers, limitations
│   └── examples.ipynb                           ← JSON mode in practice with parsing
│
├── 03_pydantic_schemas/
│   ├── theory.md                                ← Pydantic v2 as schema source-of-truth
│   └── examples.ipynb                           ← Defining, nested, and validated schemas
│
├── 04_openai_structured_outputs/
│   ├── theory.md                                ← OpenAI response_format, strict JSON schema
│   └── examples.ipynb                           ← OpenAI structured outputs full API
│
├── 05_structured_outputs_across_providers/
│   ├── theory.md                                ← OpenAI vs Anthropic vs Gemini differences
│   └── examples.ipynb                           ← Same task on 3 providers side-by-side
│
├── 06_output_validation_and_repair/
│   ├── theory.md                                ← Validation strategies, repair patterns
│   └── examples.ipynb                           ← Robust parsing, Pydantic validation, auto-repair
│
└── 07_structured_outputs_in_agents/
    ├── theory.md                                ← Structured outputs in agentic pipelines
    └── examples.ipynb                           ← Production agent with structured decisions
```

---

## 📚 Topics Covered

| # | Subfolder | Core Concept |
|---|---|---|
| 1 | `01_what_are_structured_outputs` | Motivation, what changes, core categories |
| 2 | `02_json_mode` | JSON mode — the simplest structured output primitive |
| 3 | `03_pydantic_schemas` | Pydantic v2 models as structured output schemas |
| 4 | `04_openai_structured_outputs` | OpenAI `response_format` with strict JSON Schema |
| 5 | `05_structured_outputs_across_providers` | Cross-provider API differences and LiteLLM normalization |
| 6 | `06_output_validation_and_repair` | Parsing failures, validation, LLM self-repair |
| 7 | `07_structured_outputs_in_agents` | Structured outputs as agent decision interfaces |

---

## ⏱️ Estimated Time

| Activity | Time |
|---|---|
| Reading all 7 theory files | 3–4 hours |
| Running all 7 notebooks | 3–4 hours |
| **Total** | **~7 hours** |

---

## 🔧 Setup

```bash
pip install openai anthropic google-generativeai pydantic litellm python-dotenv rich instructor
```

```env
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

---

## 🔗 Prerequisites

- ✅ Module 02 → `01_prompt_engineering_for_agents`
- ✅ Module 02 → `02_function_calling_tool_use`
- Familiarity with JSON and Python dataclasses is helpful but not required
