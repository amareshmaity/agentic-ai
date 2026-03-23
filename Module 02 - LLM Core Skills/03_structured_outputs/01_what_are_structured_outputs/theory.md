# 01 — What Are Structured Outputs?

> *The shift from "text that looks like JSON" to "guaranteed machine-readable data."*

---

## 1.1 The Problem with Unstructured LLM Output

By default, an LLM returns raw text. When you need that text to drive code, you're always guessing:

```python
# ❌ What you send
prompt = "Extract the customer name and order amount from this email."

# ❌ What you might get back (unpredictable!)
# "The customer is John Doe and his order was $245.00"
# OR: "Name: John Doe\nAmount: $245"
# OR: "{'name': 'John Doe', 'amount': 245}"  ← not quite JSON
# OR: "```json\n{\"name\": \"John Doe\", \"amount\": 245}\n```"
```

Every variation requires different parsing logic, and **one unexpected format breaks your pipeline**.

### The Hidden Tax of Unstructured Output

| Problem | Cost |
|---|---|
| Inconsistent format | Brittle regex / string parsing code |
| Missing fields | Silent data loss, hard bugs |
| Wrong types | `"245"` vs `245` — type errors in downstream code |
| Markdown wrapping | LLM adds ```json``` — your parser fails |
| Hallucinated keys | LLM invents extra fields you didn't ask for |

---

## 1.2 What Are Structured Outputs?

**Structured outputs** are LLM responses that are guaranteed to conform to a pre-defined schema — typically JSON — with correct field names, correct types, and no extra content.

```python
# ✅ Same task, with structured output
from pydantic import BaseModel
from openai import OpenAI

class OrderExtraction(BaseModel):
    customer_name: str
    order_amount: float

client = OpenAI()
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Extract: John Doe ordered $245.00"}],
    response_format=OrderExtraction
)

result = response.choices[0].message.parsed
print(result.customer_name)   # "John Doe"   — always a string
print(result.order_amount)    # 245.0         — always a float
```

This is a **schema contract**: you define the shape, the LLM must fill it.

---

## 1.3 The Three Categories of Structured Output Techniques

There are three distinct approaches, each with different reliability guarantees:

### Category 1: Prompt-Based (Unreliable)
Tell the LLM in the prompt to return JSON. The LLM *tries* but isn't *forced*.

```
"Return your answer as JSON with the keys 'name' and 'amount'."
```

- ✅ Works with any model, no special API support needed
- ❌ No guarantee — LLM may deviate, add prose, or use wrong types
- ❌ Still need `try/except json.loads()`

### Category 2: JSON Mode (Semi-Reliable)
Instruct the API to return valid JSON. Format is guaranteed; schema is not.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},  # Forces valid JSON
    messages=[...]
)
```

- ✅ Output is always valid JSON (no syntax errors)
- ❌ Field names and types are still up to the LLM
- ❌ No schema enforcement — extra/missing fields possible

### Category 3: Strict Structured Outputs (Fully Reliable)
Provide a full JSON Schema. The LLM is constrained at the **token level** to conform.

```python
response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    response_format=MyPydanticModel,    # Full schema enforcement
    messages=[...]
)
```

- ✅ Output is always valid JSON *and* matches your schema exactly
- ✅ Correct field names, correct types, no extra fields
- ✅ Parse directly to Python objects
- ⚠️ Requires models that support it (GPT-4o with structured output support)

---

## 1.4 When to Use Each Approach

| Scenario | Best Approach |
|---|---|
| Quick prototype, any model | Prompt-based |
| Need valid JSON, field names flexible | JSON mode |
| Production pipeline, type safety critical | Strict structured outputs |
| Extraction from documents | Strict + Pydantic |
| Agent decision interfaces | Strict + Pydantic |
| Classification tasks | Strict (use `enum` fields) |
| Data transformation pipelines | Strict + Pydantic |

---

## 1.5 The Evolution of Structured Output in LLM APIs

```
2022  Prompt engineering    → "Return JSON with keys X, Y, Z"
                              (unreliable, no guarantees)

2023  JSON Mode             → response_format: {"type": "json_object"}
                              (valid JSON guaranteed, schema not enforced)

2023  Function Calling hack → Force a tool call with a schema as parameters
                              (schema-enforced, but semantically hacky)

2024  Strict Structured     → response_format: {json_schema: ...}
      Outputs               → .parse() with Pydantic models
                              (true schema enforcement at token level)

2024+ Instructor library    → Unified Pydantic interface across providers
                              (cross-provider schema enforcement)
```

---

## 1.6 How Strict Structured Outputs Work Internally

OpenAI's strict mode works by **constraining the token sampling** process using a context-free grammar derived from your JSON schema.

```
Normal LLM sampling:
  At each token position, pick from all possible tokens.

  Token probabilities: {"name": 0.3, "The": 0.2, "I": 0.1, ...}
  → Can generate anything.

Structured output sampling:
  At each token position, only tokens valid under the current schema position are allowed.

  Schema expects: opening { then "customer_name" key then : then a string...
  Token probabilities: {"{": 1.0}  → forced to {
  Next position: {"\"customer_name\"": 1.0}  → forced key
  ...continues until schema is fully satisfied.
```

This is why structured output adds a small latency — the model must track schema state at each step.

---

## 1.7 Core Vocabulary

| Term | Meaning |
|---|---|
| **JSON Schema** | Standard for describing the structure of JSON data (types, required fields, etc.) |
| **Pydantic** | Python library for defining data models with automatic validation |
| **response_format** | OpenAI API parameter controlling output format |
| **Strict mode** | Schema enforced at token-sampling level — output guaranteed to match |
| **Instructor** | Python library providing Pydantic-based structured output for any LLM provider |
| **Parsing** | Converting raw LLM string output into Python objects |
| **Validation** | Checking that a parsed object satisfies constraints (types, ranges, etc.) |

---

## 1.8 Real-World Use Cases

```python
# 1. Document extraction
class InvoiceData(BaseModel):
    vendor: str
    invoice_number: str
    line_items: list[LineItem]
    total_amount: float
    due_date: str

# 2. Sentiment + category classification
class ContentClassification(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    category: Literal["complaint", "praise", "question", "other"]
    urgency: int  # 1-5
    requires_human: bool

# 3. Agent planning step
class AgentAction(BaseModel):
    thought: str                    # Chain-of-thought reasoning
    action: Literal["search", "calculate", "answer", "ask_user"]
    action_input: str
    confidence: float               # 0.0 - 1.0

# 4. Meeting notes summarization
class MeetingNotes(BaseModel):
    summary: str
    decisions: list[str]
    action_items: list[ActionItem]
    next_meeting_date: str | None
```

---

## 📌 Key Takeaways

1. **Unstructured output is unreliable** — format can vary with any prompt, any run
2. **Three levels**: prompt-based (weakest) → JSON mode → strict structured outputs (strongest)
3. **Strict structured outputs work at token level** — schema enforced during generation, not after
4. **Pydantic is the best way to define schemas in Python** — type-safe, composable, validated
5. **Use strict mode for production** — eliminates an entire class of parsing bugs
6. **Structured outputs enable typed agent interfaces** — agents that return machine-readable decisions
