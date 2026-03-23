# 04 — OpenAI Structured Outputs API

> *OpenAI's native structured output mode: strict JSON Schema enforcement at the token level.*

---

## 4.1 Two APIs: `.create()` vs `.parse()`

OpenAI offers two ways to use structured outputs:

### Option A: `.create()` with `response_format` JSON Schema
```python
# Raw JSON Schema approach — manual parsing required
response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "product_info",       # Human-readable name
            "strict": True,               # Enforce schema at token level
            "schema": {                   # Your JSON Schema
                "type": "object",
                "properties": {
                    "name":  {"type": "string"},
                    "price": {"type": "number"}
                },
                "required": ["name", "price"],
                "additionalProperties": False
            }
        }
    },
    messages=[...]
)
raw = response.choices[0].message.content  # Still a string you parse
data = json.loads(raw)
```

### Option B: `.parse()` with Pydantic (Recommended)
```python
# Pydantic approach — returns typed Python object, no manual parsing
from pydantic import BaseModel

class ProductInfo(BaseModel):
    name: str
    price: float

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    response_format=ProductInfo,   # Pass the class directly
    messages=[...]
)
product = response.choices[0].message.parsed  # Already a ProductInfo instance
```

**Use `.parse()` whenever possible** — it's cleaner, type-safe, and Pydantic handles the schema.

---

## 4.2 What `strict: True` Actually Guarantees

When `strict=True` is set (the default with `.parse()`):

| Guarantee | Description |
|---|---|
| ✅ Valid JSON syntax | Output is always parseable with `json.loads()` |
| ✅ All required fields present | No missing keys from your schema |
| ✅ No extra fields | No extra keys the LLM decided to add |
| ✅ Correct types | `string` fields are strings, `number` fields are numbers |
| ✅ Enum values respected | `Literal["a", "b"]` always returns "a" or "b" |
| ❌ Semantic accuracy | The LLM may still misidentify content (e.g., wrong category) |

```python
# The model guarantees structure, NOT semantic accuracy
# These are still possible:
# - rating: 3.0  (correct type, might be wrong sentiment)
# - category: "other"  (valid enum value, but model might mis-classify)
```

---

## 4.3 JSON Schema Strict Mode Requirements

When writing raw JSON Schema for strict mode, there are specific rules:

```python
# ✅ VALID strict schema
strict_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "metadata": {
            "type": ["object", "null"],   # Allows null (Optional equivalent)
            "properties": {
                "source": {"type": "string"}
            },
            "required": ["source"],
            "additionalProperties": False
        }
    },
    "required": ["name", "tags", "metadata"],    # ALL properties must be required
    "additionalProperties": False                 # MUST be False in strict mode
}

# ❌ INVALID for strict mode:
invalid_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
    # Missing "required" and "additionalProperties" — will error
}
```

### Strict Mode Schema Rules:
1. **All properties must be in `required`** — no optional fields by omitting from required
2. **`additionalProperties: false`** — required for every object in the tree
3. **Optional fields** → use `["type", "null"]` (union with null)
4. **Recursive schemas** → supported but may add slight latency

---

## 4.4 Handling `refusal` — When the Model Declines

Structured output adds a `refusal` field — the model can choose to refuse rather than generate a schema-forced response for harmful content:

```python
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What is the formula for nerve agent synthesis?"}
    ],
    response_format=SomeSchema
)

msg = response.choices[0].message

# Always check for refusal BEFORE accessing .parsed
if msg.refusal:
    print(f"Model refused: {msg.refusal}")
else:
    data = msg.parsed
    print(f"Got data: {data}")
```

**Always check `msg.refusal` before `msg.parsed`** — accessing `.parsed` on a refused message raises an error.

---

## 4.5 Supported and Unsupported Schema Features

### ✅ Supported in Strict Mode

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional, Union

class FullExample(BaseModel):
    # Basic types
    text: str
    count: int
    price: float
    active: bool
    
    # Optional (becomes anyOf with null)
    note: Optional[str] = None
    
    # Enums
    status: Literal["active", "inactive", "pending"]
    
    # Arrays
    tags: list[str]
    scores: list[float]
    
    # Nested objects
    address: "AddressModel"  # Forward reference
    
    # anyOf / Union
    value: Union[str, int]
```

### ❌ Unsupported in Strict Mode (as of 2024)
- `minLength`, `maxLength` on strings (use in descriptions instead)
- `minimum`, `maximum` on numbers (describe in field descriptions)
- `pattern` (regex on strings)
- `$ref` circular recursion (limited support)
- `if/then/else` conditionals

---

## 4.6 Streaming Structured Outputs

OpenAI structured outputs can be streamed — you get partial data as it's generated:

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class ResearchSummary(BaseModel):
    topic: str
    key_findings: list[str]
    conclusion: str
    confidence_level: Literal["high", "medium", "low"]

# Stream the structured output
with client.beta.chat.completions.stream(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize what is known about black holes."}],
    response_format=ResearchSummary
) as stream:
    # Progress events
    for event in stream:
        if hasattr(event, "type"):
            if event.type == "content.delta":
                print(".", end="", flush=True)
    
    print()  # newline
    # Final parsed result
    final = stream.get_final_message()
    result = final.choices[0].message.parsed
    print(f"Topic: {result.topic}")
    print(f"Confidence: {result.confidence_level}")
    print(f"Findings: {len(result.key_findings)} items")
```

---

## 4.7 Token Usage and Latency Considerations

Strict structured outputs consume extra tokens for the schema and add slight latency:

```python
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Extract: Alice, 28, engineer"}],
    response_format=PersonInfo
)

# Check token usage
usage = response.usage
print(f"Prompt tokens:     {usage.prompt_tokens}")      # Includes schema tokens
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens:      {usage.total_tokens}")

# Latency: strict mode adds ~10-20% latency vs unstructured
# For most use cases this is negligible (< 500ms extra)
# For high-volume real-time systems: profile and benchmark first
```

**Optimization tips**:
- Keep schemas as small as needed — fewer fields = fewer schema tokens
- Use `description` on fields instead of long field names
- Remove unused fields — every field in schema costs tokens

---

## 4.8 Model Compatibility

Not all OpenAI models support strict structured outputs:

| Model | JSON Mode | Strict Structured Output |
|---|---|---|
| `gpt-4o-mini` | ✅ | ✅ |
| `gpt-4o` (2024-08-06+) | ✅ | ✅ |
| `gpt-4-turbo` | ✅ | ❌ (JSON mode only) |
| `gpt-3.5-turbo` | ✅ | ❌ (JSON mode only) |
| `o1`, `o3` | ✅ | ✅ |

Always check the [OpenAI model documentation](https://platform.openai.com/docs/models) for the latest compatibility.

---

## 4.9 Complete Production Pattern

```python
import json
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, Literal

client = OpenAI()

# Schema definition
class DocumentClassification(BaseModel):
    document_type: Literal[
        "invoice", "contract", "report", "email", 
        "resume", "legal", "technical", "other"
    ]
    language: str
    page_estimate: int = Field(ge=1)
    contains_pii: bool
    summary: str = Field(description="One sentence summary of document content")
    key_entities: list[str] = Field(
        default_factory=list,
        description="Key people, organizations, or products mentioned"
    )
    action_required: Optional[str] = None

def classify_document(text: str) -> DocumentClassification | None:
    """Classify a document with full error handling."""
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a document classifier. Analyze the provided document."
                },
                {
                    "role": "user",
                    "content": f"Classify this document:\n\n{text}"
                }
            ],
            response_format=DocumentClassification,
            temperature=0.0
        )
        
        msg = response.choices[0].message
        
        # Check for refusal first
        if msg.refusal:
            print(f"Model refused to classify: {msg.refusal}")
            return None
        
        return msg.parsed
    
    except Exception as e:
        print(f"Structured output failed: {type(e).__name__}: {e}")
        return None

# Usage
doc = """
INVOICE #2024-1234
Billing to: Acme Corporation
For: Software Development Services - November 2024
Amount Due: $12,500.00
Payment Due: December 31, 2024
"""

result = classify_document(doc)
if result:
    print(f"Type:      {result.document_type}")
    print(f"Language:  {result.language}")
    print(f"Contains PII: {result.contains_pii}")
    print(f"Summary:   {result.summary}")
    print(f"Entities:  {result.key_entities}")
```

---

## 📌 Key Takeaways

1. **Use `.parse()` not `.create()`** — cleaner API, typed return values, no manual parsing
2. **`strict=True` guarantees** schema shape — all required fields, no extra fields, correct types
3. **Always check `msg.refusal`** before `msg.parsed` — model can refuse harmful requests
4. **Strict schema rules**: all properties required, `additionalProperties: False`, optional via union with null
5. **Streaming**: use `client.beta.chat.completions.stream()` for long structured outputs
6. **Model compatibility**: `gpt-4o-mini` and `gpt-4o` (2024-08-06+) support strict mode
7. **Semantic accuracy not guaranteed** — schema shape is enforced, not factual correctness
