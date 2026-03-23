# 05 — Structured Outputs Across Providers

> *OpenAI, Anthropic, and Google Gemini each handle structured output differently — here's how to navigate all three.*

---

## 5.1 Provider Comparison Overview

| Feature | OpenAI | Anthropic (Claude) | Google Gemini |
|---|---|---|---|
| **JSON Mode** | ✅ Native (`json_object`) | ❌ Prompt-only | ✅ Native (`application/json`) |
| **Strict Structured Output** | ✅ Native (`.parse()` / `json_schema`) | ❌ Not native | ⚠️ Via `response_schema` (partial) |
| **Pydantic Integration** | ✅ Direct (`.parse()`) | ⚠️ Via `instructor` | ⚠️ Via `instructor` |
| **Schema Enforcement** | ✅ Token-level | ❌ Prompt-based | ⚠️ Best-effort |
| **Streaming Structured** | ✅ | ❌ | ⚠️ |
| **Cross-provider tool** | `instructor`, `litellm` | `instructor`, `litellm` | `instructor`, `litellm` |

---

## 5.2 OpenAI — Native Structured Output (Recap)

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class ProductInfo(BaseModel):
    name: str
    price: float
    category: str

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Extract: iPhone 15 Pro, $999, electronics"}],
    response_format=ProductInfo
)
product = response.choices[0].message.parsed  # ← Typed object
print(product.name, product.price)
```

---

## 5.3 Anthropic Claude — Prompt Engineering Approach

Claude doesn't have a native structured output API (as of 2024). Use prompt engineering + `instructor`:

### Option A: Prompt-Based (No library)
```python
import anthropic, json, re

client = anthropic.Anthropic()

EXTRACT_SYSTEM = """You are a data extractor. Always respond with ONLY valid JSON.
No explanation, no markdown fences, no prose. Just the raw JSON object."""

def extract_with_claude(text: str, schema_description: str) -> dict:
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        system=EXTRACT_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Schema: {schema_description}\n\nExtract from: {text}"
        }]
    )
    
    raw = response.content[0].text.strip()
    
    # Strip markdown fences if Claude adds them anyway
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"error": "Failed to parse", "raw": cleaned}

result = extract_with_claude(
    "Order by Sarah Johnson: 3x Coffee at $4.50 each, Total: $13.50",
    '{"customer": str, "items": [{"product": str, "qty": int, "price": float}], "total": float}'
)
print(result)
```

### Option B: `instructor` Library (Recommended for Anthropic)
```python
import anthropic
import instructor
from pydantic import BaseModel

# Patch the Anthropic client with instructor
anthropic_client = instructor.from_anthropic(anthropic.Anthropic())

class ProductInfo(BaseModel):
    name: str
    price: float
    category: str

# instructor handles all the prompt engineering and parsing
product = anthropic_client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1024,
    messages=[{"role": "user", "content": "iPhone 15 Pro, $999, electronics category"}],
    response_model=ProductInfo  # instructor's equivalent of response_format
)
print(product.name, product.price)  # Typed Pydantic object!
```

---

## 5.4 Google Gemini — Native JSON + `response_schema`

Gemini supports JSON mode via `response_mime_type` and schema via `response_schema`:

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Option A: JSON mode only (valid JSON, no schema enforcement)
model_json = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=genai.types.GenerationConfig(
        response_mime_type="application/json"
    )
)

response = model_json.generate_content(
    "Extract {name: str, price: float} from: 'MacBook Pro costs $2,499'"
)
import json
data = json.loads(response.text)  # Always valid JSON
print(data)

# Option B: response_schema (Gemini's schema enforcement - partial)
import typing_extensions

class ProductInfo(typing_extensions.TypedDict):
    name: str
    price: float
    category: str

model_schema = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=genai.types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=ProductInfo      # Schema guidance (not strict enforcement)
    )
)

response = model_schema.generate_content(
    "iPad Air costs $599, tablet category"
)
data = json.loads(response.text)
print(data)
```

---

## 5.5 The `instructor` Library — Unified Pydantic Interface

`instructor` is the most popular library for adding Pydantic structured output to any provider:

```bash
pip install instructor
```

```python
import instructor
from openai import OpenAI
from anthropic import Anthropic
from pydantic import BaseModel
from typing import Literal

# Same schema for all providers
class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrase: str

TEXT = "The new product launch was an absolute disaster. Sales dropped 40%."

# ── OpenAI with instructor ────────────────────────────────────────────────
openai_client = instructor.from_openai(OpenAI())
openai_result = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": TEXT}],
    response_model=SentimentResult
)
print(f"OpenAI:    {openai_result.sentiment} ({openai_result.confidence:.0%}) - '{openai_result.key_phrase}'")

# ── Anthropic with instructor ─────────────────────────────────────────────
anthropic_client = instructor.from_anthropic(Anthropic())
claude_result = anthropic_client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=256,
    messages=[{"role": "user", "content": TEXT}],
    response_model=SentimentResult
)
print(f"Claude:    {claude_result.sentiment} ({claude_result.confidence:.0%}) - '{claude_result.key_phrase}'")
```

---

## 5.6 LiteLLM — One API for All Providers

LiteLLM provides a unified OpenAI-compatible interface for 100+ models:

```python
from litellm import completion
import json

def extract_with_litellm(text: str, model: str, schema_hint: str) -> dict:
    """Use any model via LiteLLM's unified interface."""
    
    response = completion(
        model=model,
        response_format={"type": "json_object"},  # JSON mode where supported
        messages=[
            {"role": "system", "content": f"Extract as JSON. Schema: {schema_hint}"},
            {"role": "user",   "content": text}
        ]
    )
    
    raw = response.choices[0].message.content
    return json.loads(raw)

# Same code, different models
models_to_test = [
    "gpt-4o-mini",                            # OpenAI
    "claude-3-haiku-20240307",                # Anthropic
    "gemini/gemini-1.5-flash",                # Gemini
]

schema = "{product: str, price: float, brand: str}"
text = "Samsung Galaxy S24 Ultra from Samsung, priced at $1,299"

for model in models_to_test:
    try:
        result = extract_with_litellm(text, model, schema)
        print(f"{model:<35}: {result}")
    except Exception as e:
        print(f"{model:<35}: Error - {type(e).__name__}")
```

---

## 5.7 Cross-Provider Differences: Practical Cheat Sheet

### JSON Mode
```python
# OpenAI
response_format={"type": "json_object"}   # Built-in

# Anthropic  
system="Return ONLY valid JSON."           # Prompt-based, strip fences

# Gemini
generation_config={"response_mime_type": "application/json"}  # Built-in
```

### Strict Schema Enforcement
```python
# OpenAI — native
client.beta.chat.completions.parse(model="gpt-4o-mini", response_format=MyModel)

# Anthropic — via instructor
instructor.from_anthropic(Anthropic()).messages.create(response_model=MyModel)

# Gemini — partial, via response_schema
generation_config={"response_schema": TypedDict}  # Best-effort, not strict

# Gemini via instructor — better enforcement
instructor.from_gemini(genai.GenerativeModel("gemini-1.5-flash")).create(response_model=MyModel)
```

---

## 5.8 Reliability Ranking

Based on real-world usage:

| Provider | Reliability | Speed | Cost | Best For |
|---|---|---|---|---|
| OpenAI (gpt-4o-mini) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Production structured output |
| Anthropic (Claude) + instructor | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Complex reasoning + extraction |
| Gemini Flash | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High-volume, cost-sensitive |
| LiteLLM + any | ⭐⭐⭐ | Varies | Varies | Flexibility, fallback routing |

---

## 📌 Key Takeaways

1. **OpenAI has the best native structured output** — use `.parse()` with Pydantic
2. **Anthropic Claude has no native schema enforcement** — use `instructor` library or prompt engineering
3. **Gemini has native JSON mode** via `response_mime_type` and partial schema via `response_schema`
4. **`instructor`** provides a unified Pydantic interface across all major providers
5. **`litellm`** provides a unified OpenAI-compatible API for 100+ models
6. **For production multi-provider systems**: standardize on `instructor` + retry logic
7. **Prompt-based JSON for Claude**: always strip markdown fences, add strict instructions in system prompt
