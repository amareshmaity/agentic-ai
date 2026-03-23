# 02 — JSON Mode

> *The first step toward structured output — valid JSON guaranteed, schema optional.*

---

## 2.1 What Is JSON Mode?

JSON mode is a parameter you set on the LLM API call that instructs the model to **always return syntactically valid JSON**. Unlike prompt-based approaches, if JSON mode is enabled, the output will never have markdown fences, prose, or syntax errors.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},  # ← JSON mode
    messages=[
        {"role": "system", "content": "You are a data extractor. Return JSON."},
        {"role": "user",   "content": "Extract name and age from: Alex is 28 years old."}
    ]
)

raw = response.choices[0].message.content
data = json.loads(raw)  # This will NEVER fail with a JSONDecodeError
print(data)  # {"name": "Alex", "age": 28}
```

---

## 2.2 JSON Mode vs Strict Structured Output

| Feature | JSON Mode | Strict Structured Output |
|---|---|---|
| Output always valid JSON | ✅ Yes | ✅ Yes |
| Field names guaranteed | ❌ No | ✅ Yes |
| Data types guaranteed | ❌ No | ✅ Yes |
| Required fields guaranteed | ❌ No | ✅ Yes |
| No extra fields | ❌ No | ✅ Yes |
| Model support required | Any OpenAI | GPT-4o+ with structured output |
| Works with Anthropic/Gemini | ✅ Yes (via prompt) | ⚠️ Varies |
| Best for | Simple extraction, prototyping | Production pipelines |

---

## 2.3 The Critical Rule: System Prompt Must Mention JSON

OpenAI's JSON mode **requires** the word "JSON" (or "json") to appear in your messages — usually in the system prompt. If you omit it, you get an error:

```python
# ❌ This will raise: BadRequestError
response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[{"role": "user", "content": "What is 2 + 2?"}]  # No mention of JSON!
)

# ✅ This works
response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "You are a calculator. Always return JSON with key 'result'."},
        {"role": "user",   "content": "What is 2 + 2?"}
    ]
)
```

**Rule**: Always include "Return JSON" or similar in your system prompt when using JSON mode.

---

## 2.4 JSON Mode Request and Response Anatomy

```python
# Full request
response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},  # ← JSON mode flag
    messages=[
        {
            "role": "system",
            "content": "Extract product info. Return JSON with keys: name (str), price (number), in_stock (boolean)."
        },
        {
            "role": "user",
            "content": "The Dell XPS 15 laptop costs $1,499.99 and is currently available."
        }
    ],
    temperature=0.1  # Low temperature for more consistent structured output
)

# Response inspection
msg = response.choices[0].message
print(f"Content type: {type(msg.content)}")       # Always str
print(f"finish_reason: {response.choices[0].finish_reason}")  # "stop"

# Parse — this will ALWAYS succeed in JSON mode
data = json.loads(msg.content)
print(f"name:     {data['name']}")                 # str
print(f"price:    {data['price']}")                # number (could be str or int — LLM decides)
print(f"in_stock: {data['in_stock']}")             # LLM decides — may be bool or "true"
```

---

## 2.5 Controlling Schema via the System Prompt

Since JSON mode doesn't enforce a schema, your only control is through the system prompt. Use precise instructions:

```python
# ❌ Weak schema description — LLM may invent field names
system = "Extract the order details as JSON."

# ✅ Strong schema description — specific field names and types
system = """Extract order details from the user message.
Return a JSON object with EXACTLY these fields:
{
    "customer_name": string,
    "order_id": string (include the # prefix),
    "items": [{"product": string, "quantity": integer, "unit_price": number}],
    "total": number,
    "status": one of ["pending", "processing", "shipped", "delivered"]
}
Do not include any other fields."""
```

---

## 2.6 Parsing and Type Safety After JSON Mode

```python
import json
from dataclasses import dataclass

def safe_parse_json(raw: str) -> dict | None:
    """Safely parse JSON mode output with fallback."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Should never happen in JSON mode, but defensive coding is good
        return None

# After parsing, always validate types manually
def extract_product(text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Extract product JSON: {name: str, price: float, quantity: int}"},
            {"role": "user",   "content": text}
        ]
    )
    data = safe_parse_json(response.choices[0].message.content)
    
    if data is None:
        return {"error": "Failed to parse JSON"}
    
    # Coerce types defensively — JSON mode doesn't guarantee types
    return {
        "name":     str(data.get("name", "")),
        "price":    float(data.get("price", 0)),
        "quantity": int(data.get("quantity", 0))
    }

result = extract_product("5 units of Premium Coffee Beans at $12.99 each")
print(result)
print(f"Types: name={type(result['name']).__name__}, price={type(result['price']).__name__}")
```

---

## 2.7 JSON mode Across Providers

### OpenAI
```python
# OpenAI — native JSON mode
response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[...]
)
```

### Anthropic (Claude)
```python
# Anthropic — no native JSON mode; use prompt + post-processing
import anthropic

anthropic_client = anthropic.Anthropic()

response = anthropic_client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1024,
    system="Always return valid JSON. No prose, no markdown fences.",
    messages=[{"role": "user", "content": "Extract: {name, price} from 'iPhone 15 costs $999'"}]
)
raw = response.content[0].text

# Strip potential ```json fences from Claude
import re
cleaned = re.sub(r'^```json\s*|\s*```$', '', raw.strip())
data = json.loads(cleaned)
print(data)
```

### Google Gemini
```python
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={"response_mime_type": "application/json"}  # ← Gemini's JSON mode
)

response = model.generate_content(
    "Extract {name: str, price: float} from: 'MacBook Pro costs $2,499'"
)
data = json.loads(response.text)
print(data)
```

---

## 2.8 JSON Mode Limitations — When NOT to Use It

❌ **Don't use JSON mode when you need**:
- Guaranteed field names — use Strict Structured Outputs instead
- Type-safe fields — LLM may return `"price": "12.99"` instead of `"price": 12.99`
- Required fields guaranteed present — LLM may omit fields
- No extra fields — LLM may add commentary keys like `"note"` or `"explanation"`
- Nested complex schemas — harder to prompt-engineer reliably

✅ **JSON mode is ideal when**:
- You're using a model without strict structured output support
- The schema is simple (2-4 top-level keys)
- You're prototyping and will add proper schemas later
- The consumer code does its own validation anyway

---

## 2.9 JSON Lines — Streaming Structured Data

JSON mode also works with streamed outputs. For large structured outputs, you can stream and parse after:

```python
# Stream a large JSON object
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "Return a JSON object with 5 product recommendations."},
        {"role": "user",   "content": "Recommend 5 Python books for beginners."}
    ],
    stream=True
)

# Accumulate the full JSON string
full_content = ""
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        full_content += delta.content
        print(".", end="", flush=True)  # Progress indicator

print()  # newline
# Parse the complete JSON
data = json.loads(full_content)
print(f"Got {len(data)} top-level keys: {list(data.keys())}")
```

---

## 📌 Key Takeaways

1. **JSON mode guarantees valid JSON syntax** — `json.loads()` will always succeed
2. **JSON mode does NOT guarantee schema** — field names and types are still up to the LLM
3. **Always mention "JSON" in your messages** — OpenAI requires it, or raises an error
4. **Low temperature helps** — `temperature=0.0` to `0.2` makes output more consistent
5. **Always coerce types after parsing** — `float(data["price"])` not `data["price"]`
6. **Cross-provider**: OpenAI has native JSON mode; Anthropic needs prompt engineering; Gemini has `response_mime_type`
7. **Step up to strict mode** when your schema matters — JSON mode is a stepping stone, not a destination
