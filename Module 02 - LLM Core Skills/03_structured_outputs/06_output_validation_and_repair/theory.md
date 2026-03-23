# 06 — Output Validation and Repair

> *What happens when structured output breaks — and how to detect, validate, and automatically repair it.*

---

## 6.1 Why Structured Output Still Fails

Even with strict mode enabled, structured output can fail in several ways:

```
1. Semantic errors      → LLM returns wrong data (right type, wrong value)
2. Refusals             → Model declines to generate output
3. Truncation           → max_tokens reached mid-schema
4. Model not supported  → Strict mode used on incompatible model
5. Schema too complex   → Circular refs, unsupported schema features
6. Prompt-based gaps    → JSON mode produces valid JSON but wrong schema
7. Validation failures  → Custom Pydantic validators reject the output
```

A robust production system handles all seven cases.

---

## 6.2 Validation Layers

Think of validation as a three-layer stack:

```
Layer 3: Business Logic Validation   ← Your custom rules (e.g., total matches items)
    ↑
Layer 2: Pydantic Model Validation   ← Types, ranges, Literal constraints
    ↑
Layer 1: JSON Syntax Validation      ← Can json.loads() parse this?
```

```python
from pydantic import BaseModel, Field, model_validator
from typing import Literal
import json

class OrderSummary(BaseModel):
    # Layer 1 is handled by json.loads() before even reaching Pydantic
    
    # Layer 2: Pydantic type + range validation
    order_id: str
    item_count: int = Field(ge=1, le=1000)
    unit_price: float = Field(ge=0)
    total: float = Field(ge=0)
    status: Literal["pending", "processing", "shipped", "delivered", "cancelled"]
    
    # Layer 3: Business logic — total must equal item_count * unit_price
    @model_validator(mode="after")
    def validate_total_matches(self):
        expected = round(self.item_count * self.unit_price, 2)
        actual   = round(self.total, 2)
        if abs(expected - actual) > 0.01:  # Allow 1 cent rounding error
            raise ValueError(
                f"Total ${actual} doesn't match {self.item_count} × ${self.unit_price} = ${expected}"
            )
        return self
```

---

## 6.3 Detecting and Catching Validation Errors

```python
from openai import OpenAI
from pydantic import BaseModel, ValidationError
import json

client = OpenAI()

def safe_structured_extract(text: str, schema: type[BaseModel]) -> tuple[BaseModel | None, str | None]:
    """
    Extract with full error classification.
    Returns (result, error_message) — one will always be None.
    """
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": text}],
            response_format=schema
        )
        
        msg = response.choices[0].message
        
        # Check refusal
        if msg.refusal:
            return None, f"REFUSAL: {msg.refusal}"
        
        # Check truncation (finish_reason = "length")
        if response.choices[0].finish_reason == "length":
            return None, "TRUNCATED: max_tokens reached before schema completion"
        
        # Validate the parsed result
        result = msg.parsed
        if result is None:
            return None, "PARSE_FAILED: Parsed object is None"
        
        return result, None
    
    except ValidationError as e:
        # Pydantic validation failed
        field_errors = [(err["loc"][0], err["msg"]) for err in e.errors()]
        return None, f"VALIDATION_ERROR: {field_errors}"
    
    except Exception as e:
        return None, f"UNEXPECTED: {type(e).__name__}: {e}"
```

---

## 6.4 The Self-Repair Pattern — Ask the LLM to Fix Its Own Output

When validation fails, send the error back to the LLM and ask it to repair:

```python
from pydantic import BaseModel, ValidationError
from openai import OpenAI
import json

client = OpenAI()

def extract_with_self_repair(
    text: str,
    schema: type[BaseModel],
    max_repair_attempts: int = 2
) -> BaseModel | None:
    """
    Extract structured data. If validation fails, ask LLM to repair.
    """
    messages = [
        {"role": "system", "content": "You are a careful data extractor."},
        {"role": "user",   "content": f"Extract structured data from:\n{text}"}
    ]
    
    for attempt in range(max_repair_attempts + 1):
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=schema
        )
        
        msg = response.choices[0].message
        
        if msg.refusal:
            print(f"Model refused: {msg.refusal}")
            return None
        
        try:
            result = msg.parsed
            if result:
                if attempt > 0:
                    print(f"✅ Repaired successfully on attempt {attempt + 1}")
                return result
        except ValidationError as e:
            if attempt >= max_repair_attempts:
                print(f"❌ Validation failed after {attempt + 1} attempts: {e}")
                return None
            
            # Add the error to the conversation so LLM can self-repair
            error_feedback = f"Your output failed validation: {e.json()}\nPlease fix and resubmit."
            messages.append(msg)  # The assistant's bad response
            messages.append({"role": "user", "content": error_feedback})
            print(f"⚠️  Attempt {attempt + 1} failed — asking LLM to repair...")
    
    return None
```

---

## 6.5 The Fallback Chain — Multiple Parsing Strategies

```python
from pydantic import BaseModel, ValidationError
from openai import OpenAI
import json, re

client = OpenAI()

class EventInfo(BaseModel):
    name: str
    date: str
    location: str
    capacity: int

def extract_event_with_fallback(text: str) -> EventInfo | None:
    """
    Try extraction strategies in order of reliability.
    """
    
    # Strategy 1: Strict structured output (most reliable)
    try:
        r = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Extract event info: {text}"}],
            response_format=EventInfo
        )
        if r.choices[0].message.parsed:
            print("✅ Strategy 1 succeeded (strict structured output)")
            return r.choices[0].message.parsed
    except Exception as e:
        print(f"⚠️  Strategy 1 failed: {e}")
    
    # Strategy 2: JSON mode + manual Pydantic parse
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Extract event: {name, date, location, capacity_int}"},
                {"role": "user",   "content": text}
            ]
        )
        raw_dict = json.loads(r.choices[0].message.content)
        result = EventInfo(**raw_dict)  # Manually construct and validate
        print("✅ Strategy 2 succeeded (JSON mode + Pydantic)")
        return result
    except Exception as e:
        print(f"⚠️  Strategy 2 failed: {e}")
    
    # Strategy 3: Prompt-based extraction + regex post-processing
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Extract from text. Return raw JSON only:\n{text}\n\nFields: name, date, location, capacity (integer)"
            }]
        )
        raw = r.choices[0].message.content
        # Aggressively strip any non-JSON content
        cleaned = re.search(r'\{.*\}', raw, re.DOTALL)
        if cleaned:
            data = json.loads(cleaned.group())
            result = EventInfo(
                name=str(data.get("name", "Unknown")),
                date=str(data.get("date", "TBD")),
                location=str(data.get("location", "Unknown")),
                capacity=int(data.get("capacity", 0))
            )
            print("✅ Strategy 3 succeeded (prompt-based + aggressive parsing)")
            return result
    except Exception as e:
        print(f"⚠️  Strategy 3 failed: {e}")
    
    print("❌ All strategies exhausted — returning None")
    return None
```

---

## 6.6 Output Sanitization — Cleaning Before Validation

Sometimes LLM output is structurally correct but contains dirty data:

```python
from pydantic import BaseModel, field_validator
from typing import Optional
import re

class ContactExtraction(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    
    @field_validator("email")
    @classmethod
    def clean_email(cls, v: str) -> str:
        """Normalize email — LLM may add extra spaces or mixed case."""
        return v.strip().lower()
    
    @field_validator("phone")
    @classmethod
    def clean_phone(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        # Keep only valid phone characters
        cleaned = re.sub(r"[^\d\+\-\(\)\s]", "", v).strip()
        return cleaned if len(cleaned) >= 7 else None
    
    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        """Remove extra whitespace, fix casing."""
        return " ".join(v.split()).title()

# Test sanitization on messy LLM output
messy_contact = ContactExtraction(
    name="  john   DOE  ",
    email="  JOHN.DOE@EXAMPLE.COM  ",
    phone="(555) 123-4567 ext. 89"    # ext. is not valid phone chars
)
print(f"name:  {messy_contact.name!r}")    # 'John Doe'
print(f"email: {messy_contact.email!r}")  # 'john.doe@example.com'
print(f"phone: {messy_contact.phone!r}")  # '(555) 123-4567 . 89' but cleaned
```

---

## 6.7 Partial Output Recovery — When Truncation Happens

```python
def extract_with_truncation_recovery(
    text: str, schema: type[BaseModel], max_tokens: int = 4096
) -> BaseModel | None:
    """
    Handle max_tokens truncation by re-requesting with more budget.
    """
    for token_budget in [max_tokens, max_tokens * 2, max_tokens * 4]:
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": text}],
                response_format=schema,
                max_tokens=token_budget
            )
            
            finish_reason = response.choices[0].finish_reason
            
            if finish_reason == "length":
                print(f"⚠️  Truncated at {token_budget} tokens — retrying with {token_budget * 2}")
                continue
            
            result = response.choices[0].message.parsed
            if result:
                return result
        except Exception as e:
            print(f"Error at {token_budget} tokens: {e}")
            break
    
    return None
```

---

## 6.8 Validation Metrics — Tracking Reliability in Production

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ExtractionMetrics:
    """Track structured output reliability over time."""
    total_attempts: int = 0
    successes: int = 0
    refusals: int = 0
    validation_errors: int = 0
    truncations: int = 0
    repairs_needed: int = 0
    repairs_succeeded: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return self.successes / self.total_attempts if self.total_attempts > 0 else 0
    
    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0
    
    def report(self):
        print(f"Extraction Metrics Report")
        print(f"  Total attempts:     {self.total_attempts}")
        print(f"  Success rate:       {self.success_rate:.1%}")
        print(f"  Refusals:           {self.refusals}")
        print(f"  Validation errors:  {self.validation_errors}")
        print(f"  Truncations:        {self.truncations}")
        print(f"  Repairs needed:     {self.repairs_needed}")
        print(f"  Repair success:     {self.repairs_succeeded}/{self.repairs_needed}")
        print(f"  Avg latency:        {self.avg_latency_ms:.0f}ms")

metrics = ExtractionMetrics()
# Increment as you extract...
```

---

## 📌 Key Takeaways

1. **Three layers of validation**: JSON syntax → Pydantic types → business logic
2. **Self-repair**: return the validation error to the LLM and ask it to fix
3. **Fallback chain**: strict → JSON mode → prompt-only, with graceful degradation
4. **Field validators** in Pydantic auto-sanitize dirty LLM output (emails, phones, names)
5. **Truncation recovery**: retry with larger `max_tokens` budget when `finish_reason == "length"`
6. **Track metrics** — success rate, repair rate, latency — to detect model drift in production
7. **Never crash** — always return `None` + error message rather than raising in production code
