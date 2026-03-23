# 03 — Pydantic Schemas for Structured Outputs

> *Pydantic is the idiomatic Python way to define, validate, and enforce output schemas — the industry standard for LLM structured output.*

---

## 3.1 Why Pydantic?

Before Pydantic integration with LLMs, structured output schemas were written as raw JSON Schema dictionaries — verbose, error-prone, and disconnected from Python types:

```python
# ❌ Old way — raw JSON Schema dict (verbose, no Python type checking)
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer", "minimum": 0},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age"],
    "additionalProperties": False
}

# ✅ Pydantic way — Python class with full IDE support + auto JSON Schema generation
from pydantic import BaseModel, Field
from typing import Optional

class UserProfile(BaseModel):
    name:  str
    age:   int = Field(ge=0)          # ge=0 means greater than or equal to 0
    email: Optional[str] = None       # Optional field with default None
```

Pydantic automatically generates the JSON Schema from your Python class — and validates data against it.

---

## 3.2 Pydantic v2 Essentials for LLM Output

### Basic Model

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal

class ProductReview(BaseModel):
    product_name: str
    rating: int = Field(ge=1, le=5, description="Rating from 1 to 5 stars")
    review_text: str
    sentiment: Literal["positive", "negative", "neutral"]
    verified_purchase: bool
    reviewer_age_group: Optional[str] = None  # May not always be present
```

### Field Metadata

```python
from pydantic import BaseModel, Field

class MeetingSummary(BaseModel):
    title: str = Field(description="Short title for the meeting")
    date: str  = Field(description="ISO-8601 date, e.g. 2024-11-15")
    participants: list[str] = Field(
        description="Full names of all attendees",
        min_length=1         # At least one participant
    )
    summary: str = Field(
        description="3-5 sentence summary of what was discussed",
        max_length=1000       # Limit summary length
    )
    action_items: list[str] = Field(
        default_factory=list,
        description="Specific tasks assigned, with owners if mentioned"
    )
    next_steps: Optional[str] = None
```

---

## 3.3 Nested Models

Complex schemas use nested Pydantic models:

```python
from pydantic import BaseModel, Field
from typing import Optional

class Address(BaseModel):
    street: str
    city: str
    state: str
    country: str
    zip_code: Optional[str] = None

class ContactInfo(BaseModel):
    email: str
    phone: Optional[str] = None
    preferred_contact: Literal["email", "phone", "either"] = "email"

class CompanyProfile(BaseModel):
    name: str
    industry: str
    founded_year: int = Field(ge=1800, le=2030)
    employee_count: int = Field(ge=1)
    headquarters: Address               # Nested model
    contact: ContactInfo                # Another nested model
    products: list[str] = Field(default_factory=list)
    is_public: bool = False
    stock_ticker: Optional[str] = None  # Only if public company
```

---

## 3.4 Converting Pydantic Models to JSON Schema

You can inspect the JSON Schema Pydantic generates:

```python
import json
from pydantic import BaseModel, Field
from typing import Literal

class SentimentAnalysis(BaseModel):
    text_analyzed: str
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_phrases: list[str]
    language: str = "en"

# Get the JSON Schema
schema = SentimentAnalysis.model_json_schema()
print(json.dumps(schema, indent=2))
# OpenAI uses this schema automatically when you pass the class to response_format
```

---

## 3.5 Using Pydantic with OpenAI `.parse()`

The `client.beta.chat.completions.parse()` method accepts a Pydantic model directly:

```python
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal

client = OpenAI()

class JobPosting(BaseModel):
    title: str
    company: str
    location: str
    is_remote: bool
    salary_min: float | None = None
    salary_max: float | None = None
    required_skills: list[str]
    experience_years: int = Field(ge=0, description="Minimum years of experience required")
    job_type: Literal["full-time", "part-time", "contract", "internship"]

job_post_text = """
Senior Machine Learning Engineer at TechCorp
Location: San Francisco, CA (Remote OK)
Salary: $150,000 - $200,000/year
We're looking for an ML Engineer with 5+ years of experience in Python,
TensorFlow, and PyTorch. Full-time position. Required: deep learning expertise.
"""

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a job posting parser."},
        {"role": "user",   "content": job_post_text}
    ],
    response_format=JobPosting
)

job = response.choices[0].message.parsed  # Returns a JobPosting instance
print(f"Title: {job.title}")
print(f"Company: {job.company}")
print(f"Remote: {job.is_remote}")
print(f"Salary: ${job.salary_min:,.0f} - ${job.salary_max:,.0f}")
print(f"Skills: {job.required_skills}")
print(f"Experience: {job.experience_years}+ years")
```

---

## 3.6 Model Validation — Pydantic Auto-Validates

Pydantic validates data automatically when parsing:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class ProductRating(BaseModel):
    product_id: str
    rating: float = Field(ge=1.0, le=5.0)
    review: str = Field(min_length=10, max_length=500)
    category: Literal["electronics", "clothing", "food", "books", "other"]
    
    @field_validator("product_id")
    @classmethod
    def validate_product_id_format(cls, v: str) -> str:
        """Ensure product ID starts with P- prefix."""
        if not v.startswith("P-"):
            v = f"P-{v}"  # Auto-correct the format
        return v.upper()

# Manually test validation
try:
    valid_rating = ProductRating(
        product_id="a1234",          # Will be corrected to "P-A1234"
        rating=4.5,
        review="Great product, very happy with it!",
        category="electronics"
    )
    print(f"Valid: {valid_rating.product_id}, rating={valid_rating.rating}")
except Exception as e:
    print(f"Validation error: {e}")

# Test with invalid data
try:
    bad_rating = ProductRating(
        product_id="xyz",
        rating=6.0,                  # ❌ Exceeds max of 5.0
        review="ok",                 # ❌ Too short (min 10 chars)
        category="toys"              # ❌ Not in Literal options
    )
except Exception as e:
    print(f"Expected error: {type(e).__name__}")
    for err in e.errors():
        print(f"  Field={err['loc'][0]}: {err['msg']}")
```

---

## 3.7 Optional Fields and Default Values

```python
from pydantic import BaseModel, Field
from typing import Optional

class NewsArticle(BaseModel):
    headline: str
    author: Optional[str] = None                    # May be unknown
    publication_date: Optional[str] = None          # May be missing
    category: str = "general"                       # Default category
    word_count: int = 0                             # Default 0
    tags: list[str] = Field(default_factory=list)   # Empty list default
    is_opinion: bool = False                        # Default to fact-based
    paywall: bool = False                           # Default to free
    
    # Computed check — does it have full metadata?
    @property
    def is_complete(self) -> bool:
        return self.author is not None and self.publication_date is not None
```

---

## 3.8 Using `model_dump()` — Convert Back to Dict

```python
from pydantic import BaseModel
from typing import Literal

class ExtractedEntity(BaseModel):
    entity_text: str
    entity_type: Literal["person", "organization", "location", "date", "money"]
    context: str
    confidence: float

# After LLM extraction, convert to dict for downstream processing
entity = ExtractedEntity(
    entity_text="Elon Musk",
    entity_type="person",
    context="CEO mentioned in investor call",
    confidence=0.95
)

# Convert to plain dict
as_dict = entity.model_dump()
print(as_dict)

# Convert to JSON string
as_json = entity.model_dump_json()
print(as_json)

# Exclude None fields
as_dict_no_none = entity.model_dump(exclude_none=True)
print(as_dict_no_none)
```

---

## 3.9 Lists of Pydantic Objects — Batch Extraction

```python
from pydantic import BaseModel
from typing import Literal

class ExtractedEntity(BaseModel):
    text: str
    type: Literal["person", "org", "location", "date", "product", "money"]
    start_index: int | None = None

class EntityList(BaseModel):
    """Wrapper for batch entity extraction — LLM returns a list of entities."""
    entities: list[ExtractedEntity]
    total_count: int
    extraction_confidence: float

# Use wrapper model for batch extractions
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract all named entities from the text."},
        {"role": "user",   "content": "Apple CEO Tim Cook met with Google's Sundar Pichai in San Francisco on January 15, 2024 to discuss a potential $500M partnership."}
    ],
    response_format=EntityList
)

result = response.choices[0].message.parsed
print(f"Found {result.total_count} entities (confidence: {result.extraction_confidence:.0%}):")
for e in result.entities:
    print(f"  [{e.type:10}] {e.text}")
```

---

## 📌 Key Takeaways

1. **Pydantic BaseModel is the standard** — define schema in Python, get JSON Schema for free
2. **Field()** adds constraints: `ge`, `le`, `min_length`, `max_length`, `description`
3. **Literal** creates enum-like fields — LLM must pick from fixed options
4. **Nested models** for complex hierarchical data — compose freely
5. **`Optional[T] = None`** for fields that may not be present
6. **`field_validator`** adds custom validation and auto-correction logic
7. **`.parse()`** returns a fully typed Pydantic object — access fields with dot notation
8. **`.model_dump()`** converts back to dict; `.model_dump_json()` to JSON string
