# `.with_structured_output()` — The Modern Approach

> *The cleanest, most reliable way to get structured output in LangChain. Uses native tool calling under the hood — no format instructions needed.*

---

## 🤔 What is `.with_structured_output()`?

`.with_structured_output()` is a method on ChatModels that:
1. Takes a Pydantic schema (or JSON schema)
2. Uses the **model's native tool calling** to enforce the schema
3. Returns a configured model that always outputs structured data

**No format instructions in the prompt. No parsing errors. Just works.**

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class PersonInfo(BaseModel):
    name: str
    age: int

# Standard model
llm = ChatOpenAI(model="gpt-4o-mini")

# Model configured for structured output
structured_llm = llm.with_structured_output(PersonInfo)

result = structured_llm.invoke("Alice is 30 years old.")
print(type(result))   # <class 'PersonInfo'>
print(result.name)    # "Alice"
print(result.age)     # 30
```

---

## 📦 Full Usage Pattern

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional

# 1. Define schema
class ProductInfo(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    category: str = Field(description="Product category")
    in_stock: bool = Field(description="Whether the product is in stock")
    tags: List[str] = Field(description="Product tags")

# 2. Create structured LLM — no parser needed in chain!
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(ProductInfo)

# 3. Build chain — structured_llm already handles output
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract product information from the text."),
    ("human",  "{text}")
])

chain = prompt | structured_llm  # No parser at end!

# 4. Use it
result = chain.invoke({
    "text": "The Sony WH-1000XM5 headphones cost $349.99. "
            "They are premium audio equipment, currently in stock. "
            "Tags: wireless, noise-cancelling, premium"
})

print(type(result))        # <class 'ProductInfo'>
print(result.name)         # "Sony WH-1000XM5"
print(result.price)        # 349.99  — correct float!
print(result.category)     # "premium audio equipment"
print(result.in_stock)     # True   — correct bool!
print(result.tags)         # ['wireless', 'noise-cancelling', 'premium']
```

---

## 🔄 How It Works Under the Hood

```
Traditional PydanticOutputParser:
    Prompt → (format instructions injected) → LLM generates text
           → Parser tries to extract JSON from text (can fail!)
           → Validate with Pydantic

.with_structured_output():
    Prompt → LLM (with schema as a "tool" definition)
           → Model is forced to call the "tool" with correct args
           → Always returns valid structured data (model-enforced!)
```

This is why `.with_structured_output()` is more reliable — the schema enforcement happens at the **model level**, not in post-processing.

---

## 🔧 with_structured_output Options

```python
# Option 1: Pydantic model (returns Pydantic instance)
structured_llm = llm.with_structured_output(PersonInfo)

# Option 2: JSON schema dict (returns dict)
json_schema = {
    "title": "PersonInfo",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}
structured_llm = llm.with_structured_output(json_schema)

# Option 3: TypedDict (returns dict with type hints)
from typing import TypedDict
class PersonDict(TypedDict):
    name: str
    age: int
structured_llm = llm.with_structured_output(PersonDict)

# Option 4: include_raw=True (get both raw and parsed)
structured_llm = llm.with_structured_output(PersonInfo, include_raw=True)
result = structured_llm.invoke("Alice is 30.")
print(result["raw"])     # AIMessage (raw output)
print(result["parsed"])  # PersonInfo instance
```

---

## 🌊 Streaming Structured Output

`.with_structured_output()` supports partial streaming:

```python
from pydantic import BaseModel
from typing import List

class Report(BaseModel):
    title: str
    key_findings: List[str]
    recommendation: str

structured_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(Report)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Create a brief analysis report."),
    ("human",  "Analyze the topic: {topic}")
])

chain = prompt | structured_llm

# Stream partial Report objects as they build up
for partial in chain.stream({"topic": "LangChain adoption in enterprise"}):
    print(partial)

# Output builds incrementally:
# title='LangChain' key_findings=[] recommendation=''
# title='LangChain Adoption' key_findings=['Growing rapidly'] recommendation=''
# title='LangChain Adoption in Enterprise' key_findings=['Growing rapidly', 'Strong ecosystem'] recommendation='Adopt for RAG'
```

---

## 🏗️ Real-World Extraction Patterns

### Information Extraction Pipeline

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class JobOffer(BaseModel):
    job_title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    location: str = Field(description="Job location or 'Remote'")
    salary_range: Optional[str] = Field(default=None, description="Salary range if mentioned")
    required_skills: List[str] = Field(description="Required technical skills")
    experience_years: int = Field(description="Years of experience required, default 0")
    remote_ok: bool = Field(description="Whether remote work is allowed")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(JobOffer)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract job offer details from job posting text."),
    ("human",  "{text}")
])

chain = prompt | structured_llm

job_posting = """
Senior ML Engineer at TechCorp (Remote OK)
Location: San Francisco / Remote
Salary: $150,000 - $200,000
Requirements:
- 5+ years Python experience
- Deep learning frameworks (PyTorch, TensorFlow)
- LangChain and LLM application development
- Strong MLOps experience
"""

result = chain.invoke({"text": job_posting})
print(f"Title: {result.job_title}")
print(f"Remote: {result.remote_ok}")
print(f"Skills: {result.required_skills}")
print(f"Experience: {result.experience_years} years")
```

### Classification with Enum

```python
from pydantic import BaseModel, Field
from enum import Enum

class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"

class Priority(str, Enum):
    urgent = "urgent"
    high = "high"
    medium = "medium"
    low = "low"

class SupportTicket(BaseModel):
    summary: str = Field(description="One-line summary of the issue")
    sentiment: Sentiment = Field(description="Customer sentiment")
    priority: Priority = Field(description="Ticket priority level")
    category: str = Field(description="Issue category (billing, technical, shipping, etc.)")
    requires_human: bool = Field(description="Whether this needs a human agent")

structured_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(SupportTicket)
```

---

## 📊 Choosing the Right Approach

| Scenario | Best Choice |
|---|---|
| Simple text output | `StrOutputParser` |
| Flexible JSON, unknown schema | `JsonOutputParser` |
| Fixed schema, need validation | `.with_structured_output()` ✅ |
| Fixed schema, model doesn't support tool calling | `PydanticOutputParser` |
| Need raw + parsed output | `.with_structured_output(include_raw=True)` |
| Production agents | `.with_structured_output()` ✅ |

---

## ✅ Key Takeaways

- `.with_structured_output(schema)` is the **modern preferred approach** — simpler and more reliable than parsers
- Uses **native tool calling** to enforce schema at the model level (not post-processing)
- Accepts Pydantic models, JSON schemas, or TypedDict
- Supports **streaming** of partial structured objects
- No need to add format instructions to the prompt
- Use `include_raw=True` when you also need the original `AIMessage`

---

## ⬅️ Previous
[PydanticOutputParser](./04_pydantic_output_parser.md)

## ➡️ Next
[Hands-on Examples →](./examples.ipynb)
