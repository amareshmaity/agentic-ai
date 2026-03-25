# PydanticOutputParser

> *PydanticOutputParser is the most powerful standard parser — it enforces a typed, validated schema using Pydantic models. Your chain always returns the exact object type you defined.*

---

## 🤔 What is PydanticOutputParser?

`PydanticOutputParser` takes a Pydantic model class and:
1. Generates **format instructions** telling the LLM exactly what JSON to return
2. **Parses** the LLM's JSON output into your Pydantic model
3. **Validates** all field types and constraints

```
LLM outputs: '{"name": "Alice", "age": 30, "skills": ["Python", "ML"]}'
                              ↓ PydanticOutputParser(pydantic_object=Person)
Chain returns: Person(name="Alice", age=30, skills=["Python", "ML"])
               ← Typed Pydantic object, always correct types, has validation
```

---

## 📦 Basic Usage

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

# 1. Define your schema
class PersonInfo(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age as an integer")
    occupation: str = Field(description="Person's job title")
    skills: List[str] = Field(description="List of key technical skills")

# 2. Create the parser
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# 3. Add format instructions to the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract person information.\n{format_instructions}"),
    ("human",  "{text}")
]).partial(format_instructions=parser.get_format_instructions())

# 4. Build and run chain
llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | llm | parser

result = chain.invoke({
    "text": "Alice Smith is a 30-year-old ML engineer who knows Python, TensorFlow, and LangChain."
})

# 5. Use typed output
print(type(result))          # <class 'PersonInfo'>
print(result.name)           # "Alice Smith"
print(result.age)            # 30  — guaranteed int
print(result.occupation)     # "ML engineer"
print(result.skills)         # ['Python', 'TensorFlow', 'LangChain'] — guaranteed list
print(result.age + 5)        # 35  — works! Math on guaranteed int
```

---

## 🔍 Format Instructions — What the LLM Sees

```python
print(parser.get_format_instructions())
```
```
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema.
The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
{"properties": {"name": {"title": "Name", "description": "Person's full name", "type": "string"},
"age": {"title": "Age", "description": "Person's age as an integer", "type": "integer"},
"occupation": {"title": "Occupation", "description": "Person's job title", "type": "string"},
"skills": {"title": "Skills", "description": "List of key technical skills", "type": "array", "items": {"type": "string"}}},
"required": ["name", "age", "occupation", "skills"]}
```

---

## 🏗️ Advanced Pydantic Schemas

### Nested Objects

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")

class Company(BaseModel):
    name: str = Field(description="Company name")
    founding_year: int = Field(description="Year the company was founded")
    headquarters: Address = Field(description="Company headquarters address")
    products: List[str] = Field(description="Main products or services")
    ceo: Optional[str] = Field(default=None, description="CEO name if mentioned")

parser = PydanticOutputParser(pydantic_object=Company)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract company information.\n{format_instructions}"),
    ("human",  "{text}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | parser

result = chain.invoke({"text": """
    Apple Inc. was founded in 1976. It is headquartered at One Apple Park Way,
    Cupertino, California, USA. The company makes iPhones, MacBooks, and iPads.
    Tim Cook is the CEO.
"""})

print(result.name)                      # "Apple Inc."
print(result.founding_year)             # 1976
print(result.headquarters.city)        # "Cupertino"
print(result.headquarters.country)     # "USA"
print(result.products)                  # ['iPhones', 'MacBooks', 'iPads']
print(result.ceo)                       # "Tim Cook"
```

### Field Validators

```python
from pydantic import BaseModel, Field, field_validator

class ProductReview(BaseModel):
    product_name: str = Field(description="Name of the product")
    rating: int = Field(description="Rating from 1 to 10")
    sentiment: str = Field(description="One of: positive, negative, neutral")
    summary: str = Field(description="Brief review summary under 100 words")

    @field_validator("rating")
    @classmethod
    def rating_in_range(cls, v):
        if not 1 <= v <= 10:
            raise ValueError("Rating must be between 1 and 10")
        return v

    @field_validator("sentiment")
    @classmethod
    def sentiment_valid(cls, v):
        v = v.lower()
        if v not in ["positive", "negative", "neutral"]:
            raise ValueError("Sentiment must be positive, negative, or neutral")
        return v
```

### Optional Fields and Defaults

```python
from typing import Optional, List
from pydantic import BaseModel, Field

class JobPosting(BaseModel):
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    salary_min: Optional[int] = Field(default=None, description="Minimum salary, null if not mentioned")
    salary_max: Optional[int] = Field(default=None, description="Maximum salary, null if not mentioned")
    required_skills: List[str] = Field(description="Required technical skills")
    remote: bool = Field(description="True if remote work is allowed")
    years_experience: int = Field(default=0, description="Years of experience required, 0 if not specified")
```

---

## ⚠️ Error Handling

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

try:
    result = chain.invoke({"text": "Some text..."})
except OutputParserException as e:
    print(f"Parser failed: {e}")
    # Options:
    # 1. Retry with higher temperature
    # 2. Use OutputFixingParser (auto-retry)
    # 3. Fall back to JsonOutputParser

# Auto-fix with OutputFixingParser
from langchain.output_parsers import OutputFixingParser

fixing_parser = OutputFixingParser.from_llm(
    parser=parser,
    llm=llm   # Uses LLM to fix malformed output
)
robust_chain = prompt | llm | fixing_parser
```

---

## 📊 PydanticOutputParser vs .with_structured_output()

| Feature | PydanticOutputParser | .with_structured_output() |
|---|---|---|
| **Setup** | Manual format instructions in prompt | Automatic |
| **Reliability** | Good | ✅ Better (uses native tool calling) |
| **Streaming** | ❌ Not supported | ✅ Supported |
| **Format instructions** | Must add to prompt | ✅ Automatic |
| **Works with all models** | ✅ Yes | Depends on model support |
| **Recommendation** | Fallback option | ✅ **Use this first** |

---

## ✅ Key Takeaways

- `PydanticOutputParser` + Pydantic model = **typed, validated** LLM output
- Add `{format_instructions}` to your prompt via `.partial(format_instructions=parser.get_format_instructions())`
- Supports **nested objects**, **optional fields**, **custom validators**
- Use `OutputFixingParser` for auto-retry on parse failures
- For modern code, prefer `.with_structured_output()` — it's simpler and more reliable

---

## ➡️ Next
[.with_structured_output() →](./05_with_structured_output.md)
