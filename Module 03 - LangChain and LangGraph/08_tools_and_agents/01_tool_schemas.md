# Tool Schemas

> *A tool is just a function with a schema. The schema is what the LLM reads to understand what the tool does, what arguments it takes, and when to use it.*

---

## 🤔 What is a Tool?

A **tool** is a callable function that an LLM can decide to invoke during its reasoning loop.

```
Without Tools:
  User: "What's the weather in London?"
  LLM:  "I don't have real-time data." ❌  (or it hallucinates ❌)

With Tools:
  User:  "What's the weather in London?"
  LLM:   decides → call get_weather(city="London")
  Tool:  returns {"temp": 18, "condition": "cloudy"}
  LLM:   "It's 18°C and cloudy in London right now." ✅
```

---

## 📐 Anatomy of a Tool Schema

Every tool exposes three things to the LLM:

```
name        → what to call it   e.g. "search_web"
description → when to use it   e.g. "Search the internet for current events"
arguments   → input parameters  e.g. {"query": "string", "max_results": "integer"}
```

The LLM reads the schema and decides:
1. **Should I call a tool?** (based on the user question)
2. **Which tool?** (based on the description)
3. **What arguments?** (based on the argument schema)

---

## 🏗️ Three Ways to Define Tools in LangChain

### Method 1 — `@tool` Decorator (Most Common)

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: The name of the city to get weather for.
    
    Returns:
        A string describing current weather conditions.
    """
    # In production, call a real weather API
    return f"Weather in {city}: 22°C, sunny"

# Inspect the auto-generated schema
print(get_weather.name)          # "get_weather"
print(get_weather.description)   # "Get the current weather for a city. ..."
print(get_weather.args)          # {"city": {"title": "City", "type": "string"}}
```

> **How it works**: LangChain reads your function's **type hints** and **docstring** to auto-generate the full JSON schema.

---

### Method 2 — Pydantic Schema (Precise Control)

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query to look up")
    max_results: int = Field(default=5, description="Number of results to return (1-20)")

@tool("search_web", args_schema=SearchInput)
def search_web(query: str, max_results: int = 5) -> str:
    """Search the internet for current information on any topic."""
    # Call search API here
    return f"Search results for '{query}': [result1, result2...]"

print(search_web.args)
# {
#   "query": {"title": "Query", "description": "The search query...", "type": "string"},
#   "max_results": {"title": "Max Results", "description": "Number...", "default": 5, "type": "integer"}
# }
```

---

### Method 3 — `StructuredTool` (From Existing Functions)

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

def calculate_bmi(weight_kg: float, height_m: float) -> str:
    bmi = weight_kg / (height_m ** 2)
    category = "underweight" if bmi < 18.5 else "normal" if bmi < 25 else "overweight"
    return f"BMI: {bmi:.1f} ({category})"

class BMIInput(BaseModel):
    weight_kg: float = Field(description="Weight in kilograms")
    height_m: float = Field(description="Height in meters")

bmi_tool = StructuredTool.from_function(
    func=calculate_bmi,
    name="calculate_bmi",
    description="Calculate BMI from weight and height",
    args_schema=BMIInput,
    return_direct=False  # True = return tool output directly without LLM post-processing
)
```

---

## 📋 What the LLM Actually Sees

When you bind tools to an LLM, the schema is serialized to JSON and sent as part of the API call:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "title": "City",
          "description": "The name of the city to get weather for.",
          "type": "string"
        }
      },
      "required": ["city"]
    }
  }
}
```

This is the **OpenAI function calling format** — LangChain handles the serialization automatically.

---

## 🔍 Inspecting Your Tools

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b

# Explore the tool object
print(multiply.name)            # "multiply"
print(multiply.description)     # "Multiply two integers together."
print(multiply.args)            # {"a": {"type": "integer"}, "b": {"type": "integer"}}
print(multiply.return_direct)   # False

# Run the tool directly
print(multiply.invoke({"a": 3, "b": 4}))   # 12
print(multiply.run("{'a': 3, 'b': 4}"))    # 12 (string input also works)
```

---

## ✍️ Writing Good Tool Descriptions

The description is the most important part — it's what the LLM uses to decide when to call the tool.

```python
# ❌ BAD — too vague
@tool
def search(q: str) -> str:
    """Search."""
    ...

# ✅ GOOD — tells the LLM exactly when and why to use this tool
@tool
def search_web(query: str) -> str:
    """Search the internet for current, real-time information.
    
    Use this tool when:
    - The user asks about recent events, news, or live data
    - You need information more recent than your training cutoff
    - You need specific facts, prices, or statistics
    
    Do NOT use this for general knowledge questions you already know.
    
    Args:
        query: A specific, focused search query (not a full sentence question)
    """
    ...
```

---

## 🔢 Supported Argument Types

| Python Type | JSON Schema | Example |
|---|---|---|
| `str` | `"string"` | names, queries, text |
| `int` | `"integer"` | counts, IDs |
| `float` | `"number"` | prices, measurements |
| `bool` | `"boolean"` | flags, toggles |
| `list[str]` | `"array"` | multiple values |
| `Optional[str]` | `"string"` + nullable | optional params |
| Pydantic model | `"object"` | nested structure |

---

## ✅ Key Takeaways

- A tool = function + name + description + argument schema
- The LLM reads the schema (not your code) to decide which tools to call
- Use `@tool` decorator for simplest definition — reads type hints + docstring
- Use Pydantic `args_schema` for precise field descriptions and validation
- **The description is the most critical part** — be specific about when and how to use the tool
- LangChain auto-converts your schema to OpenAI function calling format

---

## ➡️ Next
[Tool Calling →](./02_tool_calling.md)
