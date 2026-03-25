# JsonOutputParser & JSON Mode

> *JsonOutputParser converts LLM output into Python dicts. JSON mode forces the LLM to always return valid JSON. Together they give you flexible structured output.*

---

## 🤔 What is JsonOutputParser?

`JsonOutputParser` parses the LLM's text output as JSON and returns a Python `dict`.

```
LLM outputs:  '{"name": "Alice", "age": 30, "role": "engineer"}'
                              ↓ JsonOutputParser
Chain returns: {"name": "Alice", "age": 30, "role": "engineer"}  ← Python dict
```

---

## 📦 Basic Usage

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data extractor. Always respond with valid JSON only."),
    ("human",  "Extract the person's name, age, and job from: {text}")
])

chain = prompt | llm | parser

result = chain.invoke({
    "text": "John Smith is a 35-year-old software engineer at Google."
})

print(type(result))        # <class 'dict'>
print(result)              # {'name': 'John Smith', 'age': 35, 'job': 'software engineer'}
print(result["name"])      # "John Smith"
print(result["age"])       # 35  (BUT: this is whatever the LLM decides — no type guarantee)
```

---

## ⚠️ The Problem with JsonOutputParser

JsonOutputParser parses JSON but **does not validate** the schema:

```python
# What you expect:
{"name": "John", "age": 35, "job": "engineer"}

# What you might get:
{"full_name": "John Smith", "years": "35", "occupation": "software engineer at Google"}
# ^ Wrong key names!  ^ Age as string!  ^ Job has extra text!

result["age"] + 1    →  TypeError: can only concatenate str (not "int") to str
result["name"]       →  KeyError: 'name'
```

**For guaranteed schema consistency, use `PydanticOutputParser` instead.**

---

## 📋 JsonOutputParser with Schema Hint

You can add a schema hint in the prompt to guide the LLM, but it's not enforced:

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Define the JSON structure you want (as a hint)
schema_hint = """
{
  "name": "string — person's full name",
  "age": "integer — person's age as a number",
  "job": "string — person's job title only",
  "company": "string — company name or null if not mentioned"
}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""Extract information as JSON matching this schema exactly:
{schema_hint}
Return ONLY the JSON object, no explanation."""),
    ("human", "{text}")
])

parser = JsonOutputParser()
chain = prompt | llm | parser

result = chain.invoke({"text": "Sarah Chen, 28, works as ML engineer at OpenAI."})
print(result)
# {'name': 'Sarah Chen', 'age': 28, 'job': 'ML engineer', 'company': 'OpenAI'}
```

---

## 🔵 JSON Mode — Force Valid JSON Output

**JSON mode** is an LLM feature (OpenAI, Google) that guarantees the model **always** outputs valid JSON — no markdown, no explanation, just JSON.

```python
from langchain_openai import ChatOpenAI

# Enable JSON mode (OpenAI)
llm_json = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}  # ← JSON mode
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data extractor. Always return valid JSON."),
    ("human",  "Extract name, skill, and years from: {text}")
])

parser = JsonOutputParser()
chain = prompt | llm_json | parser

result = chain.invoke({
    "text": "Maria has been coding Python for 7 years."
})
print(result)   # Always valid JSON — {'name': 'Maria', 'skill': 'Python', 'years': 7}
```

> ⚠️ **JSON mode requirement**: Your prompt MUST mention JSON or the model will throw an error.

---

## 🌊 Streaming JSON Output

`JsonOutputParser` supports streaming — partial JSON is accumulated and yielded as it becomes valid:

```python
from langchain_core.output_parsers import JsonOutputParser

chain = prompt | llm | JsonOutputParser()

# Stream partial JSON as it arrives
print("Streaming JSON:")
for partial_result in chain.stream({"text": "Alice, 30, data scientist"}):
    print(partial_result)  # Shows dict building up incrementally

# Output (example):
# {}
# {'name': 'Alice'}
# {'name': 'Alice', 'age': 30}
# {'name': 'Alice', 'age': 30, 'role': 'data scientist'}
```

---

## 🔄 JsonOutputParser vs PydanticOutputParser

| Feature | JsonOutputParser | PydanticOutputParser |
|---|---|---|
| **Output type** | `dict` | Pydantic model instance |
| **Schema enforcement** | ❌ Hint only | ✅ Strict |
| **Type validation** | ❌ None | ✅ Full (int, str, List, etc.) |
| **Field validation** | ❌ None | ✅ Custom validators |
| **IDE autocomplete** | ❌ None | ✅ Full |
| **Nested objects** | Manual | ✅ Automatic |
| **Error handling** | parse error only | ✅ Pydantic ValidationError |
| **Best for** | Quick/flexible JSON | Production, typed data |

---

## 📊 When to Use JsonOutputParser

✅ **Use JsonOutputParser when:**
- You need a `dict` output but don't know the exact schema in advance
- You're doing quick prototyping
- The schema varies per request
- You need streaming JSON

❌ **Prefer PydanticOutputParser when:**
- You have a fixed schema you always need
- Type safety matters (correct int, correct list, etc.)
- You want IDE autocomplete and validation
- Building production agents

---

## ✅ Key Takeaways

- `JsonOutputParser` → `dict` from LLM text, no type validation
- **JSON mode** (`response_format: json_object`) forces the LLM to always output valid JSON
- JsonOutputParser supports streaming — partial JSON builds up incrementally
- For type safety and schema enforcement, use `PydanticOutputParser` or `.with_structured_output()`
- Always hint the desired JSON structure in your system prompt

---

## ➡️ Next
[PydanticOutputParser →](./04_pydantic_output_parser.md)
