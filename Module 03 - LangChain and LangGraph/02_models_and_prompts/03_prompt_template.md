# PromptTemplate

> *PromptTemplate is LangChain's way of creating reusable, parameterized string prompts for legacy LLMs. Understanding it builds the foundation for the more powerful ChatPromptTemplate.*

---

## 🤔 What is a PromptTemplate?

A `PromptTemplate` is a string template with **named placeholders** that get filled at runtime.

```
Template: "Write a {tone} blog post about {topic} in {language}."
              ↑              ↑               ↑
           variable      variable         variable

Filled:   "Write a professional blog post about AI in English."
```

---

## 📦 Basic Usage

```python
from langchain_core.prompts import PromptTemplate

# Method 1: from string with explicit input_variables
template = PromptTemplate(
    input_variables=["topic", "tone"],
    template="Write a {tone} article about {topic}."
)

# Method 2: from_template (auto-detects variables)
template = PromptTemplate.from_template(
    "Write a {tone} article about {topic}."
)

# Format: fills in variables → returns string
result = template.format(topic="LangChain", tone="technical")
print(result)
# "Write a technical article about LangChain."

# Invoke: fills in variables → returns StringPromptValue
result = template.invoke({"topic": "LangChain", "tone": "technical"})
print(result.to_string())
# "Write a technical article about LangChain."
```

---

## 🔗 PromptTemplate in a Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = PromptTemplate.from_template(
    "You are a {role}. Explain {concept} in simple terms."
)
llm = ChatOpenAI(model="gpt-4o-mini")

chain = template | llm | StrOutputParser()
result = chain.invoke({"role": "teacher", "concept": "recursion"})
print(result)
```

---

## 🔧 PromptTemplate Features

### Partial Variables — Pre-fill Some Variables

```python
# Partially fill a template (fix some variables now, others later)
template = PromptTemplate(
    input_variables=["topic", "language"],
    template="Explain {topic} in {language}."
)

# Fix the language now
english_template = template.partial(language="English")

# Only need to provide {topic} later
result = english_template.format(topic="machine learning")
print(result)
# "Explain machine learning in English."
```

### Validation

```python
# PromptTemplate validates that all variables are provided
template = PromptTemplate.from_template("Hello {name}, you are {age} years old.")

# Raises error if variables are missing
try:
    template.format(name="Alice")   # Missing 'age'
except KeyError as e:
    print(f"Error: {e}")  # KeyError: 'age'
```

---

## 🤔 PromptTemplate vs ChatPromptTemplate

| Feature | PromptTemplate | ChatPromptTemplate |
|---|---|---|
| **Output** | Single string | List of messages |
| **Use with** | LLM (legacy) | ChatModel (modern) |
| **System prompt** | Must bake into string | Dedicated `SystemMessage` |
| **Multi-turn history** | Manual | `MessagesPlaceholder` |
| **Tool calling** | Not compatible | ✅ Compatible |
| **Recommendation** | Use for simple strings | ✅ Use for ChatModels |

**Rule of thumb**: If you're using a ChatModel (which you should always be), use `ChatPromptTemplate`.

---

## 📝 When to Actually Use PromptTemplate

Use `PromptTemplate` when you need a **plain string** for:
- Building a part of a larger prompt (sub-template)
- Formatting instructions that get inserted into a `ChatPromptTemplate`
- Legacy code or LLM integrations

```python
# Common pattern: use PromptTemplate for format instructions
# that get inserted into a ChatPromptTemplate
format_instructions = PromptTemplate.from_template(
    "Output your response as JSON with keys: {keys}"
).format(keys="name, age, city")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", f"You are a data extractor. {format_instructions}"),
    ("human", "{text}")
])
```

---

## ✅ Key Takeaways

- `PromptTemplate` = string template with `{variable}` placeholders
- Use `.from_template("...")` — it auto-detects variables
- Use `.partial()` to pre-fill some variables
- In modern LangChain, **prefer `ChatPromptTemplate`** for use with ChatModels
- `PromptTemplate` is still useful for building sub-components of larger prompts

---

## ➡️ Next
[ChatPromptTemplate →](./04_chat_prompt_template.md)
