# ChatPromptTemplate

> *ChatPromptTemplate is the standard way to structure input for ChatModels. It maps directly to the system/human/ai message format that modern LLMs expect.*

---

## 🤔 What is a ChatPromptTemplate?

A `ChatPromptTemplate` is a template for **multiple typed messages**, not just a single string. This maps directly to how ChatModels actually work: they receive a list of messages with roles.

```
ChatPromptTemplate:
    [
        ("system", "You are a {role}."),    ← SystemMessage
        ("human",  "Explain {concept}.")    ← HumanMessage
    ]
              ↓ filled with variables
    [
        SystemMessage(content="You are a Python tutor."),
        HumanMessage(content="Explain decorators.")
    ]
              ↓ sent to ChatModel
    AIMessage(content="A decorator is...")
```

---

## 📦 Basic Usage

```python
from langchain_core.prompts import ChatPromptTemplate

# Method 1: from_messages (most common)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that speaks {language}."),
    ("human",  "{question}")
])

# Format → returns list of messages
messages = prompt.format_messages(
    language="English",
    question="What is LangChain?"
)
print(messages)
# [SystemMessage(content="You are a helpful assistant that speaks English."),
#  HumanMessage(content="What is LangChain?")]

# Invoke → returns ChatPromptValue (Runnable)
result = prompt.invoke({"language": "English", "question": "What is LangChain?"})
print(result.to_messages())  # Same as format_messages
```

---

## 💬 Message Role Options

```python
# All valid role identifiers in from_messages:
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),   # SystemMessage
    ("human",  "{user_input}"),        # HumanMessage  ← also: "user"
    ("ai",     "{ai_response}"),       # AIMessage      ← also: "assistant"
])

# Using typed message classes directly (same result)
from langchain_core.messages import SystemMessage, HumanMessage

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a Python expert."),
    HumanMessage(content="What is {concept}?")
])
```

---

## 🔗 ChatPromptTemplate in a Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Keep answers concise."),
    ("human",  "{question}")
])
llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({
    "domain": "machine learning",
    "question": "What is overfitting?"
})
print(result)
```

---

## 💬 Multi-Turn Conversations

### Static Multi-Turn (Fixed History)

```python
# Hard-code a conversation history in the template
prompt = ChatPromptTemplate.from_messages([
    ("system",  "You are a helpful assistant."),
    ("human",   "My name is Alice."),
    ("ai",      "Nice to meet you, Alice!"),
    ("human",   "What is my name?"),
])

chain = prompt | llm | StrOutputParser()
result = chain.invoke({})   # No variables needed
print(result)  # "Your name is Alice."
```

### Dynamic Multi-Turn (MessagesPlaceholder)

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# MessagesPlaceholder = a slot to inject a dynamic list of messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # ← Dynamic messages
    ("human", "{question}")
])

# Build and use with history
from langchain_core.messages import HumanMessage, AIMessage

chat_history = [
    HumanMessage(content="My favorite color is blue."),
    AIMessage(content="That's a calming color!"),
    HumanMessage(content="I also love Python programming."),
    AIMessage(content="Python is great for data science and AI!"),
]

chain = prompt | llm | StrOutputParser()
result = chain.invoke({
    "chat_history": chat_history,
    "question": "What are my two interests?"
})
print(result)
# "Based on our conversation, your two interests are blue (colors)
#  and Python programming."
```

---

## 🔧 Advanced Features

### Partial Variables

```python
# Pre-fill some variables at template creation time
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} who responds in {language}."),
    ("human",  "{question}")
])

# Fix role and language now
python_tutor_prompt = prompt.partial(
    role="Python tutor",
    language="English"
)

# Chain only needs {question} now
chain = python_tutor_prompt | llm | StrOutputParser()
result = chain.invoke({"question": "What is a decorator?"})
```

### Dynamic System Prompt with a Function

```python
from datetime import datetime

def get_system_prompt(topic: str) -> str:
    return f"You are an expert in {topic}. Today is {datetime.now().strftime('%B %d, %Y')}."

# Use a callable as a message content
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_message}"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()
result = chain.invoke({
    "system_message": get_system_prompt("LangChain"),
    "question": "What are the latest features?"
})
```

### from_template (Simple Alternative)

```python
# Quick one-liner for simple prompts
prompt = ChatPromptTemplate.from_template("What is {topic}?")
# Creates: [HumanMessage(content="What is {topic}?")]
# No system message — use from_messages for anything real
```

---

## 🔄 Template Inspection

```python
# Inspect a ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human",  "{question}")
])

print(prompt.input_variables)   # ['role', 'question']
print(prompt.messages)          # List of message templates
print(prompt.input_schema)      # Pydantic schema

# Check output schema
print(prompt.output_schema)
```

---

## 🗂️ Practical Patterns

### Pattern 1: Persona Agent

```python
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are {name}, a {role}.
Your expertise: {expertise}
Your communication style: {style}
Always stay in character."""),
    MessagesPlaceholder("history"),
    ("human", "{user_input}")
])
```

### Pattern 2: RAG Prompt

```python
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer the user's question using ONLY the context below.
If the answer isn't in the context, say 'I don't have that information.'

Context:
{context}"""),
    ("human", "{question}")
])
```

### Pattern 3: Structured Output Prompt

```python
structured_prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract information and return as valid JSON.
{format_instructions}"""),
    ("human", "Extract from this text: {text}")
])
```

---

## ✅ Key Takeaways

- `ChatPromptTemplate.from_messages([...])` is the standard way to build prompts for ChatModels
- Use `("system", ...)`, `("human", ...)`, `("ai", ...)` tuple syntax for role-based messages
- `MessagesPlaceholder` injects a **dynamic list** of messages — essential for conversation history
- Use `.partial()` to pre-fill some variables (e.g., language, role)
- Supports full Runnable interface: `.invoke()`, `.stream()`, `.batch()`

---

## ⬅️ Previous
[PromptTemplate](./03_prompt_template.md)

## ➡️ Next
[Advanced Prompting Techniques →](./05_advanced_prompting.md)
