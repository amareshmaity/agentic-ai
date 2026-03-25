# Advanced Prompting Techniques

> *These techniques separate basic LLM usage from production-grade prompt engineering — essential for reliable, consistent agent behavior.*

---

## 1️⃣ Few-Shot Prompting

Give the LLM **examples** of the input-output behavior you want. The most reliable way to get consistent output format.

### Why Few-Shot?

```
Zero-shot:  "Classify this review as positive or negative: 'Great product!'"
            → LLM might return: "Positive", "positive", "POSITIVE", "This is positive", etc.

Few-shot:   "Here are examples:
             Review: 'Love it!' → Sentiment: positive
             Review: 'Terrible' → Sentiment: negative
             Review: 'Great product!' → Sentiment:"
            → LLM reliably returns: "positive"
```

### Basic Few-Shot with ChatPromptTemplate

```python
from langchain_core.prompts import ChatPromptTemplate

# Embed examples directly in system prompt
few_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a sentiment classifier. 
Classify the sentiment of reviews as 'positive', 'negative', or 'neutral'.

Examples:
Review: "This product is amazing, I love it!"
Sentiment: positive

Review: "Terrible quality, broke after one day."
Sentiment: negative

Review: "It's okay, nothing special."
Sentiment: neutral

Respond with only one word: positive, negative, or neutral."""),
    ("human", "Review: {review}\nSentiment:")
])

chain = few_shot_prompt | llm | StrOutputParser()
print(chain.invoke({"review": "Best purchase I've made this year!"}))
# "positive"
```

### FewShotChatMessagePromptTemplate

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate
)

# Define the example format
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai",    "{output}")
])

# Define examples
examples = [
    {"input": "2 + 2",       "output": "4"},
    {"input": "10 * 5",      "output": "50"},
    {"input": "100 / 4",     "output": "25"},
    {"input": "7 squared",   "output": "49"},
]

# Build few-shot prompt
few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Wrap in full chat prompt
full_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math calculator. Only output the answer, nothing else."),
    few_shot,   # ← injects examples here
    ("human", "{question}")
])

chain = full_prompt | llm | StrOutputParser()
print(chain.invoke({"question": "8 * 9"}))  # "72"
```

---

## 2️⃣ System Prompt Design

The system prompt is your most powerful tool. Well-designed system prompts = consistent, reliable behavior.

### The 5 Elements of a Great System Prompt

```
1. ROLE:       "You are a senior Python developer with 10 years experience."
2. CONTEXT:    "You are helping a team build production LangChain applications."
3. BEHAVIOR:   "Always explain WHY before HOW. Use bullet points for lists."
4. CONSTRAINTS:"Never write code without docstrings. Always handle edge cases."
5. FORMAT:     "Structure: Overview → Code → Explanation → Gotchas"
```

### System Prompt Examples

```python
# Agent system prompt
agent_system = """You are an AI research assistant.

Your capabilities:
- Search the web for current information
- Analyze documents and extract key insights
- Synthesize information from multiple sources
- Generate well-structured reports

Your rules:
- Always cite your sources
- If unsure, say so — never fabricate information
- Ask clarifying questions if the request is ambiguous
- Keep responses focused and actionable"""

# Coding assistant
code_system = """You are a senior Python developer.

When writing code:
- Always include type hints
- Add docstrings to all functions
- Handle edge cases explicitly
- Prefer readability over cleverness
- Follow PEP 8

When reviewing code:
- Point out bugs first, then style issues
- Suggest specific improvements with examples
- Explain WHY something is wrong"""
```

---

## 3️⃣ Prompt Composition — Building Prompts from Parts

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Build reusable prompt components
tone_instruction = PromptTemplate.from_template(
    "Respond in a {tone} tone. Keep it {length}."
)

format_instruction = PromptTemplate.from_template(
    "Format your response as: {format}"
)

# Combine into a full prompt
def build_agent_prompt(tone="professional", length="concise", format="bullet points"):
    tone_str   = tone_instruction.format(tone=tone, length=length)
    format_str = format_instruction.format(format=format)

    return ChatPromptTemplate.from_messages([
        ("system", f"You are a helpful assistant.\n{tone_str}\n{format_str}"),
        ("human",  "{question}")
    ])

# Use with different configurations
casual_prompt      = build_agent_prompt(tone="casual",       format="paragraph")
technical_prompt   = build_agent_prompt(tone="technical",    format="numbered list")
executive_prompt   = build_agent_prompt(tone="executive",    format="3 bullet points max")
```

---

## 4️⃣ Dynamic Prompts with Runtime Values

```python
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate

def get_current_date():
    return datetime.now().strftime("%B %d, %Y")

def get_user_context(user_id: str) -> str:
    # In real code: fetch from DB
    return f"Premium user, joined 2023, expertise: Python"

# Build prompt dynamically at runtime
def create_personalized_prompt(user_id: str):
    user_context = get_user_context(user_id)
    current_date = get_current_date()

    return ChatPromptTemplate.from_messages([
        ("system", f"""You are a personalized AI assistant.
Current Date: {current_date}
User Profile: {user_context}
Tailor your responses to this user's expertise level."""),
        ("human", "{question}")
    ])

# Usage
prompt = create_personalized_prompt("user_123")
chain  = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "What's new in Python 3.12?"})
```

---

## 5️⃣ Prompt Chaining — Chains of Prompts

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step 1: Generate outline
outline_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a content planner. Create a brief 3-point outline."),
    ("human",  "Create an outline for a blog post about: {topic}")
])

# Step 2: Write from outline
write_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a skilled writer. Write a detailed blog post."),
    ("human",  "Write a blog post based on this outline:\n{outline}")
])

# Chain the two prompts together
outline_chain = outline_prompt | llm | StrOutputParser()
write_chain   = write_prompt   | llm | StrOutputParser()

# Full pipeline: topic → outline → blog post
full_chain = (
    outline_chain                              # Generate outline
    | (lambda outline: {"outline": outline})   # Adapt output format
    | write_chain                              # Write full post
)

result = full_chain.invoke({"topic": "LangChain for beginners"})
print(result)
```

---

## 6️⃣ Chain-of-Thought (CoT) Prompting

Guide the LLM to reason step-by-step before giving an answer.

```python
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", """Solve problems step by step.
Always:
1. Understand the problem
2. Break it into sub-problems
3. Solve each sub-problem
4. Combine for the final answer
5. Verify your answer makes sense"""),
    ("human", "{problem}")
])

chain = cot_prompt | llm | StrOutputParser()

result = chain.invoke({"problem": """
A store sells apples for $0.50 each and oranges for $0.75 each.
If someone buys 12 apples and some oranges and pays $9.00 total,
how many oranges did they buy?
"""})
print(result)
```

---

## 7️⃣ Output Format Control

```python
# JSON output instruction
json_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data extractor.
Always respond with valid JSON only — no explanation, no markdown, just JSON.
Schema: {{"name": string, "age": number, "skills": [string]}}"""),
    ("human", "Extract from: {text}")
])

# Structured list output
list_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a content analyst.
Always respond with exactly {n} bullet points.
Each bullet must be a complete sentence.
No introduction or conclusion."""),
    ("human", "{request}")
])

chain = list_prompt | llm | StrOutputParser()
result = chain.invoke({"n": 3, "request": "List benefits of LangChain"})
```

---

## 📋 Prompt Engineering Checklist

Before shipping any prompt, verify:

```
✅ Role/persona clearly defined
✅ Task/goal explicitly stated
✅ Output format specified (JSON, bullet points, etc.)
✅ Constraints listed (length, style, what NOT to do)
✅ Few-shot examples provided for format-sensitive outputs
✅ Edge cases handled ("if you don't know, say...")
✅ Language/tone specified
✅ Prompt tested with 10+ varied inputs
✅ LangSmith tracing enabled to catch failures
```

---

## ✅ Key Takeaways

- **Few-shot prompting** is the most reliable way to control output format
- **System prompts** should include: role, context, behavior, constraints, format
- **Compose prompts** from reusable components — don't repeat yourself
- **Chain-of-Thought** prompting dramatically improves reasoning quality
- **Always specify output format** in the system prompt for structured outputs
- **Test with diverse inputs** — prompts that work for one case often fail on edge cases

---

## ⬅️ Previous
[ChatPromptTemplate](./04_chat_prompt_template.md)

## ➡️ Next
[Hands-on Examples →](./examples.ipynb)
