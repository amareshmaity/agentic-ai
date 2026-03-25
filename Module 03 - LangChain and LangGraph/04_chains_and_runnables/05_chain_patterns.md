# Chain Patterns

> *Mastering these 5 patterns covers 95% of real-world LangChain use cases. Think of them as design patterns for LLM pipelines.*

---

## Pattern 1: Sequential Chain

The most basic pattern — one step flows into the next.

```
Input → Step A → Step B → Step C → Output
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step 1: Expand topic into a brief outline
outline_chain = (
    ChatPromptTemplate.from_template("Create a 3-point outline for: {topic}")
    | llm | StrOutputParser()
)

# Step 2: Write detailed content from outline
content_chain = (
    ChatPromptTemplate.from_template("Write detailed content for this outline:\n{outline}")
    | llm | StrOutputParser()
)

# Step 3: Proofread
proofread_chain = (
    ChatPromptTemplate.from_template("Proofread and fix any errors in:\n{text}")
    | llm | StrOutputParser()
)

# Chain them sequentially with data reshaping between steps
full_chain = (
    outline_chain
    | (lambda outline: {"outline": outline})
    | content_chain
    | (lambda content: {"text": content})
    | proofread_chain
)

result = full_chain.invoke({"topic": "LangChain for beginners"})
print(result)
```

---

## Pattern 2: Parallel Chain

Run multiple chains on the same input simultaneously, merge results.

```
         ┌─► Chain A (pros)    ─┐
Input ──►├─► Chain B (cons)    ─┼─► Combine → Output
         └─► Chain C (examples)─┘
```

```python
from langchain_core.runnables import RunnableParallel

analysis = RunnableParallel({
    "pros": (
        ChatPromptTemplate.from_template("List 3 pros of {topic}") | llm | StrOutputParser()
    ),
    "cons": (
        ChatPromptTemplate.from_template("List 3 cons of {topic}") | llm | StrOutputParser()
    ),
    "examples": (
        ChatPromptTemplate.from_template("Give 2 real-world examples of {topic}") | llm | StrOutputParser()
    ),
})

combine_prompt = ChatPromptTemplate.from_messages([
    ("system", "Create a balanced, well-structured analysis."),
    ("human", """Topic: {topic}

Pros: {pros}
Cons: {cons}
Examples: {examples}

Write a concise 3-sentence analysis.""")
])

# Parallel → reshape → combine
full_chain = (
    {"topic": lambda x: x["topic"], **{}}          # keep topic
    | RunnableParallel({"data": analysis, "topic": lambda x: x["topic"]})
    # Flatten for prompt
    | (lambda x: {**x["data"], "topic": x["topic"]})
    | combine_prompt | llm | StrOutputParser()
)
```

**Simpler version — preserve topic with dict expansion:**

```python
from langchain_core.runnables import RunnablePassthrough

chain_with_topic = (
    RunnableParallel({
        "pros":     ChatPromptTemplate.from_template("3 pros of {topic}") | llm | StrOutputParser(),
        "cons":     ChatPromptTemplate.from_template("3 cons of {topic}") | llm | StrOutputParser(),
        "topic":    lambda x: x["topic"]           # ← preserve input
    })
    | combine_prompt | llm | StrOutputParser()
)

result = chain_with_topic.invoke({"topic": "LangChain"})
```

---

## Pattern 3: Conditional (Routing) Chain

Route input to different chains based on some condition.

```
         ┌─► Technical Chain (if code question)
Input ──►├─► General Chain (if general question)
         └─► Math Chain (if math question)
```

```python
from langchain_core.runnables import RunnableLambda

# Classifier — determines which branch to use
classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the question type. Reply with only one word: technical, math, or general."),
    ("human", "{question}")
])
classifier = classifier_prompt | llm | StrOutputParser()

# Specialized chains
technical_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a senior software engineer. Give technical, precise answers."),
        ("human", "{question}")
    ]) | llm | StrOutputParser()
)

math_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Show your work step by step. Always verify your answer."),
        ("human", "{question}")
    ]) | llm | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Give a friendly, accessible explanation."),
        ("human", "{question}")
    ]) | llm | StrOutputParser()
)

# Router function
def route(inputs: dict):
    q_type = classifier.invoke(inputs).strip().lower()
    print(f"Routing to: {q_type}")
    if "technical" in q_type:
        return technical_chain
    elif "math" in q_type:
        return math_chain
    else:
        return general_chain

# Chain with routing
router_chain = RunnableLambda(route)
full_chain = router_chain | (lambda chain_fn: chain_fn)

# Better — use direct routing:
from langchain_core.runnables import RunnableLambda

def route_and_run(inputs: dict):
    q_type = classifier.invoke(inputs).strip().lower()
    if "technical" in q_type:
        return technical_chain.invoke(inputs)
    elif "math" in q_type:
        return math_chain.invoke(inputs)
    else:
        return general_chain.invoke(inputs)

conditional_chain = RunnableLambda(route_and_run)

# Test
questions = [
    {"question": "What is a Python decorator?"},
    {"question": "What is 128 divided by 4?"},
    {"question": "Why is the sky blue?"},
]
for q in questions:
    print(f"Q: {q['question']}")
    print(f"A: {conditional_chain.invoke(q)[:80]}...")
    print()
```

---

## Pattern 4: Iterative (Loop) Chain

Run a chain repeatedly until a condition is met — essential for ReAct-style agents.

```
Input → Generate → Evaluate → [score < threshold?] → Revise → (loop)
                                [score ≥ threshold?] → Return
```

```python
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

class ReviewedContent(BaseModel):
    content: str = Field(description="The generated content")
    score: int = Field(description="Quality score 1-10")
    feedback: str = Field(description="Specific feedback for improvement")

# Generator chain
generate_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a high-quality blog introduction. Previous feedback: {feedback}"),
    ("human", "Topic: {topic}")
])
generate_chain = generate_prompt | llm | StrOutputParser()

# Evaluator chain
evaluate_prompt = ChatPromptTemplate.from_messages([
    ("system", "Evaluate this blog introduction. Be strict."),
    ("human", "Content: {content}")
])
evaluate_chain = evaluate_prompt | llm.with_structured_output(ReviewedContent)

def iterative_improve(inputs: dict, max_iterations: int = 3, threshold: int = 8):
    topic    = inputs["topic"]
    feedback = "No previous feedback — write your best first draft."
    content  = ""

    for i in range(max_iterations):
        # Generate content
        content = generate_chain.invoke({"topic": topic, "feedback": feedback})
        print(f"\nIteration {i+1}:")
        print(f"Content: {content[:100]}...")

        # Evaluate
        reviewed = evaluate_chain.invoke({"content": content})
        print(f"Score: {reviewed.score}/10 | Feedback: {reviewed.feedback[:60]}")

        if reviewed.score >= threshold:
            print(f"✅ Quality threshold reached!")
            break
        feedback = reviewed.feedback

    return content

result = iterative_improve({"topic": "Why LangChain is the future of AI development"})
```

---

## Pattern 5: Fallback Chain

Automatically try backup chains when the primary fails.

```
Primary Chain → [fails?] → Backup Chain 1 → [fails?] → Backup Chain 2
```

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Primary: expensive, best quality
primary = (
    ChatPromptTemplate.from_template("{question}")
    | ChatOpenAI(model="gpt-4o", temperature=0)
    | StrOutputParser()
)

# Fallback 1: cheaper alternative
backup1 = (
    ChatPromptTemplate.from_template("{question}")
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

# Fallback 2: different provider
backup2 = (
    ChatPromptTemplate.from_template("{question}")
    | ChatAnthropic(model="claude-3-haiku-20240307")
    | StrOutputParser()
)

# Build fallback chain
resilient_chain = primary.with_fallbacks([backup1, backup2])

result = resilient_chain.invoke({"question": "What is LangChain?"})
# If gpt-4o fails → tries gpt-4o-mini → tries claude-3-haiku
```

---

## 🗺️ Pattern Decision Guide

| Scenario | Use Pattern |
|---|---|
| Multi-step transformation | Sequential |
| Multiple analyses of same input | Parallel |
| Different handling for different inputs | Conditional |
| Generate→Evaluate→Refine loop | Iterative |
| High availability, multi-provider | Fallback |
| RAG pipeline | Parallel (context + passthrough) |

---

## ✅ Key Takeaways

- **Sequential**: `A | B | C` — standard pipeline
- **Parallel**: `{k1: chain1, k2: chain2}` — simultaneous branches
- **Conditional**: `RunnableLambda(router)` — dynamic routing by LLM or rule
- **Iterative**: Python loop + LLM evaluator — quality control, ReAct
- **Fallback**: `.with_fallbacks([...])` — resilience, high availability
- Most real-world chains are **combinations** of these patterns

---

## ⬅️ Previous
[RunnablePassthrough & Lambda](./04_runnable_passthrough.md)

## ➡️ Next
[Hands-on Examples →](./examples.ipynb)
