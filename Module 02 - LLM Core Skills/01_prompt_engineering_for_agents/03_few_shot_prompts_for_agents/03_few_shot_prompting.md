# 03 — Few-Shot Prompting

> *Show, don't just tell — examples are more powerful than instructions for teaching LLMs complex behaviors.*

---

## 3.1 What Is Few-Shot Prompting?

**Few-shot prompting** means providing the LLM with **example input-output pairs** inside the prompt, so it learns the desired behavior by analogy rather than explicit instruction alone.

```
Zero-shot:  Pure instruction, no examples
One-shot:   1 example
Few-shot:   2–5 examples (optimal range)
Many-shot:  5+ examples (diminishing returns after ~10)
```

**Key insight**: Examples bypass the LLM's tendency to interpret instructions ambiguously. Seeing is believing — even for language models.

---

## 3.2 Zero-Shot vs Few-Shot — Side by Side

### Zero-Shot (Instruction Only)
```
System: You are a sentiment analyzer. Classify customer reviews as 
        POSITIVE, NEGATIVE, or NEUTRAL.

User: "The product arrived late but works perfectly."
```
**Risk**: The model might over-explain, format incorrectly, or be unsure about mixed sentiment.

### Few-Shot (With Examples)
```
System: You are a sentiment analyzer. Classify customer reviews.

Examples:
Input: "Absolutely amazing, best purchase I've made!"
Output: POSITIVE

Input: "Completely broken, total waste of money."
Output: NEGATIVE

Input: "It's okay, does the job but nothing special."
Output: NEUTRAL

Input: "Delivery was slow but the quality surprised me."
Output: MIXED

Now classify:
Input: "The product arrived late but works perfectly."
Output:
```
**Result**: The model now knows the exact format, handles edge cases (MIXED), and matches the label style perfectly.

---

## 3.3 Why Few-Shot Works: In-Context Learning

LLMs have a capability called **in-context learning (ICL)** — they detect the pattern in your examples and generalize it to new inputs **without any gradient updates or retraining**.

```
Pattern detection:
  "input → POSITIVE" × 2 + "input → NEGATIVE" × 2
      ↓
  Model infers: short uppercase label, no explanation, classify tone
      ↓
  Applies exact same pattern to new input
```

This works because during pre-training, the model saw billions of examples of pattern-following text.

---

## 3.4 Few-Shot for Agents: Tool Use Examples

The most powerful application for agents is **demonstrating tool calling behavior** through examples.

### Example: Teaching an Agent When to Search vs Answer Directly

```
System: You are a research agent with access to web_search.

# Examples of when to use web_search vs answer directly:

Example 1:
User: "What is 2 + 2?"
Thought: This is basic arithmetic. No search needed.
Answer: 4

Example 2:
User: "What is the current price of Tesla stock?"
Thought: Stock prices change in real time. I need current data.
Action: web_search("Tesla TSLA stock price today")
Observation: Tesla (TSLA): $248.42 USD
Answer: The current Tesla stock price is $248.42 USD.

Example 3:
User: "Who invented the telephone?"
Thought: This is historical fact from training data.
Answer: Alexander Graham Bell is credited with inventing the telephone in 1876.

Example 4:  
User: "What AI papers were published this week?"
Thought: Weekly publications change constantly. Must search.
Action: web_search("new AI papers published this week 2025")
...

Now handle: "What is the population of Tokyo as of 2025?"
```

The agent now knows exactly when to search and when not to — from examples, not just rules.

---

## 3.5 Structuring Few-Shot Examples

### Rule 1: Consistent Format
Every example must use **identical structure**. Inconsistency confuses the model.

```
❌ Inconsistent:
Example 1: Input: "X" → Output: Y
Example 2: 
  Question: "X"
  Response: Y

✅ Consistent:
Example 1:
Input: "X"
Output: Y

Example 2:
Input: "X"
Output: Y
```

### Rule 2: Cover Edge Cases
Pick examples that demonstrate:
- The easy/typical case
- The "tricky" boundary case
- The "refuse or redirect" case (if applicable)

```
Example set for a content moderation agent:
1. Clear benign input → ALLOW
2. Clear harmful input → BLOCK
3. Borderline input   → FLAG_FOR_REVIEW  ← most important to demonstrate
4. Off-topic input    → REDIRECT
```

### Rule 3: 3–5 Examples Is the Sweet Spot
- **< 3 examples**: model may not catch the pattern
- **3–5 examples**: optimal — pattern is clear without wasting tokens  
- **> 8 examples**: diminishing returns, wastes context window

### Rule 4: Put Examples Before the Target
```
[System Prompt]
[Example 1]
[Example 2]
[Example 3]
← Target Input Goes Here  (the model "continues" the pattern)
```

### Rule 5: Use Real Examples, Not Synthetic Ones
Examples from actual production data outperform made-up examples significantly. If you have real user queries → use them as few-shot demonstrations.

---

## 3.6 Dynamic Few-Shot: Selecting Examples at Runtime

**Static few-shot**: same examples hardcoded in every prompt  
**Dynamic few-shot**: select the most relevant examples from a library at runtime

```python
# Example library stored in a vector database
example_library = [
    {"input": "Tesla stock price", "output": "SEARCH", "embedding": [...]},
    {"input": "Who is Einstein?",  "output": "ANSWER", "embedding": [...]},
    # ... hundreds more
]

def get_relevant_examples(user_query: str, k: int = 3) -> list[dict]:
    """Retrieve k most similar examples to the current query."""
    query_embedding = embed(user_query)
    scored = [(ex, cosine_sim(query_embedding, ex["embedding"])) 
              for ex in example_library]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [ex for ex, score in scored[:k]]

# Build prompt dynamically
examples = get_relevant_examples(user_query)
prompt = SYSTEM_PROMPT + format_examples(examples) + user_query
```

**Benefits of dynamic few-shot**:
- Better coverage across diverse queries
- Smaller average prompt size
- Examples stay relevant as task distribution shifts

---

## 3.7 Few-Shot Templates for Common Agent Tasks

### Template 1: Classification Agent
```
Classify the following [ITEM_TYPE] into one of: [CATEGORY_1], [CATEGORY_2], [CATEGORY_3].

Examples:
Input: "[example_1]"
Category: [CATEGORY_1]

Input: "[example_2]"  
Category: [CATEGORY_2]

Input: "[example_3]"
Category: [CATEGORY_3]

Now classify:
Input: "{user_input}"
Category:
```

### Template 2: Extraction Agent
```
Extract [ENTITY_TYPE] from text. Return JSON with field: [FIELD_NAME].

Examples:
Text: "Contact John Smith at john@company.com or call +1-555-0123"
{"name": "John Smith", "email": "john@company.com", "phone": "+1-555-0123"}

Text: "Reach out to Dr. Maria Garcia, maria.garcia@hospital.org"
{"name": "Dr. Maria Garcia", "email": "maria.garcia@hospital.org", "phone": null}

Extract from:
Text: "{user_input}"
```

### Template 3: ReAct-Style Reasoning Agent
```
Answer questions using Thought → Action → Observation format.

Question: What year was Python created?
Thought: Python's creation is historical knowledge I have in training data.
Answer: Python was created in 1991 by Guido van Rossum.

Question: What is today's weather in London?
Thought: Weather changes in real-time. I need to search.
Action: web_search("London weather today")
Observation: [search result]
Answer: Based on current data, London is [weather].

Question: {user_question}
Thought:
```

---

## 3.8 Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails | Fix |
|---|---|---|
| **Inconsistent example format** | Model averages across formats, produces hybrid | Strict identical formatting |
| **Only positive examples** | Model never learns when to refuse/redirect | Include negative + edge cases |
| **Too many examples** | Wastes tokens, buries the actual task | 3–5 is optimal |
| **Examples contradict instructions** | Model confusion — examples win | Ensure examples match stated rules |
| **Made-up, unrealistic examples** | Poor generalization to real inputs | Use real data from production |
| **Examples without variety** | Overfits to narrow pattern | Cover range of difficulty |

---

## 📌 Key Takeaways

1. **Few-shot > instructions alone** for format enforcement and edge case handling
2. **3–5 examples** is the optimal range — more is not always better
3. **Consistent format** across examples is non-negotiable
4. **Cover edge cases** — the borderline example is often the most valuable
5. **Dynamic few-shot** (examples selected by similarity) outperforms static for diverse tasks
6. **Examples teach behavior** better than rules — use both together for maximum reliability
7. **Real examples > synthetic** — always prefer actual production data when available
