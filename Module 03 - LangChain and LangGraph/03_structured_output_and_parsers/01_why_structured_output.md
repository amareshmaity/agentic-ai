# Why Structured Output?

> *Unstructured LLM output is unpredictable — it breaks your application. Structured output makes your agents reliable and production-ready.*

---

## 🔴 The Problem: Unstructured LLM Output

Ask an LLM to extract data from text — and without constraints, the output format is unpredictable:

```
Prompt: "Extract the company name and founding year from:
         'Apple was founded in 1976 by Steve Jobs.'"

Run 1: "The company is Apple, founded in 1976."
Run 2: "Company: Apple\nFounded: 1976"
Run 3: "Apple (1976)"
Run 4: '{"company": "Apple", "year": 1976}'
Run 5: "Based on the text, Apple was established in 1976."
```

**Every response format is different.** You can't write reliable parsing code for this.

---

## 😱 Real-World Consequences

```python
# Trying to use unstructured output in your app
response = llm.invoke("Extract name and year from: " + text)

# Which format came back this time?
name = response.content.split("company is ")[1]  # Only works for format 1!
# → IndexError on run 2, 3, 4, 5

# Result: intermittent crashes, bad data, debugging nightmares
```

---

## ✅ The Solution: Structured Output

Force the LLM to return data in a **defined, predictable format** that your code can always parse.

```python
from pydantic import BaseModel

class CompanyInfo(BaseModel):
    company_name: str
    founding_year: int
    founder: str

# Every run returns the exact same structure
result = chain.invoke("Apple was founded in 1976 by Steve Jobs.")
print(result.company_name)   # "Apple"     — always works
print(result.founding_year)  # 1976        — always int
print(result.founder)        # "Steve Jobs" — always works
print(type(result))          # <class 'CompanyInfo'>
```

---

## 🧠 Why This Matters for Agents

Agents need to:
1. **Call tools** → tool arguments must be exact types (int, str, list)
2. **Make decisions** → needs to parse "route to agent A or B"
3. **Store data** → can't store raw strings in a database reliably
4. **Chain steps** → step 2 needs predictable output from step 1

```
Without structured output:
  Step 1: LLM output = "The score is about 8 out of 10"
  Step 2: Can't compare "about 8 out of 10" > 7  → agent breaks

With structured output:
  Step 1: LLM output = Review(score=8, sentiment="positive")
  Step 2: if result.score > 7: publish()   → agent works perfectly
```

---

## 🔧 LangChain's Structured Output Options

| Method | Returns | Validation | Best For |
|---|---|---|---|
| `StrOutputParser` | `str` | None | Simple text, no structure needed |
| `JsonOutputParser` | `dict` | None | Flexible JSON, unknown schema |
| `PydanticOutputParser` | Pydantic model | ✅ Full | Typed, validated output |
| `.with_structured_output(schema)` | Pydantic model / dict | ✅ Full | **Modern, preferred approach** |

---

## 📈 The Reliability Spectrum

```
Least reliable                              Most reliable
────────────────────────────────────────────────────────
Raw LLM    StrOutput   JsonOutput   Pydantic   .with_structured_output()
output       Parser      Parser      Parser
  ❌           ✅          ✅✅         ✅✅✅           ✅✅✅✅
(no format) (string)   (dict, no   (typed +    (native tool calling,
                        type check)  validated)  model-enforced schema)
```

---

## ✅ Key Takeaways

- Unstructured LLM output → **unpredictable** format → **fragile** apps
- Structured output → **predictable** format → **reliable** apps
- This is especially critical for **agents** that chain steps or call tools
- LangChain provides 4 parser options — choose based on your needs
- **Always use structured output** for anything beyond simple text generation

---

## ➡️ Next
[StrOutputParser →](./02_str_output_parser.md)
