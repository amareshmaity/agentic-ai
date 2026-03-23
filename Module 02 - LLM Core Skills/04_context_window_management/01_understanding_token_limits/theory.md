# 01 — Understanding Token Limits

> *You can't manage what you can't measure — token counting is the first skill of context management.*

---

## 1.1 What Is a Token?

A **token** is the basic unit that LLMs process. It's not exactly a word or a character — it sits in between:

```
Word → "hamburger"  → 3 tokens: ["ham", "bur", "ger"]
Word → "hello"      → 1 token:  ["hello"]
Word → "ChatGPT"    → 2 tokens: ["Chat", "GPT"]
Char → " "          → 1 token (space is often merged with next word)
```

**Rough rules of thumb (English):**
- 1 token ≈ 4 characters
- 1 token ≈ ¾ of a word
- 100 tokens ≈ 75 words
- 1,000 tokens ≈ 750 words
- 1 page of text ≈ 500–750 tokens

---

## 1.2 Context Window Limits by Model (2024)

| Model | Context Window | ~ Words | Best For |
|---|---|---|---|
| `gpt-4o-mini` | 128,000 tokens | ~96,000 words | Agents, high-volume |
| `gpt-4o` | 128,000 tokens | ~96,000 words | High-quality reasoning |
| `claude-3-5-sonnet` | 200,000 tokens | ~150,000 words | Long documents, analysis |
| `claude-3-haiku` | 200,000 tokens | ~150,000 words | Fast + long context |
| `gemini-1.5-flash` | 1,000,000 tokens | ~750,000 words | Massive document processing |
| `gemini-1.5-pro` | 2,000,000 tokens | ~1.5M words | Entire codebases |
| `llama-3.1-70b` | 128,000 tokens | ~96,000 words | Open source |

**Important**: Larger context ≠ better performance. Models are generally worse at retrieving information from the middle of very long contexts ("lost in the middle" problem).

---

## 1.3 What Counts Against Your Context Window

Every API call re-sends the entire conversation history plus tools. The full token budget includes:

```
Total tokens used = 
    System prompt tokens
  + All prior message tokens (user + assistant)
  + Tool/function definitions
  + Current user message
  + Reserved space for completion (output)
```

```python
# Example token budget for a typical agent call
system_prompt       = ~200  tokens
conversation history = ~3,000 tokens (10 turns × 300 each)
tool definitions    = ~400  tokens (4 tools × 100 each)
current message     = ~50   tokens
completion reserved = ~1,000 tokens (max_tokens setting)
─────────────────────────────────────────────
Total               = ~4,650 tokens per call
```

With 128k context and 4,650 tokens per call, you can sustain roughly 27 turns before you run out. In practice, much fewer because conversations grow.

---

## 1.4 Counting Tokens with `tiktoken`

OpenAI's `tiktoken` library counts tokens exactly the same way the API does:

```python
import tiktoken

# Choose the right encoder for your model
enc = tiktoken.encoding_for_model("gpt-4o-mini")
# OR use the base encoding directly
enc = tiktoken.get_encoding("cl100k_base")  # Used by GPT-4, GPT-4o, GPT-3.5

text = "Context window management is critical for agentic systems."
tokens = enc.encode(text)

print(f"Text:        {text!r}")
print(f"Token count: {len(tokens)}")
print(f"Token IDs:   {tokens}")
print(f"Decoded:     {[enc.decode([t]) for t in tokens]}")
```

### Counting Tokens in a Messages Array

```python
import tiktoken, json

def count_message_tokens(messages: list[dict], model: str = "gpt-4o-mini") -> int:
    """
    Count tokens for a full messages array the same way OpenAI counts them.
    Accounts for per-message overhead that OpenAI adds.
    """
    enc = tiktoken.encoding_for_model(model)
    
    # OpenAI adds 3 tokens per message (role + content delimiters)
    # Plus 3 tokens for the reply priming at the end
    tokens_per_message = 3
    tokens_per_name = 1
    total = 3  # Reply priming
    
    for message in messages:
        total += tokens_per_message
        for key, value in message.items():
            total += len(enc.encode(str(value)))
            if key == "name":
                total += tokens_per_name
    
    return total
```

---

## 1.5 Counting Tool Definition Tokens

Tool schemas consume significant input tokens on every call:

```python
import tiktoken, json

def count_tool_tokens(tools: list[dict], model: str = "gpt-4o-mini") -> int:
    """Estimate tokens consumed by tool definitions."""
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for tool in tools:
        schema_str = json.dumps(tool)
        total += len(enc.encode(schema_str))
    # OpenAI adds ~10-15 tokens of overhead per tools array
    total += 15
    return total

# Example
SAMPLE_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information on any topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "description": "Max results to return (1-10)"}
            },
            "required": ["query"]
        }
    }
}

tool_tokens = count_tool_tokens([SAMPLE_TOOL])
print(f"Single tool definition: ~{tool_tokens} tokens")
print(f"10 tools would add:     ~{tool_tokens * 10} tokens per call")
```

---

## 1.6 Cost Calculation — Tokens = Money

```python
# OpenAI pricing (as of early 2024, may change)
PRICING = {
    "gpt-4o-mini": {
        "input":  0.150 / 1_000_000,   # $0.150 per 1M input tokens
        "output": 0.600 / 1_000_000,   # $0.600 per 1M output tokens
    },
    "gpt-4o": {
        "input":  5.00  / 1_000_000,   # $5.00 per 1M input tokens
        "output": 15.00 / 1_000_000,   # $15.00 per 1M output tokens
    },
    "claude-3-haiku-20240307": {
        "input":  0.25  / 1_000_000,
        "output": 1.25  / 1_000_000,
    }
}

def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o-mini"
) -> float:
    """Calculate the cost in USD for a single API call."""
    pricing = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])

def estimate_agent_run_cost(
    avg_input_tokens_per_step: int,
    avg_output_tokens_per_step: int,
    num_steps: int,
    model: str = "gpt-4o-mini"
) -> dict:
    """Estimate total cost for a multi-step agent run."""
    cost_per_step = estimate_cost(avg_input_tokens_per_step, avg_output_tokens_per_step, model)
    total_cost = cost_per_step * num_steps
    return {
        "cost_per_step_usd":  round(cost_per_step, 6),
        "total_cost_usd":     round(total_cost, 4),
        "per_1000_runs_usd":  round(total_cost * 1000, 2),
        "per_million_runs":   round(total_cost * 1_000_000, 0)
    }
```

---

## 1.7 The "Lost in the Middle" Problem

Research shows LLMs perform better when relevant information is at the **beginning or end** of the context:

```
Best recall:   ████████ ░░░░░░░░░░░░░░░░░░░░ ████████
               ^Beginning                    End^

Worst recall:       ░░░░░░░░░ ████ ░░░░░░░░░
                              ^Middle^
```

**Practical implications:**
- Put system instructions at the **start** (before conversation history)
- Put the current task/question at the **end** (most recent message)
- Critical reference data: **beginning** of context
- Avoid burying key facts in the middle of long conversation history

---

## 1.8 Completion Tokens — Reserve Space for Output

Always reserve space for the model's response:

```python
def calculate_safe_max_input(
    context_limit: int,
    max_output_tokens: int,
    tool_tokens: int = 0,
    buffer: int = 200       # Safety margin
) -> int:
    """Calculate maximum safe input tokens."""
    return context_limit - max_output_tokens - tool_tokens - buffer

# Example
safe_input = calculate_safe_max_input(
    context_limit=128_000,
    max_output_tokens=4_096,
    tool_tokens=500,    # 5 tools × ~100 tokens
    buffer=200
)
print(f"Safe max input tokens: {safe_input:,}")   # ~123,204
```

---

## 1.9 Token Model Cheat Sheet

```python
# Model encodings (use with tiktoken)
ENCODING_MAP = {
    "gpt-4o":          "o200k_base",
    "gpt-4o-mini":     "o200k_base",
    "gpt-4-turbo":     "cl100k_base",
    "gpt-4":           "cl100k_base",
    "gpt-3.5-turbo":   "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
}

# Quick token count (no model needed — rough estimate)
def quick_token_estimate(text: str) -> int:
    """~4 chars per token rough estimate."""
    return len(text) // 4
```

---

## 📌 Key Takeaways

1. **1 token ≈ 4 characters ≈ ¾ word** — always count before sending
2. **tiktoken** counts tokens exactly — use `encoding_for_model()` to match the API
3. **Full context = system + all history + tools + current message + reserved output space**
4. **Tool definitions cost tokens on every call** — remove unused tools
5. **"Lost in the middle"** — put important info at start/end, not buried in history
6. **Reserve max_tokens** for output — never fill the whole context with input
7. **Monitor token usage** via `response.usage` — every response has the actual count
