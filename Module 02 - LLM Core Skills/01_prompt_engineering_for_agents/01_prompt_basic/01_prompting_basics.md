# 01 — Prompting Basics: Anatomy of a Prompt

> *Before you can engineer prompts, you must understand exactly what they are and how the LLM processes them.*

---

## 1.1 What Is a Prompt?

A **prompt** is the complete input package sent to an LLM. It is NOT just "the question you type." It is a structured, ordered set of messages that the LLM uses as its entire context to generate a response.

```
PROMPT = System Message + Conversation History + User Message + (Tool Results)
```

Everything the LLM knows at inference time comes from the prompt. There is no background knowledge retrieval at generation time — it's all in the window.

---

## 1.2 The Message Role System

Modern LLM APIs use a **chat format** with three core roles:

### `system`
- Sets the LLM's identity, behavior rules, and constraints
- Processed first — has the strongest influence on all subsequent behavior
- Written by the developer, not the user
- **In agents**: defines who the agent is, what tools it has, and how it should behave

### `user`
- Input from the human (or from another agent in multi-agent systems)
- The question, instruction, or task to perform

### `assistant`
- Previous responses from the LLM
- Included in multi-turn conversations to give the LLM memory of what it said before
- In tool-use loops: also contains tool call decisions

### `tool` (function)
- The result of a tool/function call
- Passed back to the LLM after tool execution so it can process the result

```python
messages = [
    {"role": "system",    "content": "You are a research agent..."},
    {"role": "user",      "content": "Find the latest AI papers"},
    {"role": "assistant", "content": None, "tool_calls": [...]},  # agent's tool call
    {"role": "tool",      "content": "Search results: ...", "tool_call_id": "..."},  # result
    {"role": "assistant", "content": "Here are the top papers..."},  # final answer
]
```

---

## 1.3 Anatomy of a Well-Structured Prompt

A complete agent prompt has these layers (in order):

```
┌─────────────────────────────────────────────────────────┐
│  1. IDENTITY       Who the agent is                      │
│     "You are an expert research agent..."                │
├─────────────────────────────────────────────────────────┤
│  2. CAPABILITIES   What the agent can do                 │
│     "You have access to: web_search, calculator, ..."    │
├─────────────────────────────────────────────────────────┤
│  3. CONSTRAINTS    What the agent must/must not do       │
│     "Always cite sources. Never make up facts."          │
├─────────────────────────────────────────────────────────┤
│  4. REASONING STYLE  How the agent should think          │
│     "Think step by step before taking action."           │
├─────────────────────────────────────────────────────────┤
│  5. OUTPUT FORMAT  How to structure responses            │
│     "Return a JSON object with fields: summary, sources" │
├─────────────────────────────────────────────────────────┤
│  6. EXAMPLES (optional)  2-3 demonstrations              │
│     (few-shot — covered in file 03)                      │
└─────────────────────────────────────────────────────────┘
```

---

## 1.4 Tokens: The Currency of LLMs

Everything in the prompt is counted in **tokens** — the fundamental unit the LLM processes.

### What Is a Token?
- Roughly 4 characters or ¾ of a word in English
- `"Hello world"` ≈ 2 tokens
- `"Agentic AI"` ≈ 3 tokens
- Code, JSON, and special characters tend to use more tokens per character

### Token Rules Every Agent Engineer Must Know

| Concept | Detail |
|---|---|
| **Context window** | Maximum total tokens (input + output) the model can handle |
| **Input tokens** | All messages sent to the model — you pay for these |
| **Output tokens** | The model's response — more expensive than input tokens |
| **Token budget** | You must manage input tokens to leave room for output |

### Context Window Sizes (2025)

| Model | Context Window | Practical Input Limit |
|---|---|---|
| GPT-4o | 128k tokens | ~100k tokens (leave room for output) |
| GPT-4o-mini | 128k tokens | ~100k tokens |
| Claude 3.5 Sonnet | 200k tokens | ~180k tokens |
| Gemini 1.5 Pro | 1M tokens | ~900k tokens |
| Llama 3.1 70B | 128k tokens | ~100k tokens |

### Token Cost Awareness
```
GPT-4o:      $2.50 / 1M input,  $10.00 / 1M output
GPT-4o-mini: $0.15 / 1M input,  $0.60 / 1M output
Claude Haiku:$0.25 / 1M input,  $1.25 / 1M output

Agent with 10 steps × 2000 tokens/step = 20,000 tokens per run
20,000 tokens × $0.0015/1k = $0.03 per agent run
At 1000 runs/day = $30/day — budget matters!
```

---

## 1.5 How LLMs Process Prompts

Understanding this changes how you write prompts.

### The LLM Reads Left to Right, Top to Bottom
- Earlier content has **higher influence** than later content
- Your system prompt (at the top) shapes everything that follows
- Put the most critical instructions **first**

### Attention Bias: Primacy & Recency Effects
- LLMs pay more attention to:
  - **Beginning** of the prompt (system message, first few sentences)
  - **End** of the prompt (the most recent user message or instruction)
- **Middle content gets less attention** — don't bury critical instructions in the middle of a long system prompt

### Context Doesn't Equal Understanding
- The LLM sees ALL tokens but doesn't weight them equally
- Long, complex system prompts → the model may "forget" rules buried deep inside
- **Rule of thumb**: Important rules should appear at the START and be REPEATED briefly at the END

### The LLM Has No Memory (Default)
- Each API call is **stateless** — the LLM doesn't remember previous calls
- Memory = you explicitly including past messages in the `messages` array
- This is why context window management is a core agent engineering skill

---

## 1.6 Prompt vs Hyperparameters

A prompt is not the only thing you control. These hyperparameters are equally important:

| Parameter | What It Controls | Agent Guidance |
|---|---|---|
| `temperature` | Randomness of output (0.0–2.0) | Use 0.0–0.2 for tool calling; 0.7–1.0 for creative tasks |
| `max_tokens` | Maximum output length | Always set this — prevents runaway outputs |
| `top_p` | Nucleus sampling | Usually set 1.0 with temperature; don't set both |
| `stop` | Stop sequences | Set `["Observation:", "Human:"]` to control agent loops |
| `seed` | Reproducibility | Set for deterministic testing |
| `presence_penalty` | Reduces repetition | Useful for long-running agents that repeat themselves |

```python
# Typical settings for a tool-calling agent
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=TOOLS,
    tool_choice="auto",
    temperature=0.1,      # Low: we want reliable tool selection, not creativity
    max_tokens=1024,      # Cap output per step
    seed=42               # Reproducible for testing
)
```

---

## 1.7 Common Beginner Mistakes

| Mistake | Problem | Fix |
|---|---|---|
| Prompt too vague | Agent hallucinates approach | Specify exact output format and reasoning style |
| No output format | Unparseable responses | Define JSON schema or markdown structure |
| Constraints at the end | Agent ignores them (primacy effect) | Put hard constraints at the very start |
| Temperature too high | Non-deterministic tool calls | Use 0.0–0.2 for agentic reasoning |
| No max_tokens | Agent writes forever | Always set max_tokens per step |
| Instructions in user message | User can override system rules | Put agent rules in system message only |
| Giant monolithic prompt | Model loses track of rules | Break into sections with clear headers |

---

## 📌 Key Takeaways

1. A **prompt = all messages** sent to the LLM, not just the user query
2. **Roles matter**: `system` > `user` > `assistant` > `tool` in terms of behavioral influence
3. **Tokens are the unit** — count them, budget them, track their cost per agent run
4. **Primacy & recency**: important instructions at the TOP and reiterated at the END
5. **Temperature = 0.1** for tool-calling agents; only raise it for creative generation steps
6. **Always set max_tokens** — an agent with no output cap is a runaway cost risk

---

## 🔗 Further Reading
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering Docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [Token counting with tiktoken](https://github.com/openai/tiktoken)
