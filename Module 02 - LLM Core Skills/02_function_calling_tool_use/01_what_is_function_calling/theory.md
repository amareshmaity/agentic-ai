# 01 — What Is Function Calling?

> *Understanding the protocol that transforms LLMs from text generators into action-taking agents.*

---

## 1.1 The Problem Function Calling Solves

An LLM trained up to a cutoff date has **static knowledge**. It cannot:
- Tell you today's weather
- Know the current Bitcoin price
- Read your emails
- Submit a form
- Call your company's internal API

Before function calling, developers used fragile **text parsing tricks**:

```
# Old approach (fragile)
System: "If you need to search, output EXACTLY: SEARCH: <query>"

LLM output: "SEARCH: latest AI news"

# Developer: manually parse this string, execute search, inject result back
```

**Function calling** replaced this with a **structured, reliable protocol** built directly into the model's output format. Introduced by OpenAI in June 2023.

---

## 1.2 How Function Calling Works — The Protocol

### Step-by-Step Flow

```
STEP 1: DEVELOPER DEFINES TOOLS
        → Describe available functions in JSON schema format
        → These are sent to the LLM as part of the API call

STEP 2: USER SENDS A MESSAGE
        → "What's the weather in Mumbai today?"

STEP 3: LLM REASONS & DECIDES
        → LLM sees: user question + tool schemas
        → Decides: "I need to call get_weather('Mumbai')"
        → Outputs: structured JSON tool call (not free text)

STEP 4: DEVELOPER PARSES TOOL CALL
        → Application extracts: tool name + arguments from LLM output
        → Executes the actual function

STEP 5: RESULT INJECTED BACK
        → Tool result is added to conversation as a 'tool' message
        → LLM receives result in next call

STEP 6: LLM GENERATES FINAL ANSWER
        → LLM reads tool result
        → Produces natural language response to user
```

### Visual Flow

```
User Input
    │
    ▼
┌─────────────────────────────┐
│  LLM (with tool schemas)    │
│  Reasons: "need weather"    │
│  Outputs: tool_call JSON    │
└─────────────────────────────┘
    │  tool_call: {name: "get_weather", args: {"city": "Mumbai"}}
    ▼
┌─────────────────────────────┐
│  Your Application           │
│  Parses tool call           │
│  Executes get_weather()     │
│  Gets result: "28°C, Sunny" │
└─────────────────────────────┘
    │  tool_result: "28°C, Sunny, Humidity: 65%"
    ▼
┌─────────────────────────────┐
│  LLM (with tool result)     │
│  Reads observation          │
│  Generates final answer     │
└─────────────────────────────┘
    │
    ▼
"The weather in Mumbai is 28°C and sunny today."
```

---

## 1.3 What the LLM Actually Outputs

When the LLM decides to call a tool, it does NOT output text. It outputs a structured object:

```json
{
  "id": "call_abc123",
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"city\": \"Mumbai\", \"units\": \"celsius\"}"
  }
}
```

Key details:
- `arguments` is a **JSON string** (not a JSON object) — you must `json.loads()` it
- `id` is a unique identifier for this specific call — used to match the result
- The LLM can output **multiple tool calls** in a single response (parallel calls)

---

## 1.4 The Tool Call Lifecycle

```
Message 1: User asks question
Message 2: LLM outputs tool_call (no text content)
Message 3: Tool result injected (role: "tool", tool_call_id: matches call id)
Message 4: LLM generates final answer (now has text content)
```

The conversation history after a tool call:

```python
messages = [
    # Turn 1
    {"role": "user", "content": "What's the weather in Mumbai?"},
    
    # Turn 2: LLM decides to call a tool
    {
        "role": "assistant",
        "content": None,  # No text when making tool calls
        "tool_calls": [{
            "id": "call_abc123",
            "type": "function", 
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Mumbai"}'
            }
        }]
    },
    
    # Turn 3: Tool result injected
    {
        "role": "tool",
        "tool_call_id": "call_abc123",  # Must match the call id above
        "content": "28°C, Sunny, Humidity: 65%"
    },
    
    # Turn 4: LLM generates final answer (next API call)
    {
        "role": "assistant",
        "content": "The weather in Mumbai is 28°C and sunny today with 65% humidity."
    }
]
```

---

## 1.5 Function Calling vs Tool Use — Terminology

Different providers use different terminology for the same concept:

| Term | Used by | Meaning |
|---|---|---|
| **Function Calling** | OpenAI | LLM calls your defined functions |
| **Tool Use** | Anthropic (Claude) | Same concept, different name |
| **Function Calling** | Google (Gemini) | Same concept |
| **Tools** | LangChain, LangGraph | Abstraction over all provider variants |

They all work the same way conceptually — the API format differs slightly.

---

## 1.6 When Does the LLM Decide to Call a Tool?

The LLM makes this decision based on:

1. **Your tool descriptions** — if the description matches the user's need, it calls the tool
2. **tool_choice parameter** — you can force or prevent tool use:

```python
# Let the LLM decide (recommended for most cases)
tool_choice="auto"

# Force the LLM to call a specific tool
tool_choice={"type": "function", "function": {"name": "get_weather"}}

# Force the LLM to call SOME tool (any one from the list)
tool_choice="required"

# Prevent any tool calls (direct answer only)
tool_choice="none"
```

3. **Context from the conversation** — if previous messages show a pattern, the LLM adapts

---

## 1.7 Function Calling vs Fine-Tuning vs RAG

| Approach | What it does | When to use |
|---|---|---|
| **Function Calling** | LLM calls external systems in real-time | Dynamic data; actions; APIs |
| **RAG** | Retrieves documents and injects as context | Static knowledge base; documents |
| **Fine-tuning** | Updates model weights with new data | Consistent style/format changes; behavioral patterns |

In production agents, you often use **all three together**:
- RAG to retrieve company knowledge
- Function calling to take actions (send email, query live API)
- Fine-tuning to enforce consistent output format

---

## 1.8 Security: What You Must Validate Before Executing

The LLM decides which function to call and with what arguments — but **you must validate everything before executing**:

```python
def safe_execute_tool(tool_name: str, args: dict) -> str:
    # 1. Is this tool in our approved list?
    if tool_name not in APPROVED_TOOLS:
        return f"Error: Tool '{tool_name}' is not permitted."
    
    # 2. Are the arguments valid?
    if not validate_args(tool_name, args):
        return f"Error: Invalid arguments for '{tool_name}'."
    
    # 3. Does the user have permission to use this tool?
    if not user_has_permission(tool_name):
        return f"Error: Insufficient permissions for '{tool_name}'."
    
    # 4. Execute safely
    try:
        return APPROVED_TOOLS[tool_name](**args)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"
```

**Never blindly execute whatever the LLM requests.** Validate input, check permissions, handle errors.

---

## 📌 Key Takeaways

1. Function calling is a **built-in protocol** that lets LLMs request structured tool execution
2. The LLM outputs a **JSON tool call object**, not free text, when it wants to use a tool
3. Your application executes the tool and **injects the result** back into the conversation
4. The `tool_call_id` links the result to the specific call — essential for multi-tool scenarios
5. `tool_choice` controls whether the LLM is forced, allowed, or prevented from calling tools
6. **Always validate** tool names and arguments before executing — never trust LLM output blindly
7. FC ≠ RAG ≠ fine-tuning — they solve different problems; use them together in production
