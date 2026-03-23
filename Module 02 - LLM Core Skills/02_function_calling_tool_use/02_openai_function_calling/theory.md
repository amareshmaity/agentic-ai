# 02 — OpenAI Function Calling API

> *Deep dive into the OpenAI tool-calling API — the most widely used function calling implementation.*

---

## 2.1 The Complete API Structure

A function calling request to OpenAI has these components:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],          # Conversation history
    tools=[...],             # Tool/function definitions
    tool_choice="auto",      # How the model decides to call tools
    parallel_tool_calls=True # Allow multiple tools per response
)
```

---

## 2.2 The `tools` Parameter — Defining Your Functions

Each tool in the `tools` list is a JSON object:

```python
TOOLS = [
    {
        "type": "function",       # Always "function"
        "function": {
            "name": "get_weather",                    # Exact function name (no spaces)
            "description": "Get the current weather for a city. Use this when the user asks about weather, temperature, or climate conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g., 'Mumbai', 'New York', 'London'"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Default: celsius"
                    }
                },
                "required": ["city"],                 # city is required, units is optional
                "additionalProperties": False
            },
            "strict": True   # Enables structured output mode (JSON schema strict enforcement)
        }
    }
]
```

### Parameter Types Supported

| JSON Type | Python Equivalent | Example |
|---|---|---|
| `string` | `str` | `"city": {"type": "string"}` |
| `number` | `float` | `"temperature": {"type": "number"}` |
| `integer` | `int` | `"count": {"type": "integer"}` |
| `boolean` | `bool` | `"verbose": {"type": "boolean"}` |
| `array` | `list` | `"tags": {"type": "array", "items": {"type": "string"}}` |
| `object` | `dict` | `"filters": {"type": "object", "properties": {...}}` |
| `enum` | Literal values | `"unit": {"type": "string", "enum": ["m", "km", "mi"]}` |

---

## 2.3 Parsing the Response

After the LLM call, check if it wants to call a tool or give a final answer:

```python
def handle_response(response) -> dict:
    """Parse LLM response — returns either tool_calls or final_answer."""
    msg = response.choices[0].message
    
    if msg.tool_calls:
        # LLM wants to call tools
        return {
            "type": "tool_calls",
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments)
                }
                for tc in msg.tool_calls
            ],
            "raw_message": msg
        }
    elif msg.content:
        # LLM has a final answer
        return {
            "type": "final_answer",
            "content": msg.content
        }
    else:
        return {"type": "empty", "content": ""}
```

---

## 2.4 `finish_reason` — Understanding Why the Model Stopped

```python
finish_reason = response.choices[0].finish_reason

# Possible values:
"stop"          # LLM completed naturally (final answer)
"tool_calls"    # LLM wants to call tools — must handle tool execution
"length"        # Hit max_tokens limit — output may be incomplete
"content_filter"# Output blocked by safety filter
"function_call" # (Legacy, use "tool_calls" instead)
```

Always check `finish_reason` before processing:

```python
if finish_reason == "tool_calls":
    # Execute tools and continue the loop
    pass
elif finish_reason == "stop":
    # Done — extract and return final answer
    pass
elif finish_reason == "length":
    # Incomplete output — either increase max_tokens or handle gracefully
    pass
```

---

## 2.5 Strict Mode — Guaranteed JSON Schema Compliance

When `"strict": True` is set on a function, the model **guarantees** its tool call arguments match the schema exactly. No extra fields, no missing required fields.

```python
# With strict=True
tool = {
    "type": "function",
    "function": {
        "name": "create_ticket",
        "strict": True,  # ← Enable strict mode
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "assigned_to": {"type": "string"}
            },
            "required": ["title", "priority"],
            "additionalProperties": False  # ← Required for strict mode
        }
    }
}
```

**Trade-off**: Strict mode adds slight latency but eliminates JSON parsing errors for invalid arguments.

---

## 2.6 `tool_choice` — Controlling Tool Selection

```python
# 1. Auto (default) — LLM decides whether to call tools
tool_choice = "auto"

# 2. Required — LLM MUST call at least one tool
tool_choice = "required"

# 3. None — LLM must NOT call any tools (give direct answer)
tool_choice = "none"

# 4. Force a specific function call
tool_choice = {
    "type": "function",
    "function": {"name": "get_weather"}
}
```

### When to Use Each

| Setting | Use Case |
|---|---|
| `"auto"` | General purpose — let the model decide |
| `"required"` | When you need structured output from any tool |
| `"none"` | When you need a conversational response after tool results |
| `{specific}` | Extraction/classification where you KNOW which tool is needed |

---

## 2.7 Streaming Function Calls

For agents with UIs, you can stream the tool call arguments as they're generated:

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=TOOLS,
    stream=True
)

# Accumulate streaming chunks
tool_calls_buffer = {}

for chunk in stream:
    delta = chunk.choices[0].delta
    
    if delta.tool_calls:
        for tc_delta in delta.tool_calls:
            idx = tc_delta.index
            
            # Initialize buffer for this tool call
            if idx not in tool_calls_buffer:
                tool_calls_buffer[idx] = {
                    "id": "", "name": "", "arguments": ""
                }
            
            # Accumulate fields
            if tc_delta.id:
                tool_calls_buffer[idx]["id"] += tc_delta.id
            if tc_delta.function.name:
                tool_calls_buffer[idx]["name"] += tc_delta.function.name
            if tc_delta.function.arguments:
                tool_calls_buffer[idx]["arguments"] += tc_delta.function.arguments
                # Show partial progress
                print(f"\rBuilding args: {tool_calls_buffer[idx]['arguments']}", end="")

# After stream ends, parse final tool calls
for idx, tc in tool_calls_buffer.items():
    parsed_args = json.loads(tc["arguments"])
    print(f"\nFinal call: {tc['name']}({parsed_args})")
```

---

## 2.8 Token Usage with Tool Calls

Tool definitions consume input tokens. Plan your budget accordingly:

```python
def estimate_tool_tokens(tools: list) -> int:
    """Rough estimate: each tool definition ≈ 100-300 tokens."""
    total = 0
    for tool in tools:
        schema_str = json.dumps(tool)
        # ~1 token per 4 characters (rough)
        total += len(schema_str) // 4
    return total

tools_tokens = estimate_tool_tokens(TOOLS)
print(f"Tools add ~{tools_tokens} tokens to every request")
print(f"At 1000 agent runs/day this costs: ${tools_tokens * 0.00015:.2f}/day")
```

**Optimization**: Remove unnecessary tools from the list when they're not needed for the current task. Every tool in the list costs tokens on every call.

---

## 2.9 Complete OpenAI Function Calling Example

```python
import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Define tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g., 'AAPL', 'GOOGL', 'TSLA'"
                    }
                },
                "required": ["ticker"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_percentage_change",
            "description": "Calculate percentage change between two values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_value": {"type": "number", "description": "Original value"},
                    "new_value": {"type": "number", "description": "New value"}
                },
                "required": ["old_value", "new_value"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

# Mock tool implementations
def get_stock_price(ticker: str) -> str:
    prices = {"AAPL": 185.50, "GOOGL": 175.20, "TSLA": 248.00, "MSFT": 420.00}
    price = prices.get(ticker.upper(), None)
    return f"${price:.2f}" if price else f"Ticker {ticker} not found."

def calculate_percentage_change(old_value: float, new_value: float) -> str:
    change = ((new_value - old_value) / old_value) * 100
    direction = "up" if change > 0 else "down"
    return f"Changed {direction} {abs(change):.2f}% (from {old_value} to {new_value})"

TOOL_MAP = {
    "get_stock_price": get_stock_price,
    "calculate_percentage_change": calculate_percentage_change
}

def run_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": "You are a financial assistant. Use tools to get real data."},
        {"role": "user",   "content": user_message}
    ]
    
    for step in range(5):  # Max 5 steps
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        if response.choices[0].finish_reason == "tool_calls":
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = TOOL_MAP[name](**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
        else:
            return msg.content
    
    return "Max steps reached"

# Run it
answer = run_agent("What is Apple's current stock price? If it was $150 six months ago, how much has it changed?")
print(answer)
```

---

## 📌 Key Takeaways

1. `tools` array = list of JSON schemas describing your functions
2. Parse `finish_reason` — `"tool_calls"` means execute tools; `"stop"` means final answer
3. `strict: True` guarantees schema-compliant arguments — use for production
4. `tool_choice` controls whether FC is optional, required, or forced to a specific function
5. Streaming FC is possible — accumulate argument chunks before parsing
6. Tool definitions cost tokens — remove unused tools from the list
7. Always add the raw `msg` object back to `messages` before injecting tool results
