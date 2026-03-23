# 05 — Function Calling Across Providers

> *OpenAI, Anthropic, and Google all support tool use — but with different APIs. Master the differences.*

---

## 5.1 Why Provider Differences Matter

In production, you may need to:
- Switch providers for cost optimization
- Use fallback models when primary is unavailable
- Test the same agent across models for comparison
- Use the best model for each subtask (cheap model for simple lookups, powerful for reasoning)

Understanding the differences lets you build **provider-agnostic agents**.

---

## 5.2 OpenAI vs Anthropic vs Google — Side-by-Side

| Feature | OpenAI | Anthropic (Claude) | Google (Gemini) |
|---|---|---|---|
| **Parameter name** | `tools` | `tools` | `tools` |
| **Tool type** | `"function"` | `"custom"` | Automatic |
| **Tool call format** | `tool_calls[]` in message | `tool_use` content block | `function_call` in part |
| **Tool result format** | `role: "tool"` | `role: "user"` with `tool_result` | `role: "function"` |
| **Parallel calls** | ✅ Native | ✅ Native | ✅ Native |
| **Strict mode** | ✅ Yes | ❌ No | ❌ No |
| **Streaming tool calls** | ✅ Yes | ✅ Yes | ✅ Yes |

---

## 5.3 OpenAI Tool Call Format

```python
# Request
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]
)

# Response — tool call looks like:
{
    "role": "assistant",
    "content": None,
    "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"city": "Mumbai"}'
        }
    }]
}

# Inject result:
{"role": "tool", "tool_call_id": "call_abc123", "content": "28°C, Sunny"}
```

---

## 5.4 Anthropic (Claude) Tool Use Format

```python
import anthropic

client = anthropic.Anthropic()

# Request
response = client.messages.create(
    model="claude-3-5-haiku-20241022",
    max_tokens=1024,
    tools=[{
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {           # ← "input_schema" not "parameters"
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }],
    messages=[{"role": "user", "content": "What's the weather in Mumbai?"}]
)

# Response — tool call in content block:
# response.content = [
#   ToolUseBlock(
#     type="tool_use",
#     id="toolu_abc123",
#     name="get_weather",
#     input={"city": "Mumbai"}    # ← Already a dict, not a JSON string!
#   )
# ]

# Check if it's a tool call
for block in response.content:
    if block.type == "tool_use":
        name = block.name
        args = block.input     # ← Already parsed dict (no json.loads needed)
        tool_id = block.id
        
        # Execute tool...
        result = get_weather(**args)
        
        # Inject result — note: goes back as "user" role with tool_result!
        messages_for_next_call = [
            {"role": "user", "content": "What's the weather in Mumbai?"},
            {"role": "assistant", "content": response.content},  # Full content block
            {
                "role": "user",       # ← Tool result is a USER message in Claude!
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_id,    # ← "tool_use_id" not "tool_call_id"
                    "content": result
                }]
            }
        ]
```

---

## 5.5 Google (Gemini) Function Calling Format

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Define function declaration
get_weather_func = genai.protos.FunctionDeclaration(
    name="get_weather",
    description="Get weather for a city",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "city": genai.protos.Schema(type=genai.protos.Type.STRING)
        },
        required=["city"]
    )
)

tool = genai.protos.Tool(function_declarations=[get_weather_func])
model = genai.GenerativeModel("gemini-1.5-flash", tools=[tool])

# Send message
response = model.generate_content("What's the weather in Mumbai?")

# Check for function call
for part in response.parts:
    if fn := part.function_call:
        name = fn.name
        args = dict(fn.args)     # ← Already a dict
        
        result = get_weather(**args)
        
        # Inject result
        response2 = model.generate_content([
            response.candidates[0].content,
            genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=name,
                        response={"result": result}
                    )
                )],
                role="function"     # ← role is "function" in Gemini
            )
        ])
```

---

## 5.6 LiteLLM — One API for All Providers

**LiteLLM** provides a unified OpenAI-compatible interface for 100+ models. Use it to switch providers without changing your agent code:

```python
import litellm
from litellm import completion

# Same code, different models — LiteLLM handles translation
def call_with_tools(model: str, messages: list, tools: list) -> dict:
    response = completion(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    return response

# Works identically for all:
r1 = call_with_tools("gpt-4o-mini", messages, tools)
r2 = call_with_tools("claude-3-5-haiku-20241022", messages, tools)
r3 = call_with_tools("gemini/gemini-1.5-flash", messages, tools)
```

### LiteLLM Provider Prefixes

```
OpenAI:     "gpt-4o", "gpt-4o-mini"
Anthropic:  "claude-3-5-sonnet-20241022"
Google:     "gemini/gemini-1.5-pro"
Ollama:     "ollama/llama3.1:8b"
Groq:       "groq/llama3-70b-8192"
Azure:      "azure/gpt-4o"
AWS Bedrock:"bedrock/anthropic.claude-3-5-sonnet"
```

---

## 5.7 Provider Comparison: Tool-Use Reliability

From empirical testing on standard agentic tasks (2025):

| Model | Tool Call Accuracy | Instruction Following | Speed | Cost |
|---|---|---|---|---|
| GPT-4o | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Medium | $$$ |
| GPT-4o-mini | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Fast | $ |
| Claude 3.5 Sonnet | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Medium | $$$ |
| Claude 3.5 Haiku | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Fast | $ |
| Gemini 1.5 Pro | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | $$ |
| Gemini 1.5 Flash | ⭐⭐⭐ | ⭐⭐⭐ | Very Fast | $ |
| Llama 3.1 70B (local) | ⭐⭐⭐ | ⭐⭐⭐ | Depends on hardware | Free |

**Recommendation for agents**:
- **Complex multi-step reasoning**: GPT-4o or Claude 3.5 Sonnet
- **High-volume / cost-sensitive**: GPT-4o-mini or Claude 3.5 Haiku
- **Privacy-sensitive (on-premise)**: Llama 3.1 via Ollama

---

## 5.8 Building a Provider-Agnostic Agent with LiteLLM Fallback

```python
import litellm

# Configure fallback — if primary fails, try backup
MODEL_CHAIN = [
    "gpt-4o-mini",                     # Primary (cheap, fast)
    "claude-3-5-haiku-20241022",       # Fallback 1
    "gemini/gemini-1.5-flash"          # Fallback 2
]

def robust_completion(messages, tools, attempt=0):
    if attempt >= len(MODEL_CHAIN):
        raise RuntimeError("All models failed")
    
    model = MODEL_CHAIN[attempt]
    try:
        return litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            timeout=30
        )
    except (litellm.RateLimitError, litellm.ServiceUnavailableError) as e:
        print(f"Model {model} failed: {e}. Trying fallback...")
        return robust_completion(messages, tools, attempt + 1)
```

---

## 📌 Key Takeaways

1. **OpenAI**: `tools` + `tool_calls[]` + inject as `role: "tool"` — industry standard format
2. **Anthropic**: `input_schema` instead of `parameters`; result goes back as `role: "user"` with `tool_result` block
3. **Gemini**: uses proto-based API; result injected as `role: "function"`
4. **LiteLLM** normalizes everything to OpenAI format — use it for provider-agnostic agents
5. **GPT-4o / Claude 3.5 Sonnet** = highest reliability; mini/haiku variants for cost efficiency
6. **Fallback chain**: always configure backup models for production robustness
7. Test your agent on **at least 2 providers** before choosing one for production
