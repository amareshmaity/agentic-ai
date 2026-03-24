# 05 — Streaming Tool Calls

> *How partial JSON arguments arrive across chunks — and how to assemble them correctly.*

---

## 5.1 Why Tool Call Streaming Is Tricky

When a model decides to call a function, it generates the arguments as JSON — but that JSON arrives **across multiple chunks**:

```
Chunk 1: {"id": "call_abc", "name": "get_weather", "arguments": ""}
Chunk 2: {"arguments": "{\"ci"}
Chunk 3: {"arguments": "ty\": \""}
Chunk 4: {"arguments": "Tokyo\"}"}
Chunk 5: {} (finish_reason: "tool_calls")
```

You must concatenate the `arguments` strings from every chunk **before** calling `json.loads()`.

---

## 5.2 The Accumulator Pattern

```python
def accumulate_tool_calls(stream) -> dict[int, dict]:
    """
    Correctly accumulate tool call deltas across a stream.
    Returns: {index: {"id": str, "name": str, "arguments": str}}
    """
    tool_calls = {}  # Index → accumulated fields

    for chunk in stream:
        delta = chunk.choices[0].delta
        if not delta.tool_calls:
            continue

        for tc_delta in delta.tool_calls:
            idx = tc_delta.index

            # Initialize on first encounter for this index
            if idx not in tool_calls:
                tool_calls[idx] = {"id": "", "name": "", "arguments": ""}

            # Fields only present in the FIRST chunk for this call
            if tc_delta.id:
                tool_calls[idx]["id"] += tc_delta.id
            if tc_delta.function and tc_delta.function.name:
                tool_calls[idx]["name"] += tc_delta.function.name

            # Arguments accumulate across MANY chunks
            if tc_delta.function and tc_delta.function.arguments:
                tool_calls[idx]["arguments"] += tc_delta.function.arguments

    # Parse arguments JSON after accumulation
    result = {}
    for idx, tc in tool_calls.items():
        result[idx] = {
            "id":         tc["id"],
            "name":       tc["name"],
            "arguments":  tc["arguments"],          # Raw JSON string
            "args_parsed": json.loads(tc["arguments"]) if tc["arguments"] else {},
        }
    return result
```

---

## 5.3 Multiple Parallel Tool Calls

The model can request multiple tool calls simultaneously. Each has a different `index`:

```
# Two parallel tool calls streaming:
Chunk 1: tool_calls=[{index:0, id:"call_abc", name:"get_weather", arguments:""}]
Chunk 2: tool_calls=[{index:0, arguments:'{"city"'}]
Chunk 3: tool_calls=[{index:1, id:"call_def", name:"calculate",  arguments:""}]
Chunk 4: tool_calls=[{index:0, arguments:': "Tokyo"}'}]
Chunk 5: tool_calls=[{index:1, arguments:'{"expr'}]
Chunk 6: tool_calls=[{index:1, arguments:'ession": "2+2"}'}]
Chunk 7: finish_reason="tool_calls"
```

The accumulator pattern handles this naturally via the `index` key.

---

## 5.4 Converting Accumulated Tool Calls to the Message Format

After accumulation, convert to the format required for the assistant message:

```python
def make_assistant_message(text: str, accumulated_tool_calls: dict) -> dict:
    """Build the assistant message to append to conversation history."""
    tool_calls_list = [
        {
            "id":   tc["id"],
            "type": "function",
            "function": {
                "name":      tc["name"],
                "arguments": tc["arguments"]  # Keep as raw JSON string (not parsed)
            }
        }
        for tc in accumulated_tool_calls.values()
    ]

    return {
        "role":       "assistant",
        "content":    text if text else None,  # Can be None if model only calls tools
        "tool_calls": tool_calls_list
    }


def make_tool_result_messages(accumulated_tool_calls: dict, results: dict) -> list[dict]:
    """Build tool result messages to append after execution."""
    return [
        {
            "role":        "tool",
            "tool_call_id": tc["id"],
            "content":     str(results.get(tc["name"], "Error: tool not found"))
        }
        for tc in accumulated_tool_calls.values()
    ]
```

---

## 5.5 Streaming Tool Call Arguments in Real Time

For very long tool call arguments (e.g., writing a long document), you might want to process them as they stream rather than waiting for the complete JSON:

```python
def stream_progressive_args(stream, callback=None) -> dict:
    """
    Stream tool call arguments and optionally process them progressively.
    Useful when args contain long text like document content.
    """
    tool_calls = {}

    for chunk in stream:
        delta = chunk.choices[0].delta
        if not delta.tool_calls:
            continue

        for tc in delta.tool_calls:
            idx = tc.index
            if idx not in tool_calls:
                tool_calls[idx] = {"id": "", "name": "", "arguments": ""}

            new_args = tc.function.arguments if tc.function else ""
            if new_args:
                tool_calls[idx]["arguments"] += new_args
                # Progressive callback — process partial args as they arrive
                if callback:
                    callback(idx, tool_calls[idx]["arguments"])

    return tool_calls
```

---

## 5.6 Error Handling for Malformed Arguments

Sometimes models generate invalid JSON in tool call arguments:

```python
import json

def safe_parse_args(arguments_str: str) -> tuple[dict, str | None]:
    """
    Safely parse tool call arguments.
    Returns (parsed_dict, error_message_or_None).
    """
    if not arguments_str:
        return {}, None

    try:
        return json.loads(arguments_str), None
    except json.JSONDecodeError as e:
        # Try to recover partial JSON
        try:
            # Sometimes arguments are truncated — try with closing brace
            recovered = json.loads(arguments_str + "}")
            return recovered, f"Recovered truncated JSON (added closing brace)"
        except:
            return {}, f"JSON parse error: {e}. Raw: {arguments_str[:100]}"
```

---

## 📌 Key Takeaways

1. **Tool call JSON is delivered across multiple chunks** — always accumulate before parsing
2. **Index-based accumulation** — use `tc_delta.index` to track parallel tool calls
3. **Fields appear once**: `id` and `name` in the first chunk; `arguments` across many chunks
4. **Never `json.loads()` until the stream ends** (or `finish_reason == "tool_calls"`)
5. **Multiple parallel calls** — model sets `finish_reason: "tool_calls"` even with 2+ simultaneous calls
6. **Invalid JSON** — wrap `json.loads()` in try/except; models occasionally generate malformed args
7. **Keep `arguments` as raw string** in the assistant message — don't pass the parsed dict
