# 04 — Streaming in Agent Loops

> *Stream the model's reasoning while your agent continues to act.*

---

## 4.1 The Problem: Streaming + Tools

A simple agent loop without streaming blocks on each step:

```
Step 1: call model   → wait 5s → full response → detect tool call
Step 2: execute tool → wait 1s → result
Step 3: call model   → wait 5s → full response → detect "stop"

Total wall time: 11s — user sees nothing for 5s at a time
```

With streaming in the agent loop:

```
Step 1: call model (streaming)
        → 200ms: user sees "I'll search for..."
        → 1200ms: tool call detected → execute tool immediately
Step 2: tool executes (1s)
Step 3: call model (streaming)
        → 200ms: user sees "Based on the results..."

Total perceived wait: ~200ms before first text, tool runs in parallel
```

---

## 4.2 Agent Loop with Streaming

```python
from openai import OpenAI
import json

client = OpenAI()

def run_streaming_agent(
    messages: list[dict],
    tools: list[dict],
    tool_executor: dict,   # {"tool_name": callable}
    max_steps: int = 10,
) -> str:
    """
    Agent loop where each LLM call streams.
    Tool calls are detected mid-stream and executed.
    """
    for step in range(max_steps):
        print(f"\n🤖 Step {step+1}:")

        # ── Stream the model's response ────────────────────────────────────
        full_text = ""
        tool_calls_acc = {}
        finish_reason = None

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason or finish_reason

            # Stream text to user (agent's "thought")
            if delta.content:
                full_text += delta.content
                print(delta.content, end="", flush=True)

            # Accumulate tool call arguments
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:                       tool_calls_acc[idx]["id"]        += tc.id
                    if tc.function.name:            tool_calls_acc[idx]["name"]      += tc.function.name
                    if tc.function.arguments:       tool_calls_acc[idx]["arguments"] += tc.function.arguments

        print()

        # ── Normal stop: agent is done ─────────────────────────────────────
        if finish_reason == "stop":
            messages.append({"role": "assistant", "content": full_text})
            return full_text

        # ── Tool call: execute and continue loop ───────────────────────────
        elif finish_reason == "tool_calls":
            # Add assistant message with tool_calls to history
            tool_calls_list = []
            for idx, tc in tool_calls_acc.items():
                tool_calls_list.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]}
                })
            messages.append({
                "role": "assistant",
                "content": full_text if full_text else None,
                "tool_calls": tool_calls_list
            })

            # Execute each tool
            for tc in tool_calls_list:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                print(f"   🔧 Executing {name}({args})")

                executor = tool_executor.get(name)
                result = executor(**args) if executor else f"Tool '{name}' not found"
                print(f"   ✅ Result: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result)
                })

    return "Max steps reached."
```

---

## 4.3 Streaming Agent Thoughts to Users

In production agents, you typically want to show:
1. **Thinking** — what the model "says" before calling a tool
2. **Action** — which tool was called and with what args
3. **Observation** — what the tool returned
4. **Final answer** — the model's conclusion

```python
class StreamingAgentUI:
    """Formats streaming agent output for display."""

    def on_text(self, text: str):
        """Called for each streamed text chunk."""
        print(f"\033[94m{text}\033[0m", end="", flush=True)  # Blue text

    def on_tool_call(self, name: str, args: dict):
        """Called when a tool call is detected."""
        print(f"\n\n\033[93m🔧 Tool: {name}({args})\033[0m")  # Yellow

    def on_observation(self, result: str):
        """Called when tool result is available."""
        print(f"\033[92m📋 Result: {result[:100]}\033[0m\n")  # Green

    def on_final_answer(self, answer: str):
        """Called when agent reaches final stop."""
        print(f"\n\033[1m✅ Final answer:\033[0m {answer[:200]}")
```

---

## 4.4 Partial Output Processing

Sometimes you can start processing a streamed response before it's complete:

```python
def stream_with_early_termination(prompt: str, stop_phrase: str) -> str:
    """
    Stream and stop early if a specific phrase is detected.
    Useful for: detecting "I don't know", "Error:", confidence markers, etc.
    """
    accumulated = ""
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        accumulated += content
        print(content, end="", flush=True)

        # Abort stream early if model expresses uncertainty
        if stop_phrase.lower() in accumulated.lower():
            stream.close()  # Stop receiving from server
            print(f"\n   [Stopped early: detected '{stop_phrase}']")
            break

    return accumulated
```

---

## 4.5 Multi-Turn Streaming Agent

```python
def streaming_chat_agent(tools: list[dict], tool_executor: dict):
    """
    Interactive multi-turn chat with streaming.
    Maintains conversation history across turns.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Think step by step."}
    ]

    print("🤖 Streaming Agent (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break

        messages.append({"role": "user", "content": user_input})

        print("Agent: ", end="")
        response = run_streaming_agent(
            messages=messages,
            tools=tools,
            tool_executor=tool_executor
        )
        # (messages is updated in-place by run_streaming_agent)
        print()
```

---

## 4.6 Streaming vs. Non-Streaming in Agent Loops

| Aspect | Non-streaming | Streaming |
|---|---|---|
| User experience | Long silence between steps | Immediate feedback per step |
| Tool call detection | After full response | During stream (as arguments arrive) |
| Error recovery | After full timeout | Can abort early |
| Complexity | Simple | Requires accumulation logic |
| Token cost | Identical | Identical |
| Memory usage | One large response | Small chunks (lower peak) |

---

## 📌 Key Takeaways

1. **Streaming in agent loops** = stream each step, handle `finish_reason` to decide next action
2. **Tool calls arrive mid-stream** — `arguments` must be accumulated across chunks
3. **Show agent "thoughts"** during streaming — don't hide the model's reasoning from users
4. **Early termination** — you can `stream.close()` to stop consuming tokens mid-response
5. **Maintain history correctly**: add `{role: "assistant", tool_calls: [...]}` then `{role: "tool", ...}`
6. **Multi-turn works identically** — just keep appending to `messages` across turns
