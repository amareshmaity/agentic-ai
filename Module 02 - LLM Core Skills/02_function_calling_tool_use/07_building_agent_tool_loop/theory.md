# 07 — Building a Production Agent Tool Loop

> *Bringing it all together — a complete, robust, observable agent loop from scratch.*

---

## 7.1 The Agent Tool Loop — Full Architecture

A production agent tool loop is more than just "call LLM, execute tool, repeat." It needs:

```
┌────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION AGENT LOOP                           │
│                                                                    │
│  1. INPUT VALIDATION      ← sanitize, token check                 │
│  2. SYSTEM PROMPT ASSEMBLY← dynamic context, tool definitions      │
│  3. LLM CALL              ← with retry + timeout                  │
│  4. RESPONSE PARSING      ← extract tool calls or final answer     │
│  5. TOOL EXECUTION        ← with validation, retry, circuit breaker│
│  6. RESULT INJECTION      ← wrap results, truncate if needed       │
│  7. CONTEXT MANAGEMENT    ← compress if approaching limit          │
│  8. TERMINATION CHECK     ← max steps, goal achieved, error        │
│  9. LOGGING & TRACING     ← every step observable                 │
│  10. OUTPUT VALIDATION    ← final answer meets expectations        │
└────────────────────────────────────────────────────────────────────┘
```

---

## 7.2 AgentState — Managing Agent State Across Steps

```python
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

class AgentStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_STEPS = "max_steps"
    AWAITING_HUMAN = "awaiting_human"

@dataclass
class AgentState:
    # Core state
    goal: str
    messages: list = field(default_factory=list)
    status: AgentStatus = AgentStatus.RUNNING
    
    # Tracking
    step_count: int = 0
    max_steps: int = 10
    tool_calls_made: int = 0
    
    # Results
    final_answer: str = ""
    tool_results: list = field(default_factory=list)
    
    # Cost tracking
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Timing
    start_time: float = field(default_factory=lambda: __import__("time").time())
    
    @property
    def elapsed_seconds(self) -> float:
        import time
        return time.time() - self.start_time
    
    @property
    def estimated_cost_usd(self) -> float:
        # GPT-4o-mini pricing
        return (self.input_tokens * 0.00015 + self.output_tokens * 0.0006) / 1000
    
    def should_stop(self) -> bool:
        return (
            self.step_count >= self.max_steps or 
            self.status != AgentStatus.RUNNING
        )
```

---

## 7.3 Complete Production Agent Class

```python
import os, json, time, logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("prod_agent")

class ProductionAgent:
    """
    A production-ready agent tool loop with:
    - Retry logic for LLM calls
    - Tool validation and error handling
    - Context window management
    - Full observability logging
    - Configurable termination
    """
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list,
        tool_map: dict,
        model: str = "gpt-4o-mini",
        max_steps: int = 10,
        max_context_tokens: int = 100_000,
        temperature: float = 0.1
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_map = tool_map
        self.model = model
        self.max_steps = max_steps
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature
        self.client = OpenAI()
    
    def run(self, user_input: str) -> AgentState:
        """Run the agent on a user input. Returns final AgentState."""
        
        state = AgentState(goal=user_input, max_steps=self.max_steps)
        state.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_input}
        ]
        
        logger.info(f"[{self.name}] Starting run | goal='{user_input[:50]}...'")
        
        while not state.should_stop():
            state.step_count += 1
            logger.info(f"[{self.name}] Step {state.step_count}/{self.max_steps}")
            
            # ── Step 1: Check context window ─────────────────────────────
            self._manage_context(state)
            
            # ── Step 2: Call LLM ─────────────────────────────────────────
            response = self._call_llm_with_retry(state)
            if response is None:
                state.status = AgentStatus.FAILED
                state.final_answer = "ERROR: LLM call failed after all retries."
                break
            
            # Track token usage
            if response.usage:
                state.input_tokens  += response.usage.prompt_tokens
                state.output_tokens += response.usage.completion_tokens
            
            msg = response.choices[0].message
            state.messages.append(msg)
            
            # ── Step 3: Handle response ───────────────────────────────────
            finish_reason = response.choices[0].finish_reason
            
            if finish_reason == "tool_calls":
                tool_results = self._execute_tools(msg.tool_calls, state)
                state.messages.extend(tool_results)
                
            elif finish_reason in ("stop", None):
                state.final_answer = msg.content or ""
                state.status = AgentStatus.COMPLETED
                logger.info(f"[{self.name}] Completed in {state.step_count} steps | cost=${state.estimated_cost_usd:.4f}")
                break
            
            elif finish_reason == "length":
                logger.warning(f"[{self.name}] Hit max_tokens at step {state.step_count}")
                state.final_answer = (msg.content or "") + "\n[Response truncated — output was too long]"
                state.status = AgentStatus.COMPLETED
                break
        
        if state.should_stop() and state.status == AgentStatus.RUNNING:
            state.status = AgentStatus.MAX_STEPS
            state.final_answer = f"Max steps ({self.max_steps}) reached without completing the goal."
        
        return state
    
    def _call_llm_with_retry(self, state: AgentState, max_retries: int = 3):
        """Call LLM with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=state.messages,
                    tools=self.tools if self.tools else None,
                    tool_choice="auto" if self.tools else None,
                    temperature=self.temperature,
                    max_tokens=2048,
                    timeout=60
                )
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}. Waiting {wait}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait)
        return None
    
    def _execute_tools(self, tool_calls, state: AgentState) -> list:
        """Execute all tool calls and return result messages."""
        results = []
        
        for tc in tool_calls:
            tool_name = tc.function.name
            state.tool_calls_made += 1
            
            # Validate tool exists
            if tool_name not in self.tool_map:
                result_content = f"ERROR: Tool '{tool_name}' does not exist. Available tools: {list(self.tool_map.keys())}"
                logger.error(f"Invalid tool requested: {tool_name}")
            else:
                # Parse and validate arguments
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    result_content = f"ERROR: Could not parse tool arguments: {e}"
                    args = None
                
                if args is not None:
                    start = time.time()
                    try:
                        result_content = str(self.tool_map[tool_name](**args))
                        duration = time.time() - start
                        logger.info(f"  Tool {tool_name}({args}) → {len(result_content)} chars in {duration:.2f}s")
                    except Exception as e:
                        result_content = f"ERROR executing {tool_name}: {str(e)}"
                        logger.error(f"  Tool {tool_name} failed: {e}")
                    
                    # Truncate long results to protect context
                    if len(result_content) > 6000:
                        result_content = result_content[:6000] + "\n[...truncated]"
            
            state.tool_results.append({"tool": tool_name, "result": result_content[:200]})
            results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_content
            })
        
        return results
    
    def _manage_context(self, state: AgentState):
        """Compress context if approaching token limit."""
        # Rough estimate: 4 chars ≈ 1 token
        estimated_tokens = sum(
            len(str(m.get("content", ""))) // 4 
            for m in state.messages
        )
        
        if estimated_tokens > self.max_context_tokens * 0.75:
            logger.info("Context approaching limit — compressing old messages")
            state.messages = self._compress_messages(state.messages)
    
    def _compress_messages(self, messages: list) -> list:
        """Keep system + summarize middle + keep last 4 messages."""
        if len(messages) <= 6:
            return messages
        
        system = messages[0]
        recent = messages[-4:]
        to_compress = messages[1:-4]
        
        # Build summary of compressed messages
        history_text = "\n".join([
            f"{m['role'].upper()}: {str(m.get('content', ''))[:200]}"
            for m in to_compress
        ])
        
        summary_response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Summarize this agent conversation history in 200 words, preserving: goal, key discoveries, actions taken:\n{history_text}"}],
            max_tokens=300
        )
        summary = summary_response.choices[0].message.content
        
        return [
            system,
            {"role": "assistant", "content": f"[Context Summary - earlier steps]: {summary}"}
        ] + recent
```

---

## 7.4 Termination Conditions

A reliable agent must know when to stop:

```python
def check_termination(state: AgentState, response_content: str) -> bool:
    """Check multiple termination conditions."""
    
    # 1. Natural finish — LLM says it's done
    if "TASK_COMPLETE" in response_content:
        return True
    
    # 2. Max steps guard
    if state.step_count >= state.max_steps:
        logger.warning("Stopping: max steps reached")
        return True
    
    # 3. Cost guard — stop if spending too much
    if state.estimated_cost_usd > 1.00:  # $1 limit per run
        logger.warning("Stopping: cost limit reached")
        return True
    
    # 4. Time guard
    if state.elapsed_seconds > 300:  # 5 minute limit
        logger.warning("Stopping: time limit reached")
        return True
    
    # 5. Loop detection — if last 3 tool calls are identical
    if len(state.tool_results) >= 3:
        last_3 = [r["tool"] for r in state.tool_results[-3:]]
        if len(set(last_3)) == 1:  # All 3 the same tool
            logger.warning(f"Stopping: possible loop detected — {last_3[0]} called 3× in a row")
            return True
    
    return False
```

---

## 7.5 Putting It All Together — Full Demo

```python
# Define tools
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Use for any arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Python math expression"}
                },
                "required": ["expression"]
            }
        }
    }
]

# Tool implementations
def web_search(query: str) -> str:
    # Replace with real API (Tavily, SerpAPI, etc.)
    return f"[Mock search result for: {query}] — Python 3.13 released Oct 2024 with new features."

def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

TOOL_MAP = {"web_search": web_search, "calculator": calculator}

SYSTEM = """You are a research assistant. Use tools to find accurate information.
Think step by step. When you have a complete answer, respond with it directly."""

# Create and run agent
agent = ProductionAgent(
    name="ResearchBot",
    system_prompt=SYSTEM,
    tools=TOOLS_SCHEMA,
    tool_map=TOOL_MAP,
    max_steps=8
)

state = agent.run("What is the latest Python version and what is 25% of 3,840?")

print(f"\n{'='*60}")
print(f"STATUS:     {state.status.value}")
print(f"STEPS:      {state.step_count}")
print(f"TOOL CALLS: {state.tool_calls_made}")
print(f"TOKENS:     {state.input_tokens} in / {state.output_tokens} out")
print(f"COST:       ${state.estimated_cost_usd:.4f}")
print(f"TIME:       {state.elapsed_seconds:.1f}s")
print(f"\nFINAL ANSWER:\n{state.final_answer}")
```

---

## 📌 Key Takeaways

1. **AgentState** is the source of truth — track steps, tokens, cost, timing in one object
2. **LLM call with retry** — wrap every LLM call in exponential backoff
3. **Tool validation before execution** — check name, parse args, handle exceptions
4. **Context management** — monitor token usage and compress proactively at 75% limit
5. **Multi-condition termination** — max steps + cost + time + loop detection
6. **Log every step** — step number, tool used, result size, duration, cost
7. **Return graceful errors** — never let a single tool failure crash the entire run
