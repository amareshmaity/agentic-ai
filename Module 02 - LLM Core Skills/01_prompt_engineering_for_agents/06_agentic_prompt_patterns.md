# 06 — Agentic Prompt Patterns

> *Patterns specific to multi-step, tool-using agent loops — the bridge between theory and production agents.*

---

## 6.1 Why Agents Need Special Prompt Patterns

A single-turn LLM call has one input and one output. An **agent loop** has:
- Multiple turns (each requiring a well-structured prompt)
- Tool calls that generate observations (which must be re-injected)
- A scratchpad that grows over time (context window pressure)
- A need to know when to stop (termination logic)
- Human-in-the-loop checkpoints (requires specific prompt signals)

Standard prompting techniques (CoT, few-shot) are necessary but not sufficient. You also need **agent-specific patterns**.

---

## 6.2 Pattern 1: The ReAct Prompt Pattern

**ReAct** (Reasoning + Acting) is the foundational agentic prompt pattern. It interleaves:
- **Thought**: the agent's reasoning about what to do next
- **Action**: the tool call (or final answer decision)
- **Observation**: the tool result injected back into context

### System Prompt for ReAct

```
You are an intelligent agent. You solve problems by reasoning and using tools.

For each step, follow this EXACT format:

Thought: [Your reasoning about the current situation and what needs to happen next. 
          Be specific and honest about what you know vs. what you need to find out.]

Action: [TOOL_NAME]
Input: [tool arguments as required]

You will then receive:
Observation: [result from tool]

Continue this Thought → Action → Observation cycle until you can answer.
When you have enough information, respond with:

Thought: [reasoning for why you're done]
Final Answer: [your complete response to the user]
```

### Full ReAct Loop Example

```
User: "What is the market cap of Nvidia and how does it compare to Intel?"

Thought: I need the current market cap of both companies. This is financial 
         data that changes frequently, so I should search for it.
Action: web_search
Input: "Nvidia market cap 2025"

Observation: Nvidia (NVDA) market cap: approximately $3.2 trillion USD (as of early 2025)

Thought: Good, I have Nvidia's market cap. Now I need Intel's for comparison.
Action: web_search
Input: "Intel market cap 2025"

Observation: Intel (INTC) market cap: approximately $95 billion USD

Thought: I now have both data points:
         - Nvidia: ~$3.2 trillion
         - Intel: ~$95 billion
         Nvidia is approximately 33× larger than Intel by market cap.
         I have enough data to answer.
         
Final Answer: As of early 2025, Nvidia's market cap (~$3.2 trillion) is 
approximately 33× larger than Intel's (~$95 billion), reflecting Nvidia's 
dominance in the AI chip market and Intel's ongoing challenges.
```

---

## 6.3 Pattern 2: The Scratchpad Pattern

The scratchpad is the agent's "thinking space" — persistent internal notes that build up across the agent loop. Unlike in-context CoT (which is embedded in the conversation), a scratchpad is a **dedicated section** of the agent's state.

### System Prompt with Explicit Scratchpad

```
You have a scratchpad for private reasoning. Use it freely.

Format every response as:
<scratchpad>
[Your private step-by-step reasoning, notes, partial calculations, 
 plan updates, self-corrections. This is NOT shown to the user.]
</scratchpad>

<response>
[Only what the user should see — clean, formatted, final answer]
</response>
```

### Parsing Scratchpad in Code

```python
import re

def parse_agent_output(raw_output: str) -> dict:
    """Extract scratchpad and response from agent output."""
    scratchpad_match = re.search(r'<scratchpad>(.*?)</scratchpad>', raw_output, re.DOTALL)
    response_match = re.search(r'<response>(.*?)</response>', raw_output, re.DOTALL)
    
    return {
        "scratchpad": scratchpad_match.group(1).strip() if scratchpad_match else "",
        "response": response_match.group(1).strip() if response_match else raw_output,
        "raw": raw_output
    }

# Use in agent loop
output = call_llm(messages)
parsed = parse_agent_output(output["content"])
print("Agent thinking:", parsed["scratchpad"])  # for debugging
send_to_user(parsed["response"])                # clean output
```

---

## 6.4 Pattern 3: The Plan-Then-Execute Pattern

For complex multi-step tasks, have the agent **generate the full plan first**, then execute each step, then re-plan if a step fails.

### Two-Phase System Prompt

```
## Phase 1: Planning (called first)
Given the user's goal, generate a numbered execution plan.
List each step as an atomic action you can take using your available tools.
Format:
Plan:
1. [action] → [expected output]
2. [action] → [expected output]
...
N. [compile results] → final answer

Respond ONLY with the plan. Do not execute yet.

---

## Phase 2: Execution (called once per step)
You are now executing step {step_number} of your plan:
"{step_description}"

Previous results from completed steps:
{completed_steps_summary}

Execute this step now using the appropriate tool.
If the step fails, note the failure and propose a revised plan for remaining steps.
```

### Implementation

```python
class PlanExecuteAgent:
    def __init__(self):
        self.plan = []
        self.results = []
    
    def plan_phase(self, goal: str) -> list[str]:
        """Phase 1: Generate the plan."""
        response = call_llm([
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": f"Goal: {goal}\n\nGenerate a plan:"}
        ])
        # Parse the numbered list from response
        self.plan = parse_numbered_plan(response)
        return self.plan
    
    def execute_step(self, step_idx: int) -> str:
        """Phase 2: Execute one step of the plan."""
        step = self.plan[step_idx]
        summary = format_previous_results(self.results)
        
        response = call_llm([
            {"role": "system", "content": EXECUTION_SYSTEM_PROMPT.format(
                step_number=step_idx + 1,
                step_description=step,
                completed_steps_summary=summary
            )},
            {"role": "user", "content": "Execute this step now."}
        ], tools=TOOLS)
        
        result = handle_tool_calls_if_any(response)
        self.results.append({"step": step, "result": result})
        return result
    
    def run(self, goal: str) -> str:
        self.plan_phase(goal)
        for i in range(len(self.plan)):
            self.execute_step(i)
        return compile_final_answer(self.results)
```

---

## 6.5 Pattern 4: Tool-Use Prompt Pattern

How you describe tools to the agent is just as important as the tool implementation.

### Tool Description Best Practices

```python
# ❌ WEAK tool description
{
    "name": "search",
    "description": "Search the web",
    "parameters": {"query": {"type": "string"}}
}

# ✅ STRONG tool description
{
    "name": "web_search",
    "description": """Search the web for current information. 
    Use this tool when:
    - The question involves facts that change over time (prices, events, news)
    - The question asks about events after January 2024
    - You need to verify a specific claim with an authoritative source
    
    Do NOT use this tool when:
    - The answer is basic knowledge available in training data
    - The question is about math, logic, or reasoning
    - The question is creative/hypothetical
    
    For best results: use specific, targeted queries rather than broad searches.
    Example good query: "Tesla Q3 2024 earnings per share"
    Example bad query: "Tesla stuff"
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The specific search query. Be precise. Use quotes for exact phrases."
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results (1-10). Default 5. Use 3 for targeted lookups, 10 for broad research.",
                "default": 5
            }
        },
        "required": ["query"]
    }
}
```

### Tool Result Injection Pattern

After a tool executes, results must be injected back into the conversation correctly:

```python
def inject_tool_result(messages: list, tool_call_id: str, result: str) -> list:
    """Properly inject tool result back into the conversation."""
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": result  # Always a string, even if result is JSON (stringify it)
    })
    return messages

# Full loop pattern
def agent_loop(goal: str, max_steps: int = 10) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal}
    ]
    
    for step in range(max_steps):
        response = call_llm(messages, tools=TOOLS)
        msg = response.choices[0].message
        messages.append(msg)  # Add assistant message to history
        
        # Check for tool calls
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                result = execute_tool(tool_name, tool_args)
                inject_tool_result(messages, tool_call.id, str(result))
        
        # Check for termination
        elif msg.content:
            return msg.content  # Final answer
    
    return "Max steps reached"
```

---

## 6.6 Pattern 5: HITL (Human-in-the-Loop) Prompt Pattern

Design agents to **pause and request human approval** at critical decision points.

### Detecting HITL Trigger Points in Prompts

```
## When to Pause for Human Approval

Before taking any of these actions, OUTPUT the text "AWAITING_HUMAN_APPROVAL" 
followed by a description of what you plan to do and why.

Actions requiring approval:
- Sending any email or notification
- Writing to or deleting any file
- Making any API call that modifies external state
- Any action with monetary cost > $10
- Any action that cannot be easily reversed

Example:
AWAITING_HUMAN_APPROVAL
I'm about to send the following email to john@company.com:
Subject: Q4 Report
Body: [full email text]
Reason: The user asked me to send the Q4 report automatically.
Waiting for your confirmation to proceed...
```

### Parsing HITL Signal in Code

```python
HITL_SIGNAL = "AWAITING_HUMAN_APPROVAL"

def run_agent_with_hitl(goal: str) -> str:
    messages = build_initial_messages(goal)
    
    while True:
        response = call_llm(messages)
        output = response.choices[0].message.content
        
        if HITL_SIGNAL in output:
            # Parse what the agent wants to do
            action_description = output.split(HITL_SIGNAL)[1].strip()
            
            # Show to human
            print(f"\n⚠️  AGENT PAUSE — Approval Required:")
            print(action_description)
            
            # Get human decision
            decision = input("\nApprove? (yes/no/modify): ").strip().lower()
            
            if decision == "yes":
                messages.append({"role": "user", "content": "Approved. Proceed."})
            elif decision == "no":
                messages.append({"role": "user", "content": "Rejected. Do not take this action."})
            else:
                modification = input("Describe modification: ")
                messages.append({"role": "user", "content": f"Modified instruction: {modification}"})
        
        elif response.choices[0].message.tool_calls:
            handle_tool_calls(messages, response)
        
        else:
            return output  # Final answer
```

---

## 6.7 Pattern 6: Context Compression Prompt Pattern

When agent context gets long, use this pattern to compress it before it overflows.

### Compression Trigger

```python
def should_compress(messages: list, model: str = "gpt-4o-mini") -> bool:
    """Check if context is getting too long and needs compression."""
    total_tokens = count_tokens(messages, model)
    context_limit = 128_000  # gpt-4o-mini
    compression_threshold = 0.7  # Compress when 70% full
    return total_tokens > context_limit * compression_threshold

def compress_context(messages: list) -> list:
    """Summarize old messages to free up context space."""
    # Keep system prompt and last 3 messages intact
    system = messages[0]
    recent = messages[-3:]
    to_compress = messages[1:-3]
    
    if not to_compress:
        return messages
    
    # Ask LLM to summarize the middle messages
    summary_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Summarize the following agent conversation history into a compact 
            paragraph that preserves: (1) the original goal, (2) key facts discovered, 
            (3) actions taken and their results, (4) current state.

            History:
            {format_messages(to_compress)}
            
            Summary:"""
        }]
    )
    summary = summary_response.choices[0].message.content
    
    # Replace compressed messages with summary
    return [system, {"role": "assistant", "content": f"[Context Summary]: {summary}"}] + recent
```

---

## 6.8 Pattern 7: Multi-Agent Communication Prompt Pattern

When building multi-agent systems, agents need to communicate structured messages.

### Agent-to-Agent Message Format

```
## Communication Protocol

You receive tasks from other agents and must respond in this format:

TASK_ID: {task_id}
STATUS: COMPLETE | PARTIAL | FAILED
RESULT: {your output for this task}
CONFIDENCE: HIGH | MEDIUM | LOW
NEXT_NEEDED: {what information or action is still required, if any}
HANDOFF_TO: {agent_name if task should be passed to a specialist, else NONE}
```

### Orchestrator Prompt Pattern

```
## Your Role: Orchestrator

You manage a team of specialist agents:
- researcher_agent: Web search and information gathering
- analyst_agent: Data analysis and insight extraction  
- writer_agent: Content creation and formatting
- coder_agent: Code generation and debugging

Workflow:
1. Receive the user's request
2. Decompose it into subtasks
3. Assign each subtask to the appropriate specialist using: 
   ASSIGN: {agent_name} | TASK: {specific task description}
4. Wait for results (they arrive as "AGENT_RESULT: {agent_name}" messages)
5. Synthesize all results into a final response

Rules:
- Never do a specialist's job yourself — always delegate
- If a specialist returns STATUS: FAILED, reassign to another or ask user for help
- Track which subtasks are complete before assembling final output
```

---

## 📌 Key Takeaways

1. **ReAct**: Thought → Action → Observation is the universal agent loop prompt pattern
2. **Scratchpad**: give agents private reasoning space separate from user-visible output
3. **Plan-then-Execute**: generate the full plan first, execute step by step — better for complex multi-step tasks
4. **Tool descriptions**: descriptive, with "when to use" and "when NOT to use" → dramatically improves selection
5. **HITL signals**: agents should output standardized keywords before risky actions
6. **Context compression**: check token count at each step and compress when approaching limit
7. **Multi-agent communication**: use structured message formats for reliable agent-to-agent handoffs
