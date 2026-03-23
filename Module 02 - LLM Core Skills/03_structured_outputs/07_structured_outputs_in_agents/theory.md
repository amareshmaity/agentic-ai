# 07 — Structured Outputs in Agents

> *Structured outputs are the typed interface between an agent's reasoning and the code that acts on it — the key to building reliable agentic pipelines.*

---

## 7.1 Why Structured Outputs Are Fundamental to Agents

In an agentic system, the LLM doesn't just answer questions — it **makes decisions** that drive automated code. Those decisions need to be machine-readable:

```
Without structured outputs:
  LLM → "I think we should search for recent news about Tesla, then summarize it"
  Code → How do I parse this? What tool? What query? What format?
  Result → ❌ Brittle string parsing, fragile regex, frequent failures

With structured outputs:
  LLM → AgentAction(action="search", query="Tesla latest news", tool="web_search")
  Code → action.tool, action.query — typed, reliable, no parsing needed
  Result → ✅ Deterministic, typed agent decisions
```

**Structured outputs turn agent reasoning into a typed API.**

---

## 7.2 The Agent Decision Interface Pattern

The most powerful pattern: define a Pydantic model that represents the agent's decision at each step.

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class AgentStep(BaseModel):
    """One reasoning step in the agent loop."""
    
    thought: str = Field(
        description="Internal reasoning about the current situation and what to do next"
    )
    action: Literal["search", "calculate", "read_file", "write_file", "answer", "ask_user"]
    action_input: str = Field(
        description="The input/query/content for the chosen action"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence that this is the right action (0.0 = not sure, 1.0 = certain)"
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if user input is needed before proceeding"
    )
```

At each agent step, you call `.parse()` and get a `AgentStep` object — no string parsing needed.

---

## 7.3 Building a Structured Agent Loop

```python
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional
import json

client = OpenAI()

# ── Agent Decision Schema ────────────────────────────────────────────────
class AgentAction(BaseModel):
    thought: str
    action: Literal["search", "calculate", "answer"]
    action_input: str
    is_final: bool = False

# ── Tool Implementations ─────────────────────────────────────────────────
def search(query: str) -> str:
    """Mock search tool."""
    results = {
        "tesla stock":   "Tesla (TSLA) is currently trading at $248.50, up 2.1%.",
        "python salary": "Python developers earn $95K-$145K/year in the US (2024).",
    }
    for k, v in results.items():
        if k in query.lower():
            return v
    return f"Search results for '{query}': [Latest news and data about {query}]"

def calculate(expression: str) -> str:
    """Safe arithmetic calculator."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"

TOOLS = {"search": search, "calculate": calculate}

# ── Structured Agent Loop ────────────────────────────────────────────────
def run_structured_agent(user_query: str, max_steps: int = 5) -> str:
    messages = [
        {
            "role": "system",
            "content": """You are a helpful research assistant with search and calculation tools.
At each step, decide the best action. Set is_final=True when you have the complete answer."""
        },
        {"role": "user", "content": user_query}
    ]
    
    print(f"\n🤖 Agent starting: {user_query!r}")
    print("─" * 60)
    
    for step in range(max_steps):
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=AgentAction
        )
        
        action_obj = response.choices[0].message.parsed
        
        print(f"\n[Step {step+1}] Thought: {action_obj.thought}")
        print(f"          Action: {action_obj.action}({action_obj.action_input!r})")
        
        # Final answer
        if action_obj.action == "answer" or action_obj.is_final:
            print(f"\n✅ Final Answer: {action_obj.action_input}")
            return action_obj.action_input
        
        # Execute tool
        tool_result = TOOLS[action_obj.action](action_obj.action_input)
        print(f"          Result: {tool_result}")
        
        # Add to conversation
        messages.append({
            "role": "assistant",
            "content": json.dumps(action_obj.model_dump())
        })
        messages.append({
            "role": "user",
            "content": f"Tool result: {tool_result}\nContinue if needed or give final answer."
        })
    
    return "Max steps reached"
```

---

## 7.4 Multi-Step Planning with Structured Outputs

For complex tasks, generate a full plan as structured output before executing:

```python
from pydantic import BaseModel, Field
from typing import Literal

class PlanStep(BaseModel):
    step_number: int
    description: str
    tool_to_use: Literal["search", "calculate", "read_file", "write_report", "none"]
    expected_output: str
    depends_on: list[int] = Field(default_factory=list, description="Step numbers this step depends on")

class ExecutionPlan(BaseModel):
    goal: str
    steps: list[PlanStep]
    estimated_duration_mins: int
    can_parallelize: bool

def generate_plan(task: str) -> ExecutionPlan:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a task planner. Create detailed step-by-step plans."},
            {"role": "user",   "content": f"Create an execution plan for: {task}"}
        ],
        response_format=ExecutionPlan
    )
    return response.choices[0].message.parsed

# Usage
plan = generate_plan("Research the top 3 AI companies and write a comparison report")
print(f"Goal: {plan.goal}")
print(f"Steps: {len(plan.steps)}, Can parallelize: {plan.can_parallelize}")
for step in plan.steps:
    deps = f" (depends on steps {step.depends_on})" if step.depends_on else ""
    print(f"  Step {step.step_number}: {step.description}{deps}")
    print(f"    Tool: {step.tool_to_use} → {step.expected_output}")
```

---

## 7.5 Structured Output for Classification-Based Routing

In multi-agent systems, use structured output to route tasks to specialized agents:

```python
from pydantic import BaseModel, Field
from typing import Literal

class TaskRouter(BaseModel):
    task_type: Literal[
        "data_extraction",
        "code_generation",
        "research",
        "summarization",
        "calculation",
        "creative_writing",
        "question_answering"
    ]
    complexity: Literal["simple", "moderate", "complex"]
    requires_tools: bool
    suggested_agent: Literal["extractor_agent", "coder_agent", "researcher_agent", "general_agent"]
    estimated_tokens: int = Field(ge=100, le=16000)

def route_task(user_request: str) -> TaskRouter:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Classify this user request and determine how to route it to the best agent."
            },
            {"role": "user", "content": user_request}
        ],
        response_format=TaskRouter
    )
    return response.choices[0].message.parsed

# Example routing decisions
requests = [
    "Extract all email addresses from this document: ...",
    "Write a Python function to sort a linked list",
    "What is the current market cap of NVIDIA?"
]

for req in requests:
    route = route_task(req)
    print(f"Request: {req[:55]!r}...")
    print(f"  → Agent: {route.suggested_agent} | Type: {route.task_type} | Complex: {route.complexity}")
```

---

## 7.6 Structured Observations — Typed Tool Results

Not only decisions, but tool results can use structured output:

```python
from pydantic import BaseModel, Field
from typing import Optional

class SearchObservation(BaseModel):
    """Structured result from a web search tool."""
    query: str
    num_results: int
    top_results: list[str]
    key_facts: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    source_urls: list[str] = Field(default_factory=list)

class ToolObservation(BaseModel):
    """Generic wrapper for any tool result."""
    tool_name: str
    success: bool
    result_summary: str
    data: Optional[dict] = None
    error: Optional[str] = None
    execution_time_ms: float

def execute_tool_structured(tool_name: str, tool_input: str) -> ToolObservation:
    """Execute a tool and return a structured observation."""
    import time
    start = time.perf_counter()
    
    try:
        # Execute the tool (mock here)
        raw_result = f"Results for '{tool_input}' via {tool_name}: [data here]"
        elapsed = (time.perf_counter() - start) * 1000
        
        return ToolObservation(
            tool_name=tool_name,
            success=True,
            result_summary=raw_result[:200],
            execution_time_ms=elapsed
        )
    except Exception as e:
        return ToolObservation(
            tool_name=tool_name,
            success=False,
            result_summary="Tool execution failed",
            error=str(e),
            execution_time_ms=(time.perf_counter() - start) * 1000
        )
```

---

## 7.7 Memory and State Management with Structured Outputs

Agents need persistent state. Structure that too:

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class MemoryItem(BaseModel):
    """One item in agent working memory."""
    key: str
    value: str
    source: str          # Which step added this
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: str       # ISO format

class AgentState(BaseModel):
    """Full agent state — serializable to JSON."""
    task: str
    status: Literal["running", "paused", "complete", "error"]
    steps_taken: int = 0
    memory: list[MemoryItem] = Field(default_factory=list)
    final_answer: Optional[str] = None
    
    def add_memory(self, key: str, value: str, source: str, confidence: float = 0.9):
        self.memory.append(MemoryItem(
            key=key, value=value, source=source,
            confidence=confidence, timestamp=datetime.now().isoformat()
        ))
    
    def get_memory(self, key: str) -> Optional[str]:
        for item in reversed(self.memory):  # Newest first
            if item.key == key:
                return item.value
        return None
    
    def to_context_string(self) -> str:
        """Format memory as context for the LLM."""
        if not self.memory:
            return "No previous findings."
        return "\n".join([f"- {m.key}: {m.value}" for m in self.memory[-10:]])  # Last 10 items

# Full agent with state
state = AgentState(task="Research Python web frameworks", status="running")
state.add_memory("django_users", "~2M monthly active users", source="step_1")
state.add_memory("fastapi_growth", "200% YoY growth since 2021", source="step_2")

print(f"State: {state.status} | Steps: {state.steps_taken}")
print(f"Memory context:\n{state.to_context_string()}")

# Serialize entire state for persistence
state_json = state.model_dump_json(indent=2)
print(f"\nSerializable state ({len(state_json)} chars)")
```

---

## 7.8 Complete Production Agent

```python
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional
import json

client = OpenAI()

class ResearchStep(BaseModel):
    thought: str
    action: Literal["search", "compile_findings", "answer"]
    search_query: Optional[str] = None
    final_report: Optional[str] = None
    key_finding: Optional[str] = None

def research_agent(topic: str) -> str:
    """A minimal but complete research agent using structured outputs."""
    messages = [
        {
            "role": "system",
            "content": f"""Research agent. Topic: {topic}
Steps:
1. Search for key information (action='search' with search_query)
2. Add more searches if needed
3. When ready, give the answer (action='answer' with final_report)"""
        },
        {"role": "user", "content": f"Research this topic thoroughly: {topic}"}
    ]
    
    findings = []
    
    for step_num in range(6):
        r = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=ResearchStep
        )
        step = r.choices[0].message.parsed
        
        if step.action == "answer" and step.final_report:
            return step.final_report
        
        # Execute search (mock)
        if step.action == "search" and step.search_query:
            result = f"[Search: {step.search_query}] → Key info about {step.search_query}: data found."
            findings.append(result)
            messages.append({"role": "assistant", "content": json.dumps(step.model_dump())})
            messages.append({"role": "user",      "content": f"Search result: {result}\nContinue researching or give final answer."})
    
    return "Research incomplete — max steps reached"
```

---

## 📌 Key Takeaways

1. **Agent decisions are typed** — `AgentAction` Pydantic model replaces free-text decisions
2. **Parse agent steps with `.parse()`** — no string parsing, direct attribute access
3. **Plan as structured output** — generate full `ExecutionPlan` before executing
4. **Task routing** — classification Pydantic models route to specialized agents
5. **Structured observations** — tool results wrapped in typed `ToolObservation` models
6. **Serializable state** — `AgentState` Pydantic model makes agent memory persitable as JSON
7. **Structured outputs = typed API between reasoning and code** — the foundation of reliable agents
