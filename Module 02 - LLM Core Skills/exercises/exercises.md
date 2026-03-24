# 🏋️ Module 02 — LLM Core Skills: Exercises

> Practice problems and a mini-project to solidify every concept in the module.

---

## 📋 Instructions

- Exercises are grouped by topic and ordered from **warm-up → intermediate → challenge**
- The **Mini-Project** at the end integrates all 7 topics
- Work in a Jupyter notebook or `.py` file — create your own `solutions/` folder

---

## 1️⃣ Prompt Engineering for Agents

### Warm-Up
1. Write a `system_prompt` for a travel-booking agent that: (a) has a clear persona, (b) lists its available tools by name, (c) says what it should do when it's uncertain, and (d) formats output as JSON.
2. Given this bad system prompt — "You are an AI assistant. Answer questions." — rewrite it for an agent that manages a user's calendar.

### Intermediate
3. Create a few-shot prompt for an agent that classifies customer support tickets into categories: `billing`, `technical`, `account`, `other`. Include 3 examples and test it on 5 new tickets.
4. Add a **chain-of-thought** instruction to your classification prompt from #3. Compare outputs — does CoT improve accuracy on ambiguous tickets?

### Challenge
5. Design a **ReAct-format** prompt for an agent with three tools: `search_web`, `read_calendar`, `send_email`. The agent must: think before each action, explain why it chose the tool, and summarize at the end. Test it on: "Schedule a meeting with Alice for next Thursday and send her a confirmation email."
6. Try a **prompt injection attack** on your travel-booking agent from #1: craft a user message that tries to make the agent ignore its system prompt. Then add a defense to the system prompt that resists it.

---

## 2️⃣ Function Calling & Tool Use

### Warm-Up
1. Define an OpenAI tool schema for a `get_stock_price(ticker: str, currency: str = "USD")` function. Call it and handle the response.
2. Write a complete single-turn tool call: user asks for weather → model calls `get_weather` → you return a mock result → model gives final answer.

### Intermediate
3. Build a **multi-turn tool loop** that can handle up to 5 steps. The agent must:
   - Use `search` and `calculate` tools
   - Append `assistant` + `tool` messages correctly each turn
   - Stop when `finish_reason == "stop"`
4. Define **two tools** with overlapping capabilities. Send a prompt that should trigger a **parallel tool call** (both called in one response). Verify both are executed.

### Challenge
5. Build a tool that itself calls the LLM (`llm_judge(response: str) -> dict`) — an LLM-as-judge tool. Wire it into an agent loop so the agent validates its own answers before returning.
6. Port your tool loop to **Anthropic's tool use API**. Use the same logical tools but adapt the schema and response parsing. Verify you get the same final answer.

---

## 3️⃣ Structured Outputs

### Warm-Up
1. Define a `ProductReview` Pydantic model with fields: `product`, `rating` (1–5), `pros` (list), `cons` (list), `verdict`. Extract it from 3 free-text reviews using OpenAI's structured output mode.
2. Use `json_mode` to extract `{"name": str, "age": int, "city": str}` from 5 unstructured bio sentences.

### Intermediate
3. Build a pipeline that:
   - Accepts raw job postings (free text)
   - Extracts a `JobPosting` model: `title`, `company`, `required_skills (list)`, `salary_range`, `remote (bool)`
   - Validates all fields with Pydantic validators
   - Prints a structured report for 5 postings
4. Implement **output repair**: if the LLM returns invalid JSON, catch the error and re-prompt with the error message to get a corrected output. Test with a deliberately bad prompt that encourages partial JSON.

### Challenge
5. Use `instructor` to extract a **nested Pydantic model**: `ResearchPaper(title, authors: list[Author], abstract_summary, citations: list[Citation])`. Parse 3 paragraph-long paper descriptions.
6. Build a **structured classification agent** that routes customer questions to: `FAQ`, `HUMAN_AGENT`, or `AUTOMATED_RESPONSE`. The output must be a Pydantic model including `category`, `confidence` (0–1), and `reasoning`.

---

## 4️⃣ Context Window Management

### Warm-Up
1. Write a function `count_tokens(messages: list[dict], model: str) -> int` using `tiktoken`. Verify it against the OpenAI API's `usage.prompt_tokens`.
2. Given a 10-message conversation, implement a **fixed-window** context manager that keeps only the last N messages within a token budget.

### Intermediate
3. Implement a **sliding window with summarization**: when the window exceeds 3,000 tokens, summarize the oldest half before dropping it. Preserve the system prompt always.
4. Build a **token budget allocator** that accepts roles (`system`, `context`, `history`, `response_reserve`) and a total budget, then calculates max tokens per role with a priority order.

### Challenge
5. Simulate a **10-turn agent conversation** (use mock tool calls) and implement a context manager that: (a) keeps the system prompt, (b) summarizes older turns, (c) always keeps the last 3 full turns, (d) never exceeds 4,000 tokens.
6. Implement **RAG-style context injection**: given a user question, retrieve the 3 most relevant chunks from a 20-chunk document using `tiktoken` + cosine similarity on TF-IDF, then inject them into context.

---

## 5️⃣ LLM Selection Guide

### Warm-Up
1. Using the OpenAI and a free Gemini API key, run the same prompt (`"Summarize the history of AI in 3 sentences."`) on `gpt-4o-mini` and `gemini-1.5-flash`. Compare: latency, output quality, and cost.
2. Write a `ModelScorecard` function that scores a model on: speed, cost, context window, and tool support (1–5 each), then ranks them.

### Intermediate
3. Design a **task-to-model mapping** function: given `{"task": str, "max_cost_per_call": float, "needs_tools": bool, "max_latency_ms": int}`, return the best model from a predefined list with justification.
4. Run a **benchmark**: send the same 5 prompts to 2 models, measure latency + cost, and score outputs on a 1–5 rubric. Display as a comparison table.

### Challenge
5. Build a **dynamic model selector** that tracks observed latency and quality for each model over 10 requests and updates its recommendations based on real performance data.

---

## 6️⃣ LLM Routing & Fallback

### Warm-Up
1. Write a `token_router(messages, cheap_model, expensive_model, threshold=2000)` function that routes based on token count.
2. Implement a `retry_with_backoff(fn, max_retries=3)` decorator using `tenacity` with exponential backoff on `RateLimitError`.

### Intermediate
3. Implement a **circuit breaker** class with states: `CLOSED → OPEN (after 3 failures) → HALF_OPEN (after 30s) → CLOSED`. Test by injecting artificial failures.
4. Build a **3-provider fallback chain**: `gpt-4o-mini → claude-haiku → gemini-flash`. Use a mock that raises `APIError` on the first two. Verify the third is called.

### Challenge
5. Build a `ComposableRouter` that chains: `TokenRouter → KeywordRouter → CostRouter → FallbackRouter`. Each router can either handle the request or pass it to the next. Test with 10 diverse prompts.
6. Configure **LiteLLM Router** with 3 models, retry policy, and fallback chain. Run 20 simultaneous requests and display a metrics table (model used, latency, cost, fallback%).

---

## 7️⃣ Streaming Responses

### Warm-Up
1. Write a `stream_and_count(prompt)` function that streams a response and returns: full text, chunk count, TTFT, total latency.
2. Use `stream_options={"include_usage": True}` to get token counts from a streaming call. Compute and print the cost.

### Intermediate
3. Build a **streaming tool call handler**: stream a response that triggers a tool call, correctly accumulate the `arguments` JSON across chunks, and execute the tool after the stream ends.
4. Write an `AsyncStreamPool` that takes a list of prompts and streams all of them concurrently using `AsyncOpenAI` + `asyncio.gather`. Print each result labeled with its prompt index.

### Challenge
5. Build a **streaming agent loop** that: (a) streams reasoning text token-by-token, (b) detects tool calls mid-stream, (c) executes tools, (d) continues streaming. Support up to 5 steps.
6. Create a FastAPI endpoint `POST /chat/stream` that: accepts `{"messages": [...], "model": str}`, streams the response as SSE, handles client disconnect, and sends a final `done` event with token count and cost.

---

## 🏆 Mini-Project: Build an Agentic Research Assistant

> Integrate **all 7 topics** into a single production-quality system.

### Requirements

Your `ResearchAssistant` class must:

**Prompt Engineering**
- [ ] System prompt with clear persona, tool descriptions, reasoning format, and output schema
- [ ] Few-shot examples for when to use each tool

**Function Calling**
- [ ] At least 3 tools: `search_web(query)`, `summarize_text(text)`, `save_note(title, content)`
- [ ] Multi-step tool loop (up to 8 steps)

**Structured Outputs**
- [ ] Final answer must be a validated Pydantic model: `ResearchResult(topic, summary, sources, confidence, notes)`

**Context Window Management**
- [ ] Automatically summarize context when > 6,000 tokens
- [ ] Always preserve the system prompt

**LLM Selection**
- [ ] Use `gpt-4o-mini` for tool calls, `gpt-4o` for final synthesis
- [ ] Justify the model choice in a comment

**Routing & Fallback**
- [ ] Wrap all LLM calls with retry (3 attempts, exponential backoff)
- [ ] Circuit breaker on the primary model; fallback to `gpt-4o-mini` if primary fails

**Streaming**
- [ ] Stream all reasoning text token-by-token to the terminal
- [ ] In the final synthesis step, stream the structured output as it builds

### Deliverable

A single Python script or notebook `mini_project/research_assistant.py` that:
1. Accepts a research topic from the user
2. Runs an agentic loop (streaming output visible)
3. Returns a validated `ResearchResult`
4. Prints a summary table: steps taken, tools used, total cost, total latency

### Stretch Goals
- [ ] Expose it as a FastAPI `/research` streaming SSE endpoint
- [ ] Add a simple HTML page that renders the streamed output in the browser
- [ ] Add async support so 3 research topics run concurrently

---

## 📊 Self-Assessment Checklist

After completing the exercises, you should be able to answer **yes** to all of these:

- [ ] Can I write a production-quality agentic system prompt from scratch?
- [ ] Can I implement a complete tool-calling agent loop (multi-turn, multi-tool)?
- [ ] Can I enforce structured outputs with Pydantic across OpenAI, Anthropic, and Gemini?
- [ ] Can I implement a sliding-window context manager with summarization?
- [ ] Can I select the right model for a given cost/latency/quality constraint?
- [ ] Can I build a fallback chain with circuit breakers and retry logic?
- [ ] Can I stream LLM output token-by-token and handle tool calls mid-stream?
- [ ] Can I expose all of the above via a production-grade FastAPI SSE endpoint?

**If you can answer yes to all 8 — you are ready for Module 03: Building Agents with LangChain & LangGraph.** 🚀
