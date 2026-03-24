# 01 — The LLM Landscape

> *Before you can choose the right model, you need to know what's out there — specs, strengths, and trade-offs.*

---

## 1.1 The Major LLM Families (2024–2025)

### OpenAI GPT Family

| Model | Context | Strengths | Best For |
|---|---|---|---|
| `gpt-4o` | 128k | All-round best, vision, fast | Production agents, reasoning, code |
| `gpt-4o-mini` | 128k | 15× cheaper, nearly as good | High-volume operations, bulk tasks |
| `o1` | 128k | Best reasoning, slow | Hard math, logic, complex planning |
| `o1-mini` | 128k | Cheaper reasoning | Moderate reasoning tasks |

### Anthropic Claude Family

| Model | Context | Strengths | Best For |
|---|---|---|---|
| `claude-3-5-sonnet` | 200k | Best coding, long context | Coding, analysis, long documents |
| `claude-3-5-haiku` | 200k | Fast, cheap, capable | High-volume, real-time apps |
| `claude-3-opus` | 200k | Very high quality | Complex reasoning, nuanced tasks |

### Google Gemini Family

| Model | Context | Strengths | Best For |
|---|---|---|---|
| `gemini-1.5-pro` | 2M tokens | Massive context, multimodal | Entire codebases, long videos |
| `gemini-1.5-flash` | 1M tokens | Fast, cheap, large context | Bulk document processing |
| `gemini-2.0-flash` | 1M tokens | Newest, multimodal, fast | Current best Gemini for agents |

### Meta Llama Family (Open Source)

| Model | Context | Strengths | Best For |
|---|---|---|---|
| `llama-3.1-405b` | 128k | Competitive with GPT-4, open | High-quality, open-source |
| `llama-3.1-70b` | 128k | Great quality-to-cost | Self-hosted production |
| `llama-3.1-8b` | 128k | Tiny, fast, very cheap | Edge/embedded, ultra-fast |

### Other Notable Models

| Model | Provider | Context | Notable Feature |
|---|---|---|---|
| `mixtral-8x22b` | Mistral AI | 64k | MoE, very fast |
| `mistral-large` | Mistral AI | 128k | Strong coding, EU data residency |
| `deepseek-v3` | DeepSeek | 128k | Best coding, very cheap |
| `qwen2.5-72b` | Alibaba | 128k | Multilingual, open source |
| `command-r-plus` | Cohere | 128k | Best RAG-optimized |

---

## 1.2 Context Window Deep Dive

Context window size determines what the model can "see" at once:

```
Model                    Context (tokens)   What it can hold
────────────────────────────────────────────────────────────
GPT-4o, GPT-4o-mini      128,000           ~500 pages
Claude 3.5 Sonnet        200,000           ~800 pages
Gemini 1.5 Flash         1,000,000         ~4,000 pages
Gemini 1.5 Pro           2,000,000         ~8,000 pages
```

**Practical guidance:**
- 128k ≈ a full novel, a large codebase, or a 2-hour meeting transcript
- 200k ≈ multiple books, a company's quarterly reports, a full technical spec
- 1M+ ≈ entire codebases (Linux kernel = ~17M tokens), hours of video transcripts

**Warning**: Larger context ≠ better retrieval. The "lost in the middle" problem means models often fail to use information buried in the middle of very long contexts.

---

## 1.3 Capability Dimensions

Think of models on five dimensions:

```
Dimension          What It Means
──────────────────────────────────────────────────────────────
REASONING          Multi-step logic, math, planning, chains of thought
CODING             Writing, debugging, explaining, refactoring code
INSTRUCTION-FOLLOW Adhering precisely to complex system prompts
LONG-CONTEXT       Accurately using information across a long document
MULTIMODAL         Processing images, audio, video in addition to text
```

### Rough Rankings (as of early 2025)

| Model | Reasoning | Coding | Instruction | Long-Context | Cost |
|---|---|---|---|---|---|
| `o1` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 💵💵💵💵💵 |
| `claude-3-5-sonnet` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 💵💵💵 |
| `gpt-4o` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 💵💵💵 |
| `gemini-1.5-pro` | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 💵💵 |
| `gpt-4o-mini` | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 💵 |
| `llama-3.1-70b` | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 💵 (self-hosted) |
| `claude-3-5-haiku` | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 💵 |

---

## 1.4 Key Benchmarks to Know

### MMLU (Massive Multitask Language Understanding)
Tests knowledge across 57 subjects. Good proxy for general capability.

### HumanEval / SWE-Bench
Tests code generation. HumanEval = write functions. SWE-Bench = fix GitHub issues.

### GPQA (Graduate-level Professional QA)
PhD-level science questions. Tests deep reasoning.

### LongBench
Tests retrieval and understanding over long contexts.

```python
# Rough benchmark scores (approximate, as of early 2025)
BENCHMARKS = {
    "gpt-4o":             {"MMLU": 88.7, "HumanEval": 90.2, "GPQA": 53.6},
    "claude-3-5-sonnet":  {"MMLU": 88.3, "HumanEval": 92.0, "GPQA": 59.4},
    "gemini-1.5-pro":     {"MMLU": 85.9, "HumanEval": 84.1, "GPQA": 46.2},
    "llama-3.1-70b":      {"MMLU": 83.6, "HumanEval": 80.5, "GPQA": 38.9},
    "gpt-4o-mini":        {"MMLU": 82.0, "HumanEval": 87.2, "GPQA": 40.2},
}
```

---

## 1.5 Choosing by Use Case — Quick Reference

| Use Case | Recommended Model | Reason |
|---|---|---|
| Complex reasoning / math | `o1` or `o1-mini` | Extended thinking |
| Code generation / review | `claude-3-5-sonnet` or `gpt-4o` | Best on SWE-Bench |
| Long document analysis | `claude-3-5-sonnet` (200k) | Large context + quality |
| Massive document / codebase | `gemini-1.5-pro` (2M) | Largest context |
| High-volume extraction | `gpt-4o-mini` or `claude-3-5-haiku` | Cheap + reliable |
| Real-time / low latency | `gemini-2.0-flash` or `claude-3-5-haiku` | Fastest |
| On-premise / privacy | `llama-3.1-70b` via Ollama/vLLM | Self-hosted |
| Multilingual | `gpt-4o` or `qwen2.5-72b` | Best multilingual |
| Image + text | `gpt-4o` or `gemini-1.5-pro` | Native multimodal |

---

## 📌 Key Takeaways

1. **No single "best" model** — best depends on task type, budget, latency, privacy
2. **Context window ≠ quality** — bigger window doesn't mean smarter reasoning
3. **mini/haiku/flash = 80% quality, 10% cost** — use them for bulk operations
4. **Benchmarks are starting points** — always test on your specific task
5. **Claude leads on coding**, **o1 leads on hard reasoning**, **Gemini leads on long context**
6. **Open source (Llama 3.1 70B) = competitive with GPT-4-turbo** — viable for self-hosting
