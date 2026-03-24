# 06 — Open Source Models

> *Open source LLMs give you privacy, cost control, and customization — at the price of infrastructure complexity.*

---

## 6.1 Why Open Source?

| Situation | Open Source Wins Because |
|---|---|
| **Data privacy** | Data never leaves your servers |
| **Regulatory compliance** | HIPAA, GDPR, SOC2 data residency requirements |
| **Cost at extreme scale** | Hardware amortizes over billions of calls |
| **Customization** | Fine-tune on proprietary data |
| **No vendor lock-in** | Switch hardware, stay with same weights |
| **Offline/edge** | Run without internet connection |

---

## 6.2 Top Open Source Models (2024–2025)

### Meta Llama 3.1 Series (Best Overall)

| Model | Parameters | Context | Quality Level |
|---|---|---|---|
| `llama-3.1-405b` | 405B | 128k | ≈ GPT-4-turbo |
| `llama-3.1-70b` | 70B | 128k | ≈ GPT-4o-mini (strong) |
| `llama-3.1-8b` | 8B | 128k | ≈ GPT-3.5 |

- **License**: Llama 3 Community License (commercial use allowed for most)
- **Best for**: General tasks, reasoning, instruction following
- **Where to run**: Groq (fastest), Together AI, Replicate, Ollama (local)

### Mistral / Mixtral Series

| Model | Parameters | Context | Specialty |
|---|---|---|---|
| `mistral-large` | ~47B | 128k | Strong coding, EU hosted |
| `mixtral-8x22b` | 141B (MoE) | 64k | Very fast, competitive quality |
| `mistral-7b` | 7B | 32k | Tiny, very fast |

- **License**: Apache 2.0 (fully open, commercial-friendly)
- **Best for**: European data residency, coding tasks
- **MoE (Mixture of Experts)**: 141B total parameters, only 39B active per token — efficient

### DeepSeek Models

| Model | Parameters | Context | Specialty |
|---|---|---|---|
| `deepseek-v3` | 671B (MoE) | 128k | Best coding, very cheap API |
| `deepseek-r1` | 671B (MoE) | 128k | Reasoning (o1 competitive) |

- **License**: MIT (fully open source)
- **Best for**: Coding tasks at low cost
- **Caveat**: Data may route through China — check for compliance

### Google Gemma Series

| Model | Parameters | Context | Notes |
|---|---|---|---|
| `gemma-2-27b` | 27B | 8k | Strong instruction following |
| `gemma-2-9b` | 9B | 8k | Efficient, great quality/size ratio |

- **License**: Gemma Terms of Use (commercial OK, no competing model training)

---

## 6.3 Running Models Locally with Ollama

Ollama is the easiest way to run LLMs locally:

```bash
# Install Ollama
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
# Windows: download from ollama.ai

# Pull a model
ollama pull llama3.1:8b          # 4.7 GB
ollama pull llama3.1:70b         # 40 GB — needs good GPU
ollama pull mistral:7b           # 4.1 GB
ollama pull deepseek-v2:16b      # 9 GB

# Run interactively
ollama run llama3.1:8b
```

### Using Ollama via OpenAI-Compatible API

```python
from openai import OpenAI

# Ollama serves an OpenAI-compatible API at localhost:11434
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Required but ignored
)

response = ollama_client.chat.completions.create(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Explain quantum entanglement briefly."}
    ],
    max_tokens=200
)

print(response.choices[0].message.content)
```

---

## 6.4 Cloud-Hosted Open Source — Best of Both Worlds

For production scale without managing GPU infrastructure:

```python
# Together AI — Llama, Mistral, Qwen, etc.
from openai import OpenAI

together_client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.getenv("TOGETHER_API_KEY")
)

response = together_client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "What is Python?"}],
    max_tokens=200
)

# ─────────────────────────────────────────────

# Groq — Fastest inference, OpenAI-compatible
from groq import Groq

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = groq_client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=[{"role": "user", "content": "What is Python?"}],
    max_tokens=200
)

# Speed metrics unique to Groq:
print(f"Queue time: {response.usage.queue_time:.3f}s")
print(f"Prompt time: {response.usage.prompt_time:.3f}s")
print(f"Completion time: {response.usage.completion_time:.3f}s")
```

---

## 6.5 Fine-Tuning Open Source Models

One of the biggest advantages over proprietary models:

```python
# Fine-tuning workflow (using HuggingFace + PEFT/LoRA)

# 1. Prepare dataset
training_data = [
    {"instruction": "Classify this email", "input": "...", "output": "SPAM"},
    # ... thousands of examples
]

# 2. Fine-tune with LoRA (Low-Rank Adaptation) — efficient on a single GPU
# pip install transformers peft trl datasets

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Configure LoRA (trains only 0.1-1% of parameters)
config = LoraConfig(
    r=8,             # Rank of update matrices
    lora_alpha=32,   # LoRA scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# trainable params: 6,815,744 || all params: 6,745,088,000 || trainable%: 0.10%
```

---

## 6.6 When NOT to Use Open Source

Open source isn't always the right choice:

| Constraint | Use Proprietary Instead |
|---|---|
| **Small team** | Infrastructure overhead is too high |
| **Low volume** | Cloud APIs cheaper than GPU hardware |
| **Best-in-class quality needed** | GPT-4o/Claude still lead on many tasks |
| **Fast iteration** | Model upgrades are automatic with APIs |
| **No ML expertise** | Operating inference servers requires skill |

**Rule of thumb**: Open source makes economic sense when > 1M tokens/day or strict privacy requirements.

---

## 📌 Key Takeaways

1. **Llama 3.1 70B** = commercial-grade quality, fully open, Apache-compatible
2. **Ollama** = easiest local setup; OpenAI-compatible API out of the box
3. **Groq** = fastest cloud inference for open models (300-750 tokens/sec)
4. **Together AI, Replicate** = easy cloud hosting without managing GPU servers
5. **Fine-tuning with LoRA** = train on your domain data with minimal GPU resources
6. **Cost breaks even at ~1M tokens/day** — below that, cloud APIs are usually cheaper
7. **DeepSeek-V3** = surprisingly competitive coding model, MIT licensed, very cheap API
