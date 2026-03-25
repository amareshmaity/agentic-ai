# Model Providers in LangChain

> *LangChain's key power: swap any LLM with zero chain code change. One interface, 100+ providers.*

---

## 🌐 Overview of Supported Providers

```
Cloud Providers (API-based):
    OpenAI    → ChatOpenAI          (GPT-4o, GPT-4o-mini)
    Anthropic → ChatAnthropic       (Claude 3.5 Sonnet, Haiku)
    Google    → ChatGoogleGenerativeAI (Gemini 1.5 Pro, Flash)
    Cohere    → ChatCohere
    Mistral   → ChatMistralAI
    Groq      → ChatGroq            (Llama 3, Mistral — ultra fast)

Local Models (run on your machine):
    Ollama    → ChatOllama          (Llama 3, Mistral, Phi, etc.)
    HuggingFace → HuggingFacePipeline
```

---

## 1️⃣ OpenAI — `ChatOpenAI`

```python
from langchain_openai import ChatOpenAI

# Standard setup
llm = ChatOpenAI(
    model="gpt-4o-mini",   # or "gpt-4o", "gpt-4-turbo"
    temperature=0,
    # api_key read from OPENAI_API_KEY env var
)

# Available models
# gpt-4o           → Best quality, most expensive
# gpt-4o-mini      → Great quality, very cheap — use for most tasks
# gpt-4-turbo      → Legacy
# gpt-3.5-turbo    → Fast & cheap, weaker reasoning

response = llm.invoke("Hello!")
print(response.content)
```

### OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"   # or "text-embedding-3-large"
)

vector = embeddings.embed_query("LangChain is a framework")
print(f"Embedding dimension: {len(vector)}")   # 1536
```

---

## 2️⃣ Anthropic — `ChatAnthropic`

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",  # Best quality
    # or "claude-3-haiku-20240307"       # Fast & cheap
    temperature=0,
    max_tokens=1000,
    # api_key read from ANTHROPIC_API_KEY env var
)

# Same interface as OpenAI
response = llm.invoke("What is LangChain?")
print(response.content)
```

### Claude Model Comparison

| Model | Speed | Quality | Cost | Best For |
|---|---|---|---|---|
| Claude 3.5 Sonnet | Medium | ⭐⭐⭐⭐⭐ | $$$ | Complex reasoning, coding |
| Claude 3 Haiku | Fast | ⭐⭐⭐ | $ | Simple tasks, high volume |
| Claude 3 Opus | Slow | ⭐⭐⭐⭐⭐ | $$$$ | Most complex tasks |

---

## 3️⃣ Google — `ChatGoogleGenerativeAI`

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   # Fast & cheap
    # or "gemini-1.5-pro"       # Best quality, 1M context
    temperature=0,
    # api_key read from GOOGLE_API_KEY env var
)

response = llm.invoke("What is LangChain?")
print(response.content)

# Gemini's superpower: 1M token context window
# Great for very long documents, codebases, etc.
```

### Google Embeddings

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
```

---

## 4️⃣ Groq — `ChatGroq` (Fastest Inference)

```python
from langchain_groq import ChatGroq

# Groq runs open-source models at extreme speeds (fast inference chip)
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Ultra fast
    # or "llama-3.1-70b-versatile" # Better quality
    # or "mixtral-8x7b-32768"
    temperature=0,
    # api_key read from GROQ_API_KEY env var
)

response = llm.invoke("Hello!")
print(response.content)
# Groq can do 800+ tokens/second vs ~50 for standard APIs
```

---

## 5️⃣ Ollama — `ChatOllama` (Local Models)

Run models **completely locally** — no API key, no cost, full privacy.

```bash
# Install Ollama first: https://ollama.com
ollama pull llama3.2        # Download model (~2GB)
ollama pull mistral         # Alternative
ollama serve                # Start server at localhost:11434
```

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",           # Must be pulled first
    temperature=0,
    base_url="http://localhost:11434"  # Default
)

response = llm.invoke("What is LangChain?")
print(response.content)  # Runs 100% locally!
```

### Popular Ollama Models

| Model | Size | Best For |
|---|---|---|
| `llama3.2` | 2GB | General use, fast |
| `llama3.1:8b` | 5GB | Good balance |
| `llama3.1:70b` | 40GB | High quality (needs GPU) |
| `mistral` | 4GB | Coding, reasoning |
| `phi3` | 2GB | Lightweight tasks |
| `codellama` | 4GB | Code generation |
| `nomic-embed-text` | 270MB | Embeddings locally |

---

## 🔄 Swapping Providers — Zero Code Change

This is the beauty of LangChain's unified interface:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# Build your chain ONCE
def build_chain(llm):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ])
    return prompt | llm | StrOutputParser()

# Swap the LLM — chain code doesn't change at all
llm_openai  = ChatOpenAI(model="gpt-4o-mini")
llm_claude  = ChatAnthropic(model="claude-3-haiku-20240307")
llm_gemini  = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm_local   = ChatOllama(model="llama3.2")

question = {"question": "What is LangChain?"}

# All use the exact same chain code
chain = build_chain(llm_openai)
print("OpenAI:", chain.invoke(question)[:50])

chain = build_chain(llm_claude)
print("Claude:", chain.invoke(question)[:50])

chain = build_chain(llm_local)
print("Local: ", chain.invoke(question)[:50])
```

---

## 📊 Provider Comparison

| Provider | Best Model | Context | Speed | Cost | Use When |
|---|---|---|---|---|---|
| **OpenAI** | gpt-4o-mini | 128k | Fast | $$ | Default choice |
| **Anthropic** | Claude 3.5 Sonnet | 200k | Medium | $$$ | Complex reasoning |
| **Google** | Gemini 1.5 Pro | 1M | Fast | $$ | Long documents |
| **Groq** | Llama 3.1 70b | 32k | Ultra-fast | $ | Speed-critical |
| **Ollama** | Llama 3.2 | 128k | Local | Free | Privacy, offline |

---

## 🔧 Environment Setup

```bash
# Install provider packages
pip install langchain-openai      # OpenAI
pip install langchain-anthropic   # Anthropic / Claude
pip install langchain-google-genai # Google / Gemini
pip install langchain-groq        # Groq
pip install langchain-ollama      # Ollama (local)
```

```env
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
GROQ_API_KEY=gsk_...
# Ollama: no key needed (local)
```

---

## ✅ Key Takeaways

- **One interface** works across all providers — only the import and model name changes
- **OpenAI `gpt-4o-mini`** is the best default — cheap, fast, capable
- **Anthropic Claude** excels at long-form reasoning and following complex instructions
- **Google Gemini 1.5 Pro** wins for extremely long context (1M tokens)
- **Groq** is fastest — great when latency is critical
- **Ollama** is free and private — use for sensitive data or cost-zero development

---

## ⬅️ Previous
[LLM vs ChatModel](./01_llm_vs_chatmodel.md)

## ➡️ Next
[PromptTemplate →](./03_prompt_template.md)
