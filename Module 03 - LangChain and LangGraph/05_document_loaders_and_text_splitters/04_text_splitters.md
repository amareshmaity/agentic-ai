# Text Splitters

> *LLMs have a context window limit. Text splitters break large documents into smaller chunks that fit — while preserving enough context to make each chunk useful.*

---

## 🤔 Why Split Documents?

```
Problem:
    Context window of gpt-4o-mini ≈ 128,000 tokens
    But your users ask about ONE specific part of a 500-page PDF
    Sending all 500 pages = expensive AND noisy (LLM gets confused)

Solution — Chunk → Embed → Retrieve:
    Split into 1000-char chunks
    Embed each chunk as a vector
    At query time: retrieve only the 3-5 most relevant chunks
    → LLM gets focused, relevant context at low cost
```

---

## 🔢 Core Concepts

### chunk_size
Maximum number of characters (or tokens) per chunk.

### chunk_overlap
How many characters are shared between adjacent chunks. Prevents missing context that straddles a boundary.

```
Without overlap:
    Chunk 1: [====sentence A====][==start of B==]
    Chunk 2:                    [==end of B===][====sentence C====]
    → "start of B" and "end of B" are in different chunks
    → A query about B might miss the full sentence

With overlap (chunk_overlap=100):
    Chunk 1: [====sentence A====][==start of B==]
    Chunk 2:          [==start of B==][==end of B===][====sentence C====]
    → B appears in both chunks → always retrieved together
```

---

## 1️⃣ RecursiveCharacterTextSplitter — **Always Use This First**

The best general-purpose splitter. Tries to split on natural boundaries first:

```
Tries in order: ["\n\n", "\n", " ", ""]
  Step 1: Try to split on double newlines (paragraph boundaries)
  Step 2: If chunk still too big, split on single newlines
  Step 3: If still too big, split on spaces (word boundaries)
  Step 4: If still too big, split on individual characters
```

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Max characters per chunk
    chunk_overlap=200,      # Characters shared between consecutive chunks
    length_function=len,    # How to measure length (characters by default)
    is_separator_regex=False,
)

# Split from text
text = "Your long document text here..."
chunks = splitter.create_documents([text])

# Split from existing Documents (preserves metadata!)
docs = loader.load()     # List[Document]
chunks = splitter.split_documents(docs)

print(f"Original: {len(docs)} documents")
print(f"Chunks:   {len(chunks)} chunks")
print(f"Avg size: {sum(len(c.page_content) for c in chunks)//len(chunks)} chars")
```

### Custom Separators

```python
# For Python code — split on class/function boundaries
python_splitter = RecursiveCharacterTextSplitter(
    separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
    chunk_size=2000,
    chunk_overlap=200
)

# For Markdown
markdown_splitter = RecursiveCharacterTextSplitter(
    separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    chunk_size=1500,
    chunk_overlap=150
)
```

---

## 2️⃣ CharacterTextSplitter — Simple, Single Separator

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",    # ONLY splits on this (double newline)
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

chunks = splitter.split_documents(docs)
```

**Use when:** Your text has a clean, consistent separator (e.g., double newlines between sections).

**Avoid if:** Text paragraphs are uneven — some may still exceed `chunk_size`.

---

## 3️⃣ TokenTextSplitter — Token-Based (Matches LLM Context)

```python
from langchain_text_splitters import TokenTextSplitter

# Split by actual tokens (not characters)
splitter = TokenTextSplitter(
    chunk_size=256,       # Max TOKENS per chunk (not chars)
    chunk_overlap=20,     # Token overlap
    encoding_name="cl100k_base"  # OpenAI's tokenizer
)

chunks = splitter.split_documents(docs)
# More accurate — 1 token ≈ 4 chars on average
```

**Use when:** You need precise token control — e.g., staying within a specific model's context.

```python
# Token count ≠ character count
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

text = "LangChain is a framework for building LLM applications."
chars  = len(text)                    # 56 chars
tokens = len(enc.encode(text))        # ~12 tokens

print(f"chars: {chars}, tokens: {tokens}")
```

---

## 4️⃣ MarkdownHeaderTextSplitter — Structure-Aware

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """
# Introduction
This is the intro section.

## LangChain
LangChain is a framework.

### Installation
Run: pip install langchain

## LangGraph
LangGraph adds stateful agents.
"""

headers_to_split_on = [
    ("#",  "h1"),    # Level 1 heading
    ("##", "h2"),    # Level 2 heading
    ("###","h3"),    # Level 3 heading
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False   # Keep headers in content
)

chunks = splitter.split_text(markdown_text)

for chunk in chunks:
    print("Content:", chunk.page_content[:60])
    print("Metadata:", chunk.metadata)   # {"h1": "Introduction", "h2": "LangChain", ...}
    print()
```

**Use when:** Your documents are Markdown (documentation sites, README files, wikis).

---

## 5️⃣ HTML Header Text Splitter

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html_text = """
<html>
<body>
<h1>LangChain Guide</h1>
<h2>Installation</h2>
<p>Run pip install langchain...</p>
<h2>Usage</h2>
<p>Import and build chains...</p>
</body>
</html>
"""

headers_to_split_on = [("h1", "header1"), ("h2", "header2")]
splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(html_text)

for chunk in chunks:
    print(chunk.metadata)      # {"header1": "LangChain Guide", "header2": "Installation"}
    print(chunk.page_content[:80])
```

---

## 📊 Splitter Comparison

| Splitter | Splits On | Good For |
|---|---|---|
| `RecursiveCharacterTextSplitter` | `\n\n`, `\n`, ` `, `""` | **Default — everything** |
| `CharacterTextSplitter` | Single separator | Clean text with consistent structure |
| `TokenTextSplitter` | Token boundaries | Precise LLM context management |
| `MarkdownHeaderTextSplitter` | Markdown headers | Docs, README files |
| `HTMLHeaderTextSplitter` | HTML tags | HTML pages, scraped web content |

---

## ✅ Key Takeaways

- **Always start with `RecursiveCharacterTextSplitter`** — it handles most cases well
- `chunk_size` and `chunk_overlap` are the two most impactful parameters
- Token-based splitting is more accurate than character-based for LLM context limits
- Use structure-aware splitters (Markdown, HTML) when documents have clear headings
- Metadata from the original document is **preserved** through splitting

---

## ➡️ Next
[Chunking Strategies →](./05_chunking_strategies.md)
