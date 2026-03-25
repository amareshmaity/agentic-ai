# Web & Other Loaders

> *Beyond PDFs — load web pages, CSVs, JSON, YouTube transcripts, Wikipedia, and more. Every source becomes the same `Document` object.*

---

## 1️⃣ WebBaseLoader — Load Web Pages

```python
from langchain_community.document_loaders import WebBaseLoader

# Single URL
loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
docs = loader.load()

print(f"Loaded {len(docs)} document")           # 1 document per URL
print(f"Length: {len(docs[0].page_content)}")
print(f"Metadata: {docs[0].metadata}")
# {'source': 'https://...', 'title': 'Intro | LangChain', 'language': 'en'}
```

### Load Multiple URLs

```python
# Multiple URLs in one loader
urls = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/tutorials/",
]

loader = WebBaseLoader(urls)
docs = loader.load()
print(f"Loaded {len(docs)} pages")  # One Document per URL

# Parallel loading (faster)
loader = WebBaseLoader(urls, requests_per_second=2)
docs = loader.load()
```

### Control What Gets Extracted

```python
import bs4

# Filter to only specific HTML elements (removes nav, ads, footers)
loader = WebBaseLoader(
    web_paths=["https://python.langchain.com/docs/introduction/"],
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=("post-content", "post-title", "article-body")
        )
    }
)
docs = loader.load()
# Much cleaner content — only the actual article text
```

---

## 2️⃣ CSVLoader — Tabular Data

```python
from langchain_community.document_loaders import CSVLoader

# Basic: each row = one Document
loader = CSVLoader("data.csv")
docs = loader.load()

print(f"Rows loaded: {len(docs)}")
print(docs[0].page_content)
# "column1: value1\ncolumn2: value2\ncolumn3: value3"
print(docs[0].metadata)
# {'source': 'data.csv', 'row': 0}
```

```python
# Specify which column contains the main text
loader = CSVLoader(
    file_path="products.csv",
    source_column="product_description",  # This column = page_content
    encoding="utf-8",
    csv_args={"delimiter": ",", "quotechar": '"'}
)
docs = loader.load()
```

### DataFrameLoader — Load from Pandas

```python
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader

df = pd.read_csv("data.csv")

loader = DataFrameLoader(
    data_frame=df,
    page_content_column="description"  # Column to use as page_content
)
docs = loader.load()
# Other columns → metadata automatically
```

---

## 3️⃣ JSONLoader — Structured JSON

```python
from langchain_community.document_loaders import JSONLoader

# Load JSON with jq selector for what to extract
loader = JSONLoader(
    file_path="data.json",
    jq_schema=".[]",                     # Extract all items in array
    text_content=False                   # Don't convert to string
)
docs = loader.load()

# Extract specific field as content
loader = JSONLoader(
    file_path="products.json",
    jq_schema=".products[].description",  # Extract description field
    metadata_func=lambda record, meta: {**meta, "product_id": record.get("id")}
)
```

---

## 4️⃣ YoutubeLoader — Video Transcripts

```python
from langchain_community.document_loaders import YoutubeLoader

# Load transcript of a YouTube video
loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=your_video_id",
    add_video_info=True,      # Include title, author, length, views
    language=["en", "hi"],    # Try English, fallback to Hindi
    translation="en",         # Auto-translate to English
)
docs = loader.load()

print(docs[0].metadata)
# {'source': 'video_id', 'title': '...', 'description': '...', 'author': '...'}
print(docs[0].page_content[:300])   # Transcript text
```

---

## 5️⃣ WikipediaLoader — Wikipedia Articles

```python
from langchain_community.document_loaders import WikipediaLoader

loader = WikipediaLoader(
    query="LangChain AI framework",
    load_max_docs=2,              # Max articles to load
    doc_content_chars_max=5000,   # Truncate long articles
    lang="en",
)
docs = loader.load()

for doc in docs:
    print(f"Title: {doc.metadata['title']}")
    print(f"Content: {doc.page_content[:200]}\n")
```

---

## 6️⃣ TextLoader — Plain Text Files

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    file_path="notes.txt",
    encoding="utf-8",
    autodetect_encoding=True  # Auto-detect if encoding unknown
)
docs = loader.load()
# 1 Document containing the entire file content
```

---

## 7️⃣ UnstructuredFileLoader — Any File Type

```python
from langchain_community.document_loaders import UnstructuredFileLoader

# Works for PDF, DOCX, PPTX, XLSX, HTML, EML, MSG, and more
loader = UnstructuredFileLoader(
    "document.docx",
    mode="elements"     # or "single", "paged"
)
docs = loader.load()
```

---

## 8️⃣ Creating Documents Manually

Sometimes you already have text and just need to wrap it as Documents:

```python
from langchain_core.documents import Document

# Create from scratch
docs = [
    Document(
        page_content="LangChain is a framework for building LLM applications.",
        metadata={"source": "intro", "topic": "langchain", "level": "beginner"}
    ),
    Document(
        page_content="LangGraph adds stateful graph-based orchestration to LangChain.",
        metadata={"source": "intro", "topic": "langgraph", "level": "intermediate"}
    ),
]

# Or from a list of strings
texts = ["text 1...", "text 2...", "text 3..."]
docs = [Document(page_content=t, metadata={"index": i}) for i, t in enumerate(texts)]
```

---

## 📊 Loader Comparison

| Source | Loader | Documents Returned |
|---|---|---|
| PDF file | `PyPDFLoader` | 1 per page |
| Web page | `WebBaseLoader` | 1 per URL |
| CSV | `CSVLoader` | 1 per row |
| JSON | `JSONLoader` | 1 per item (via jq) |
| YouTube | `YoutubeLoader` | 1 per video |
| Wikipedia | `WikipediaLoader` | 1 per article |
| `.txt` file | `TextLoader` | 1 per file |
| Any file | `UnstructuredFileLoader` | 1+ depending on mode |

---

## ✅ Key Takeaways

- All loaders return `List[Document]` — same format regardless of source
- `WebBaseLoader` with `bs4.SoupStrainer` gives clean article text (removes nav/ads)
- `YoutubeLoader` gives you video transcripts as text — great for RAG on video content
- `CSVLoader`'s `source_column` lets you specify which column is the main text
- `UnstructuredFileLoader` handles almost any file type — great fallback
- Always check the number of Documents and character count after loading

---

## ➡️ Next
[Text Splitters →](./04_text_splitters.md)
