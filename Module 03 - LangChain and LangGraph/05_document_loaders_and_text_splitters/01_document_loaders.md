# Document Loaders

> *Document loaders are how LangChain ingests data from the outside world — turning files, web pages, databases, and APIs into a unified `Document` object.*

---

## 📄 The `Document` Object

Everything a loader produces is a `Document`. It has two fields:

```python
from langchain_core.documents import Document

doc = Document(
    page_content="This is the actual text content of the document chunk.",
    metadata={
        "source":   "report.pdf",       # Where it came from
        "page":     3,                   # Page number (for PDFs)
        "author":   "John Smith",        # Optional custom metadata
        "created":  "2024-01-15",
    }
)

print(doc.page_content)   # The text
print(doc.metadata)       # Dict of metadata
print(len(doc.page_content))  # Character count
```

**Why metadata matters:**
- After retrieval, you can show the user **which source** the answer came from
- Filter searches by source, date, author, or any custom field
- Track provenance in production systems

---

## 🗂️ Loader Categories

```
File Loaders:
    PDF          → PyPDFLoader, PDFMinerLoader, UnstructuredPDFLoader
    Word (DOCX)  → Docx2txtLoader, UnstructuredWordDocumentLoader
    CSV          → CSVLoader, DataFrameLoader
    JSON         → JSONLoader
    TXT          → TextLoader
    Markdown     → UnstructuredMarkdownLoader
    PowerPoint   → UnstructuredPowerPointLoader

Web Loaders:
    Web page     → WebBaseLoader
    Sitemap      → SitemapLoader
    YouTube      → YoutubeLoader (transcript)
    Wikipedia    → WikipediaLoader
    ArXiv        → ArxivLoader

Database Loaders:
    SQL          → SQLDatabaseLoader
    MongoDB      → MongodbLoader

Code Loaders:
    GitHub       → GithubFileLoader
    Git repo     → GitLoader

API Loaders:
    Notion       → NotionDirectoryLoader
    Google Drive → GoogleDriveLoader
    Slack        → SlackDirectoryLoader
```

---

## 📦 Installation

```bash
# Core loaders
pip install langchain-community

# PDF loaders (choose based on your needs)
pip install pypdf          # PyPDFLoader — default choice
pip install pdfminer.six   # PDFMinerLoader — better text extraction
pip install unstructured   # UnstructuredPDFLoader — best for complex PDFs

# Web loader
pip install beautifulsoup4 # Required by WebBaseLoader

# YouTube transcript
pip install youtube-transcript-api

# Document parsing helpers
pip install python-docx    # Word docs
pip install openpyxl       # Excel
```

---

## 🔧 Common Loader Interface

All loaders implement the same interface:

```python
# Every loader has these methods:
loader.load()       # → List[Document] — load all at once
loader.lazy_load()  # → Iterator[Document] — memory-efficient streaming
loader.load_and_split(text_splitter)  # → List[Document] — load + split in one step
```

---

## 🔍 Examining Loaded Documents

```python
# After loading, always inspect what you got
docs = loader.load()

print(f"Documents loaded: {len(docs)}")
print(f"First doc length: {len(docs[0].page_content)} chars")
print(f"Metadata keys:    {list(docs[0].metadata.keys())}")
print(f"Content preview:  {docs[0].page_content[:200]}")

# Check all pages
for i, doc in enumerate(docs):
    print(f"Doc {i}: {len(doc.page_content)} chars | {doc.metadata}")
```

---

## ✅ Key Takeaways

- `Document(page_content=..., metadata=...)` is the universal data unit in LangChain
- Metadata is crucial — keep `source`, `page`, and any domain-specific fields
- All loaders expose `.load()` and `.lazy_load()` for memory efficiency
- `langchain-community` contains 100+ loaders — check docs for your specific source
- Always inspect loaded documents before proceeding to splitting

---

## ➡️ Next
[PDF Loaders →](./02_pdf_loaders.md)
