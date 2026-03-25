# PDF Loaders

> *PDFs are the most common document format in RAG systems. LangChain provides multiple PDF loaders — each with different trade-offs in speed, accuracy, and layout handling.*

---

## 🔧 Three Main PDF Loaders

| Loader | Package | Speed | Quality | Layout | Best For |
|---|---|---|---|---|---|
| `PyPDFLoader` | `pypdf` | ⚡ Fast | Good | Basic | Default choice |
| `PDFMinerLoader` | `pdfminer.six` | Medium | Better | Good | Text-heavy PDFs |
| `UnstructuredPDFLoader` | `unstructured` | Slow | Best | Excellent | Complex/scanned PDFs |

---

## 1️⃣ PyPDFLoader — Default Choice

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("path/to/document.pdf")

# Load all pages
docs = loader.load()

print(f"Pages loaded: {len(docs)}")          # One Document per page
print(f"First page length: {len(docs[0].page_content)}")
print(f"Metadata: {docs[0].metadata}")
# {'source': 'document.pdf', 'page': 0}

# Preview first page
print(docs[0].page_content[:500])
```

**Characteristics:**
- One `Document` per PDF page
- Metadata includes `source` and `page` number (0-indexed)
- Fast and reliable for standard text PDFs
- May struggle with complex layouts or scanned PDFs

---

## 2️⃣ PDFMinerLoader — Better Text Extraction

```python
from langchain_community.document_loaders import PDFMinerLoader

loader = PDFMinerLoader("path/to/document.pdf")
docs = loader.load()

# Returns the entire PDF as ONE Document (not per-page)
# Better at preserving text flow across columns and complex layouts
print(f"Documents: {len(docs)}")    # Usually 1 — whole PDF
print(f"Length: {len(docs[0].page_content)} chars")
```

**Use when:** Text order matters (multi-column layout, heavily formatted PDFs)

---

## 3️⃣ UnstructuredPDFLoader — Best for Complex PDFs

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

# Mode 1: single — whole PDF as one Document
loader = UnstructuredPDFLoader("document.pdf", mode="single")

# Mode 2: elements — each text element as separate Document
loader = UnstructuredPDFLoader("document.pdf", mode="elements")
docs = loader.load()

# Elements mode gives granular control
# Each element has type metadata: NarrativeText, Title, Table, etc.
for doc in docs[:5]:
    print(doc.metadata.get("category"), "→", doc.page_content[:80])
```

**Use when:** Scanned PDFs (needs OCR), PDFs with tables, forms, or mixed content.

---

## 4️⃣ Loading Entire Directories

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Load ALL PDFs in a folder
loader = DirectoryLoader(
    path="./documents/",         # Folder path
    glob="**/*.pdf",             # Pattern: all PDFs, including subfolders
    loader_cls=PyPDFLoader,      # Which loader to use
    show_progress=True,          # Progress bar
    use_multithreading=True,     # Load files in parallel
)

docs = loader.load()
print(f"Total documents: {len(docs)}")

# Group by source file
from collections import Counter
sources = Counter(doc.metadata["source"] for doc in docs)
for src, count in sources.items():
    print(f"  {src}: {count} pages")
```

---

## 5️⃣ Memory-Efficient Loading with `lazy_load()`

```python
# For large PDFs — don't load everything into memory at once
loader = PyPDFLoader("huge_document.pdf")

# lazy_load() returns a generator — processes one page at a time
for doc in loader.lazy_load():
    process_page(doc)   # Process and discard, no memory buildup
```

---

## 6️⃣ Load + Split in One Step

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("document.pdf")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Load and split in one convenient call
chunks = loader.load_and_split(text_splitter=splitter)
print(f"Total chunks: {len(chunks)}")
print(f"Sample metadata: {chunks[0].metadata}")
# {'source': 'document.pdf', 'page': 0}  ← page metadata preserved!
```

---

## 📋 Metadata After Loading

| Loader | Metadata Fields |
|---|---|
| `PyPDFLoader` | `source`, `page` |
| `PDFMinerLoader` | `source`, `file_path` |
| `UnstructuredPDFLoader` | `source`, `page_number`, `category`, `filename` |

**Tip: Add custom metadata after loading:**

```python
docs = loader.load()

# Enrich with custom metadata
for doc in docs:
    doc.metadata["author"]   = "John Smith"
    doc.metadata["subject"]  = "Machine Learning"
    doc.metadata["date"]     = "2024-01-15"
    doc.metadata["version"]  = "v2.3"

# Now you can filter by these fields in your vector store
```

---

## ✅ Key Takeaways

- **`PyPDFLoader`** is the default — one Document per page, fast, reliable
- **`PDFMinerLoader`** for better text extraction in complex layouts
- **`UnstructuredPDFLoader`** for scanned/mixed content PDFs
- **`DirectoryLoader`** loads entire folders with multithreading
- **`lazy_load()`** for memory efficiency on large files
- Always preserve and enrich `metadata` — it's crucial for retrieval attribution

---

## ➡️ Next
[Web & Other Loaders →](./03_web_and_other_loaders.md)
