# 05 — Document Loaders & Text Splitters

> **Data is the fuel for RAG. Document loaders bring data into LangChain; text splitters break it into chunks the LLM can actually use.**

---

## 📚 Subtopics in This Section

| File | Topic |
|---|---|
| [`01_document_loaders.md`](./01_document_loaders.md) | What is a Document, loader categories, installed packages |
| [`02_pdf_loaders.md`](./02_pdf_loaders.md) | PyPDFLoader, PDFMinerLoader, UnstructuredPDFLoader |
| [`03_web_and_other_loaders.md`](./03_web_and_other_loaders.md) | WebBaseLoader, CSVLoader, JSONLoader, YouTubeLoader |
| [`04_text_splitters.md`](./04_text_splitters.md) | RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter |
| [`05_chunking_strategies.md`](./05_chunking_strategies.md) | chunk_size, overlap, how to choose, semantic chunking |
| [`examples.ipynb`](./examples.ipynb) | Hands-on: load PDFs, web pages, split with different strategies |

---

## 🎯 Learning Objectives

By the end of this section you will be able to:

- Load data from PDFs, web pages, CSVs, YouTube, and more
- Understand the `Document` object and its metadata
- Split long documents into chunks using `RecursiveCharacterTextSplitter`
- Compare different text splitters and choose the right one
- Tune chunk size and overlap for optimal RAG performance

---

## ⚡ Quick Summary

```
RAG Pipeline:
  1. Load    → PyPDFLoader("file.pdf").load()          → List[Document]
  2. Split   → RecursiveCharacterTextSplitter().split_documents(docs)
  3. Embed   → OpenAIEmbeddings() (covered in Module 06)
  4. Store   → Chroma.from_documents(chunks, embeddings)
  5. Retrieve→ vectorstore.as_retriever()

The Document object:
  Document(
      page_content = "Text content of this chunk...",
      metadata     = {"source": "file.pdf", "page": 3}
  )
```

---

## ⬅️ Previous
[04 — Chains & Runnables](../04_chains_and_runnables/theory.md)

## ➡️ Next Subtopic
[06 — Vector Stores & Retrievers](../06_vector_stores_and_retrievers/theory.md)
