# Chatbot Project — PDF Knowledge Base

> *Put it all together: a fully functional RAG chatbot that can answer questions from any PDF document, with conversation memory and streaming responses.*

---

## 🎯 Project Overview

We'll build a **PDF Q&A Chatbot** that:
- 📄 Ingests any PDF document as knowledge base
- 💬 Answers questions about its content
- 🧠 Remembers conversation history (multi-turn)
- ⚡ Streams responses token by token
- 📎 Shows source citations

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────┐
                    │         INDEXING (one-time)         │
                    │                                     │
                    │  PDF → chunks → embeddings → Chroma │
                    └─────────────────────────────────────┘

Query time:
  User input  ──► Contextualize (with history) ──► Retriever
                                                       │
                                                   Context docs
                                                       │
  Chat history ──► RAG Prompt ──► LLM ──► Streaming answer
```

---

## 📦 Dependencies

```bash
pip install langchain langchain-openai langchain-community \
            langchain-chroma langchain-text-splitters \
            pypdf chromadb python-dotenv
```

---

## 🔨 Full Implementation

### `rag_chatbot.py`

```python
"""
RAG Chatbot — PDF Knowledge Base with Conversational Memory
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL       = "gpt-4o-mini"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
TOP_K           = 4
PERSIST_DIR     = "./rag_chatbot_db"


class PDFChatbot:
    """Conversational RAG chatbot for PDF documents."""

    def __init__(self, pdf_path: str, reset_db: bool = False):
        self.pdf_path    = pdf_path
        self.chat_history: list = []
        self.embeddings  = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.llm         = ChatOpenAI(model=LLM_MODEL, temperature=0, streaming=True)

        # Build or load vector store
        if reset_db or not Path(PERSIST_DIR).exists():
            print("📄 Indexing PDF...")
            self.vectorstore = self._index_pdf()
        else:
            print("📦 Loading existing index...")
            self.vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=self.embeddings
            )
        print(f"✅ Ready! ({self.vectorstore._collection.count()} chunks indexed)")

        # Build chain
        self.chain = self._build_chain()

    def _index_pdf(self) -> Chroma:
        """Load, split, embed, and store the PDF."""
        # Load
        loader   = PyPDFLoader(self.pdf_path)
        raw_docs = loader.load()
        print(f"   Loaded {len(raw_docs)} pages")

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True
        )
        chunks = splitter.split_documents(raw_docs)
        print(f"   Created {len(chunks)} chunks")

        # Embed & Store
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=PERSIST_DIR
        )

    def _build_chain(self):
        """Build the conversational retrieval chain."""
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 4, "lambda_mult": 0.5}
        )

        # Prompt 1: Contextualize question using history
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given the chat history and the latest question, 
reformulate it as a standalone question without the chat history context.
Return it unchanged if it's already standalone."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_prompt
        )

        # Prompt 2: Answer using context
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert assistant for document Q&A.
Answer ONLY using the provided context. Be concise and accurate.
If the answer isn't in the context, say: "I don't have that information."
Always cite your sources when possible.

Context:
{context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        document_chain = create_stuff_documents_chain(self.llm, rag_prompt)
        return create_retrieval_chain(history_aware_retriever, document_chain)

    def chat(self, question: str, show_sources: bool = True) -> str:
        """Ask a question and get a streaming answer."""
        print(f"\n🤔 You: {question}")
        print("💡 Bot: ", end="", flush=True)

        full_answer = ""
        context_docs = []

        # Stream the response
        for chunk in self.chain.stream({
            "input":        question,
            "chat_history": self.chat_history
        }):
            if "context" in chunk:
                context_docs = chunk["context"]
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)
                full_answer += chunk["answer"]
        print()  # newline after streaming

        # Show sources
        if show_sources and context_docs:
            print(f"\n📚 Sources:")
            seen = set()
            for doc in context_docs:
                source = doc.metadata.get("source", "unknown")
                page   = doc.metadata.get("page", "")
                key    = f"{source}:{page}"
                if key not in seen:
                    seen.add(key)
                    loc = f" (page {page})" if page != "" else ""
                    print(f"   • {Path(source).name}{loc}")

        # Update history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=full_answer))

        return full_answer

    def reset_history(self):
        """Start a new conversation."""
        self.chat_history = []
        print("🔄 Conversation history cleared.")

    def summary(self):
        """Print conversation stats."""
        turns = len(self.chat_history) // 2
        print(f"\n📊 Conversation: {turns} turn(s), {len(self.chat_history)} messages")


# ─────────────────────────────────────────────
# Main — interactive chat loop
# ─────────────────────────────────────────────
def main():
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "document.pdf"

    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        print("Usage: python rag_chatbot.py your_document.pdf")
        return

    bot = PDFChatbot(pdf_path)

    print("\n" + "="*50)
    print("  PDF Chatbot — Ready!")
    print("  Commands: 'quit', 'reset', 'stats'")
    print("="*50)

    while True:
        try:
            question = input("\nYou: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if question.lower() == "reset":
                bot.reset_history()
                continue
            if question.lower() == "stats":
                bot.summary()
                continue
            bot.chat(question)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
```

---

## 🚀 Running the Chatbot

```bash
# Basic usage
python rag_chatbot.py my_document.pdf

# Example session:
# You: What is this document about?
# 💡 Bot: This document covers...
# 📚 Sources: • my_document.pdf (page 1)

# You: What are the main topics?
# 💡 Bot: The main topics are...

# You: Can you explain the first one in more detail?
# 💡 Bot: [Uses chat history to resolve "the first one"] ...
```

---

## 🔧 Variations & Extensions

### Use a Different Model Provider

```python
# Switch to Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

# Switch to local Ollama
from langchain_ollama import ChatOllama, OllamaEmbeddings
llm        = ChatOllama(model="llama3.2")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

### Add Multiple PDFs

```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("./docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
all_docs = loader.load()  # All PDFs loaded together
```

### Gradio Web UI

```python
import gradio as gr

bot = PDFChatbot("document.pdf")

def respond(message, history):
    return bot.chat(message, show_sources=False)

demo = gr.ChatInterface(
    respond,
    title="📄 PDF Chatbot",
    description="Ask questions about your PDF document",
)
demo.launch()
```

---

## ✅ Key Takeaways

- **PDFChatbot** class encapsulates the full conversational RAG pipeline
- Indexing is skipped if the database already exists (use `reset_db=True` to re-index)
- Streaming is done with `.stream()` — extracting `"answer"` chunk key
- History is maintained as `List[HumanMessage | AIMessage]`
- Source attribution from `result["context"]` doc metadata
- Easily swap LLM provider and embeddings by changing one line each

---

## ⬅️ Previous
[Conversational RAG](./04_conversational_rag.md)

## ➡️ Next
[Hands-on Examples →](./examples.ipynb)
