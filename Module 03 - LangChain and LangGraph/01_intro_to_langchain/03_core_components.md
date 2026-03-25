# LangChain Core Components

> *LangChain's power comes from its 9 core component types — each solving a distinct part of building LLM applications.*

---

## 🗺️ Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LANGCHAIN COMPONENTS                      │
│                                                             │
│  INPUT SIDE              CORE ENGINE          OUTPUT SIDE   │
│  ─────────               ───────────          ──────────    │
│  Document Loaders   →    Models (LLMs)   →    Output Parsers│
│  Text Splitters     →    Prompts         →    (to your app) │
│  Vector Stores      →    Chains                             │
│  Retrievers         →    Agents                             │
│  Memory             →    Tools                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ Models (The Brain)

LangChain wraps all LLM providers behind a **unified interface**.

### Two Types of Models

| Type | Class | Output | Example |
|---|---|---|---|
| **LLM** | `BaseLLM` | Raw string | `OpenAI`, `Cohere` |
| **ChatModel** | `BaseChatModel` | `AIMessage` | `ChatOpenAI`, `ChatAnthropic` |

> **Use ChatModels** — they support structured messages (system/human/ai), tool calling, and multimodal inputs. LLMs are legacy.

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Unified interface — same methods for all
gpt    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
claude = ChatAnthropic(model="claude-3-haiku-20240307")
gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# All work the same way
response = gpt.invoke("What is LangChain?")
print(response.content)          # "LangChain is..."
print(response.usage_metadata)   # {'input_tokens': 12, ...}

# Swap models with zero code change
response = claude.invoke("What is LangChain?")  # same!
```

### Key Parameters

```python
ChatOpenAI(
    model="gpt-4o-mini",   # Model version
    temperature=0,          # 0 = deterministic, 1 = creative
    max_tokens=1000,        # Limit output length
    streaming=True,         # Enable streaming
    timeout=30,             # Request timeout in seconds
)
```

---

## 2️⃣ Prompts (The Input)

**Prompts** are templates that structure the input to the LLM.

### PromptTemplate (for LLMs)

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["product", "tone"],
    template="Write a {tone} description for: {product}"
)

# Format the prompt
result = template.format(product="AI Agent", tone="professional")
# → "Write a professional description for: AI Agent"
```

### ChatPromptTemplate (for ChatModels — use this)

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Always respond in {language}."),
    ("human",  "{user_input}")
])

# Returns a list of formatted messages
messages = prompt.format_messages(
    role="Python expert",
    language="English",
    user_input="What is a decorator?"
)

# Use in a chain
chain = prompt | llm
result = chain.invoke({
    "role": "Python expert",
    "language": "English",
    "user_input": "What is a decorator?"
})
```

### Prompt Types

| Prompt Type | Use Case |
|---|---|
| `PromptTemplate` | Single-string LLMs |
| `ChatPromptTemplate` | ChatModels (most common) |
| `FewShotPromptTemplate` | Few-shot examples |
| `FewShotChatMessagePromptTemplate` | Few-shot for ChatModels |
| `MessagesPlaceholder` | Inject dynamic message history |

---

## 3️⃣ Output Parsers (Structured Output)

LLMs return raw text. Output Parsers convert that into **structured Python objects**.

```python
from langchain_core.output_parsers import (
    StrOutputParser,        # Raw string → string
    JsonOutputParser,       # Raw string → dict
    PydanticOutputParser,   # Raw string → Pydantic model
)

# 1. StrOutputParser — most common, extracts .content from AIMessage
parser = StrOutputParser()
chain = prompt | llm | parser
result = chain.invoke(...)   # Returns plain string

# 2. PydanticOutputParser — enforce a schema
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating 1-10")
    summary: str = Field(description="Brief summary")

parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt_with_format = ChatPromptTemplate.from_messages([
    ("system", "Extract movie info.\n{format_instructions}"),
    ("human", "{text}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt_with_format | llm | parser
review = chain.invoke({"text": "Inception is a masterpiece. 10/10"})
print(review.title)    # "Inception"
print(review.rating)   # 10
```

---

## 4️⃣ Document Loaders (Data Ingestion)

Load data from **any source** into LangChain `Document` objects.

```python
from langchain_community.document_loaders import (
    PyPDFLoader,           # PDFs
    WebBaseLoader,         # Web pages
    CSVLoader,             # CSV files
    JSONLoader,            # JSON files
    YoutubeLoader,         # YouTube transcripts
    GitLoader,             # Git repositories
    NotionDirectoryLoader, # Notion pages
    UnstructuredFileLoader # Any file type
)

# Load a PDF
loader = PyPDFLoader("document.pdf")
docs = loader.load()
# Returns: List[Document(page_content="...", metadata={"page": 0})]

# Load a web page
loader = WebBaseLoader("https://docs.langchain.com")
docs = loader.load()

# Load a CSV
loader = CSVLoader("data.csv", source_column="text")
docs = loader.load()
```

### The `Document` Object

```python
from langchain_core.documents import Document

doc = Document(
    page_content="This is the text content...",
    metadata={
        "source": "document.pdf",
        "page": 0,
        "author": "John Smith"
    }
)
```

---

## 5️⃣ Text Splitters (Chunking)

Documents are often too large for a context window. Split them into **chunks**.

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,  # Best general-purpose
    CharacterTextSplitter,           # Simple character-based
    TokenTextSplitter,               # Token-based (matches LLM limits)
    MarkdownHeaderTextSplitter,      # Structure-aware for markdown
)

# RecursiveCharacterTextSplitter is the default choice
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Max characters per chunk
    chunk_overlap=200,    # Overlap between chunks (preserves context)
    separators=["\n\n", "\n", " ", ""]  # Try these in order
)

# Split documents
docs = loader.load()
chunks = splitter.split_documents(docs)

print(f"Original: 1 document")
print(f"Chunked:  {len(chunks)} chunks")
print(f"Sample chunk:\n{chunks[0].page_content[:200]}")
```

### Why Overlap?

```
Chunk 1: [...paragraph A...][...start of para B...]
                         ↑ overlap region ↑
Chunk 2:             [...end of para A...][...paragraph B...]

Without overlap: "start of para B" and "end of para A" are in different
chunks → retrieval misses context that spans a boundary.
With overlap: Important context near boundaries appears in both chunks.
```

---

## 6️⃣ Embeddings & Vector Stores (Semantic Memory)

**Embeddings** convert text into numeric vectors. **Vector Stores** index and search them.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

# Step 1: Create an embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Step 2: Create a vector store from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Save to disk
)

# Step 3: Search semantically
results = vectorstore.similarity_search("What is LangGraph?", k=3)
# Returns top 3 most semantically similar chunks

# Step 4: Or use as a retriever (Runnable interface)
retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr" for diversity
    search_kwargs={"k": 5}
)
docs = retriever.invoke("What is LangGraph?")
```

---

## 7️⃣ Retrievers (Semantic Search Interface)

Retrievers are the **Runnable interface** over vector stores. They take a string query and return `List[Document]`.

```python
from langchain_core.retrievers import BaseRetriever

# Basic vector store retriever
retriever = vectorstore.as_retriever()

# MMR retriever (maximize relevance + diversity)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
)

# Use in a chain
chain = retriever | format_docs | llm | StrOutputParser()
```

| Retriever Type | Description |
|---|---|
| `VectorStoreRetriever` | Basic similarity search |
| `MultiQueryRetriever` | Generates multiple queries, merges results |
| `ContextualCompressionRetriever` | Compresses retrieved docs to fit context |
| `EnsembleRetriever` | Combines multiple retrievers (BM25 + vector) |
| `SelfQueryRetriever` | LLM generates the filter + query |

---

## 8️⃣ Memory (Conversation History)

Memory gives your agent the ability to **remember past turns** in a conversation.

```python
from langchain.memory import (
    ConversationBufferMemory,         # Store all messages
    ConversationBufferWindowMemory,   # Keep last K messages
    ConversationSummaryMemory,        # Summarize older messages
    ConversationSummaryBufferMemory,  # Summary + recent buffer
    VectorStoreRetrieverMemory,       # Store in vector DB, retrieve by relevance
)

# Buffer memory — simplest
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

memory.save_context(
    {"input": "My name is Alice"},
    {"output": "Nice to meet you, Alice!"}
)

memory.load_memory_variables({})
# {"chat_history": [HumanMessage("My name is Alice"), AIMessage("Nice...")]}

# Window memory — last 3 exchanges only
memory = ConversationBufferWindowMemory(k=3)
```

> **Note**: In modern LangChain/LangGraph, memory is typically managed via **State** in the graph, not standalone memory objects. But understanding these helps you understand what's happening under the hood.

---

## 9️⃣ Tools (Agent Actions)

**Tools** are functions that agents can call. They give the LLM **hands** to interact with the world.

```python
from langchain_core.tools import tool

# Define a tool with @tool decorator
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid Python math expression."""
    return str(eval(expression))

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # In real code: call a weather API
    return f"Weather in {city}: 22°C, Sunny"

# Tools expose a schema that the LLM uses to understand when/how to call it
print(calculate.name)           # "calculate"
print(calculate.description)    # "Evaluate a mathematical..."
print(calculate.args_schema)    # Pydantic schema

# Bind tools to a model
llm_with_tools = llm.bind_tools([calculate, get_weather])
response = llm_with_tools.invoke("What is 25 * 17?")
print(response.tool_calls)
# [{'name': 'calculate', 'args': {'expression': '25 * 17'}, 'id': '...'}]
```

### Built-in Tools

```python
from langchain_community.tools import (
    TavilySearchResults,     # Web search
    WikipediaQueryRun,       # Wikipedia
    PubmedQueryRun,          # Medical papers
    YahooFinanceNewsTool,    # Finance news
    ShellTool,               # Execute shell commands
    PythonREPLTool,          # Execute Python code
)
```

---

## 🔗 How Components Connect: End-to-End RAG Example

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load → 2. Split → 3. Embed → 4. Store
loader   = PyPDFLoader("langchain_docs.pdf")
docs     = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks   = splitter.split_documents(docs)
store    = Chroma.from_documents(chunks, OpenAIEmbeddings())

# 5. Retrieve
retriever = store.as_retriever(search_kwargs={"k": 3})

# 6. Prompt + 7. LLM + 8. Parse
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context below:\n\n{context}"),
    ("human",  "{question}")
])
llm    = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 9. Chain it all together with LCEL
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

answer = chain.invoke("What is LCEL?")
print(answer)
```

---

## ✅ Component Quick Reference

| Component | Import | Role |
|---|---|---|
| ChatOpenAI | `langchain_openai` | LLM (brain) |
| ChatPromptTemplate | `langchain_core.prompts` | Structure LLM input |
| StrOutputParser | `langchain_core.output_parsers` | Raw output → string |
| PydanticOutputParser | `langchain_core.output_parsers` | Output → typed object |
| PyPDFLoader | `langchain_community.document_loaders` | Load PDFs |
| RecursiveCharacterTextSplitter | `langchain_text_splitters` | Chunk documents |
| OpenAIEmbeddings | `langchain_openai` | Create embeddings |
| Chroma | `langchain_community.vectorstores` | Store + search embeddings |
| ConversationBufferMemory | `langchain.memory` | Store chat history |
| @tool | `langchain_core.tools` | Define agent tools |

---

## ⬅️ Previous
[LangChain Architecture](./02_architecture.md)

## ➡️ Next
[LangChain Ecosystem →](./04_ecosystem.md)
