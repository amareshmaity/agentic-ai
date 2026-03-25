# Retrieval Chain API

> *LangChain provides high-level helper functions to assemble RAG chains with less boilerplate — `create_retrieval_chain` and `create_stuff_documents_chain`.*

---

## 🔧 Two Key Functions

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
```

| Function | Role |
|---|---|
| `create_stuff_documents_chain(llm, prompt)` | Takes a list of docs + question → LLM answer |
| `create_retrieval_chain(retriever, document_chain)` | Retrieves docs then passes to document_chain |

---

## 1️⃣ `create_stuff_documents_chain`

"Stuff" means: stuff all retrieved documents into the prompt at once.

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt MUST have {context} and {input} variables
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant. Answer using ONLY the provided context.
If the answer is not in the context, say "I don't have that information."

<context>
{context}
</context>"""),
    ("human", "{input}")
])

# Create the document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Can use directly with a list of docs
from langchain_core.documents import Document

test_docs = [
    Document(page_content="LangChain is a framework for building LLM applications."),
    Document(page_content="LangSmith provides tracing and monitoring for LangChain apps."),
]

result = document_chain.invoke({
    "input": "What is LangSmith?",
    "context": test_docs               # ← List[Document] injected here
})
print(result)
```

---

## 2️⃣ `create_retrieval_chain` — End-to-End

Connects retriever + document chain into a complete RAG pipeline.

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Components
embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore  = Chroma(persist_directory="./rag_db", embedding_function=embeddings)
retriever    = vectorstore.as_retriever(search_kwargs={"k": 4})
llm          = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only this context:\n\n{context}"),
    ("human",  "{input}")
])

# Chain assembly
document_chain   = create_stuff_documents_chain(llm, prompt)
retrieval_chain  = create_retrieval_chain(retriever, document_chain)

# Invoke — returns a dict!
result = retrieval_chain.invoke({"input": "What is LangChain?"})

print("Answer:",  result["answer"])                          # LLM answer
print("Sources:", [d.metadata for d in result["context"]])   # Retrieved docs
```

### Output Structure

```python
result = retrieval_chain.invoke({"input": "question"})
# Returns a dict:
{
    "input":   "question",              # Original question
    "context": [Document(...), ...],    # Retrieved docs (List[Document])
    "answer":  "LLM's answer..."        # Generated answer (str)
}
```

---

## 3️⃣ Streaming with Retrieval Chain

```python
# Stream the answer token by token
for chunk in retrieval_chain.stream({"input": "What is LangChain?"}):
    if "answer" in chunk:
        print(chunk["answer"], end="", flush=True)
print()
```

---

## 4️⃣ Source Attribution

Show users exactly which source documents were used:

```python
def ask_with_sources(chain, question: str) -> None:
    result = chain.invoke({"input": question})

    print(f"🤔 Question: {question}")
    print(f"\n💡 Answer:\n{result['answer']}")

    print(f"\n📚 Sources used ({len(result['context'])} chunks):")
    for i, doc in enumerate(result["context"], 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "")
        loc    = f"page {page}" if page else ""
        print(f"  [{i}] {source} {loc}")
        print(f"       ...{doc.page_content[:100]}...")

ask_with_sources(retrieval_chain, "What are the key features of LangChain?")
```

---

## 5️⃣ Adding a System Context / Persona

```python
# Customize for a specific domain
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are TechBot, an expert assistant for our technical documentation.

Guidelines:
- Answer ONLY from the provided context
- Use bullet points for lists
- Include code examples when relevant  
- If unsure, say "I don't have enough information"

Context:
{context}"""),
    ("human", "{input}")
])

domain_chain = create_retrieval_chain(
    retriever,
    create_stuff_documents_chain(llm, prompt)
)
```

---

## 6️⃣ Custom Prompts for Different Use Cases

```python
# Q&A — factual, concise
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer factually in 1-3 sentences using only:\n{context}"),
    ("human",  "{input}")
])

# Summary — comprehensive
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Provide a comprehensive summary based on:\n{context}"),
    ("human",  "Summarize information about: {input}")
])

# Comparison
compare_prompt = ChatPromptTemplate.from_messages([
    ("system", "Compare the topics using a structured table. Context:\n{context}"),
    ("human",  "Compare: {input}")
])

# Swap prompts depending on intent
qa_chain      = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, qa_prompt))
summary_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, summary_prompt))
```

---

## ✅ Key Takeaways

- `create_stuff_documents_chain(llm, prompt)` — document chain that injects `{context}` list
- `create_retrieval_chain(retriever, doc_chain)` — auto-retrieves then generates
- Output is a dict: `{"input": ..., "context": [...], "answer": "..."}`
- Use `result["context"]` for source attribution
- Supports streaming with `.stream()` — access answer chunks via `chunk["answer"]`

---

## ➡️ Next
[Conversational RAG →](./04_conversational_rag.md)
