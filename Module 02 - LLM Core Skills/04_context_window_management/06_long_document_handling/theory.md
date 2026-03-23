# 06 — Long Document Handling

> *When the document is larger than your context window — chunking, overlapping, and map-reduce patterns.*

---

## 6.1 The Long Document Problem

Context windows, even at 128k tokens, can't always fit an entire document:

```
128k tokens ≈ 96,000 words ≈ ~380 pages

But real documents can be:
- Legal contracts:    50–500+ pages
- Technical manuals: 200–2000+ pages
- Codebases:         100k–10M+ lines
- Research papers:   Often 50–100 pages with dense content
- Meeting transcripts: Hours of speech → thousands of words
```

When a document exceeds the window, you need strategies to process it across multiple calls.

---

## 6.2 Strategy 1 — Simple Chunking

Split documents into fixed-size chunks and process each independently:

```python
import tiktoken

def chunk_text_by_tokens(
    text: str,
    chunk_size: int = 2000,
    model: str = "gpt-4o-mini"
) -> list[str]:
    """
    Split text into chunks of at most `chunk_size` tokens.
    Splits at sentence boundaries when possible.
    """
    enc = tiktoken.encoding_for_model(model)
    
    # Split into sentences first
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk_tokens = []
    current_token_count = 0
    
    for sentence in sentences:
        sentence_tokens = enc.encode(sentence)
        
        if current_token_count + len(sentence_tokens) <= chunk_size:
            # Add sentence to current chunk
            current_chunk_tokens.extend(sentence_tokens)
            current_token_count += len(sentence_tokens)
        else:
            # Save current chunk and start a new one
            if current_chunk_tokens:
                chunks.append(enc.decode(current_chunk_tokens))
            current_chunk_tokens = sentence_tokens
            current_token_count = len(sentence_tokens)
    
    # Add the last chunk
    if current_chunk_tokens:
        chunks.append(enc.decode(current_chunk_tokens))
    
    return chunks
```

---

## 6.3 Strategy 2 — Overlapping Chunks (Sliding Window)

Overlapping prevents losing context at chunk boundaries:

```python
def chunk_with_overlap(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 200,
    model: str = "gpt-4o-mini"
) -> list[str]:
    """
    Split text into overlapping chunks.
    Overlap ensures that information at chunk boundary is not lost.
    
    Example with chunk_size=100, overlap=20:
    Chunk 1: tokens [0:100]
    Chunk 2: tokens [80:180]   ← 20 token overlap
    Chunk 3: tokens [160:260]
    """
    enc = tiktoken.encoding_for_model(model)
    token_ids = enc.encode(text)
    
    chunks = []
    start = 0
    step = chunk_size - overlap
    
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_tokens = token_ids[start:end]
        chunks.append(enc.decode(chunk_tokens))
        
        if end >= len(token_ids):
            break
        
        start += step
    
    return chunks
```

---

## 6.4 Strategy 3 — Map-Reduce over Documents

Process each chunk independently (map), then combine results (reduce):

```python
from openai import OpenAI

client = OpenAI()

def map_reduce_summarize(
    text: str,
    chunk_size: int = 3000,
    overlap: int = 200,
    map_prompt: str = "Summarize this section of the document in 2-3 sentences:",
    reduce_prompt: str = "Combine these section summaries into a coherent final summary:",
    model: str = "gpt-4o-mini"
) -> str:
    """
    Summarize a long document using map-reduce:
    1. MAP:    Summarize each chunk independently
    2. REDUCE: Combine all chunk summaries into a final summary
    """
    # Step 1: Chunk the document
    chunks = chunk_with_overlap(text, chunk_size, overlap, model)
    print(f"Document split into {len(chunks)} chunks")
    
    # Step 2: MAP — summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Processing chunk {i}/{len(chunks)}...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a document summarizer."},
                {"role": "user",   "content": f"{map_prompt}\n\n{chunk}"}
            ],
            max_tokens=200,
            temperature=0.0
        )
        chunk_summaries.append(response.choices[0].message.content)
    
    # Step 3: REDUCE — combine all summaries
    combined = "\n\n".join(
        f"Section {i+1}:\n{summary}"
        for i, summary in enumerate(chunk_summaries)
    )
    
    print(f"  Combining {len(chunk_summaries)} section summaries...")
    final_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a document synthesizer."},
            {"role": "user",   "content": f"{reduce_prompt}\n\n{combined}"}
        ],
        max_tokens=500,
        temperature=0.0
    )
    
    return final_response.choices[0].message.content
```

---

## 6.5 Strategy 4 — MapReduce QA (Answer Questions Over Long Docs)

```python
def answer_over_long_doc(
    document: str,
    question: str,
    chunk_size: int = 3000,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Answer a question about a document too large to fit in one context window.
    
    Steps:
    1. Chunk the document
    2. For each chunk: extract relevant info for the question
    3. Synthesize final answer from all relevant extracts
    """
    chunks = chunk_with_overlap(document, chunk_size, overlap=150, model=model)
    print(f"Document: {len(chunks)} chunks")
    
    # Step 1: Extract relevant information from each chunk
    extracts = []
    for i, chunk in enumerate(chunks, 1):
        r = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract information relevant to the question. If chunk has no relevant info, respond 'NONE'."
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nChunk {i}/{len(chunks)}:\n{chunk}"
                }
            ],
            max_tokens=200,
            temperature=0.0
        )
        extract = r.choices[0].message.content
        if extract.strip().upper() != "NONE":
            extracts.append(f"[Chunk {i}]: {extract}")
    
    print(f"Found relevant info in {len(extracts)}/{len(chunks)} chunks")
    
    if not extracts:
        return "Could not find relevant information in the document."
    
    # Step 2: Synthesize final answer
    combined_extracts = "\n\n".join(extracts)
    final = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Answer the question by synthesizing the provided evidence. Be concise and direct."
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nEvidence from document:\n{combined_extracts}"
            }
        ],
        max_tokens=400,
        temperature=0.0
    )
    
    return final.choices[0].message.content
```

---

## 6.6 Semantic Chunking — Split at Meaning Boundaries

Instead of fixed-size chunks, split where meaning changes:

```python
def semantic_chunk(
    text: str,
    min_chunk_tokens: int = 200,
    max_chunk_tokens: int = 1500,
    model: str = "gpt-4o-mini"
) -> list[str]:
    """
    Split at paragraph/section boundaries for more natural chunks.
    Falls back to token-size splitting if paragraph is too large.
    """
    enc = tiktoken.encoding_for_model(model)
    
    # Split by double newlines (natural paragraph boundaries)
    paragraphs = text.split("\n\n")
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_tokens = len(enc.encode(para))
        
        if current_tokens + para_tokens > max_chunk_tokens and current_chunk:
            # Save current chunk (reached max size)
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        elif current_tokens + para_tokens >= min_chunk_tokens:
            # Good chunk size — save and reset
            current_chunk.append(para)
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        else:
            # Keep building the chunk
            current_chunk.append(para)
            current_tokens += para_tokens
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks
```

---

## 6.7 Chunking Strategy Comparison

| Strategy | Best For | Overlap | Boundary Awareness |
|---|---|---|---|
| **Fixed token chunks** | Any text, simple | Optional | ❌ None |
| **Overlapping chunks** | Continuous text (articles, transcripts) | ✅ Yes | ❌ Mid-sentence |
| **Paragraph/semantic** | Structured documents | Natural | ✅ Paragraph |
| **Section-based** | Reports, manuals | Natural | ✅ Section heading |
| **Sentence-aware** | Technical or legal text | Optional | ✅ Sentence |

---

## 📌 Key Takeaways

1. **Chunking = splitting docs > context window into processable pieces**
2. **Always overlap chunks** (10-15% of chunk size) to avoid losing boundary context  
3. **Map-Reduce**: process chunks independently (map) → combine results (reduce)
4. **Semantic chunking** at paragraphs/sections is better than fixed-size for structured docs
5. **QA over long docs**: extract per-chunk, synthesize cross-chunk — never fit everything in one call
6. **Choose chunk size** based on task: summarization (large chunks) vs extraction (small chunks)
7. **Track chunk provenance** — know which chunk yielded which result for auditability
