# 03 — Semantic Routing

> *Route by what the user means, not just what words they use.*

---

## 3.1 Why Keywords Aren't Enough

Rule-based keyword routing breaks on natural variation:

```
"Fix this bug in my Python code"          → keyword "python" → ✅ correct
"There's an issue with my ML pipeline"    → no keyword match → ❌ wrong model
"Help me debug this neural network"       → "debug" → ✅ correct
"My transformer architecture is failing" → no keyword match → ❌ wrong model
```

The user always means the same type of task, but the words differ. Semantic routing solves this by embedding the request into a vector space and routing based on **meaning similarity**, not exact string match.

---

## 3.2 How Semantic Routing Works

```
User Request
    │
    ▼
Embedding Model (text-embedding-3-small)
    │
    ├─→ vector: [0.12, -0.45, 0.87, ...]  (1536 dimensions)
    │
    ▼
Compare to route anchor embeddings:
    - "coding tasks"     → similarity: 0.92  ← highest
    - "math reasoning"   → similarity: 0.43
    - "creative writing" → similarity: 0.21
    - "general QA"       → similarity: 0.38
    │
    ▼
Route to model registered for "coding tasks"
    → claude-3-5-sonnet
```

---

## 3.3 Route Anchors

A **route anchor** is a set of example sentences that define what a particular route "looks like" in semantic space:

```python
ROUTE_ANCHORS = {
    "coding": {
        "model": "claude-3-5-sonnet-20241022",
        "examples": [
            "Write a Python function to parse JSON",
            "Debug this JavaScript code",
            "Implement a binary search algorithm",
            "Refactor this class to use dependency injection",
            "Fix the bug in my async/await code",
            "How do I implement a REST API in FastAPI?",
        ]
    },
    "math_reasoning": {
        "model": "o1-mini",
        "examples": [
            "Calculate the integral of x squared",
            "Prove that the sum of angles in a triangle is 180 degrees",
            "Solve this differential equation",
            "What is the probability of drawing two aces in a row?",
            "Find the eigenvalues of this matrix",
        ]
    },
    "creative_writing": {
        "model": "claude-3-5-sonnet-20241022",
        "examples": [
            "Write a short story about a robot",
            "Help me write a poem about autumn",
            "Create a fictional dialogue between two scientists",
            "Give me a creative opening for my novel",
        ]
    },
    "general_qa": {
        "model": "gpt-4o-mini",
        "examples": [
            "What is the capital of France?",
            "Who invented the telephone?",
            "When did World War II end?",
            "What does GDP stand for?",
        ]
    }
}
```

---

## 3.4 Building the Semantic Router

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts using OpenAI's embedding model."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([e.embedding for e in response.data])

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

class SemanticRouter:
    """
    Routes requests to models based on semantic similarity to route anchors.
    """

    def __init__(self, anchors: dict, default_model: str = "gpt-4o-mini"):
        self.default_model = default_model
        self.routes: dict[str, dict] = {}  # route_name → {model, embedding_matrix}
        self._build_index(anchors)

    def _build_index(self, anchors: dict):
        """Embed all anchor examples and store as matrices."""
        print("🔨 Building semantic index...")
        for route_name, config in anchors.items():
            embeddings = embed(config["examples"])
            self.routes[route_name] = {
                "model": config["model"],
                "embeddings": embeddings,      # shape: (n_examples, 1536)
                "centroid": embeddings.mean(axis=0),  # average embedding
            }
        print(f"✅ Indexed {len(self.routes)} routes")

    def route(self, query: str, threshold: float = 0.4) -> tuple[str, str, float]:
        """
        Route a query to the best-matching model.
        Returns (model, route_name, similarity_score).
        """
        query_embedding = embed([query])[0]

        best_route = None
        best_score = -1.0

        for route_name, route_data in self.routes.items():
            # Compare to each anchor example, take the max similarity
            scores = [
                cosine_similarity(query_embedding, anchor_emb)
                for anchor_emb in route_data["embeddings"]
            ]
            route_score = max(scores)  # Best match within this route

            if route_score > best_score:
                best_score = route_score
                best_route = route_name

        if best_route is None or best_score < threshold:
            return self.default_model, "default", best_score

        return self.routes[best_route]["model"], best_route, best_score
```

---

## 3.5 Similarity Strategy: Max vs. Centroid

There are two ways to score a route:

### Max Similarity (recommended)
```python
# Compare query to each anchor example, take the highest score
scores = [cosine_similarity(query_emb, anchor) for anchor in route["embeddings"]]
route_score = max(scores)
```
✅ Works well — even one very similar anchor is enough to trigger the route

### Centroid Similarity
```python
# Compare query to the "average" of all anchor examples
route_score = cosine_similarity(query_emb, route["centroid"])
```
⚠️ Less robust — the centroid can become "blurry" when anchors are diverse

---

## 3.6 Classification-Based Routing (Alternative Approach)

Instead of embedding similarity, use an LLM to classify the request type:

```python
CLASSIFIER_SYSTEM_PROMPT = """
Classify the user's request into exactly one category.
Categories:
- coding: code generation, debugging, software implementation
- math: calculations, proofs, equations, statistics
- creative: stories, poems, creative writing, fiction
- general: factual questions, explanations, information lookup

Respond with ONLY the category name, nothing else.
"""

def classify_request(user_message: str) -> str:
    """Use a cheap, fast model to classify the request type."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # Fast and cheap for classification
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        max_tokens=10,
        temperature=0
    )
    return response.choices[0].message.content.strip()

CATEGORY_MODEL_MAP = {
    "coding":   "claude-3-5-sonnet-20241022",
    "math":     "o1-mini",
    "creative": "claude-3-5-sonnet-20241022",
    "general":  "gpt-4o-mini",
}

def llm_classify_and_route(user_message: str) -> tuple[str, str]:
    category = classify_request(user_message)
    model = CATEGORY_MODEL_MAP.get(category, "gpt-4o-mini")
    return model, category
```

**Tradeoff**: LLM classification adds ~200-500ms latency and costs tokens, but is more flexible than embeddings.

---

## 3.7 Hybrid Approach: Rules First, Semantics as Fallback

```python
def hybrid_route(request: str, context: dict) -> tuple[str, str]:
    """
    1. Apply hard rules first (context length, latency budget)
    2. Use semantic routing for content-based decisions
    3. Fall back to default model
    """
    # Hard rules (fast, no extra API call)
    if context.get("input_tokens", 0) > 100_000:
        return "claude-3-5-sonnet-20241022", "rule:long_context"
    if context.get("max_latency_ms", float("inf")) < 800:
        return "claude-3-5-haiku-20241022", "rule:realtime"

    # Semantic routing (requires embedding API call)
    model, route, score = semantic_router.route(request)
    return model, f"semantic:{route} (score={score:.2f})"
```

This is the recommended pattern for production:
- **Hard constraints** (context, latency) → fast rule check, no extra cost
- **Task type** → semantic routing for flexibility

---

## 3.8 When to Use Semantic Routing

| Use semantic routing when | Use rule-based routing when |
|---|---|
| Task types vary widely | Task types are well-defined and stable |
| Keywords alone are unreliable | Simple keyword detection works |
| Multi-language support needed | Single language, English-only |
| Flexibility is important | Latency budget is very tight |
| You have diverse anchor examples | You have fixed categories |

---

## 📌 Key Takeaways

1. **Semantic routing** uses embeddings to match *meaning*, not just words
2. **Route anchors** = labeled example sentences per route (5-20 examples per route)
3. **Max similarity** is better than centroid for anchor matching
4. **Threshold** controls precision vs. recall — tune based on your use case
5. **LLM classification** is an alternative — more flexible, higher latency/cost
6. **Hybrid approach** = rules for hard constraints + semantics for task type
7. **Warm the index** at startup — embedding all anchors once, not per request
