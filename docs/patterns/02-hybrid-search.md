# Pattern 02: Hybrid Search

## The Problem

A user searches for "error code E-4012" and gets irrelevant results because
the embedding model treats "E-4012" as a generic token with no meaningful
semantic content. Vector search excels at meaning but fails at exact string
matching. BM25 keyword search handles exact matches but misses semantic
similarity. You need both.

## How It Works

```
Question
   |
   v
[Embed Query] --> query vector
   |
   +-------------------+
   |                   |
   v                   v
[Vector Search]   [BM25 Keyword Search]
   |                   |
   +-------------------+
   |
   v
[Reciprocal Rank Fusion (RRF)]
   |
   v
Top-K fused chunks
   |
   v
[LLM Generate] --> answer
```

Both search methods run against the same corpus. Reciprocal Rank Fusion combines
their ranked lists: `score(doc) = alpha / (k + rank_vector) + (1 - alpha) / (k + rank_bm25)`
where `k=60` is the RRF constant. The `alpha` parameter controls the
vector-vs-keyword weight (default 0.5 = equal).

## When to Use

- Queries contain product codes, error codes, legal citations, API names, or
  other exact strings
- Your corpus mixes technical identifiers with natural language descriptions
- You want a low-overhead improvement over naive without adding LLM calls

## When NOT to Use

- Queries are purely conceptual ("explain how caching works") with no specific
  terms -- the BM25 side adds noise without benefit
- Memory is constrained -- the BM25 index roughly doubles the storage footprint
- Your vector store already has native hybrid search and you want to use its
  implementation instead

## Code Example

```python
from rag_playbook import create_pattern

pattern = create_pattern("hybrid_search", llm=llm, embedder=embedder, store=store)
result = await pattern.query("What does error code E-4012 mean?")
print(result.answer)
```

## Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `hybrid_search_alpha` | 0.5 | Weight balance: 0.0 = pure keyword, 1.0 = pure vector |
| `default_top_k` | 5 | Final number of fused chunks returned |

## Production Tips

- Start with `alpha=0.5` and tune based on your query distribution. Keyword-heavy
  domains (support tickets, legal) benefit from `alpha=0.3`.
- The BM25 implementation is a fallback for stores without native keyword search.
  If your production store (e.g. Elasticsearch, Qdrant) has built-in hybrid,
  prefer that for performance.
- Log which retrieval method (vector vs keyword vs hybrid) produced the top-ranked
  chunk. This tells you whether the hybrid approach is actually helping.
