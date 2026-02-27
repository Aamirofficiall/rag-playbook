# Pattern 03: Reranking

## The Problem

Vector search retrieves 5 chunks, but the second and fourth are irrelevant -- they
matched on surface-level vocabulary, not actual relevance to the question. The LLM
generates an answer polluted by those low-quality chunks. You need a second pass
that scores each chunk against the full question jointly, not independently.

## How It Works

```
Question
   |
   v
[Embed Query] --> query vector
   |
   v
[Vector Search] --> top_k * 5 candidates (broad net)
   |
   v
[Score Each Chunk] --> LLM rates relevance 0.0-1.0
   |                   (or cross-encoder in production)
   v
[Sort by Score, Take top_k]
   |
   v
[LLM Generate] --> answer from reranked chunks
```

Stage 1 retrieves 5x more candidates than needed using fast vector search.
Stage 2 scores each candidate against the question using an LLM-based relevance
prompt (production systems use a cross-encoder model like
`cross-encoder/ms-marco-MiniLM-L-6-v2`). Only the top-K highest-scored chunks
proceed to generation.

## When to Use

- Precision matters more than latency -- you need the absolute best chunks
- Your corpus has many similar-looking chunks and vector search alone cannot
  distinguish the most relevant ones
- You are building a system where answer quality directly impacts business
  outcomes (customer support, medical, legal)

## When NOT to Use

- Latency budget is tight -- reranking adds one LLM call per candidate chunk
  (5x top_k calls by default)
- Your corpus is small (under 50 chunks) -- oversampling has no benefit when
  there are few candidates to choose from
- Naive retrieval already produces high-quality results (scores consistently
  above 0.85)

## Code Example

```python
from rag_playbook import create_pattern

pattern = create_pattern("reranking", llm=llm, embedder=embedder, store=store)
result = await pattern.query("What are the SLA guarantees for enterprise tier?")
print(result.answer)
```

## Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `default_top_k` | 5 | Final number of chunks after reranking |
| Oversample factor | 5 (hardcoded) | Multiplier for initial retrieval candidates |

## Production Tips

- Replace the LLM-based scoring with a dedicated cross-encoder model for 10-50x
  faster reranking. Install `sentence-transformers` and use
  `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Batch the scoring calls when possible -- sending all chunks in a single prompt
  reduces round-trips.
- Monitor the score distribution. If most candidates score similarly, reranking
  is not adding value and you should investigate chunking quality instead.
