# Pattern 06: HyDE (Hypothetical Document Embeddings)

## The Problem

A user types "caching" -- two words, zero context. The embedding of this query
sits in "question space," far from the "answer space" where your documents live.
Vector search returns vaguely related chunks because there is not enough signal in
the query to match the detailed language of your documents.

## How It Works

```
Short/Ambiguous Question
   |
   v
[LLM Generate Hypothetical Answer]
   "Caching is a technique where frequently accessed data is stored
    in a fast-access layer to reduce latency and database load..."
   |
   v
[Embed Hypothetical Answer] --> vector in "answer space"
   |
   v
[Vector Search] --> chunks that match the hypothetical answer
   |
   v
[LLM Generate] --> final answer from real retrieved chunks
```

Instead of embedding the raw question, HyDE first generates a hypothetical answer
using the LLM. This hypothetical answer uses the same vocabulary and structure as
real documents, so its embedding lands in the right region of the vector space.
The search then finds chunks that are similar to what a good answer looks like.

## When to Use

- Queries are short (1-3 words) or lack enough detail for effective embedding
- The domain uses specialized jargon that the embedding model was not trained on
  -- the LLM's prior knowledge bridges the vocabulary gap
- You want better retrieval without changing your chunking or embedding model

## When NOT to Use

- Queries are already well-formed and specific -- HyDE adds latency without
  improving retrieval quality
- The LLM's prior knowledge conflicts with your corpus (e.g., your documents
  contain proprietary definitions that differ from common usage) -- the
  hypothetical answer will steer retrieval toward wrong chunks
- You need deterministic, reproducible results -- the hypothetical answer varies
  between runs

## Code Example

```python
from rag_playbook import create_pattern

pattern = create_pattern("hyde", llm=llm, embedder=embedder, store=store)
result = await pattern.query("caching")
print(result.answer)
```

## Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `default_top_k` | 5 | Chunks retrieved using the hypothetical embedding |
| `default_llm_model` | gpt-4o-mini | Model that generates the hypothetical answer |

## Production Tips

- Log the hypothetical answer alongside the query. When retrieval quality drops,
  inspect whether the hypothesis steered search in the wrong direction.
- Keep the hypothesis prompt short. A paragraph-length hypothetical is enough --
  longer outputs waste tokens and do not improve embedding quality.
- Consider combining HyDE with hybrid search: embed the hypothetical for vector
  search, but use the original query for keyword search. This gives you the best
  of both worlds.
