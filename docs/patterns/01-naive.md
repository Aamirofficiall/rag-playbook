# Pattern 01: Naive RAG

## The Problem

You have documents and a question. You need a baseline that works with minimum
complexity. Without a baseline, you cannot measure whether a more advanced pattern
actually improves anything -- you are just adding cost and latency on faith.

Example: "What is the return policy?" against a product FAQ. A simple
embed-retrieve-generate pipeline answers this correctly and fast.

## How It Works

```
Question
   |
   v
[Embed Query] --> query vector
   |
   v
[Vector Search] --> top-K chunks by cosine similarity
   |
   v
[LLM Generate] --> answer from chunks + question
   |
   v
Answer
```

No preprocessing, no postprocessing, no validation. The query goes straight to
embedding, retrieval returns the top-K nearest chunks, and the LLM generates from
those chunks verbatim.

## When to Use

- As a starting point to establish baseline metrics (latency, cost, accuracy)
- Simple factual lookups against a well-organized corpus
- Prototyping and development before investing in advanced patterns

## When NOT to Use

- Queries contain exact identifiers (product codes, legal citations) that
  embeddings miss -- use hybrid_search instead
- Questions are multi-part or compound -- use query_decomposition
- Hallucination risk is unacceptable -- use self_correcting

## Code Example

```python
from rag_playbook import create_pattern

pattern = create_pattern("naive", llm=llm, embedder=embedder, store=store)
result = await pattern.query("What is the return policy?")
print(result.answer)
```

## Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `default_top_k` | 5 | Number of chunks retrieved |
| `default_llm_model` | gpt-4o-mini | Model used for generation |
| `embedding_model` | text-embedding-3-small | Model used for query embedding |

## Production Tips

- Always benchmark naive first. If it scores above your quality threshold, stop.
- Tune `top_k` before switching patterns -- sometimes 10 chunks with naive beats
  5 chunks with reranking.
- Monitor retrieval scores. If the top chunk consistently scores below 0.7, the
  problem is likely in chunking or embedding, not the pattern.
