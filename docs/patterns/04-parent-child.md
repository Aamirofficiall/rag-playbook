# Pattern 04: Parent-Child Retrieval

## The Problem

A 50-page contract is chunked into 512-token segments. The user asks about a
specific clause. Vector search finds the right chunk, but the chunk ends
mid-sentence and lacks the surrounding context (definitions, cross-references)
needed for a correct answer. Small chunks give precise search; large chunks give
the LLM enough context. You need both.

## How It Works

```
Question
   |
   v
[Embed Query] --> query vector
   |
   v
[Vector Search] --> top-K small child chunks (precise match)
   |
   v
[Expand to Parent Context]
   For each child chunk:
     - Find adjacent chunks (same document_id, nearby chunk_index)
     - Concatenate child + neighbours into parent window
     - Deduplicate overlapping parents
   |
   v
[LLM Generate] --> answer from expanded parent chunks
```

The retrieval targets small, precise chunks. The postprocessing step is designed to
expand each matched chunk by including surrounding chunks from the same document.

> **Note:** The current implementation returns child chunks as-is without
> expanding to parent context. A production implementation would query the store
> for adjacent chunks (by `document_id` and `chunk_index`) and concatenate them.
> The `_PARENT_WINDOW=2` constant defines how many chunks before/after to include
> once parent expansion is implemented.

## When to Use

- Documents are long (10+ pages) with clear structure -- contracts, research
  papers, technical manuals, regulatory filings
- Answers require surrounding context (definitions, cross-references, section
  headers) that a single chunk cannot capture
- Your chunking produces small segments (256-512 tokens) for retrieval precision

## When NOT to Use

- Documents are already short (emails, chat messages, FAQ entries) -- there is
  no parent context to expand into
- Chunks are large enough (1000+ tokens) that expansion would waste the LLM
  context window
- Your store does not maintain document-level ordering -- the parent expansion
  depends on chunk_index adjacency

## Code Example

```python
from rag_playbook import create_pattern

pattern = create_pattern("parent_child", llm=llm, embedder=embedder, store=store)
result = await pattern.query("What are the termination conditions in section 7?")
print(result.answer)
```

## Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `default_top_k` | 5 | Number of child chunks retrieved |
| `_PARENT_WINDOW` | 2 (hardcoded) | Chunks before/after child to include in parent |
| `default_chunk_size` | 512 | Smaller values give more precise retrieval |

## Production Tips

- Use smaller chunk sizes (256-384 tokens) for retrieval when using parent-child.
  The parent expansion compensates for the narrow window.
- Store parent-child relationships explicitly in your vector store metadata
  rather than reconstructing them at query time. This makes expansion O(1) instead
  of scanning all chunks by document_id.
- Combine parent-child with reranking: retrieve small chunks, expand to parents,
  then rerank the parents before sending to the LLM.
