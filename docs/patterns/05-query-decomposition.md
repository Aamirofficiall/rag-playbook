# Pattern 05: Query Decomposition

## The Problem

A user asks: "Compare the pricing of Plan A and Plan B, and explain which one
includes 24/7 support." This is three questions in one. A single embedding of the
full question lands somewhere in a vague middle ground and retrieves chunks that
partially match multiple topics but fully match none. The answer is incomplete or
confused.

## How It Works

```
Complex Question
   |
   v
[LLM Decompose] --> ["What is Plan A pricing?",
   |                  "What is Plan B pricing?",
   |                  "Which plan includes 24/7 support?"]
   v
For each sub-query:
   [Embed] --> [Vector Search] --> chunks
   |
   v
[Merge All Retrieved Chunks]
   |
   v
[LLM Generate] --> synthesized answer from all sub-query chunks
```

The LLM decomposes the question into 2-4 self-contained sub-queries (returned as
a JSON array). Each sub-query is embedded and retrieved independently. All
retrieved chunks are merged and passed to the final generation step, which
synthesizes across all topics.

## When to Use

- Users ask compound questions that span multiple topics or document sections
- Questions contain "compare", "and", "also", "what about" -- signals of multiple
  information needs
- Recall is more important than latency -- you want to find all relevant chunks
  even at the cost of extra LLM calls

## When NOT to Use

- Queries are simple and single-topic -- decomposition adds an LLM call that
  returns the original question unchanged
- Latency budget is under 2 seconds -- the decomposition step alone takes 0.5-1s
- The LLM tends to over-decompose simple questions into redundant sub-queries,
  wasting retrieval budget

## Code Example

```python
from rag_playbook import create_pattern

pattern = create_pattern("query_decomposition", llm=llm, embedder=embedder, store=store)
result = await pattern.query("Compare Plan A and Plan B pricing, and which has 24/7 support?")
print(result.answer)
```

## Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `default_top_k` | 5 | Chunks retrieved per sub-query |
| Max sub-queries | 4 (prompt-enforced) | Bounds the decomposition to 2-4 queries |

## Production Tips

- Monitor the decomposition output. If the LLM frequently returns the original
  question as-is, your queries are already simple and this pattern adds no value.
- Cap the total chunks across all sub-queries to avoid blowing the LLM context
  window. With 4 sub-queries at top_k=5, that is 20 chunks -- deduplicate and
  trim.
- Combine with reranking: decompose, retrieve broadly for each sub-query, then
  rerank the merged set before generation.
