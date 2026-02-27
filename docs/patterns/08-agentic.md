# Pattern 08: Agentic RAG

## The Problem

A researcher asks: "What are the environmental impacts of lithium mining, and how
do different countries regulate it?" The system does not know which angle to
search first, whether the initial results are sufficient, or whether it needs to
search for regulations separately. A fixed pipeline retrieves once and hopes for
the best. An agentic system decides when, what, and how to search.

## How It Works

```
Question
   |
   v
[LLM with Tool Access]
   |
   +-- LLM decides: search("lithium mining environmental impacts")
   |   |
   |   v
   |   [Embed + Vector Search] --> chunks
   |   |
   |   v
   |   Results fed back to LLM
   |
   +-- LLM decides: search("lithium mining regulations by country")
   |   |
   |   v
   |   [Embed + Vector Search] --> more chunks
   |   |
   |   v
   |   Results fed back to LLM
   |
   +-- LLM decides: answer("Based on the retrieved information...")
       |
       v
       Final Answer (with all accumulated chunks as sources)
```

The LLM operates in a tool-calling loop. It can issue `search(query)` calls to
retrieve chunks, inspect the results, and decide whether to search again with a
refined query or provide a final `answer(text)`. The loop runs up to 5 iterations
to bound cost. If iterations are exhausted, a forced generation from all gathered
chunks produces the final answer.

Tool calls are JSON: `{"tool": "search", "args": {"query": "..."}}` or
`{"tool": "answer", "args": {"text": "..."}}`.

## When to Use

- Intent is unclear and the system needs to explore the corpus
- Research-style queries that require multiple searches with different angles
- The first retrieval is unlikely to be sufficient -- the system needs to iterate
- You want the LLM to formulate its own search strategy

## When NOT to Use

- You need predictable latency and cost -- the number of search iterations varies
  per query (1 to 5 LLM calls + 1 to 5 search calls)
- Simple factual queries that a single retrieval answers correctly
- Budget constraints -- agentic is the most expensive pattern by design
- You need deterministic behavior for testing or compliance

## Code Example

```python
from rag_playbook import create_pattern

pattern = create_pattern("agentic", llm=llm, embedder=embedder, store=store)
result = await pattern.query("What are the environmental impacts of lithium mining?")
print(result.answer)
print(f"Search iterations: {len(result.metadata.steps)}")
```

## Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `_MAX_ITERATIONS` | 5 (hardcoded) | Maximum search-decide loops |
| `default_top_k` | 5 | Chunks per search call |

## Production Tips

- Log every iteration's search query and chunk count. This telemetry shows
  whether the LLM is searching effectively or spinning on redundant queries.
- Set a total token budget and abort early if exceeded. The 5-iteration cap
  bounds calls but not token usage per call.
- Consider combining agentic search with reranking: let the agent decide what to
  search, but rerank the results before feeding them back. This reduces the
  chance of the agent re-searching due to low-quality initial results.
