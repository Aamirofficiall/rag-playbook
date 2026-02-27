# Pattern 07: Self-Correcting RAG

## The Problem

The LLM generates a confident answer that sounds correct but contains a claim not
supported by the retrieved context. In a medical, legal, or financial setting,
this hallucination has real consequences. You need a verification step that
catches unsupported claims and retries before the answer reaches the user.

## How It Works

```
Question
   |
   v
[Embed] --> [Retrieve] --> chunks
   |
   v
[LLM Generate] --> candidate answer
   |
   v
[Faithfulness Check]
   LLM evaluates: is every claim in the answer supported by the context?
   Returns: {is_faithful, unsupported_claims, confidence}
   |
   +-- YES (faithful) --> return answer
   |
   +-- NO (hallucination detected)
       |
       v
       [Retry Generation] --> new candidate answer
       |
       v
       [Faithfulness Check again]  (max 2 retries)
       |
       +-- Return best answer after retries
```

After the initial generation, a separate LLM call checks whether every claim in
the answer is grounded in the retrieved chunks. If the check fails, the system
regenerates and rechecks, up to 2 retries. The confidence score from the
faithfulness check is logged for observability.

## When to Use

- High-stakes domains: medical, legal, financial, compliance -- where an
  unsupported claim has real-world consequences
- You need auditable answers with a faithfulness score attached
- The cost of a wrong answer exceeds the cost of 2-3 extra LLM calls

## When NOT to Use

- Latency budget is tight -- each validation adds a full LLM round-trip
- Cost is a primary concern -- worst case is 3x the generation cost (1 initial +
  2 retries, each with a validation call)
- For casual Q&A or internal tools where occasional imprecision is acceptable

## Code Example

```python
from rag_playbook import create_pattern

pattern = create_pattern("self_correcting", llm=llm, embedder=embedder, store=store)
result = await pattern.query("What are the contraindications for drug X?")
print(result.answer)
print(result.metadata.steps)  # see validation attempts
```

## Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `_MAX_RETRIES` | 2 (hardcoded) | Maximum regeneration attempts on hallucination |
| `default_top_k` | 5 | Chunks used as context for generation and validation |

## Production Tips

- Log the confidence score from every faithfulness check. A system that
  consistently scores below 0.8 has a retrieval problem, not a generation problem.
- If retries exhaust without passing, consider falling back to "I don't have
  enough information" rather than returning a low-confidence answer.
- Combine with reranking: higher-quality input chunks reduce the chance of
  hallucination, meaning fewer retries and lower total cost.
