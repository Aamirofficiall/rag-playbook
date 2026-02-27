# Which RAG Pattern Should I Use?

This guide helps you pick the right pattern for your use case. Start with the
flowchart, then read the detailed recommendations below.

## Quick Decision Flowchart

```
Is your query simple (one topic, straightforward)?
|
+-- YES --> Is retrieval quality good enough with naive?
|   |
|   +-- YES --> Use naive (Pattern 01)
|   |
|   +-- NO --> Do queries contain specific codes/IDs/names?
|       |
|       +-- YES --> Use hybrid_search (Pattern 02)
|       |
|       +-- NO --> Use reranking (Pattern 03)
|
+-- NO --> Is it a multi-part question?
    |
    +-- YES --> Use query_decomposition (Pattern 05)
    |
    +-- NO --> Is the query very short or ambiguous?
        |
        +-- YES --> Use hyde (Pattern 06)
        |
        +-- NO --> Does accuracy matter more than speed/cost?
            |
            +-- YES --> Use self_correcting (Pattern 07)
            |
            +-- NO --> Use agentic (Pattern 08)

Where does parent_child (Pattern 04) fit?
--> Use it when documents are long (10+ pages) with clear sections.
    Parent-child is a chunking strategy that combines with any pattern.
```

## Pattern Comparison Table

| # | Pattern               | Best For                              | Latency | Cost   | Accuracy | Complexity |
|---|-----------------------|---------------------------------------|---------|--------|----------|------------|
| 01 | naive                | Baselines, simple factual lookups     | Low     | Low    | Baseline | Minimal    |
| 02 | hybrid_search        | Queries with IDs, codes, exact names  | Low     | Low    | Better   | Low        |
| 03 | reranking            | Precision-critical retrieval           | Medium  | Medium | High     | Low        |
| 04 | parent_child         | Long documents with clear structure   | Low     | Low    | Better   | Medium     |
| 05 | query_decomposition  | Multi-part or compound questions       | Medium  | Medium | High     | Low        |
| 06 | hyde                 | Short, vague, or ambiguous queries     | Medium  | Medium | Better   | Low        |
| 07 | self_correcting      | High-stakes answers, zero hallucination| High    | High   | Highest  | Medium     |
| 08 | agentic              | Exploratory, open-ended research       | High    | High   | High     | High       |

## Detailed Recommendations

### Pattern 01: naive

**When to use:** You need a baseline, your queries are simple factual lookups,
or your corpus is small and well-organized. Start here and only move to a more
complex pattern when naive measurably falls short.

**When NOT to use:** Queries contain exact identifiers that vector search misses.
Questions are multi-part or ambiguous. You need verifiable accuracy.

### Pattern 02: hybrid_search

**When to use:** Users search with product codes, legal citation numbers, API
endpoint names, or other strings where exact lexical match matters alongside
semantic similarity. The BM25 + vector fusion (Reciprocal Rank Fusion) catches
what pure embeddings miss.

**When NOT to use:** Your queries are purely conceptual with no specific terms.
The added BM25 index doubles memory usage without benefit if there are no
keyword-sensitive queries.

### Pattern 03: reranking

**When to use:** You have more candidate chunks than you can feed to the LLM and
need to pick the best ones. The two-stage approach (broad retrieve, then
cross-encoder rerank) is the single biggest accuracy win for most retrieval
systems.

**When NOT to use:** Latency budget is tight -- reranking adds an LLM call per
chunk. Your top-K from naive already has high relevance scores. Corpus is tiny
(fewer than 50 chunks) and oversampling has no benefit.

### Pattern 04: parent_child

**When to use:** Documents are long (10+ pages) -- think legal contracts, research
papers, or technical manuals with sections. Small chunks give precise retrieval;
parent context gives the LLM enough surrounding text to generate a coherent
answer.

**When NOT to use:** Documents are already short (emails, tweets, FAQ entries).
Parent expansion would just duplicate content and waste tokens.

### Pattern 05: query_decomposition

**When to use:** Users ask compound questions like "Compare X and Y, and explain
how Z relates." The LLM splits the question into sub-queries, retrieves for each
independently, then synthesizes. Dramatically improves recall for multi-topic
questions.

**When NOT to use:** Queries are simple and single-topic. Decomposition adds an
extra LLM call that provides no benefit for straightforward lookups.

### Pattern 06: hyde

**When to use:** Queries are short (1-3 words) or ambiguous. By generating a
hypothetical answer first and embedding that, HyDE bridges the gap between
question-space and answer-space embeddings. Particularly effective for domain
jargon that the embedding model was not trained on.

**When NOT to use:** Queries are already well-formed and detailed. The
hypothetical answer might introduce bias if the LLM's prior knowledge conflicts
with your corpus.

### Pattern 07: self_correcting

**When to use:** Accuracy is non-negotiable -- medical, legal, financial use
cases. The pattern generates an answer, checks it for faithfulness against the
retrieved context, and retries if hallucination is detected (up to 2 retries).

**When NOT to use:** Latency and cost constraints are tight. Each validation
round adds an LLM call. For casual Q&A or internal tools where occasional
imprecision is acceptable, this is overkill.

### Pattern 08: agentic

**When to use:** The intent is unclear and the system needs to explore. The LLM
decides when, what, and how to search in a tool-calling loop (up to 5
iterations). Good for research-style queries where the first retrieval is
unlikely to be sufficient.

**When NOT to use:** You need predictable latency and cost. Agentic patterns are
inherently non-deterministic in how many searches they perform. Simple factual
queries do not benefit from the overhead.
