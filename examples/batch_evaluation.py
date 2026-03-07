"""Batch evaluation -- run multiple queries against a pattern and aggregate metrics.

Useful for benchmarking a pattern against a test set of questions
with known expected answers.

Set your API key before running:
    export RAG_OPENAI_API_KEY="sk-..."

Usage:
    python examples/batch_evaluation.py
"""

from __future__ import annotations

import asyncio
import statistics

from rag_playbook import Document, Settings, create_pattern
from rag_playbook.core.embedder import create_embedder
from rag_playbook.core.llm import create_llm
from rag_playbook.core.models import EmbeddedChunk, RAGResult
from rag_playbook.core.vector_store import InMemoryVectorStore

# Knowledge base
DOCS = [
    Document(id="faq-1", content="Our API rate limit is 1000 requests per minute per API key."),
    Document(id="faq-2", content="Authentication uses Bearer tokens. Include your token in the Authorization header."),
    Document(id="faq-3", content="The /users endpoint supports GET, POST, and DELETE methods."),
    Document(id="faq-4", content="Pagination uses cursor-based pagination with 'next_cursor' in the response."),
    Document(id="faq-5", content="Webhooks are sent via POST to your configured endpoint with HMAC-SHA256 signatures."),
    Document(id="faq-6", content="The API returns JSON responses with UTF-8 encoding. Error responses include a 'code' and 'message' field."),
]

# Test set: queries you want to evaluate
TEST_QUERIES = [
    "What is the rate limit?",
    "How do I authenticate API requests?",
    "What HTTP methods does the users endpoint support?",
    "How does pagination work?",
    "How are webhooks secured?",
]


async def ingest(embedder, store) -> None:
    for doc in DOCS:
        vecs = await embedder.embed([doc.content])
        await store.add([
            EmbeddedChunk(
                id=doc.id, document_id=doc.id,
                content=doc.content, embedding=vecs[0],
            )
        ])


async def main() -> None:
    settings = Settings()
    llm = create_llm(settings)
    embedder = create_embedder(settings)
    store = InMemoryVectorStore()

    await ingest(embedder, store)
    print(f"Ingested {await store.count()} chunks.\n")

    pattern_name = "reranking"
    pattern = create_pattern(pattern_name, llm=llm, embedder=embedder, store=store)

    # Run all queries and collect metrics
    results: list[RAGResult] = []
    for query in TEST_QUERIES:
        result = await pattern.query(query)
        results.append(result)
        print(f"Q: {query}")
        print(f"A: {result.answer[:100]}...")
        print(f"   Latency: {result.metadata.latency_ms:.0f}ms | Cost: ${result.metadata.cost_usd:.6f}\n")

    # Aggregate metrics
    latencies = [r.metadata.latency_ms for r in results]
    costs = [r.metadata.cost_usd for r in results]
    total_cost = sum(costs)

    print("=" * 60)
    print(f"Pattern: {pattern_name}")
    print(f"Queries: {len(results)}")
    print(f"Avg latency: {statistics.mean(latencies):.0f}ms (p50={statistics.median(latencies):.0f}ms)")
    print(f"Total cost:  ${total_cost:.6f}")
    print(f"Avg cost:    ${total_cost / len(results):.6f} per query")


if __name__ == "__main__":
    asyncio.run(main())
