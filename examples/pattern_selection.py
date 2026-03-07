"""Pattern selection -- choose the right RAG pattern for your query type.

Demonstrates how different query types benefit from different patterns,
and how to programmatically select the best pattern.

Set your API key before running:
    export RAG_OPENAI_API_KEY="sk-..."

Usage:
    python examples/pattern_selection.py
"""

from __future__ import annotations

import asyncio

from rag_playbook import Document, Settings, create_pattern
from rag_playbook.core.embedder import create_embedder
from rag_playbook.core.llm import create_llm
from rag_playbook.core.models import EmbeddedChunk
from rag_playbook.core.vector_store import InMemoryVectorStore

DOCS = [
    Document(id="d1", content="The TechCorp API uses OAuth 2.0 with Bearer tokens for authentication."),
    Document(id="d2", content="Rate limits are 1000 req/min for free tier, 10000 req/min for enterprise."),
    Document(id="d3", content="Error code E-4012 means the webhook signature verification failed."),
    Document(id="d4", content="Pagination uses cursor-based approach. Pass next_cursor from the response."),
    Document(id="d5", content="The /users endpoint returns user profiles with fields: id, name, email, role."),
    Document(id="d6", content="Batch operations are supported via POST /batch with up to 100 items per request."),
]


def select_pattern(query: str) -> str:
    """Simple heuristic to pick the best pattern for a query type.

    In production, use `rag-playbook recommend` for LLM-powered selection.
    """
    query_lower = query.lower()

    # Exact codes, IDs, or technical terms -> hybrid search
    if any(marker in query_lower for marker in ["error code", "e-", "endpoint", "status"]):
        return "hybrid_search"

    # Multi-part questions -> query decomposition
    if " and " in query_lower or "?" in query[:-1]:
        return "query_decomposition"

    # Short/ambiguous queries -> HyDE
    if len(query.split()) <= 4:
        return "hyde"

    # High-stakes queries -> self-correcting
    if any(word in query_lower for word in ["security", "compliance", "legal"]):
        return "self_correcting"

    # Default -> naive (fast and cheap)
    return "naive"


async def main() -> None:
    settings = Settings()
    llm = create_llm(settings)
    embedder = create_embedder(settings)
    store = InMemoryVectorStore()

    # Ingest
    for doc in DOCS:
        vecs = await embedder.embed([doc.content])
        await store.add([
            EmbeddedChunk(id=doc.id, document_id=doc.id, content=doc.content, embedding=vecs[0])
        ])

    # Different query types -> different patterns
    queries = [
        "What does error code E-4012 mean?",         # -> hybrid_search
        "How do auth and rate limits work?",          # -> query_decomposition
        "batch operations",                           # -> hyde (short query)
        "What security measures protect the API?",    # -> self_correcting
        "How does pagination work in the API?",       # -> naive
    ]

    print("Automatic pattern selection demo\n")
    for query in queries:
        pattern_name = select_pattern(query)
        pattern = create_pattern(pattern_name, llm=llm, embedder=embedder, store=store)
        result = await pattern.query(query)

        print(f"Q: {query}")
        print(f"   Pattern:  {pattern_name}")
        print(f"   Answer:   {result.answer[:100]}...")
        print(f"   Latency:  {result.metadata.latency_ms:.0f}ms | Cost: ${result.metadata.cost_usd:.6f}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
