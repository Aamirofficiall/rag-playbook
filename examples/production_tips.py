"""Production tips for deploying rag-playbook patterns.

Demonstrates cost tracking, error handling, and logging configuration
for production use cases.

Usage:
    python examples/production_tips.py
"""

from __future__ import annotations

import asyncio
import logging

import structlog

from rag_playbook import Settings, create_pattern
from rag_playbook.core.embedder import create_embedder
from rag_playbook.core.exceptions import GenerationError, ProviderError
from rag_playbook.core.llm import create_llm
from rag_playbook.core.models import Document, EmbeddedChunk
from rag_playbook.core.vector_store import InMemoryVectorStore


def configure_logging() -> None:
    """Set up structured logging for production."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(level=logging.INFO)


async def main() -> None:
    configure_logging()
    log = structlog.get_logger()

    settings = Settings()
    llm = create_llm(settings)
    embedder = create_embedder(settings)
    store = InMemoryVectorStore()

    # Ingest
    docs = [
        Document(id="policy-1", content="Refunds are processed within 14 business days."),
        Document(id="policy-2", content="Contact support@example.com for refund requests."),
    ]
    for doc in docs:
        embeddings = await embedder.embed([doc.content])
        chunk = EmbeddedChunk(
            id=doc.id, document_id=doc.id, content=doc.content, embedding=embeddings[0]
        )
        await store.add([chunk])

    pattern = create_pattern("self_correcting", llm=llm, embedder=embedder, store=store)

    # Tip 1: Always handle provider errors gracefully
    try:
        result = await pattern.query("What is the refund policy?")
        log.info(
            "query_success",
            pattern=result.metadata.pattern,
            latency_ms=result.metadata.latency_ms,
            cost_usd=result.metadata.cost_usd,
            tokens=result.metadata.tokens_used,
        )
        print(f"Answer: {result.answer}")
    except GenerationError as e:
        log.error("llm_generation_failed", error=str(e))
        print(f"LLM call failed: {e}")
    except ProviderError as e:
        log.error("provider_error", error=str(e))
        print(f"Provider error: {e}")

    # Tip 2: Track cumulative costs across queries
    total_cost = 0.0
    cost_limit = 1.00  # $1 budget

    queries = ["What is the refund timeline?", "How do I contact support?"]
    for query in queries:
        if total_cost >= cost_limit:
            log.warning("cost_limit_reached", total_cost=total_cost, limit=cost_limit)
            print(f"Cost limit ${cost_limit:.2f} reached. Stopping.")
            break

        try:
            result = await pattern.query(query)
            total_cost += result.metadata.cost_usd
            print(f"Q: {query}")
            print(f"A: {result.answer}")
            print(f"   Cost so far: ${total_cost:.6f}\n")
        except ProviderError:
            log.error("query_failed", query=query)

    print(f"\nTotal session cost: ${total_cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
