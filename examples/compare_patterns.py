"""Compare multiple RAG patterns on the same query.

Ingests a small document set, runs several patterns, and prints a
side-by-side comparison table with latency, cost, and chunk counts.

Set your API key before running:
    export RAG_OPENAI_API_KEY="sk-..."

Usage:
    python examples/compare_patterns.py
"""

import asyncio

from rag_playbook import Document, Settings, create_pattern
from rag_playbook.core.embedder import create_embedder
from rag_playbook.core.llm import create_llm
from rag_playbook.core.models import EmbeddedChunk, RAGResult
from rag_playbook.core.vector_store import create_vector_store

DOCS = [
    Document(id="d1", content="RAG combines retrieval with generation to ground answers in facts."),
    Document(id="d2", content="Reranking improves precision by scoring retrieved chunks with a cross-encoder."),
    Document(id="d3", content="HyDE generates a hypothetical answer, embeds it, and retrieves similar real chunks."),
    Document(id="d4", content="Query decomposition breaks complex questions into simpler sub-queries."),
    Document(id="d5", content="Self-correcting RAG validates answers and retries when hallucinations are detected."),
]

PATTERNS_TO_COMPARE = ["naive", "reranking", "hyde", "self_correcting"]


async def ingest(embedder, store) -> None:
    """Embed all documents and add them to the store."""
    for doc in DOCS:
        vecs = await embedder.embed([doc.content])
        await store.add([
            EmbeddedChunk(
                id=doc.id, document_id=doc.id,
                content=doc.content, embedding=vecs[0],
            )
        ])


def print_table(question: str, results: dict[str, RAGResult]) -> None:
    """Print a simple comparison table."""
    header = f"{'Pattern':<20} {'Latency (ms)':>12} {'Cost ($)':>10} {'Chunks':>8}"
    print(f"\nQuery: {question}\n")
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        m = result.metadata
        print(f"{name:<20} {m.latency_ms:>12.0f} {m.cost_usd:>10.6f} {m.final_chunk_count:>8}")
    print()

    # Show each answer
    for name, result in results.items():
        print(f"[{name}] {result.answer[:120]}{'...' if len(result.answer) > 120 else ''}")
    print()


async def main() -> None:
    settings = Settings()
    llm = create_llm(settings)
    embedder = create_embedder(settings)
    store = create_vector_store("memory")

    await ingest(embedder, store)
    print(f"Ingested {await store.count()} chunks.")

    question = "How does RAG improve answer quality?"
    results: dict[str, RAGResult] = {}

    for name in PATTERNS_TO_COMPARE:
        pattern = create_pattern(name, llm=llm, embedder=embedder, store=store)
        results[name] = await pattern.query(question)

    print_table(question, results)


if __name__ == "__main__":
    asyncio.run(main())
