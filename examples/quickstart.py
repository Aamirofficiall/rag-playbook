"""Quickstart -- ingest documents and query with Naive RAG.

Set your API key before running:
    export RAG_OPENAI_API_KEY="sk-..."

Usage:
    python examples/quickstart.py
"""

import asyncio

from rag_playbook import Document, Settings, create_pattern
from rag_playbook.core.embedder import create_embedder
from rag_playbook.core.llm import create_llm
from rag_playbook.core.vector_store import create_vector_store


# -- Sample documents --------------------------------------------------------

DOCS = [
    Document(
        id="doc-1",
        content=(
            "Retrieval-Augmented Generation (RAG) grounds LLM responses in "
            "external knowledge. It reduces hallucinations by retrieving "
            "relevant context before generating an answer."
        ),
    ),
    Document(
        id="doc-2",
        content=(
            "Chunking splits large documents into smaller pieces so they fit "
            "within embedding model token limits. Common strategies include "
            "fixed-size, recursive, and structural chunking."
        ),
    ),
    Document(
        id="doc-3",
        content=(
            "Vector stores index embeddings for fast approximate nearest "
            "neighbor search. Popular options include ChromaDB, Qdrant, and "
            "pgvector for Postgres."
        ),
    ),
]


async def main() -> None:
    # 1. Load settings from environment / .env file
    settings = Settings()

    # 2. Create shared infrastructure
    llm = create_llm(settings)
    embedder = create_embedder(settings)
    store = create_vector_store("memory")

    # 3. Create the naive (baseline) pattern
    pattern = create_pattern("naive", llm=llm, embedder=embedder, store=store)

    # 4. Embed and ingest documents
    for doc in DOCS:
        embeddings = await embedder.embed([doc.content])
        from rag_playbook.core.models import EmbeddedChunk

        chunk = EmbeddedChunk(
            id=doc.id,
            document_id=doc.id,
            content=doc.content,
            embedding=embeddings[0],
        )
        await store.add([chunk])

    print(f"Ingested {await store.count()} chunks into the vector store.\n")

    # 5. Query
    question = "How does RAG reduce hallucinations?"
    result = await pattern.query(question)

    # 6. Print results
    print(f"Question: {question}")
    print(f"Answer:   {result.answer}\n")
    print(f"Pattern:  {result.metadata.pattern}")
    print(f"Model:    {result.metadata.model}")
    print(f"Latency:  {result.metadata.latency_ms:.0f} ms")
    print(f"Cost:     ${result.metadata.cost_usd:.6f}")
    print(f"Sources:  {result.metadata.final_chunk_count} chunks")


if __name__ == "__main__":
    asyncio.run(main())
