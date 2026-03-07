"""Streaming ingestion -- load documents from a directory and ingest incrementally.

Shows how to use the chunker and embedder to process documents one at a time,
which is useful for large document sets that don't fit in memory.

Set your API key before running:
    export RAG_OPENAI_API_KEY="sk-..."

Usage:
    python examples/streaming_ingest.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rag_playbook import Settings, create_pattern
from rag_playbook.core.chunker import create_chunker
from rag_playbook.core.embedder import create_embedder
from rag_playbook.core.llm import create_llm
from rag_playbook.core.models import Document, EmbeddedChunk
from rag_playbook.core.vector_store import InMemoryVectorStore


async def ingest_directory(
    directory: Path,
    embedder,
    store,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> int:
    """Ingest all .txt and .md files from a directory."""
    chunker = create_chunker(strategy="recursive", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    total_chunks = 0

    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix not in {".txt", ".md", ".markdown"}:
            continue

        doc = Document(id=path.stem, content=path.read_text(encoding="utf-8"))
        chunks = chunker.chunk(doc)

        if not chunks:
            continue

        texts = [c.content for c in chunks]
        vectors = await embedder.embed(texts)

        embedded = [
            EmbeddedChunk(
                id=c.id,
                document_id=c.document_id,
                content=c.content,
                metadata=c.metadata,
                embedding=vec,
            )
            for c, vec in zip(chunks, vectors)
        ]
        await store.add(embedded)
        total_chunks += len(embedded)
        print(f"  {path.name}: {len(chunks)} chunks")

    return total_chunks


async def main() -> None:
    settings = Settings()
    llm = create_llm(settings)
    embedder = create_embedder(settings)
    store = InMemoryVectorStore()

    # Ingest sample_docs/ (or any directory you like)
    docs_dir = Path(__file__).resolve().parent.parent / "sample_docs"
    if not docs_dir.exists():
        print(f"Directory not found: {docs_dir}")
        print("Create some .md or .txt files there, or point to your own docs.")
        return

    print(f"Ingesting from {docs_dir}...\n")
    total = await ingest_directory(docs_dir, embedder, store)
    print(f"\nTotal: {total} chunks in store.\n")

    # Query the ingested documents
    pattern = create_pattern("naive", llm=llm, embedder=embedder, store=store)
    question = "What is the refund policy?"
    result = await pattern.query(question)

    print(f"Q: {question}")
    print(f"A: {result.answer}\n")
    print(f"Latency: {result.metadata.latency_ms:.0f}ms")
    print(f"Cost:    ${result.metadata.cost_usd:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
