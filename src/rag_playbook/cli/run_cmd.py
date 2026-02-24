"""CLI command: run a single RAG pattern."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from rag_playbook.cli.formatters import print_single_result

console = Console()


def run(
    pattern_name: Annotated[str, typer.Argument(help="Pattern to run (e.g. 'reranking')")],
    query: Annotated[str, typer.Option("--query", "-q", help="Question to ask")],
    data: Annotated[Path, typer.Option("--data", help="Path to documents")] = Path("."),
    top_k: Annotated[int, typer.Option("--top-k", help="Number of chunks to retrieve")] = 5,
) -> None:
    """Run a single RAG pattern and display the result."""
    import asyncio

    asyncio.run(_run_async(pattern_name, query, data, top_k))


async def _run_async(pattern_name: str, query: str, data: Path, top_k: int) -> None:
    from rag_playbook.cli.ingest_cmd import _load_file
    from rag_playbook.core.chunker import create_chunker
    from rag_playbook.core.config import Settings
    from rag_playbook.core.embedder import create_embedder
    from rag_playbook.core.llm import create_llm
    from rag_playbook.core.models import EmbeddedChunk
    from rag_playbook.core.vector_store import create_vector_store
    from rag_playbook.patterns import create_pattern

    settings = Settings()
    llm = create_llm(settings)
    embedder = create_embedder(settings)
    store = create_vector_store(settings.vector_store_provider)

    # Load and index documents
    documents = []
    if data.is_file():
        documents.append(_load_file(data))
    elif data.is_dir():
        for path in sorted(data.rglob("*")):
            if path.is_file() and path.suffix in {".txt", ".md", ".markdown"}:
                documents.append(_load_file(path))

    if not documents:
        console.print("[red]No documents found.[/red]")
        raise typer.Exit(1)

    chunker = create_chunker()
    all_chunks = chunker.chunk_many(documents)
    texts = [c.content for c in all_chunks]
    embeddings = await embedder.embed(texts)

    embedded_chunks = [
        EmbeddedChunk(
            id=chunk.id,
            document_id=chunk.document_id,
            content=chunk.content,
            metadata=chunk.metadata,
            embedding=emb,
        )
        for chunk, emb in zip(all_chunks, embeddings, strict=True)
    ]
    await store.add(embedded_chunks)

    # Run pattern
    pattern = create_pattern(pattern_name, llm=llm, embedder=embedder, store=store, config=settings)

    with console.status(f"Running {pattern_name}..."):
        result = await pattern.query(query, top_k=top_k)

    print_single_result(result)
