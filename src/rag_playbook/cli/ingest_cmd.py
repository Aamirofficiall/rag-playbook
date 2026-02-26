"""CLI command: ingest documents into the vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import track

from rag_playbook.core.chunker import create_chunker
from rag_playbook.core.config import Settings
from rag_playbook.core.models import Document

console = Console()


def ingest(
    data: Annotated[Path, typer.Option("--data", help="Path to documents directory or file")],
    chunker: Annotated[str, typer.Option("--chunker", help="Chunking strategy")] = "fixed",
    chunk_size: Annotated[int, typer.Option("--chunk-size", help="Tokens per chunk")] = 512,
    chunk_overlap: Annotated[int, typer.Option("--chunk-overlap", help="Overlap tokens")] = 50,
) -> None:
    """Load documents into the vector store."""
    import asyncio

    asyncio.run(_ingest_async(data, chunker, chunk_size, chunk_overlap))


async def _ingest_async(data: Path, chunker_name: str, chunk_size: int, chunk_overlap: int) -> None:
    if not data.exists():
        console.print(f"[red]Error:[/red] Path not found: {data}")
        raise typer.Exit(1)

    settings = Settings()
    chunker = create_chunker(chunker_name, chunk_size, chunk_overlap)

    # Load documents
    documents: list[Document] = []
    if data.is_file():
        documents.append(_load_file(data))
    elif data.is_dir():
        for path in sorted(data.rglob("*")):
            if path.is_file() and path.suffix in {".txt", ".md", ".markdown"}:
                documents.append(_load_file(path))

    if not documents:
        console.print("[yellow]No documents found.[/yellow]")
        raise typer.Exit(1)

    console.print(f"Found {len(documents)} documents")

    # Chunk documents
    all_chunks = []
    for doc in track(documents, description="Chunking..."):
        all_chunks.extend(chunker.chunk(doc))

    console.print(f"Created {len(all_chunks)} chunks ({chunker_name}, size={chunk_size})")

    # Embed and store
    from rag_playbook.core.embedder import create_embedder
    from rag_playbook.core.models import EmbeddedChunk
    from rag_playbook.core.vector_store import create_vector_store

    embedder = create_embedder(settings)
    store = create_vector_store(settings.vector_store_provider)

    texts = [c.content for c in all_chunks]
    console.print("Embedding chunks...")
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
    total = await store.count()
    console.print(f"[green]Ingested {total} chunks into {settings.vector_store_provider}[/green]")


def _load_file(path: Path) -> Document:
    """Load a single file as a Document."""
    content = path.read_text(encoding="utf-8", errors="replace")
    return Document(
        id=str(path),
        content=content,
    )
