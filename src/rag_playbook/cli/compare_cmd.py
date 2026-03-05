"""CLI command: compare all RAG patterns side by side.

This is the killer feature — the comparison table that gets
screenshots shared on social media.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console

if TYPE_CHECKING:
    from rag_playbook.core.models import RAGResult
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag_playbook.cli.formatters import (
    print_comparison_header,
    print_comparison_table,
    print_recommendation,
    print_step_breakdown,
)

console = Console()


def compare(
    query: Annotated[str, typer.Option("--query", "-q", help="Question to ask")],
    data: Annotated[Path, typer.Option("--data", help="Path to documents")] = Path("."),
    pattern_filter: Annotated[
        str | None, typer.Option("--patterns", "-p", help="Comma-separated pattern names")
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", help="Chunks to retrieve")] = 5,
    evaluate: Annotated[bool, typer.Option("--evaluate", help="Run LLM quality metrics")] = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Save results to JSON")
    ] = None,
) -> None:
    """Run ALL patterns and compare results side by side."""
    import logging

    import structlog

    logging.disable(logging.CRITICAL)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    )
    asyncio.run(_compare_async(query, data, pattern_filter, top_k, evaluate, output))
    logging.disable(logging.NOTSET)


async def _compare_async(
    query: str,
    data: Path,
    pattern_filter: str | None,
    top_k: int,
    evaluate: bool,
    output: Path | None,
) -> None:
    from rag_playbook.cli.ingest_cmd import _load_file
    from rag_playbook.core.chunker import create_chunker
    from rag_playbook.core.config import Settings
    from rag_playbook.core.embedder import create_embedder
    from rag_playbook.core.llm import create_llm
    from rag_playbook.core.models import EmbeddedChunk
    from rag_playbook.core.vector_store import create_vector_store
    from rag_playbook.patterns import PATTERN_REGISTRY, create_pattern

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

    print_comparison_header(query, len(all_chunks), str(data))

    # Determine which patterns to run
    if pattern_filter:
        names = [n.strip() for n in pattern_filter.split(",")]
    else:
        names = sorted(PATTERN_REGISTRY.keys())

    # Run all patterns
    results: dict[str, RAGResult] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running patterns...", total=len(names))
        for name in names:
            progress.update(task, description=f"Running {name}...")
            pattern = create_pattern(name, llm=llm, embedder=embedder, store=store, config=settings)
            result = await pattern.query(query, top_k=top_k)
            results[name] = result
            progress.advance(task)

    # Find recommendation (best quality/cost tradeoff)
    recommended = _recommend(results)

    # Print results
    print_comparison_table(results, recommended=recommended, show_eval=evaluate)
    print_recommendation(recommended, _recommendation_reasoning(recommended, results))
    print_step_breakdown(recommended, results[recommended])

    # Save if requested
    if output:
        _save_results(output, query, results, recommended)
        console.print(f"Saved results to {output}")


def _recommend(results: dict[str, RAGResult]) -> str:
    """Pick the best pattern using a weighted scoring function."""
    if not results:
        return "naive"

    max_latency = max(r.metadata.latency_ms for r in results.values()) or 1.0
    max_cost = max(r.metadata.cost_usd for r in results.values()) or 0.001

    scores: dict[str, float] = {}
    for name, result in results.items():
        latency_score = 1 - (result.metadata.latency_ms / max_latency)
        cost_score = 1 - (result.metadata.cost_usd / max_cost)
        chunk_score = min(result.metadata.final_chunk_count / 5, 1.0)
        scores[name] = chunk_score * 0.4 + cost_score * 0.3 + latency_score * 0.3

    return max(scores, key=lambda k: scores[k])


def _recommendation_reasoning(name: str, results: dict[str, RAGResult]) -> str:
    """Generate reasoning for the recommendation."""
    result = results[name]
    return (
        f"Best quality/cost tradeoff. "
        f"Latency: {result.metadata.latency_ms:.0f}ms, "
        f"Cost: ${result.metadata.cost_usd:.4f}, "
        f"Chunks: {result.metadata.final_chunk_count}."
    )


def _save_results(path: Path, query: str, results: dict[str, RAGResult], recommended: str) -> None:
    """Save comparison results to a JSON file."""
    data = {
        "query": query,
        "recommended": recommended,
        "results": {name: result.model_dump() for name, result in results.items()},
    }
    path.write_text(json.dumps(data, indent=2, default=str))
