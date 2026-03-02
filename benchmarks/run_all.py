"""Main benchmark runner for rag-playbook.

Runs every selected RAG pattern against every selected dataset, collects
metrics, and writes a comparison table to JSON.

Usage:
    python -m benchmarks.run_all
    python -m benchmarks.run_all --datasets squad_200 hotpotqa_100
    python -m benchmarks.run_all --patterns naive_rag sentence_window
    python -m benchmarks.run_all --output-dir benchmarks/results/experiment_1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from rag_playbook import Settings, create_pattern
from rag_playbook.core.evaluator import Evaluator
from rag_playbook.core.models import Document, RAGResult
from rag_playbook.patterns import available_pattern_names

from benchmarks.datasets.download import DATASETS, load_dataset

console = Console()

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "latest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_documents(context_docs: list[str]) -> list[Document]:
    """Wrap raw context strings into ``Document`` objects."""
    return [
        Document(content=text, metadata={"source": f"bench_doc_{i}"})
        for i, text in enumerate(context_docs)
    ]


async def _run_pattern_on_dataset(
    pattern_name: str,
    dataset_name: str,
    settings: Settings,
    evaluator: Evaluator,
    progress: Progress,
) -> dict[str, Any]:
    """Run a single pattern against all entries in a dataset.

    Returns a dict with aggregated metrics.
    """
    entries = load_dataset(dataset_name)
    pattern = create_pattern(pattern_name, settings=settings)

    task_id = progress.add_task(
        f"  {pattern_name} / {dataset_name}",
        total=len(entries),
    )

    latencies: list[float] = []
    relevance_scores: list[float] = []
    costs: list[float] = []
    results: list[RAGResult] = []

    for query, context_docs, expected_answer in entries:
        documents = _build_documents(context_docs)

        start = time.perf_counter()
        result: RAGResult = await pattern.aquery(query, documents=documents)
        elapsed = time.perf_counter() - start

        latencies.append(elapsed)

        # Evaluate quality using the project evaluator.
        eval_result = evaluator.evaluate(
            query=query,
            answer=result.answer,
            expected_answer=expected_answer,
            contexts=[d.content for d in documents],
        )
        relevance_scores.append(eval_result.relevance)
        costs.append(getattr(result, "cost", 0.0) or 0.0)
        results.append(result)

        progress.advance(task_id)

    n = len(entries)
    return {
        "pattern": pattern_name,
        "dataset": dataset_name,
        "num_queries": n,
        "avg_latency_s": sum(latencies) / n if n else 0.0,
        "avg_relevance": sum(relevance_scores) / n if n else 0.0,
        "avg_cost_usd": sum(costs) / n if n else 0.0,
        "total_cost_usd": sum(costs),
        "p95_latency_s": sorted(latencies)[int(n * 0.95)] if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_benchmarks(
    dataset_names: list[str],
    pattern_names: list[str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Execute the full benchmark suite and persist results."""

    settings = Settings()
    evaluator = Evaluator(settings=settings)
    all_results: list[dict[str, Any]] = []

    console.rule("[bold]RAG Playbook Benchmark Suite[/bold]")
    console.print(f"Datasets : {', '.join(dataset_names)}")
    console.print(f"Patterns : {', '.join(pattern_names)}")
    console.print(f"Output   : {output_dir}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for dataset_name in dataset_names:
            for pattern_name in pattern_names:
                metrics = await _run_pattern_on_dataset(
                    pattern_name=pattern_name,
                    dataset_name=dataset_name,
                    settings=settings,
                    evaluator=evaluator,
                    progress=progress,
                )
                all_results.append(metrics)

    # Persist -----------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "comparison_table.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    console.print(f"\n[green]Results saved to {out_path}[/green]")

    # Pretty-print summary table ----------------------------------------
    table = Table(title="Benchmark Summary")
    table.add_column("Pattern", style="cyan")
    table.add_column("Dataset", style="magenta")
    table.add_column("Queries", justify="right")
    table.add_column("Avg Latency (s)", justify="right")
    table.add_column("Avg Relevance", justify="right")
    table.add_column("Avg Cost ($)", justify="right")

    for row in all_results:
        table.add_row(
            row["pattern"],
            row["dataset"],
            str(row["num_queries"]),
            f"{row['avg_latency_s']:.3f}",
            f"{row['avg_relevance']:.4f}",
            f"{row['avg_cost_usd']:.5f}",
        )
    console.print(table)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG Playbook benchmarks across patterns and datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS.keys()),
        help="Dataset names to benchmark (default: all).",
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=None,
        help="Pattern names to benchmark (default: all available).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write results (default: benchmarks/results/latest/).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    pattern_names: list[str] = args.patterns or list(available_pattern_names())

    asyncio.run(
        run_benchmarks(
            dataset_names=args.datasets,
            pattern_names=pattern_names,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
