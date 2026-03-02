"""CLI command: run the full benchmark suite."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()


def bench(
    datasets: Annotated[
        str | None, typer.Option("--datasets", "-d", help="Comma-separated dataset names")
    ] = None,
    pattern_filter: Annotated[
        str | None, typer.Option("--patterns", "-p", help="Comma-separated pattern names")
    ] = None,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Directory for benchmark results")
    ] = Path("benchmark_results"),
) -> None:
    """Run full benchmark suite across datasets and patterns."""
    try:
        from benchmarks.run_all import run_benchmarks  # type: ignore[import-untyped]
    except ImportError:
        console.print("[red]Benchmarks package is not installed.[/red]")
        console.print("Install with: pip install -e '.[benchmarks]'")
        raise typer.Exit(1) from None

    ds_list = [s.strip() for s in datasets.split(",")] if datasets else None
    pt_list = [s.strip() for s in pattern_filter.split(",")] if pattern_filter else None

    console.print("[bold]Running benchmarks...[/bold]")
    console.print(f"  Datasets: {ds_list or 'all'}")
    console.print(f"  Patterns: {pt_list or 'all'}")
    console.print(f"  Output:   {output_dir}\n")

    run_benchmarks(datasets=ds_list, patterns=pt_list, output_dir=output_dir)

    console.print(f"\n[green]Results saved to {output_dir}[/green]")
