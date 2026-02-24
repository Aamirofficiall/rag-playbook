"""Rich terminal formatters for CLI output.

Handles comparison tables, progress bars, step breakdowns, and
the dot-visualization that makes the comparison table screenshot-worthy.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_playbook.core.models import RAGResult

console = Console()

_DOTS = ["○", "●"]


def _score_to_dots(score: float, max_dots: int = 5) -> str:
    """Map a 0-1 score to filled/empty dot visualization."""
    filled = round(score * max_dots)
    return "".join(_DOTS[1] if i < filled else _DOTS[0] for i in range(max_dots))


def print_comparison_header(query: str, chunk_count: int, doc_source: str) -> None:
    """Print the comparison command header panel."""
    console.print(
        Panel(
            f"[bold]rag-playbook compare[/bold]\n"
            f'  Query: "{query}"\n'
            f"  Documents: {chunk_count} chunks from {doc_source}",
            expand=False,
        )
    )


def print_comparison_table(
    results: dict[str, RAGResult],
    recommended: str | None = None,
    show_eval: bool = False,
) -> None:
    """Print the main comparison table with pattern results."""
    table = Table(show_header=True, header_style="bold", show_lines=True)

    table.add_column("Pattern", style="cyan", min_width=20)
    if show_eval:
        table.add_column("Relevance", justify="center", min_width=10)
        table.add_column("Faithful", justify="center", min_width=10)
    table.add_column("Latency", justify="right", min_width=9)
    table.add_column("Cost", justify="right", min_width=8)
    table.add_column("Chunks", justify="center", min_width=7)

    for name, result in results.items():
        marker = " [bold yellow]★[/bold yellow]" if name == recommended else ""
        pattern_cell = f"{name}{marker}"

        row = [pattern_cell]
        if show_eval:
            row.extend(["—", "—"])

        latency = f"{result.metadata.latency_ms:,.0f}ms"
        cost = f"${result.metadata.cost_usd:.4f}"
        chunks = str(result.metadata.final_chunk_count)

        row.extend([latency, cost, chunks])
        table.add_row(*row)

    console.print(table)


def print_recommendation(name: str, reasoning: str) -> None:
    """Print the recommendation section below the table."""
    console.print(f"\n[bold yellow]★[/bold yellow] Recommended: [bold]{name}[/bold]")
    console.print(f"  {reasoning}\n")


def print_step_breakdown(name: str, result: RAGResult) -> None:
    """Print per-step timing breakdown for a pattern."""
    console.print(f"[dim]Step breakdown for {name}:[/dim]")
    for step in result.metadata.steps:
        detail = f" │ {step.detail}" if step.detail else ""
        console.print(f"  {step.step:<14} │ {step.latency_ms:>7.0f}ms{detail}")
    console.print()


def print_single_result(result: RAGResult) -> None:
    """Print the result of a single pattern run."""
    console.print(Panel(f"[bold]{result.metadata.pattern}[/bold]", expand=False))
    console.print(f"\n[bold]Answer:[/bold]\n{result.answer}\n")

    console.print(f"[dim]Model: {result.metadata.model}[/dim]")
    console.print(f"[dim]Latency: {result.metadata.latency_ms:.0f}ms[/dim]")
    console.print(f"[dim]Cost: ${result.metadata.cost_usd:.4f}[/dim]")
    console.print(f"[dim]Tokens: {result.metadata.tokens_used}[/dim]")
    console.print(f"[dim]Chunks: {result.metadata.final_chunk_count}[/dim]\n")

    if result.sources:
        console.print("[bold]Sources:[/bold]")
        for i, source in enumerate(result.sources, 1):
            preview = source.content[:120].replace("\n", " ")
            console.print(f"  {i}. [{source.score:.2f}] {preview}...")
        console.print()

    print_step_breakdown(result.metadata.pattern, result)


def print_patterns_list(patterns: list[tuple[str, str]]) -> None:
    """Print available patterns with descriptions."""
    console.print("\n[bold]Available patterns:[/bold]\n")
    for name, desc in patterns:
        console.print(f"  [cyan]{name:<22}[/cyan] {desc}")
    console.print()
    console.print(
        "[dim]Run `rag-playbook run <pattern> --help` for pattern-specific options.[/dim]"
    )
    console.print("[dim]See docs/WHEN_TO_USE.md for the decision guide.[/dim]\n")
