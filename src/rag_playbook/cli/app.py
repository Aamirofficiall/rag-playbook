"""CLI entry point for rag-playbook.

Uses Typer for the command framework with Rich terminal output.
Each command is a separate module for isolation and testability.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="rag-playbook",
    help="Compare RAG patterns with real benchmarks.",
    add_completion=False,
)

# Import and register commands
from rag_playbook.cli.compare_cmd import compare  # noqa: E402
from rag_playbook.cli.ingest_cmd import ingest  # noqa: E402
from rag_playbook.cli.patterns_cmd import patterns  # noqa: E402
from rag_playbook.cli.recommend_cmd import recommend  # noqa: E402
from rag_playbook.cli.run_cmd import run  # noqa: E402

app.command()(run)
app.command()(compare)
app.command()(recommend)
app.command()(ingest)
app.command()(patterns)


if __name__ == "__main__":
    app()
