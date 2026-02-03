"""CLI entry point for rag-playbook."""

import typer

app = typer.Typer(
    name="rag-playbook",
    help="Compare RAG patterns with real benchmarks.",
    add_completion=False,
)


@app.command()
def patterns() -> None:
    """List all available RAG patterns."""
    typer.echo("No patterns registered yet. Coming soon!")


if __name__ == "__main__":
    app()
