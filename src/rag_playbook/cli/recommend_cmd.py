"""CLI command: recommend a pattern for a given query."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

console = Console()


def recommend(
    query: Annotated[str, typer.Option("--query", "-q", help="Question to analyze")],
) -> None:
    """Analyze a query and suggest the best RAG pattern."""
    import asyncio

    asyncio.run(_recommend_async(query))


async def _recommend_async(query: str) -> None:
    from rag_playbook.core.config import Settings
    from rag_playbook.core.llm import Message, create_llm
    from rag_playbook.core.prompts import RECOMMEND_PROMPT

    settings = Settings()
    llm = create_llm(settings)

    prompt = RECOMMEND_PROMPT.format(question=query)
    result = await llm.generate_json([Message(role="user", content=prompt)])

    pattern = result.get("pattern", "naive")
    reasoning = result.get("reasoning", "No reasoning provided.")
    confidence = result.get("confidence", 0.0)

    console.print("\n[bold]Query analysis:[/bold]")
    console.print(f'  "{query}"\n')
    console.print(f"[bold]Recommended pattern:[/bold] [cyan]{pattern}[/cyan]")
    console.print(f"  Reasoning: {reasoning}")
    console.print(f"  Confidence: {confidence:.0%}\n")
