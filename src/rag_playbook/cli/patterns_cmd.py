"""CLI command: list available patterns."""

from __future__ import annotations

from rag_playbook.cli.formatters import print_patterns_list
from rag_playbook.patterns import PATTERN_REGISTRY


def patterns() -> None:
    """List all available RAG patterns."""
    items = [(name, cls.__new__(cls).description) for name, cls in sorted(PATTERN_REGISTRY.items())]
    print_patterns_list(items)
