"""Pattern registry and factory.

Every pattern decorates itself with ``@register_pattern`` which adds
it to the global registry. The factory creates patterns by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_playbook.core.exceptions import ConfigurationError
from rag_playbook.patterns.base import BaseRAGPattern

if TYPE_CHECKING:
    from rag_playbook.core.config import Settings
    from rag_playbook.core.embedder import BaseEmbedder
    from rag_playbook.core.llm import BaseLLM
    from rag_playbook.core.vector_store import BaseVectorStore

PATTERN_REGISTRY: dict[str, type[BaseRAGPattern]] = {}


def register_pattern(cls: type[BaseRAGPattern]) -> type[BaseRAGPattern]:
    """Class decorator to register a pattern in the global registry."""
    # Instantiate temporarily to read the property
    # Instead, use a class-level attribute or convention
    name = cls.__dict__.get("_pattern_name", None)
    if name is None:
        # Fallback: use a temporary approach to get the property value
        # We'll use a naming convention: class must define _pattern_name
        raise ConfigurationError(
            f"Pattern {cls.__name__} must define a '_pattern_name' class attribute"
        )
    PATTERN_REGISTRY[name] = cls
    return cls


def create_pattern(
    name: str,
    *,
    llm: BaseLLM,
    embedder: BaseEmbedder,
    store: BaseVectorStore,
    config: Settings | None = None,
) -> BaseRAGPattern:
    """Factory: create a pattern by name."""
    if name not in PATTERN_REGISTRY:
        available = ", ".join(sorted(PATTERN_REGISTRY.keys()))
        raise ConfigurationError(f"Unknown pattern '{name}'. Available: {available}")
    return PATTERN_REGISTRY[name](llm=llm, embedder=embedder, store=store, config=config)


def all_patterns(
    *,
    llm: BaseLLM,
    embedder: BaseEmbedder,
    store: BaseVectorStore,
    config: Settings | None = None,
) -> list[BaseRAGPattern]:
    """Create all registered patterns with shared infrastructure."""
    return [
        cls(llm=llm, embedder=embedder, store=store, config=config)
        for cls in PATTERN_REGISTRY.values()
    ]


def available_pattern_names() -> list[str]:
    """Return sorted list of registered pattern names."""
    return sorted(PATTERN_REGISTRY.keys())


# Import all patterns to trigger registration
from rag_playbook.patterns import (  # noqa: E402, F401
    agentic,
    hybrid_search,
    hyde,
    naive,
    parent_child,
    query_decomposition,
    reranking,
    self_correcting,
)
