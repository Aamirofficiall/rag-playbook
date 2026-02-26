"""Pattern registry and factory.

All pattern classes define a ``_pattern_name`` class attribute.
This module imports them and registers them in the global registry.
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


def _register(cls: type[BaseRAGPattern]) -> None:
    """Register a pattern class by its _pattern_name attribute."""
    name = cls.__dict__.get("_pattern_name")
    if name is None:
        raise ConfigurationError(f"Pattern {cls.__name__} must define '_pattern_name'")
    PATTERN_REGISTRY[name] = cls


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


# Import all pattern classes and register them
from rag_playbook.patterns.agentic import AgenticRAG  # noqa: E402
from rag_playbook.patterns.hybrid_search import HybridSearchRAG  # noqa: E402
from rag_playbook.patterns.hyde import HyDERAG  # noqa: E402
from rag_playbook.patterns.naive import NaiveRAG  # noqa: E402
from rag_playbook.patterns.parent_child import ParentChildRAG  # noqa: E402
from rag_playbook.patterns.query_decomposition import QueryDecompositionRAG  # noqa: E402
from rag_playbook.patterns.reranking import RerankingRAG  # noqa: E402
from rag_playbook.patterns.self_correcting import SelfCorrectingRAG  # noqa: E402

for _cls in [
    NaiveRAG,
    HybridSearchRAG,
    RerankingRAG,
    ParentChildRAG,
    QueryDecompositionRAG,
    HyDERAG,
    SelfCorrectingRAG,
    AgenticRAG,
]:
    _register(_cls)
