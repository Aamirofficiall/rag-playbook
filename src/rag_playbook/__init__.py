"""rag-playbook: Compare RAG patterns with real benchmarks."""

__version__ = "0.2.1"

from rag_playbook.core.config import Settings
from rag_playbook.core.models import (
    Chunk,
    ComparisonResult,
    Document,
    EmbeddedChunk,
    RAGResult,
    RetrievedChunk,
)
from rag_playbook.patterns import create_pattern

__all__ = [
    "Chunk",
    "ComparisonResult",
    "Document",
    "EmbeddedChunk",
    "RAGResult",
    "RetrievedChunk",
    "Settings",
    "create_pattern",
]
