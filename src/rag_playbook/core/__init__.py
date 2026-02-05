"""Core infrastructure layer — shared across all RAG patterns."""

from rag_playbook.core.config import Settings
from rag_playbook.core.exceptions import (
    ChunkingError,
    ConfigurationError,
    EmbeddingError,
    EvaluationError,
    GenerationError,
    PatternError,
    ProviderError,
    RAGPlaybookError,
    VectorStoreError,
)
from rag_playbook.core.models import (
    Chunk,
    ChunkMetadata,
    ChunkStrategy,
    ComparisonResult,
    Document,
    DocumentMetadata,
    EmbeddedChunk,
    QueryMetadata,
    RAGResult,
    RetrievalMethod,
    RetrievedChunk,
    StepTiming,
)

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "ChunkStrategy",
    "ChunkingError",
    "ComparisonResult",
    "ConfigurationError",
    "Document",
    "DocumentMetadata",
    "EmbeddedChunk",
    "EmbeddingError",
    "EvaluationError",
    "GenerationError",
    "PatternError",
    "ProviderError",
    "QueryMetadata",
    "RAGPlaybookError",
    "RAGResult",
    "RetrievalMethod",
    "RetrievedChunk",
    "Settings",
    "StepTiming",
    "VectorStoreError",
]
