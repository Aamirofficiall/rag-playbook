"""Custom exception hierarchy for rag-playbook.

All exceptions inherit from RAGPlaybookError to allow callers to catch
the full family with a single except clause while still being able to
handle specific failure modes when needed.

Hierarchy:
    RAGPlaybookError
    ├── ConfigurationError      — invalid settings / missing env vars
    ├── ProviderError           — LLM or embedding provider failures
    │   ├── GenerationError     — LLM generation failed
    │   └── EmbeddingError      — embedding call failed
    ├── VectorStoreError        — storage layer failures
    ├── ChunkingError           — document chunking failures
    ├── EvaluationError         — metric evaluation failures
    └── PatternError            — RAG pattern execution failures
"""

from __future__ import annotations


class RAGPlaybookError(Exception):
    """Base exception for all rag-playbook errors."""


class ConfigurationError(RAGPlaybookError):
    """Raised when configuration is invalid or incomplete."""


class ProviderError(RAGPlaybookError):
    """Base for external provider failures (LLM, embedding)."""


class GenerationError(ProviderError):
    """Raised when an LLM generation call fails after retries."""


class EmbeddingError(ProviderError):
    """Raised when an embedding call fails after retries."""


class VectorStoreError(RAGPlaybookError):
    """Raised when the vector store operation fails."""


class ChunkingError(RAGPlaybookError):
    """Raised when document chunking fails."""


class EvaluationError(RAGPlaybookError):
    """Raised when RAG evaluation metrics fail to compute."""


class PatternError(RAGPlaybookError):
    """Raised when a RAG pattern fails during execution."""
