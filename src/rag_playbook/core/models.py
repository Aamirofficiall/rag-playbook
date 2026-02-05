"""Pydantic data models for the RAG pipeline.

Design principle: separate types for each pipeline stage. A Chunk is not an
EmbeddedChunk. Distinct types let the type checker catch pipeline ordering
mistakes at development time rather than producing silent runtime failures.

All models use frozen=True for immutability — pipeline data flows forward,
never mutated in place.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RetrievalMethod(StrEnum):
    """How a chunk was retrieved from the vector store."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    RERANKED = "reranked"


class ChunkStrategy(StrEnum):
    """Available chunking strategies."""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"


# ---------------------------------------------------------------------------
# Metadata value objects
# ---------------------------------------------------------------------------


class DocumentMetadata(BaseModel):
    """Metadata attached to a source document."""

    model_config = ConfigDict(frozen=True)

    title: str = ""
    source: str = ""
    format: str = ""
    page_count: int = 0


class ChunkMetadata(BaseModel):
    """Metadata attached to a chunk."""

    model_config = ConfigDict(frozen=True)

    page_num: int | None = None
    section_title: str = ""
    chunk_index: int = 0
    overlap_tokens: int = 0


# ---------------------------------------------------------------------------
# Pipeline stage models
# ---------------------------------------------------------------------------


class Document(BaseModel):
    """Raw input — a file, a page, a text blob."""

    model_config = ConfigDict(frozen=True)

    id: str
    content: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class Chunk(BaseModel):
    """A piece of a document, pre-embedding."""

    model_config = ConfigDict(frozen=True)

    id: str
    document_id: str
    content: str
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)


class EmbeddedChunk(BaseModel):
    """A chunk with its vector attached. Separate type enforces embedding before storage."""

    model_config = ConfigDict(frozen=True)

    id: str
    document_id: str
    content: str
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    embedding: list[float]

    @model_validator(mode="after")
    def _embedding_must_not_be_empty(self) -> Self:
        if not self.embedding:
            raise ValueError("embedding must not be empty")
        return self


class RetrievedChunk(BaseModel):
    """A chunk returned from search, with relevance score."""

    model_config = ConfigDict(frozen=True)

    id: str
    document_id: str
    content: str
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    score: float = Field(ge=0.0, le=1.0)
    retrieval_method: RetrievalMethod


# ---------------------------------------------------------------------------
# Timing & observability
# ---------------------------------------------------------------------------


class StepTiming(BaseModel):
    """Timing for a single pipeline step."""

    model_config = ConfigDict(frozen=True)

    step: str
    latency_ms: float = Field(ge=0.0)
    detail: str | None = None


class QueryMetadata(BaseModel):
    """Observability data attached to every RAG result."""

    model_config = ConfigDict(frozen=True)

    pattern: str
    model: str
    embedding_model: str
    tokens_used: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    cost_usd: float = Field(ge=0.0)
    retrieval_count: int = Field(ge=0)
    final_chunk_count: int = Field(ge=0)
    steps: list[StepTiming] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class RAGResult(BaseModel):
    """The output of any pattern. Same shape, always."""

    model_config = ConfigDict(frozen=True)

    answer: str
    sources: list[RetrievedChunk]
    metadata: QueryMetadata


class ComparisonResult(BaseModel):
    """Output of the compare command — multiple RAGResults side by side."""

    model_config = ConfigDict(frozen=True)

    query: str
    results: dict[str, RAGResult]
    recommendation: str
    recommendation_reasoning: str
