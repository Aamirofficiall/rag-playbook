"""Pluggable vector store using the Repository pattern.

Every store implements the same interface including ``hybrid_search()``.
If a backend lacks native keyword search (e.g. ChromaDB), we implement
a BM25 fallback so all patterns work with all stores.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter

import structlog

from rag_playbook.core.exceptions import ConfigurationError
from rag_playbook.core.models import EmbeddedChunk, RetrievalMethod, RetrievedChunk

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseVectorStore(ABC):
    """Repository interface for vector storage."""

    @abstractmethod
    async def add(self, chunks: list[EmbeddedChunk]) -> None:
        """Insert embedded chunks into the store."""

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        """Semantic search by vector similarity."""

    @abstractmethod
    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> list[RetrievedChunk]:
        """Combined vector + keyword search with score fusion."""

    @abstractmethod
    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ID."""

    @abstractmethod
    async def count(self) -> int:
        """Return total number of stored chunks."""

    @abstractmethod
    async def reset(self) -> None:
        """Remove all data. Used between pattern runs in compare mode."""


# ---------------------------------------------------------------------------
# In-memory implementation (for testing and lightweight usage)
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class InMemoryVectorStore(BaseVectorStore):
    """Dict-backed vector store with BM25 hybrid search.

    Suitable for testing, small datasets, and the default zero-config
    experience. Not intended for production scale.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, EmbeddedChunk] = {}

    async def add(self, chunks: list[EmbeddedChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        logger.debug("vector_store.add", count=len(chunks), total=len(self._chunks))

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        if not self._chunks:
            return []

        scored: list[tuple[float, EmbeddedChunk]] = []
        for chunk in self._chunks.values():
            sim = _cosine_similarity(query_embedding, chunk.embedding)
            scored.append((sim, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            RetrievedChunk(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=max(0.0, min(1.0, score)),
                retrieval_method=RetrievalMethod.VECTOR,
            )
            for score, chunk in scored[:top_k]
        ]

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion of vector and BM25 scores."""
        if not self._chunks:
            return []

        vector_results = await self.search(query_embedding, top_k=top_k * 2)
        keyword_results = self._bm25_search(query_text, top_k=top_k * 2)

        rrf_scores: dict[str, float] = {}
        k = 60  # RRF constant

        for rank, chunk in enumerate(vector_results):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0) + alpha / (k + rank + 1)

        for rank, chunk in enumerate(keyword_results):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0) + (1 - alpha) / (k + rank + 1)

        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        results: list[RetrievedChunk] = []
        max_score = max(rrf_scores.values()) if rrf_scores else 1.0
        for chunk_id in sorted_ids[:top_k]:
            chunk = self._chunks[chunk_id]  # type: ignore[assignment]
            normalized = rrf_scores[chunk_id] / max_score if max_score > 0 else 0.0
            results.append(
                RetrievedChunk(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=max(0.0, min(1.0, normalized)),
                    retrieval_method=RetrievalMethod.HYBRID,
                )
            )

        return results

    def _bm25_search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Simple BM25 scoring for keyword matching."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        doc_count = len(self._chunks)
        avg_dl = sum(len(_tokenize(c.content)) for c in self._chunks.values()) / max(doc_count, 1)

        # IDF
        df: Counter[str] = Counter()
        for chunk in self._chunks.values():
            unique_tokens = set(_tokenize(chunk.content))
            for token in unique_tokens:
                df[token] += 1

        k1 = 1.5
        b = 0.75
        scores: list[tuple[float, EmbeddedChunk]] = []

        for chunk in self._chunks.values():
            tokens = _tokenize(chunk.content)
            tf: Counter[str] = Counter(tokens)
            dl = len(tokens)
            score = 0.0

            for qt in query_tokens:
                if qt not in df:
                    continue
                idf = math.log((doc_count - df[qt] + 0.5) / (df[qt] + 0.5) + 1)
                tf_norm = (tf[qt] * (k1 + 1)) / (tf[qt] + k1 * (1 - b + b * dl / avg_dl))
                score += idf * tf_norm

            scores.append((score, chunk))

        scores.sort(key=lambda x: x[0], reverse=True)

        max_score = scores[0][0] if scores and scores[0][0] > 0 else 1.0
        return [
            RetrievedChunk(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=max(0.0, min(1.0, s / max_score)),
                retrieval_method=RetrievalMethod.KEYWORD,
            )
            for s, chunk in scores[:top_k]
            if s > 0
        ]

    async def delete(self, chunk_ids: list[str]) -> None:
        for cid in chunk_ids:
            self._chunks.pop(cid, None)

    async def count(self) -> int:
        return len(self._chunks)

    async def reset(self) -> None:
        self._chunks.clear()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_vector_store(provider: str = "memory") -> BaseVectorStore:
    """Create a vector store by provider name."""
    match provider:
        case "memory":
            return InMemoryVectorStore()
        case "chromadb":
            raise ConfigurationError(
                "ChromaDB store not yet implemented. Install chromadb and use 'memory' for now."
            )
        case _:
            raise ConfigurationError(f"Unknown vector store provider: {provider}")
