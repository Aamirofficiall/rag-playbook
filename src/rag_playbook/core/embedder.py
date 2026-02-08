"""Embedding providers with a caching decorator layer.

The CachedEmbedder wraps any BaseEmbedder so that identical texts are never
re-embedded. This is critical for the ``compare`` command: the same documents
get embedded once and reused across all 8 patterns, cutting cost by ~8x.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field

from rag_playbook.core.config import Settings
from rag_playbook.core.cost import embedding_cost
from rag_playbook.core.exceptions import ConfigurationError, EmbeddingError

logger = structlog.get_logger(__name__)

_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class EmbeddingResponse(BaseModel):
    """Result of an embedding call with cost tracking."""

    model_config = ConfigDict(frozen=True)

    embeddings: list[list[float]]
    model: str
    token_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    cost_usd: float = Field(ge=0.0)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseEmbedder(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this model."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...


# ---------------------------------------------------------------------------
# Caching decorator
# ---------------------------------------------------------------------------


class CachedEmbedder(BaseEmbedder):
    """Decorator that wraps any embedder with an in-memory SHA-256 cache.

    Cache key = ``sha256(text + model_name)``. On a compare run, patterns
    2-8 get their embeddings for free since pattern 1 already cached them.
    """

    def __init__(self, inner: BaseEmbedder) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}

    def _cache_key(self, text: str) -> str:
        raw = f"{text}:{self._inner.model_name}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            new_embeddings = await self._inner.embed(uncached_texts)
            for idx, text, emb in zip(
                uncached_indices, uncached_texts, new_embeddings, strict=True
            ):
                key = self._cache_key(text)
                self._cache[key] = emb
                results[idx] = emb

        logger.debug(
            "embedder.cache",
            total=len(texts),
            cache_hits=len(texts) - len(uncached_texts),
            cache_misses=len(uncached_texts),
        )

        return [r for r in results if r is not None]

    def dimension(self) -> int:
        return self._inner.dimension()

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding API client with automatic batching."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for OpenAI embeddings")
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            timeout=httpx.Timeout(30),
        )
        self._model = settings.embedding_model
        self._dimension = settings.embedding_dimension

    @property
    def model_name(self) -> str:
        return self._model

    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total_tokens = 0
        start = time.perf_counter()

        for batch_start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[batch_start : batch_start + _BATCH_SIZE]
            payload: dict[str, Any] = {
                "model": self._model,
                "input": batch,
            }

            try:
                resp = await self._client.post("/embeddings", json=payload)
                resp.raise_for_status()
            except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
                raise EmbeddingError(f"Embedding API call failed: {exc}") from exc

            data = resp.json()
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            all_embeddings.extend(item["embedding"] for item in sorted_data)
            total_tokens += data["usage"]["total_tokens"]

        elapsed_ms = (time.perf_counter() - start) * 1000
        cost = embedding_cost(self._model, total_tokens)

        logger.info(
            "embedder.embed",
            model=self._model,
            texts=len(texts),
            tokens=total_tokens,
            latency_ms=round(elapsed_ms, 1),
            cost_usd=round(cost, 6),
        )

        return all_embeddings


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_embedder(settings: Settings, *, cached: bool = True) -> BaseEmbedder:
    """Create an embedder based on configuration.

    Args:
        settings: Application settings.
        cached: Wrap with CachedEmbedder (default True).
    """
    match settings.embedding_provider:
        case "openai":
            inner = OpenAIEmbedder(settings)
        case provider:
            raise ConfigurationError(f"Unknown embedding provider: {provider}")

    if cached:
        return CachedEmbedder(inner)
    return inner
