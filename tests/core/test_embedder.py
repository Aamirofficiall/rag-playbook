"""Tests for the embedding layer.

Verifies caching, batching, and the decorator pattern behaviour.
"""

import pytest

from rag_playbook.core.config import Settings
from rag_playbook.core.embedder import BaseEmbedder, CachedEmbedder
from rag_playbook.core.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Fake embedder for unit tests
# ---------------------------------------------------------------------------


class FakeEmbedder(BaseEmbedder):
    """Deterministic embedder that tracks call count."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.call_count = 0
        self.texts_embedded: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-model"

    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.texts_embedded.extend(texts)
        return [[hash(t) % 100 / 100.0] * self._dim for t in texts]


# ---------------------------------------------------------------------------
# CachedEmbedder
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCachedEmbedder:
    async def test_cache_hit_skips_inner_call(self) -> None:
        inner = FakeEmbedder()
        cached = CachedEmbedder(inner)

        result1 = await cached.embed(["hello", "world"])
        result2 = await cached.embed(["hello", "world"])

        assert result1 == result2
        assert inner.call_count == 1  # second call served from cache

    async def test_cache_miss_calls_inner(self) -> None:
        inner = FakeEmbedder()
        cached = CachedEmbedder(inner)

        await cached.embed(["hello"])
        await cached.embed(["different"])

        assert inner.call_count == 2

    async def test_partial_cache_hit(self) -> None:
        inner = FakeEmbedder()
        cached = CachedEmbedder(inner)

        await cached.embed(["hello"])
        await cached.embed(["hello", "world"])

        assert inner.call_count == 2
        assert inner.texts_embedded == ["hello", "world"]  # only "world" on second call

    async def test_cache_key_is_deterministic(self) -> None:
        inner = FakeEmbedder()
        cached = CachedEmbedder(inner)

        key1 = cached._cache_key("hello")
        key2 = cached._cache_key("hello")
        assert key1 == key2

    async def test_different_texts_different_keys(self) -> None:
        inner = FakeEmbedder()
        cached = CachedEmbedder(inner)

        key1 = cached._cache_key("hello")
        key2 = cached._cache_key("world")
        assert key1 != key2

    async def test_empty_input_returns_empty(self) -> None:
        inner = FakeEmbedder()
        cached = CachedEmbedder(inner)
        result = await cached.embed([])
        assert result == []

    async def test_dimension_delegates_to_inner(self) -> None:
        inner = FakeEmbedder(dim=768)
        cached = CachedEmbedder(inner)
        assert cached.dimension() == 768

    async def test_model_name_delegates_to_inner(self) -> None:
        inner = FakeEmbedder()
        cached = CachedEmbedder(inner)
        assert cached.model_name == "fake-model"

    async def test_cache_size_tracks_entries(self) -> None:
        inner = FakeEmbedder()
        cached = CachedEmbedder(inner)
        assert cached.cache_size == 0
        await cached.embed(["a", "b", "c"])
        assert cached.cache_size == 3

    async def test_clear_cache(self) -> None:
        inner = FakeEmbedder()
        cached = CachedEmbedder(inner)
        await cached.embed(["hello"])
        assert cached.cache_size == 1
        cached.clear_cache()
        assert cached.cache_size == 0

    async def test_result_order_preserved(self) -> None:
        inner = FakeEmbedder()
        cached = CachedEmbedder(inner)

        await cached.embed(["b"])
        result = await cached.embed(["a", "b", "c"])

        direct = await inner.embed(["a", "b", "c"])
        assert result == direct


# ---------------------------------------------------------------------------
# Factory validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbedderFactory:
    def test_openai_embedder_requires_api_key(self) -> None:
        from rag_playbook.core.embedder import create_embedder

        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            create_embedder(Settings(openai_api_key=""))
