"""Tests for vector store implementations.

Verifies CRUD, search ranking, hybrid search with RRF, and reset behaviour.
"""

import pytest

from rag_playbook.core.models import ChunkMetadata, EmbeddedChunk, RetrievalMethod
from rag_playbook.core.vector_store import InMemoryVectorStore, create_vector_store

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    content: str,
    embedding: list[float],
    doc_id: str = "doc-1",
) -> EmbeddedChunk:
    return EmbeddedChunk(
        id=chunk_id,
        document_id=doc_id,
        content=content,
        metadata=ChunkMetadata(),
        embedding=embedding,
    )


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def sample_chunks() -> list[EmbeddedChunk]:
    return [
        _make_chunk("c1", "refund policy allows returns within 30 days", [1.0, 0.0, 0.0, 0.0]),
        _make_chunk("c2", "shipping takes 5-7 business days", [0.0, 1.0, 0.0, 0.0]),
        _make_chunk("c3", "customer support is available 24/7", [0.0, 0.0, 1.0, 0.0]),
        _make_chunk("c4", "premium refund for VIP customers", [0.9, 0.1, 0.0, 0.0]),
        _make_chunk("c5", "international shipping rates vary", [0.1, 0.9, 0.0, 0.0]),
    ]


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCRUD:
    async def test_add_and_count(
        self, store: InMemoryVectorStore, sample_chunks: list[EmbeddedChunk]
    ) -> None:
        assert await store.count() == 0
        await store.add(sample_chunks)
        assert await store.count() == 5

    async def test_add_duplicate_overwrites(self, store: InMemoryVectorStore) -> None:
        chunk = _make_chunk("c1", "original", [1.0, 0.0])
        await store.add([chunk])
        assert await store.count() == 1

        updated = _make_chunk("c1", "updated", [0.0, 1.0])
        await store.add([updated])
        assert await store.count() == 1

    async def test_delete_removes_chunks(
        self, store: InMemoryVectorStore, sample_chunks: list[EmbeddedChunk]
    ) -> None:
        await store.add(sample_chunks)
        await store.delete(["c1", "c2"])
        assert await store.count() == 3

    async def test_delete_nonexistent_is_noop(self, store: InMemoryVectorStore) -> None:
        await store.delete(["nonexistent"])
        assert await store.count() == 0

    async def test_reset_clears_all(
        self, store: InMemoryVectorStore, sample_chunks: list[EmbeddedChunk]
    ) -> None:
        await store.add(sample_chunks)
        await store.reset()
        assert await store.count() == 0


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorSearch:
    async def test_search_returns_top_k(
        self, store: InMemoryVectorStore, sample_chunks: list[EmbeddedChunk]
    ) -> None:
        await store.add(sample_chunks)
        results = await store.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2

    async def test_search_ranks_by_similarity(
        self, store: InMemoryVectorStore, sample_chunks: list[EmbeddedChunk]
    ) -> None:
        await store.add(sample_chunks)
        results = await store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert results[0].id == "c1"
        assert results[1].id == "c4"

    async def test_search_returns_retrieval_method(
        self, store: InMemoryVectorStore, sample_chunks: list[EmbeddedChunk]
    ) -> None:
        await store.add(sample_chunks)
        results = await store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results[0].retrieval_method == RetrievalMethod.VECTOR

    async def test_search_empty_store(self, store: InMemoryVectorStore) -> None:
        results = await store.search([1.0, 0.0], top_k=5)
        assert results == []

    async def test_scores_between_0_and_1(
        self, store: InMemoryVectorStore, sample_chunks: list[EmbeddedChunk]
    ) -> None:
        await store.add(sample_chunks)
        results = await store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHybridSearch:
    async def test_hybrid_search_returns_results(
        self, store: InMemoryVectorStore, sample_chunks: list[EmbeddedChunk]
    ) -> None:
        await store.add(sample_chunks)
        results = await store.hybrid_search(
            query_text="refund policy",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            top_k=3,
        )
        assert len(results) > 0
        assert len(results) <= 3

    async def test_hybrid_search_uses_hybrid_method(
        self, store: InMemoryVectorStore, sample_chunks: list[EmbeddedChunk]
    ) -> None:
        await store.add(sample_chunks)
        results = await store.hybrid_search(
            query_text="refund",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            top_k=3,
        )
        for r in results:
            assert r.retrieval_method == RetrievalMethod.HYBRID

    async def test_hybrid_search_empty_store(self, store: InMemoryVectorStore) -> None:
        results = await store.hybrid_search("query", [1.0, 0.0], top_k=5)
        assert results == []

    async def test_keyword_match_boosts_relevant_results(
        self, store: InMemoryVectorStore, sample_chunks: list[EmbeddedChunk]
    ) -> None:
        await store.add(sample_chunks)
        results = await store.hybrid_search(
            query_text="refund",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            top_k=5,
        )
        top_ids = [r.id for r in results[:2]]
        assert "c1" in top_ids
        assert "c4" in top_ids


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorStoreFactory:
    def test_creates_memory_store(self) -> None:
        store = create_vector_store("memory")
        assert isinstance(store, InMemoryVectorStore)

    def test_unknown_provider_raises(self) -> None:
        from rag_playbook.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            create_vector_store("unknown")
