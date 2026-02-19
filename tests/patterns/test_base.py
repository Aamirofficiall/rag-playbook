"""Tests for the base RAG pattern Template Method."""

import pytest

from rag_playbook.core.models import RAGResult
from rag_playbook.core.vector_store import InMemoryVectorStore
from rag_playbook.patterns.naive import NaiveRAG
from tests.conftest import MockEmbedder, MockLLM


@pytest.mark.unit
class TestBasePattern:
    """Test the Template Method orchestration via NaiveRAG (simplest subclass)."""

    async def test_query_returns_rag_result(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = NaiveRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert isinstance(result, RAGResult)
        assert len(result.answer) > 0

    async def test_metadata_tracks_pattern_name(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = NaiveRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert result.metadata.pattern == "naive"

    async def test_metadata_tracks_model_names(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = NaiveRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert result.metadata.model == "mock-llm"
        assert result.metadata.embedding_model == "mock-embedder"

    async def test_metadata_has_timing_steps(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = NaiveRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        step_names = [s.step for s in result.metadata.steps]
        assert step_names == ["preprocess", "retrieve", "postprocess", "generate", "validate"]

    async def test_latency_is_positive(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = NaiveRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert result.metadata.latency_ms > 0

    async def test_sources_are_retrieved_chunks(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = NaiveRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert len(result.sources) > 0
        assert all(hasattr(s, "score") for s in result.sources)

    async def test_top_k_limits_results(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = NaiveRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?", top_k=2)
        assert len(result.sources) <= 2

    async def test_token_count_tracked(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = NaiveRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert result.metadata.tokens_used > 0
