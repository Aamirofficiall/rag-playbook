"""Tests for Pattern 05: Query Decomposition RAG."""

import pytest

from rag_playbook.core.vector_store import InMemoryVectorStore
from rag_playbook.patterns.query_decomposition import QueryDecompositionRAG
from tests.conftest import MockEmbedder, MockLLM


@pytest.mark.unit
class TestQueryDecompositionRAG:
    def test_pattern_name(self) -> None:
        pattern = QueryDecompositionRAG.__new__(QueryDecompositionRAG)
        assert pattern.pattern_name == "query_decomposition"

    async def test_decomposes_into_sub_queries(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = QueryDecompositionRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("How does our refund policy compare to competitors?")
        # MockLLM returns 2 sub-queries for decompose prompts
        assert result.metadata.retrieval_count > 0

    async def test_retrieves_for_each_sub_query(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = QueryDecompositionRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("Compare refund and shipping policies")
        # Should have more chunks than a single retrieval
        assert result.metadata.retrieval_count >= 5

    async def test_preprocess_step_shows_multiple_queries(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = QueryDecompositionRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("Compare refund and shipping")
        preprocess_step = result.metadata.steps[0]
        assert preprocess_step.step == "preprocess"
        assert preprocess_step.detail is not None
        assert "2 queries" in preprocess_step.detail

    async def test_multiple_llm_calls(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = QueryDecompositionRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        await pattern.query("Compare policies")
        # 1 for decomposition + 1 for generation
        assert mock_llm.call_count == 2

    async def test_returns_answer(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = QueryDecompositionRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What about refunds?")
        assert len(result.answer) > 0
