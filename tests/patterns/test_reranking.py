"""Tests for Pattern 03: Reranking RAG."""

import pytest

from rag_playbook.core.models import RetrievalMethod
from rag_playbook.core.vector_store import InMemoryVectorStore
from rag_playbook.patterns.reranking import RerankingRAG
from tests.conftest import MockEmbedder, MockLLM


@pytest.mark.unit
class TestRerankingRAG:
    def test_pattern_name(self) -> None:
        pattern = RerankingRAG.__new__(RerankingRAG)
        assert pattern.pattern_name == "reranking"

    async def test_oversamples_retrieval(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = RerankingRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?", top_k=2)
        # Should retrieve more than top_k, then rerank down
        assert result.metadata.retrieval_count >= result.metadata.final_chunk_count

    async def test_reranked_chunks_marked(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = RerankingRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        for source in result.sources:
            assert source.retrieval_method == RetrievalMethod.RERANKED

    async def test_multiple_llm_calls_for_scoring(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = RerankingRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        await pattern.query("What is the refund policy?")
        # 1 call per chunk for scoring + 1 for generation
        assert mock_llm.call_count > 1

    async def test_returns_answer(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = RerankingRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert len(result.answer) > 0

    async def test_parse_score_valid(self) -> None:
        assert RerankingRAG._parse_score("0.85") == pytest.approx(0.85)

    async def test_parse_score_clamps(self) -> None:
        assert RerankingRAG._parse_score("1.5") == 1.0
        assert RerankingRAG._parse_score("-0.5") == pytest.approx(0.5)

    async def test_parse_score_fallback(self) -> None:
        assert RerankingRAG._parse_score("no number") == 0.5
