"""Tests for Pattern 02: Hybrid Search RAG."""

import pytest

from rag_playbook.core.models import RetrievalMethod
from rag_playbook.core.vector_store import InMemoryVectorStore
from rag_playbook.patterns.hybrid_search import HybridSearchRAG
from tests.conftest import MockEmbedder, MockLLM


@pytest.mark.unit
class TestHybridSearchRAG:
    def test_pattern_name(self) -> None:
        pattern = HybridSearchRAG.__new__(HybridSearchRAG)
        assert pattern.pattern_name == "hybrid_search"

    async def test_uses_hybrid_retrieval(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = HybridSearchRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("refund policy")
        assert any(s.retrieval_method == RetrievalMethod.HYBRID for s in result.sources)

    async def test_returns_results(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = HybridSearchRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("refund policy")
        assert len(result.answer) > 0
        assert len(result.sources) > 0

    async def test_single_llm_call(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = HybridSearchRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        await pattern.query("refund policy")
        assert mock_llm.call_count == 1
