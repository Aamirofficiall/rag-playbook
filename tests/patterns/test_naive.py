"""Tests for Pattern 01: Naive RAG."""

import pytest

from rag_playbook.core.vector_store import InMemoryVectorStore
from rag_playbook.patterns.naive import NaiveRAG
from tests.conftest import MockEmbedder, MockLLM


@pytest.mark.unit
class TestNaiveRAG:
    def test_pattern_name(self) -> None:
        pattern = NaiveRAG.__new__(NaiveRAG)
        assert pattern.pattern_name == "naive"

    def test_description(self) -> None:
        pattern = NaiveRAG.__new__(NaiveRAG)
        assert "baseline" in pattern.description.lower()

    async def test_uses_vector_search(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = NaiveRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert result.metadata.retrieval_count > 0

    async def test_no_postprocessing(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = NaiveRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert result.metadata.retrieval_count == result.metadata.final_chunk_count

    async def test_single_llm_call(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = NaiveRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        await pattern.query("What is the refund policy?")
        assert mock_llm.call_count == 1
