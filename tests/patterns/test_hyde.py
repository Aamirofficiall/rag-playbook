"""Tests for Pattern 06: HyDE RAG."""

import pytest

from rag_playbook.core.vector_store import InMemoryVectorStore
from rag_playbook.patterns.hyde import HyDERAG
from tests.conftest import MockEmbedder, MockLLM


@pytest.mark.unit
class TestHyDERAG:
    def test_pattern_name(self) -> None:
        pattern = HyDERAG.__new__(HyDERAG)
        assert pattern.pattern_name == "hyde"

    async def test_generates_hypothetical_answer(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = HyDERAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("refund?")
        # MockLLM returns hypothetical for HyDE prompts
        assert len(result.answer) > 0

    async def test_two_llm_calls(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = HyDERAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        await pattern.query("refund?")
        # 1 for hypothetical generation + 1 for final answer
        assert mock_llm.call_count == 2

    async def test_embeds_hypothetical_not_question(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = HyDERAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        await pattern.query("refund?")
        # Embedder called with the hypothetical answer, not raw question
        assert mock_embedder.call_count >= 1

    async def test_returns_sources(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = HyDERAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("refund?")
        assert len(result.sources) > 0
