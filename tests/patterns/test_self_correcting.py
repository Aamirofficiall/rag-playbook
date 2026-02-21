"""Tests for Pattern 07: Self-Correcting RAG."""

import pytest

from rag_playbook.core.vector_store import InMemoryVectorStore
from rag_playbook.patterns.self_correcting import SelfCorrectingRAG
from tests.conftest import MockEmbedder, MockLLM


@pytest.mark.unit
class TestSelfCorrectingRAG:
    def test_pattern_name(self) -> None:
        pattern = SelfCorrectingRAG.__new__(SelfCorrectingRAG)
        assert pattern.pattern_name == "self_correcting"

    async def test_returns_answer(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = SelfCorrectingRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert len(result.answer) > 0

    async def test_faithful_answer_passes_on_first_check(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = SelfCorrectingRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        await pattern.query("What is the refund policy?")
        # MockLLM returns is_faithful=True, so no retries
        # 1 generate + 1 faithfulness check
        assert mock_llm.call_count == 2

    async def test_validate_step_in_metadata(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = SelfCorrectingRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        step_names = [s.step for s in result.metadata.steps]
        assert "validate" in step_names

    async def test_sources_returned(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = SelfCorrectingRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert len(result.sources) > 0
