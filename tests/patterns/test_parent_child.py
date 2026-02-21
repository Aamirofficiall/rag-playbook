"""Tests for Pattern 04: Parent-Child RAG."""

import pytest

from rag_playbook.core.vector_store import InMemoryVectorStore
from rag_playbook.patterns.parent_child import ParentChildRAG
from tests.conftest import MockEmbedder, MockLLM


@pytest.mark.unit
class TestParentChildRAG:
    def test_pattern_name(self) -> None:
        pattern = ParentChildRAG.__new__(ParentChildRAG)
        assert pattern.pattern_name == "parent_child"

    async def test_returns_answer(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = ParentChildRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert len(result.answer) > 0

    async def test_deduplicates_parent_content(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = ParentChildRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        contents = [s.content for s in result.sources]
        assert len(contents) == len(set(contents))

    async def test_final_chunk_count_lte_retrieval(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = ParentChildRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert result.metadata.final_chunk_count <= result.metadata.retrieval_count

    async def test_single_llm_call(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = ParentChildRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        await pattern.query("What is the refund policy?")
        assert mock_llm.call_count == 1
