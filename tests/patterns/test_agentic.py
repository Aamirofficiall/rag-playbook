"""Tests for Pattern 08: Agentic RAG."""

import pytest

from rag_playbook.core.vector_store import InMemoryVectorStore
from rag_playbook.patterns.agentic import AgenticRAG
from tests.conftest import MockEmbedder, MockLLM


@pytest.mark.unit
class TestAgenticRAG:
    def test_pattern_name(self) -> None:
        pattern = AgenticRAG.__new__(AgenticRAG)
        assert pattern.pattern_name == "agentic"

    async def test_returns_answer(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = AgenticRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert len(result.answer) > 0

    async def test_multiple_iterations(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = AgenticRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        # Should have at least 1 search + 1 answer iteration
        assert len(result.metadata.steps) >= 2

    async def test_collects_sources(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = AgenticRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert len(result.sources) > 0

    async def test_deduplicates_chunks(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = AgenticRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        chunk_ids = [s.id for s in result.sources]
        assert len(chunk_ids) == len(set(chunk_ids))

    async def test_metadata_tracks_iterations(
        self, mock_llm: MockLLM, mock_embedder: MockEmbedder, mock_store: InMemoryVectorStore
    ) -> None:
        pattern = AgenticRAG(llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")
        assert result.metadata.tokens_used > 0
        assert result.metadata.cost_usd >= 0

    async def test_parse_tool_call_valid_json(self) -> None:
        result = AgenticRAG._parse_tool_call('{"tool": "search", "args": {"query": "test"}}')
        assert result is not None
        assert result["tool"] == "search"

    async def test_parse_tool_call_embedded_json(self) -> None:
        result = AgenticRAG._parse_tool_call(
            'I will search for that. {"tool": "search", "args": {"query": "test"}}'
        )
        assert result is not None
        assert result["tool"] == "search"

    async def test_parse_tool_call_invalid(self) -> None:
        assert AgenticRAG._parse_tool_call("no json here") is None
