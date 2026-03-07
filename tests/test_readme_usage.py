"""Tests that verify the usage examples from the README actually work.

Uses mock LLM/embedder from conftest to avoid real API calls.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from rag_playbook import Document, create_pattern
from rag_playbook.core.models import EmbeddedChunk, RAGResult
from rag_playbook.core.vector_store import InMemoryVectorStore
from rag_playbook.patterns import PATTERN_REGISTRY, available_pattern_names
from tests.conftest import SAMPLE_CHUNKS, MockEmbedder, MockLLM

# ---------------------------------------------------------------------------
# Library usage (README "Use as a Library" section)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLibraryUsage:
    """Verify the library API shown in README works end-to-end."""

    async def test_create_pattern_and_query(self, mock_llm, mock_embedder, mock_store) -> None:
        pattern = create_pattern("naive", llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")

        assert isinstance(result, RAGResult)
        assert result.answer
        assert result.metadata.cost_usd >= 0
        assert result.metadata.latency_ms >= 0
        assert result.metadata.pattern == "naive"

    async def test_result_has_sources(self, mock_llm, mock_embedder, mock_store) -> None:
        pattern = create_pattern("naive", llm=mock_llm, embedder=mock_embedder, store=mock_store)
        result = await pattern.query("What is the refund policy?")

        assert result.sources is not None
        assert len(result.sources) > 0
        assert result.metadata.final_chunk_count > 0

    async def test_result_metadata_fields(
        self, mock_llm, mock_embedder, mock_store,
    ) -> None:
        pattern = create_pattern(
            "reranking", llm=mock_llm, embedder=mock_embedder, store=mock_store,
        )
        result = await pattern.query("What is the refund policy?")

        assert result.metadata.model == "mock-llm"
        assert result.metadata.tokens_used > 0
        assert result.metadata.steps  # step breakdown is populated


# ---------------------------------------------------------------------------
# All 8 patterns (README "Patterns" table)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAllPatternsRunnable:
    """Every pattern listed in the README must be queryable."""

    EXPECTED_PATTERNS: ClassVar[list[str]] = [
        "naive",
        "hybrid_search",
        "reranking",
        "parent_child",
        "query_decomposition",
        "hyde",
        "self_correcting",
        "agentic",
    ]

    def test_all_eight_patterns_registered(self) -> None:
        for name in self.EXPECTED_PATTERNS:
            assert name in PATTERN_REGISTRY, f"Pattern '{name}' not registered"
        assert len(PATTERN_REGISTRY) == 8

    def test_available_pattern_names(self) -> None:
        names = available_pattern_names()
        assert sorted(names) == sorted(self.EXPECTED_PATTERNS)

    @pytest.mark.parametrize("pattern_name", EXPECTED_PATTERNS)
    async def test_each_pattern_returns_result(
        self, pattern_name, mock_llm, mock_embedder, mock_store
    ) -> None:
        pattern = create_pattern(
            pattern_name, llm=mock_llm, embedder=mock_embedder, store=mock_store
        )
        result = await pattern.query("What is the refund policy?")

        assert isinstance(result, RAGResult)
        assert result.answer
        assert result.metadata.pattern == pattern_name


# ---------------------------------------------------------------------------
# Document ingestion flow (README quickstart + ingest)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIngestionFlow:
    """Test the document -> chunk -> embed -> store -> query pipeline."""

    async def test_ingest_and_query(self) -> None:
        llm = MockLLM()
        embedder = MockEmbedder()
        store = InMemoryVectorStore()

        # Simulate ingestion
        docs = [
            Document(id="d1", content="Refunds are processed within 14 business days."),
            Document(id="d2", content="Shipping takes 5-7 business days."),
        ]

        for doc in docs:
            vecs = await embedder.embed([doc.content])
            chunk = EmbeddedChunk(
                id=doc.id, document_id=doc.id, content=doc.content, embedding=vecs[0]
            )
            await store.add([chunk])

        assert await store.count() == 2

        # Query
        pattern = create_pattern("naive", llm=llm, embedder=embedder, store=store)
        result = await pattern.query("What is the refund policy?")

        assert result.answer
        assert result.metadata.final_chunk_count > 0

    async def test_ingest_with_chunker(self) -> None:
        from rag_playbook.core.chunker import create_chunker

        doc = Document(
            id="long-doc",
            content=(
                "First paragraph about refunds.\n\n"
                "Second paragraph about shipping.\n\n"
                "Third paragraph about support."
            ),
        )
        chunker = create_chunker(strategy="recursive", chunk_size=20, chunk_overlap=0)
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1
        for c in chunks:
            assert c.content
            assert c.document_id == "long-doc"


# ---------------------------------------------------------------------------
# Vector store operations
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorStoreUsage:
    async def test_in_memory_store_add_search_count(self) -> None:
        store = InMemoryVectorStore()
        await store.add(SAMPLE_CHUNKS[:3])

        assert await store.count() == 3

        results = await store.search(SAMPLE_CHUNKS[0].embedding, top_k=2)
        assert len(results) == 2
        assert results[0].score >= results[1].score

    async def test_store_reset(self) -> None:
        store = InMemoryVectorStore()
        await store.add(SAMPLE_CHUNKS)
        assert await store.count() == 5

        await store.reset()
        assert await store.count() == 0

    async def test_store_delete(self) -> None:
        store = InMemoryVectorStore()
        await store.add(SAMPLE_CHUNKS[:2])
        assert await store.count() == 2

        await store.delete([SAMPLE_CHUNKS[0].id])
        assert await store.count() == 1


# ---------------------------------------------------------------------------
# create_pattern factory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreatePatternFactory:
    def test_unknown_pattern_raises(self, mock_llm, mock_embedder, mock_store) -> None:
        from rag_playbook.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="Unknown pattern"):
            create_pattern("nonexistent", llm=mock_llm, embedder=mock_embedder, store=mock_store)

    def test_pattern_has_description(self, mock_llm, mock_embedder, mock_store) -> None:
        for name in available_pattern_names():
            pattern = create_pattern(name, llm=mock_llm, embedder=mock_embedder, store=mock_store)
            assert pattern.description, f"Pattern '{name}' has no description"


# ---------------------------------------------------------------------------
# CLI commands (verify they at least parse and show help)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCLICommands:
    """Verify all CLI commands from README are registered and have help text."""

    COMMANDS: ClassVar[list[str]] = [
        "compare", "run", "recommend", "ingest", "bench", "patterns",
    ]

    @pytest.mark.parametrize("cmd", COMMANDS)
    def test_command_help(self, cmd) -> None:
        from typer.testing import CliRunner

        from rag_playbook.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, f"'{cmd} --help' failed: {result.output}"


# ---------------------------------------------------------------------------
# JSON code fence stripping (generate_json fix)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStripCodeFences:
    def test_strips_json_fence(self) -> None:
        from rag_playbook.core.llm import _strip_code_fences

        raw = '```json\n{"pattern": "naive"}\n```'
        assert _strip_code_fences(raw) == '{"pattern": "naive"}'

    def test_strips_plain_fence(self) -> None:
        from rag_playbook.core.llm import _strip_code_fences

        raw = '```\n{"key": "value"}\n```'
        assert _strip_code_fences(raw) == '{"key": "value"}'

    def test_passes_through_plain_json(self) -> None:
        from rag_playbook.core.llm import _strip_code_fences

        raw = '{"pattern": "naive"}'
        assert _strip_code_fences(raw) == '{"pattern": "naive"}'

    def test_strips_surrounding_whitespace(self) -> None:
        from rag_playbook.core.llm import _strip_code_fences

        raw = '  {"key": "val"}  '
        assert _strip_code_fences(raw) == '{"key": "val"}'

    def test_handles_fence_with_extra_whitespace(self) -> None:
        from rag_playbook.core.llm import _strip_code_fences

        raw = '```json\n  {"a": 1}  \n```'
        assert _strip_code_fences(raw) == '{"a": 1}'
