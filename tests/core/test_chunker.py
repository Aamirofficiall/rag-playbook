"""Tests for document chunking strategies.

Verifies chunk sizes, overlap, edge cases, and strategy-specific behaviour.
"""

import pytest

from rag_playbook.core.chunker import (
    FixedChunker,
    RecursiveChunker,
    StructuralChunker,
    create_chunker,
)
from rag_playbook.core.exceptions import ChunkingError
from rag_playbook.core.models import ChunkStrategy, Document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(content: str, doc_id: str = "doc-1") -> Document:
    return Document(id=doc_id, content=content)


def _long_text(words: int = 500) -> str:
    return " ".join(f"word{i}" for i in range(words))


# ---------------------------------------------------------------------------
# BaseChunker validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunkerValidation:
    def test_rejects_zero_chunk_size(self) -> None:
        with pytest.raises(ChunkingError, match="chunk_size must be positive"):
            FixedChunker(chunk_size=0)

    def test_rejects_negative_overlap(self) -> None:
        with pytest.raises(ChunkingError, match="chunk_overlap must be non-negative"):
            FixedChunker(chunk_size=100, chunk_overlap=-1)

    def test_rejects_overlap_gte_chunk_size(self) -> None:
        with pytest.raises(ChunkingError, match="chunk_overlap must be less than"):
            FixedChunker(chunk_size=100, chunk_overlap=100)

    def test_empty_document_returns_empty(self) -> None:
        chunker = FixedChunker(chunk_size=100)
        assert chunker.chunk(_doc("")) == []
        assert chunker.chunk(_doc("   ")) == []


# ---------------------------------------------------------------------------
# FixedChunker
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFixedChunker:
    def test_short_text_single_chunk(self) -> None:
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(_doc("Short text"))
        assert len(chunks) == 1
        assert chunks[0].content == "Short text"

    def test_long_text_produces_multiple_chunks(self) -> None:
        chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(_doc(_long_text(500)))
        assert len(chunks) > 1

    def test_chunk_ids_contain_document_id(self) -> None:
        chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(_doc(_long_text(200), doc_id="my-doc"))
        for c in chunks:
            assert c.document_id == "my-doc"
            assert c.id.startswith("my-doc:")

    def test_chunk_index_metadata(self) -> None:
        chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(_doc(_long_text(200)))
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i

    def test_first_chunk_has_zero_overlap(self) -> None:
        chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(_doc(_long_text(200)))
        assert chunks[0].metadata.overlap_tokens == 0

    def test_subsequent_chunks_record_overlap(self) -> None:
        chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(_doc(_long_text(200)))
        if len(chunks) > 1:
            assert chunks[1].metadata.overlap_tokens == 10

    def test_strategy_is_fixed(self) -> None:
        assert FixedChunker().strategy == ChunkStrategy.FIXED

    def test_chunk_many(self) -> None:
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        docs = [_doc("First doc", "d1"), _doc("Second doc", "d2")]
        chunks = chunker.chunk_many(docs)
        assert len(chunks) == 2
        assert chunks[0].document_id == "d1"
        assert chunks[1].document_id == "d2"


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecursiveChunker:
    def test_short_text_single_chunk(self) -> None:
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(_doc("Short text"))
        assert len(chunks) == 1

    def test_splits_on_paragraphs_first(self) -> None:
        para1 = " ".join(f"word{i}" for i in range(30))
        para2 = " ".join(f"term{i}" for i in range(30))
        text = f"{para1}\n\n{para2}"
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        chunks = chunker.chunk(_doc(text))
        assert len(chunks) >= 2

    def test_long_text_produces_chunks(self) -> None:
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(_doc(_long_text(500)))
        assert len(chunks) > 1

    def test_strategy_is_recursive(self) -> None:
        assert RecursiveChunker().strategy == ChunkStrategy.RECURSIVE


# ---------------------------------------------------------------------------
# StructuralChunker
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStructuralChunker:
    def test_splits_on_markdown_headings(self) -> None:
        text = "# Introduction\nSome intro text.\n\n## Details\nDetail content here."
        chunker = StructuralChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(_doc(text))
        assert len(chunks) == 2

    def test_preserves_heading_in_chunk(self) -> None:
        text = "# Title\nContent under title.\n\n## Section\nMore content."
        chunker = StructuralChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(_doc(text))
        assert chunks[0].content.startswith("# Title")

    def test_preamble_before_first_heading(self) -> None:
        text = "Some preamble text.\n\n# First Section\nContent."
        chunker = StructuralChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(_doc(text))
        assert len(chunks) == 2
        assert "preamble" in chunks[0].content

    def test_no_headings_falls_back_to_fixed(self) -> None:
        chunker = StructuralChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(_doc(_long_text(200)))
        assert len(chunks) > 1

    def test_strategy_is_structural(self) -> None:
        assert StructuralChunker().strategy == ChunkStrategy.STRUCTURAL


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunkerFactory:
    def test_creates_fixed(self) -> None:
        chunker = create_chunker("fixed")
        assert isinstance(chunker, FixedChunker)

    def test_creates_recursive(self) -> None:
        chunker = create_chunker("recursive")
        assert isinstance(chunker, RecursiveChunker)

    def test_creates_structural(self) -> None:
        chunker = create_chunker("structural")
        assert isinstance(chunker, StructuralChunker)

    def test_respects_size_params(self) -> None:
        chunker = create_chunker("fixed", chunk_size=256, chunk_overlap=25)
        assert chunker._chunk_size == 256
        assert chunker._chunk_overlap == 25

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ChunkingError, match="Unknown chunking strategy"):
            create_chunker("magical")
