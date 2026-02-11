"""Document chunking strategies.

Four strategies with a shared interface. All use tiktoken for accurate
token counting rather than naive whitespace splitting.
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from typing import ClassVar

import tiktoken

from rag_playbook.core.exceptions import ChunkingError
from rag_playbook.core.models import Chunk, ChunkMetadata, ChunkStrategy, Document


def _get_encoder(model: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(model)


def _token_count(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text))


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseChunker(ABC):
    """Abstract chunker. Subclasses implement the splitting strategy."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        if chunk_size < 1:
            raise ChunkingError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ChunkingError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._encoder = _get_encoder()

    @property
    @abstractmethod
    def strategy(self) -> ChunkStrategy: ...

    @abstractmethod
    def _split(self, text: str) -> list[str]:
        """Split text into raw string segments."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks with metadata."""
        if not document.content.strip():
            return []

        segments = self._split(document.content)
        chunks: list[Chunk] = []

        for i, segment in enumerate(segments):
            if not segment.strip():
                continue
            chunks.append(
                Chunk(
                    id=f"{document.id}:{uuid.uuid4().hex[:8]}",
                    document_id=document.id,
                    content=segment,
                    metadata=ChunkMetadata(
                        chunk_index=i,
                        overlap_tokens=self._chunk_overlap if i > 0 else 0,
                    ),
                )
            )

        return chunks

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        """Chunk multiple documents."""
        result: list[Chunk] = []
        for doc in documents:
            result.extend(self.chunk(doc))
        return result


# ---------------------------------------------------------------------------
# Fixed-size chunker
# ---------------------------------------------------------------------------


class FixedChunker(BaseChunker):
    """Split by token count with overlap. Simple and predictable."""

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.FIXED

    def _split(self, text: str) -> list[str]:
        tokens = self._encoder.encode(text)
        segments: list[str] = []
        start = 0
        step = self._chunk_size - self._chunk_overlap

        while start < len(tokens):
            end = min(start + self._chunk_size, len(tokens))
            segment = self._encoder.decode(tokens[start:end])
            segments.append(segment)
            if end >= len(tokens):
                break
            start += step

        return segments


# ---------------------------------------------------------------------------
# Recursive chunker
# ---------------------------------------------------------------------------


class RecursiveChunker(BaseChunker):
    """Split by paragraph → newline → sentence → word boundaries.

    Mirrors LangChain's RecursiveCharacterTextSplitter approach but
    operates on token counts via tiktoken.
    """

    _SEPARATORS: ClassVar[list[str]] = ["\n\n", "\n", ". ", " "]

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.RECURSIVE

    def _split(self, text: str) -> list[str]:
        return self._recursive_split(text, self._SEPARATORS)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if _token_count(text, self._encoder) <= self._chunk_size:
            return [text]

        if not separators:
            return FixedChunker(self._chunk_size, self._chunk_overlap)._split(text)

        sep = separators[0]
        parts = text.split(sep)
        segments: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if _token_count(candidate, self._encoder) <= self._chunk_size:
                current = candidate
            else:
                if current:
                    segments.append(current)
                if _token_count(part, self._encoder) > self._chunk_size:
                    segments.extend(self._recursive_split(part, separators[1:]))
                    current = ""
                else:
                    current = part

        if current:
            segments.append(current)

        return segments


# ---------------------------------------------------------------------------
# Structural chunker
# ---------------------------------------------------------------------------


class StructuralChunker(BaseChunker):
    """Split on Markdown/HTML headings. Each section becomes a chunk."""

    _HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.STRUCTURAL

    def _split(self, text: str) -> list[str]:
        headings = list(self._HEADING_PATTERN.finditer(text))
        if not headings:
            return FixedChunker(self._chunk_size, self._chunk_overlap)._split(text)

        segments: list[str] = []

        # Content before first heading
        if headings[0].start() > 0:
            preamble = text[: headings[0].start()].strip()
            if preamble:
                segments.append(preamble)

        for i, match in enumerate(headings):
            start = match.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            section = text[start:end].strip()
            if section:
                if _token_count(section, self._encoder) > self._chunk_size:
                    sub_chunks = FixedChunker(self._chunk_size, self._chunk_overlap)._split(section)
                    segments.extend(sub_chunks)
                else:
                    segments.append(section)

        return segments


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_chunker(
    strategy: str = "fixed",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> BaseChunker:
    """Create a chunker by strategy name."""
    match strategy:
        case "fixed":
            return FixedChunker(chunk_size, chunk_overlap)
        case "recursive":
            return RecursiveChunker(chunk_size, chunk_overlap)
        case "structural":
            return StructuralChunker(chunk_size, chunk_overlap)
        case _:
            raise ChunkingError(f"Unknown chunking strategy: {strategy}")
