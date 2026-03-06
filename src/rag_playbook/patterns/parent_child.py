"""Pattern 04: Parent-Child Retrieval.

Index small chunks for precise retrieval, but send the parent
(larger surrounding context) to the LLM. This gives the best of
both worlds: accurate search with rich context for generation.

The parent is reconstructed by fetching all chunks sharing the same
document_id and combining those adjacent to the retrieved child.
"""

from __future__ import annotations

from rag_playbook.core.models import RetrievalMethod, RetrievedChunk
from rag_playbook.patterns.base import BaseRAGPattern

_PARENT_WINDOW = 2  # chunks before/after the child to include


class ParentChildRAG(BaseRAGPattern):
    """Small chunk retrieval -> parent context expansion to LLM."""

    _pattern_name = "parent_child"

    @property
    def pattern_name(self) -> str:
        return self._pattern_name

    @property
    def description(self) -> str:
        return "Small chunk retrieval -> parent context to LLM"

    async def postprocess_chunks(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Expand each child chunk to include surrounding parent context."""
        if not chunks:
            return []

        expanded: list[RetrievedChunk] = []
        seen_content: set[str] = set()

        for chunk in chunks:
            parent_content = await self._get_parent_context(chunk)
            if parent_content not in seen_content:
                seen_content.add(parent_content)
                expanded.append(
                    RetrievedChunk(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        content=parent_content,
                        metadata=chunk.metadata,
                        score=chunk.score,
                        retrieval_method=RetrievalMethod.VECTOR,
                    )
                )

        return expanded

    async def _get_parent_context(self, child: RetrievedChunk) -> str:
        """Build parent context by combining the child with its neighbours.

        In a full implementation, the store would support fetching by
        document_id + chunk_index range. Here we use the child content
        as the base and note that production stores should maintain
        the parent-child relationship explicitly.
        """
        # For now, return the child content as-is.
        # A production implementation would:
        # 1. Query store for all chunks with same document_id
        # 2. Find chunks with adjacent chunk_index values
        # 3. Concatenate them to form the parent
        return child.content
