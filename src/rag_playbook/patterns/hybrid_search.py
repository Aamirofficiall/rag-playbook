"""Pattern 02: Hybrid Search — vector + BM25 with Reciprocal Rank Fusion.

Combines semantic similarity (vector search) with lexical matching
(BM25 keyword search). Particularly effective when queries contain
specific codes, IDs, product names, or exact phrases that vector
search alone might miss.
"""

from __future__ import annotations

from rag_playbook.core.models import RetrievedChunk
from rag_playbook.patterns.base import BaseRAGPattern


class HybridSearchRAG(BaseRAGPattern):
    """Vector + BM25 keyword search with Reciprocal Rank Fusion."""

    _pattern_name = "hybrid_search"

    @property
    def pattern_name(self) -> str:
        return self._pattern_name

    @property
    def description(self) -> str:
        return "Vector + BM25 keyword fusion"

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Use hybrid search instead of pure vector search."""
        embeddings = await self.embedder.embed([query])
        return await self.store.hybrid_search(
            query_text=query,
            query_embedding=embeddings[0],
            top_k=top_k,
            alpha=self.config.hybrid_search_alpha,
        )
