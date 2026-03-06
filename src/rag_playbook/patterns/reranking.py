"""Pattern 03: Re-ranking with Cross-Encoder.

Two-stage retrieval: first cast a wide net (retrieve top_k * 5
candidates via fast vector search), then rerank with a cross-encoder
model for precision. The cross-encoder scores query-chunk pairs
jointly, which is far more accurate than independent embedding
similarity but too slow for the initial retrieval.
"""

from __future__ import annotations

from rag_playbook.core.models import RetrievalMethod, RetrievedChunk
from rag_playbook.patterns.base import BaseRAGPattern

_OVERSAMPLE_FACTOR = 5


class RerankingRAG(BaseRAGPattern):
    """Broad retrieve -> cross-encoder rerank for precision."""

    _pattern_name = "reranking"

    @property
    def pattern_name(self) -> str:
        return self._pattern_name

    @property
    def description(self) -> str:
        return "Retrieve broadly, rerank with cross-encoder for precision"

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Retrieve top_k * OVERSAMPLE_FACTOR candidates for reranking."""
        embeddings = await self.embedder.embed([query])
        return await self.store.search(embeddings[0], top_k=top_k * _OVERSAMPLE_FACTOR)

    async def postprocess_chunks(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Rerank chunks using LLM-based scoring.

        In production, this would use a cross-encoder model like
        ``cross-encoder/ms-marco-MiniLM-L-6-v2``. For now, we use a
        lightweight LLM-based relevance scoring approach that works
        without the sentence-transformers dependency.
        """
        if not chunks:
            return []

        top_k = self.config.default_top_k
        scored: list[tuple[float, RetrievedChunk]] = []

        for chunk in chunks:
            score = await self._score_relevance(question, chunk.content)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            RetrievedChunk(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=max(0.0, min(1.0, score)),
                retrieval_method=RetrievalMethod.RERANKED,
            )
            for score, chunk in scored[:top_k]
        ]

    async def _score_relevance(self, question: str, chunk_text: str) -> float:
        """Score a single chunk's relevance to the question via LLM."""
        from rag_playbook.core.llm import Message

        prompt = (
            f"Rate the relevance of this text to the query on a scale of 0.0 to 1.0.\n\n"
            f"Query: {question}\n\n"
            f"Text: {chunk_text[:500]}\n\n"
            f"Respond with only a number between 0.0 and 1.0."
        )
        response = await self.llm.generate([Message(role="user", content=prompt)])
        return self._parse_score(response.content)

    @staticmethod
    def _parse_score(text: str) -> float:
        """Extract a float score from LLM output."""
        import re

        match = re.search(r"\d+\.?\d*", text.strip())
        if match:
            return max(0.0, min(1.0, float(match.group())))
        return 0.5
