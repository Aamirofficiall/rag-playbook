"""Pattern 06: Hypothetical Document Embeddings (HyDE).

Instead of embedding the raw question (which lives in "question space"),
generate a hypothetical answer first, then embed that answer (which
lives in "answer space"). Since documents are also in answer space,
this bridges the semantic gap and improves retrieval for short or
ambiguous queries.
"""

from __future__ import annotations

from rag_playbook.core.llm import Message
from rag_playbook.core.prompts import HYDE_PROMPT
from rag_playbook.patterns.base import BaseRAGPattern


class HyDERAG(BaseRAGPattern):
    """Hypothetical Document Embeddings — question→answer space bridging."""

    _pattern_name = "hyde"

    @property
    def pattern_name(self) -> str:
        return self._pattern_name

    @property
    def description(self) -> str:
        return "Hypothetical Document Embeddings"

    async def preprocess_query(self, question: str) -> list[str]:
        """Generate a hypothetical answer and use it as the search query."""
        prompt = HYDE_PROMPT.format(question=question)
        response = await self.llm.generate([Message(role="user", content=prompt)])

        hypothetical = response.content.strip()
        self.logger.info(
            "hyde.generated",
            question=question,
            hypothetical_length=len(hypothetical),
        )
        return [hypothetical]
