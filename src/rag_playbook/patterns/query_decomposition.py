"""Pattern 05: Query Decomposition.

Break complex multi-part questions into 2-4 simpler sub-queries,
retrieve for each independently, then synthesize the results. This
dramatically improves recall for questions that span multiple topics
or require information from different document sections.
"""

from __future__ import annotations

import json

from rag_playbook.core.llm import Message
from rag_playbook.core.prompts import DECOMPOSE_PROMPT
from rag_playbook.patterns.base import BaseRAGPattern


class QueryDecompositionRAG(BaseRAGPattern):
    """Break complex questions into sub-queries for independent retrieval."""

    _pattern_name = "query_decomposition"

    @property
    def pattern_name(self) -> str:
        return self._pattern_name

    @property
    def description(self) -> str:
        return "Break complex questions into sub-queries"

    async def preprocess_query(self, question: str) -> list[str]:
        """Use LLM to decompose the question into sub-queries."""
        prompt = DECOMPOSE_PROMPT.format(question=question)
        response = await self.llm.generate([Message(role="user", content=prompt)])

        try:
            sub_queries = json.loads(response.content)
            if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                self.logger.info(
                    "query_decomposition.split",
                    original=question,
                    sub_queries=sub_queries,
                )
                return sub_queries
        except (json.JSONDecodeError, TypeError):
            pass

        self.logger.warning(
            "query_decomposition.fallback",
            reason="LLM did not return valid JSON array",
        )
        return [question]
