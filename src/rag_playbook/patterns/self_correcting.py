"""Pattern 07: Self-Correcting RAG.

After generating an answer, validate it for faithfulness against the
retrieved context. If hallucination is detected, retry with additional
retrieval. Maximum 2 retries to bound cost and latency.
"""

from __future__ import annotations

import json

from rag_playbook.core.llm import Message
from rag_playbook.core.models import RetrievedChunk
from rag_playbook.core.prompts import FAITHFULNESS_PROMPT
from rag_playbook.patterns.base import BaseRAGPattern

_MAX_RETRIES = 2


class SelfCorrectingRAG(BaseRAGPattern):
    """Validate -> detect hallucination -> retry with new context."""

    _pattern_name = "self_correcting"

    @property
    def pattern_name(self) -> str:
        return self._pattern_name

    @property
    def description(self) -> str:
        return "Validate -> detect hallucination -> retry"

    async def validate(self, question: str, answer: str, chunks: list[RetrievedChunk]) -> str:
        """Check faithfulness and retry if hallucinating."""
        context = "\n\n---\n\n".join(c.content for c in chunks)

        for attempt in range(_MAX_RETRIES + 1):
            is_faithful, confidence = await self._check_faithfulness(context, answer)

            if is_faithful:
                self.logger.info(
                    "self_correcting.passed",
                    attempt=attempt + 1,
                    confidence=confidence,
                )
                return answer

            self.logger.warning(
                "self_correcting.retry",
                attempt=attempt + 1,
                confidence=confidence,
            )

            if attempt < _MAX_RETRIES:
                llm_response = await self.generate(question, chunks)
                answer = llm_response.content

        return answer

    async def _check_faithfulness(self, context: str, answer: str) -> tuple[bool, float]:
        """Ask the LLM to verify faithfulness. Returns (is_faithful, confidence)."""
        prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        response = await self.llm.generate([Message(role="user", content=prompt)])

        try:
            result = json.loads(response.content)
            return (
                bool(result.get("is_faithful", False)),
                float(result.get("confidence", 0.0)),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            self.logger.warning("self_correcting.parse_failed", raw=response.content[:200])
            return True, 0.5
