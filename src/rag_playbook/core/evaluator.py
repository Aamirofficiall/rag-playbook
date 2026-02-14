"""RAG quality evaluation using LLM-as-judge and deterministic metrics.

Metrics are split into two categories:
- **Free metrics** (latency, cost, chunk utilization) — computed from metadata
- **Judge metrics** (relevance, faithfulness) — require an LLM call per eval
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_playbook.core.llm import BaseLLM, Message
from rag_playbook.core.models import RAGResult, RetrievedChunk
from rag_playbook.core.prompts import (
    ANSWER_FAITHFULNESS_JUDGE_PROMPT,
    ANSWER_RELEVANCE_JUDGE_PROMPT,
    RELEVANCE_JUDGE_PROMPT,
)


@dataclass(frozen=True)
class EvalScores:
    """Evaluation scores for a single RAG result."""

    retrieval_relevance: float | None = None
    answer_faithfulness: float | None = None
    answer_relevance: float | None = None
    chunk_utilization: float | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None


class Evaluator:
    """Computes quality metrics for RAG results.

    Free metrics are always available. Judge metrics require a ``judge_llm``
    and cost extra — use a cheap model like gpt-4o-mini for judging.
    """

    def __init__(self, judge_llm: BaseLLM | None = None) -> None:
        self._judge = judge_llm

    # -- Free metrics ---------------------------------------------------------

    @staticmethod
    def chunk_utilization(result: RAGResult) -> float:
        """Fraction of retrieved chunks that appear referenced in the answer."""
        if not result.sources:
            return 0.0
        answer_lower = result.answer.lower()
        referenced = sum(
            1
            for chunk in result.sources
            if any(word in answer_lower for word in chunk.content.lower().split()[:5])
        )
        return referenced / len(result.sources)

    @staticmethod
    def free_metrics(result: RAGResult) -> EvalScores:
        """Compute all metrics that don't require an LLM call."""
        return EvalScores(
            chunk_utilization=Evaluator.chunk_utilization(result),
            latency_ms=result.metadata.latency_ms,
            cost_usd=result.metadata.cost_usd,
        )

    # -- Judge metrics --------------------------------------------------------

    async def retrieval_relevance(self, question: str, chunks: list[RetrievedChunk]) -> float:
        """Average LLM-judged relevance (1-5) normalized to 0-1."""
        if not self._judge or not chunks:
            return 0.0

        total = 0.0
        for chunk in chunks:
            prompt = RELEVANCE_JUDGE_PROMPT.format(question=question, chunk=chunk.content)
            response = await self._judge.generate([Message(role="user", content=prompt)])
            score = self._parse_score(response.content, min_val=1, max_val=5)
            total += (score - 1) / 4  # normalize 1-5 → 0-1

        return total / len(chunks)

    async def answer_faithfulness(self, context: str, answer: str) -> float:
        """LLM-judged faithfulness score (0-1)."""
        if not self._judge:
            return 0.0

        prompt = ANSWER_FAITHFULNESS_JUDGE_PROMPT.format(context=context, answer=answer)
        response = await self._judge.generate([Message(role="user", content=prompt)])
        return self._parse_float(response.content)

    async def answer_relevance(self, question: str, answer: str) -> float:
        """LLM-judged answer relevance (1-5) normalized to 0-1."""
        if not self._judge:
            return 0.0

        prompt = ANSWER_RELEVANCE_JUDGE_PROMPT.format(question=question, answer=answer)
        response = await self._judge.generate([Message(role="user", content=prompt)])
        score = self._parse_score(response.content, min_val=1, max_val=5)
        return (score - 1) / 4

    async def full_eval(self, question: str, result: RAGResult) -> EvalScores:
        """Run all metrics including LLM-judged ones."""
        context = "\n\n---\n\n".join(c.content for c in result.sources)

        retrieval_rel = await self.retrieval_relevance(question, result.sources)
        faithfulness = await self.answer_faithfulness(context, result.answer)
        answer_rel = await self.answer_relevance(question, result.answer)

        return EvalScores(
            retrieval_relevance=retrieval_rel,
            answer_faithfulness=faithfulness,
            answer_relevance=answer_rel,
            chunk_utilization=self.chunk_utilization(result),
            latency_ms=result.metadata.latency_ms,
            cost_usd=result.metadata.cost_usd,
        )

    # -- Parsing helpers ------------------------------------------------------

    @staticmethod
    def _parse_score(text: str, min_val: int, max_val: int) -> int:
        """Extract an integer score from LLM output, clamping to range."""
        import re

        match = re.search(r"\d+", text.strip())
        if match:
            return max(min_val, min(max_val, int(match.group())))
        return min_val

    @staticmethod
    def _parse_float(text: str) -> float:
        """Extract a float from LLM output, clamping to 0-1."""
        import re

        match = re.search(r"\d+\.?\d*", text.strip())
        if match:
            return max(0.0, min(1.0, float(match.group())))
        return 0.0
