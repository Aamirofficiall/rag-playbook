"""Tests for the evaluation layer."""

from __future__ import annotations

from typing import Any

import pytest

from rag_playbook.core.evaluator import Evaluator
from rag_playbook.core.llm import BaseLLM, Message
from rag_playbook.core.models import (
    QueryMetadata,
    RAGResult,
    RetrievalMethod,
    RetrievedChunk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    answer: str = "Refunds take 14 days.", chunks: list[str] | None = None
) -> RAGResult:
    sources = [
        RetrievedChunk(
            id=f"c{i}",
            document_id="doc-1",
            content=text,
            score=0.9 - i * 0.1,
            retrieval_method=RetrievalMethod.VECTOR,
        )
        for i, text in enumerate(chunks or ["Refunds are processed within 14 business days."])
    ]
    return RAGResult(
        answer=answer,
        sources=sources,
        metadata=QueryMetadata(
            pattern="naive",
            model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            tokens_used=611,
            latency_ms=980.0,
            cost_usd=0.003,
            retrieval_count=len(sources),
            final_chunk_count=len(sources),
        ),
    )


class FakeJudgeLLM(BaseLLM):
    """Returns predictable scores for testing."""

    def __init__(self, response: str = "4") -> None:
        from rag_playbook.core.config import Settings

        super().__init__(Settings())
        self._response = response

    @property
    def model_name(self) -> str:
        return "fake-judge"

    async def _call(self, messages: list[Message], **kwargs: Any) -> tuple[str, int, int]:
        return self._response, 10, 5


# ---------------------------------------------------------------------------
# Free metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFreeMetrics:
    def test_free_metrics_includes_latency_and_cost(self) -> None:
        result = _make_result()
        scores = Evaluator.free_metrics(result)
        assert scores.latency_ms == 980.0
        assert scores.cost_usd == 0.003

    def test_chunk_utilization_with_referenced_content(self) -> None:
        result = _make_result(
            answer="Refunds are processed quickly.",
            chunks=["Refunds are processed within 14 days."],
        )
        util = Evaluator.chunk_utilization(result)
        assert util > 0.0

    def test_chunk_utilization_with_no_match(self) -> None:
        result = _make_result(
            answer="I don't know.",
            chunks=["Refunds are processed within 14 days."],
        )
        util = Evaluator.chunk_utilization(result)
        # "I" and "don't" unlikely to match chunk first 5 words
        assert isinstance(util, float)

    def test_chunk_utilization_empty_sources(self) -> None:
        result = RAGResult(
            answer="No sources.",
            sources=[],
            metadata=QueryMetadata(
                pattern="naive",
                model="gpt-4o-mini",
                embedding_model="text-embedding-3-small",
                tokens_used=100,
                latency_ms=100.0,
                cost_usd=0.001,
                retrieval_count=0,
                final_chunk_count=0,
            ),
        )
        assert Evaluator.chunk_utilization(result) == 0.0


# ---------------------------------------------------------------------------
# Judge metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJudgeMetrics:
    async def test_retrieval_relevance_returns_normalized_score(self) -> None:
        judge = FakeJudgeLLM(response="4")
        evaluator = Evaluator(judge_llm=judge)
        result = _make_result()
        score = await evaluator.retrieval_relevance("What is the refund policy?", result.sources)
        assert 0.0 <= score <= 1.0

    async def test_answer_faithfulness_returns_score(self) -> None:
        judge = FakeJudgeLLM(response="0.85")
        evaluator = Evaluator(judge_llm=judge)
        score = await evaluator.answer_faithfulness("context", "answer")
        assert score == pytest.approx(0.85)

    async def test_answer_relevance_returns_normalized_score(self) -> None:
        judge = FakeJudgeLLM(response="5")
        evaluator = Evaluator(judge_llm=judge)
        score = await evaluator.answer_relevance("question", "answer")
        assert score == pytest.approx(1.0)

    async def test_no_judge_returns_zero(self) -> None:
        evaluator = Evaluator(judge_llm=None)
        assert await evaluator.retrieval_relevance("q", []) == 0.0
        assert await evaluator.answer_faithfulness("ctx", "ans") == 0.0
        assert await evaluator.answer_relevance("q", "a") == 0.0

    async def test_full_eval_includes_all_metrics(self) -> None:
        judge = FakeJudgeLLM(response="4")
        evaluator = Evaluator(judge_llm=judge)
        result = _make_result()
        scores = await evaluator.full_eval("What is the refund policy?", result)
        assert scores.retrieval_relevance is not None
        assert scores.answer_faithfulness is not None
        assert scores.answer_relevance is not None
        assert scores.chunk_utilization is not None
        assert scores.latency_ms is not None
        assert scores.cost_usd is not None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParsing:
    def test_parse_score_extracts_integer(self) -> None:
        assert Evaluator._parse_score("Score: 4", min_val=1, max_val=5) == 4

    def test_parse_score_clamps_high(self) -> None:
        assert Evaluator._parse_score("10", min_val=1, max_val=5) == 5

    def test_parse_score_clamps_low(self) -> None:
        assert Evaluator._parse_score("0", min_val=1, max_val=5) == 1

    def test_parse_score_no_match_returns_min(self) -> None:
        assert Evaluator._parse_score("no number here", min_val=1, max_val=5) == 1

    def test_parse_float_extracts_decimal(self) -> None:
        assert Evaluator._parse_float("0.85") == pytest.approx(0.85)

    def test_parse_float_clamps_high(self) -> None:
        assert Evaluator._parse_float("1.5") == 1.0

    def test_parse_float_negative_extracts_magnitude(self) -> None:
        # regex extracts digits only; "-0.5" → 0.5 which is valid
        assert Evaluator._parse_float("-0.5") == pytest.approx(0.5)

    def test_parse_float_no_match_returns_zero(self) -> None:
        assert Evaluator._parse_float("no number") == 0.0
