"""Tests for the compare CLI command's internal functions."""

import pytest

from rag_playbook.core.models import QueryMetadata, RAGResult, RetrievalMethod, RetrievedChunk


def _make_result(
    pattern: str, latency: float = 1000.0, cost: float = 0.003, chunks: int = 5
) -> RAGResult:
    sources = [
        RetrievedChunk(
            id=f"c{i}",
            document_id="doc-1",
            content=f"chunk {i}",
            score=0.9,
            retrieval_method=RetrievalMethod.VECTOR,
        )
        for i in range(chunks)
    ]
    return RAGResult(
        answer="Test answer",
        sources=sources,
        metadata=QueryMetadata(
            pattern=pattern,
            model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            tokens_used=500,
            latency_ms=latency,
            cost_usd=cost,
            retrieval_count=chunks,
            final_chunk_count=chunks,
        ),
    )


@pytest.mark.unit
class TestRecommendation:
    def test_recommend_picks_best_tradeoff(self) -> None:
        from rag_playbook.cli.compare_cmd import _recommend

        results = {
            "naive": _make_result("naive", latency=500, cost=0.001, chunks=5),
            "expensive": _make_result("expensive", latency=3000, cost=0.020, chunks=5),
        }
        recommended = _recommend(results)
        assert recommended == "naive"

    def test_recommend_with_single_pattern(self) -> None:
        from rag_playbook.cli.compare_cmd import _recommend

        results = {"naive": _make_result("naive")}
        assert _recommend(results) == "naive"

    def test_recommendation_reasoning_includes_metrics(self) -> None:
        from rag_playbook.cli.compare_cmd import _recommendation_reasoning

        results = {"naive": _make_result("naive", latency=980, cost=0.003)}
        reasoning = _recommendation_reasoning("naive", results)
        assert "980ms" in reasoning
        assert "$0.003" in reasoning


@pytest.mark.unit
class TestSaveResults:
    def test_save_results_creates_json(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import json

        from rag_playbook.cli.compare_cmd import _save_results

        output = tmp_path / "results.json"
        results = {"naive": _make_result("naive")}
        _save_results(output, "test query", results, "naive")

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["query"] == "test query"
        assert data["recommended"] == "naive"
        assert "naive" in data["results"]
