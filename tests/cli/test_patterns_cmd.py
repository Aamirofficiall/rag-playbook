"""Tests for the patterns CLI command."""

import pytest
from typer.testing import CliRunner

from rag_playbook.cli.app import app

runner = CliRunner()


@pytest.mark.unit
class TestPatternsCommand:
    def test_lists_all_patterns(self) -> None:
        result = runner.invoke(app, ["patterns"])
        assert result.exit_code == 0
        assert "naive" in result.output
        assert "hybrid_search" in result.output
        assert "reranking" in result.output
        assert "hyde" in result.output
        assert "agentic" in result.output

    def test_shows_descriptions(self) -> None:
        result = runner.invoke(app, ["patterns"])
        assert "baseline" in result.output.lower()

    def test_shows_all_eight_patterns(self) -> None:
        result = runner.invoke(app, ["patterns"])
        expected = [
            "naive",
            "hybrid_search",
            "reranking",
            "parent_child",
            "query_decomposition",
            "hyde",
            "self_correcting",
            "agentic",
        ]
        for name in expected:
            assert name in result.output
