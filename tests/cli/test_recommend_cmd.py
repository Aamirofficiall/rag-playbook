"""Tests for the recommend CLI command."""

import pytest
from typer.testing import CliRunner

from rag_playbook.cli.app import app

runner = CliRunner()


@pytest.mark.unit
class TestRecommendCommand:
    def test_recommend_help_output(self) -> None:
        result = runner.invoke(app, ["recommend", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output.lower()
        assert "pattern" in result.output.lower()

    def test_recommend_requires_query(self) -> None:
        result = runner.invoke(app, ["recommend"])
        assert result.exit_code != 0
