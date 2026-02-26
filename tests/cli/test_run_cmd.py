"""Tests for the run CLI command."""

import pytest
from typer.testing import CliRunner

from rag_playbook.cli.app import app

runner = CliRunner()


@pytest.mark.unit
class TestRunCommand:
    def test_help_shows_arguments(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "pattern" in result.output.lower()
        assert "--query" in result.output

    def test_missing_query_shows_error(self) -> None:
        result = runner.invoke(app, ["run", "naive"])
        assert result.exit_code != 0
