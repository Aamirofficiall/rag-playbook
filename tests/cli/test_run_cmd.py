"""Tests for the run CLI command."""

import re

import pytest
from typer.testing import CliRunner

from rag_playbook.cli.app import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


@pytest.mark.unit
class TestRunCommand:
    def test_help_shows_arguments(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        assert "pattern" in plain.lower()
        assert "--query" in plain

    def test_missing_query_shows_error(self) -> None:
        result = runner.invoke(app, ["run", "naive"])
        assert result.exit_code != 0
