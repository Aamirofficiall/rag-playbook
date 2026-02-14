"""Tests for the timer context manager."""

import time

import pytest

from rag_playbook.utils.timer import timer


@pytest.mark.unit
class TestTimer:
    def test_measures_elapsed_time(self) -> None:
        with timer() as t:
            time.sleep(0.01)
        assert t["elapsed_ms"] >= 10

    def test_zero_work_is_near_zero(self) -> None:
        with timer() as t:
            pass
        assert t["elapsed_ms"] < 5

    def test_records_time_on_exception(self) -> None:
        with pytest.raises(ValueError), timer() as t:
            raise ValueError("boom")
        assert t["elapsed_ms"] >= 0
