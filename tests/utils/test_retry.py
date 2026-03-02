"""Tests for the retry utility."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from rag_playbook.utils.retry import retry, retry_with_backoff


@pytest.mark.unit
class TestRetryWithBackoff:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self) -> None:
        fn = AsyncMock(return_value="ok")
        result = await retry_with_backoff(fn, max_retries=3)
        assert result == "ok"
        assert fn.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self) -> None:
        fn = AsyncMock(side_effect=[ValueError("boom"), "ok"])
        result = await retry_with_backoff(
            fn, max_retries=3, base_delay=0.0, retryable_errors=(ValueError,)
        )
        assert result == "ok"
        assert fn.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries(self) -> None:
        fn = AsyncMock(side_effect=ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            await retry_with_backoff(
                fn, max_retries=3, base_delay=0.0, retryable_errors=(ValueError,)
            )
        assert fn.call_count == 3

    @pytest.mark.asyncio
    async def test_jitter_adds_randomness(self) -> None:
        """Verify that the delay includes a random jitter component."""
        fn = AsyncMock(side_effect=[ValueError("fail"), "ok"])
        delays: list[float] = []

        async def capture_sleep(seconds: float) -> None:
            delays.append(seconds)

        with patch("rag_playbook.utils.retry.asyncio.sleep", side_effect=capture_sleep):
            await retry_with_backoff(
                fn, max_retries=3, base_delay=1.0, retryable_errors=(ValueError,)
            )

        assert len(delays) == 1
        # base_delay * 2^0 = 1.0, plus jitter in [0, 1.0] -> total in [1.0, 2.0)
        assert 1.0 <= delays[0] < 2.0


@pytest.mark.unit
class TestRetryDecorator:
    @pytest.mark.asyncio
    async def test_decorator_retries(self) -> None:
        call_count = 0

        @retry(max_retries=3, base_delay=0.0, retryable_errors=(RuntimeError,))
        async def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("not yet")
            return "done"

        result = await flaky()
        assert result == "done"
        assert call_count == 2
