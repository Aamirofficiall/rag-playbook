"""Exponential backoff with jitter retry utility for async functions."""

from __future__ import annotations

import asyncio
import functools
import random
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


async def retry_with_backoff(
    fn: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_errors: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> T:
    """Call an async function with exponential backoff and jitter.

    Args:
        fn: Async callable to invoke.
        max_retries: Maximum number of attempts.
        base_delay: Initial delay in seconds before jitter.
        max_delay: Upper bound on delay in seconds.
        retryable_errors: Exception types that trigger a retry.

    Returns:
        The return value of *fn*.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except retryable_errors as exc:
            last_error = exc
            if attempt == max_retries:
                break
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            jitter = random.uniform(0, base_delay)
            await asyncio.sleep(delay + jitter)

    raise last_error  # type: ignore[misc]


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_errors: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """Decorator form of :func:`retry_with_backoff`."""

    def decorator(
        fn: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_with_backoff(
                fn,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                retryable_errors=retryable_errors,
                **kwargs,
            )

        return wrapper

    return decorator
