"""Context manager for precise pipeline step timing."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    """Context manager that measures elapsed time in milliseconds.

    Usage::

        with timer() as t:
            do_work()
        print(f"Took {t['elapsed_ms']:.1f}ms")
    """
    result: dict[str, float] = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000
