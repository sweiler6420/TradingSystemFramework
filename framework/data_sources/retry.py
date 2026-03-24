"""Small stdlib-only retry with exponential backoff."""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int = 5,
    base_sleep_seconds: float = 1.0,
    jitter: float = 0.1,
) -> T:
    """
    Call ``fn`` until it succeeds or ``max_retries`` is exhausted.

    Sleeps ``base_sleep_seconds * 2**attempt`` between attempts, plus random jitter
    (fraction of sleep) to reduce thundering herd.
    """
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt == max_retries - 1:
                raise
            sleep_s = base_sleep_seconds * (2**attempt)
            sleep_s += random.random() * jitter * sleep_s
            time.sleep(sleep_s)
    assert last is not None
    raise last
