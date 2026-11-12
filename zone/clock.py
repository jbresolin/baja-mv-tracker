"""Clocks: the real one, and a fake one that tests move forward by hand.

Everything in `zone` that needs the time asks a clock object instead of calling `time.monotonic()` directly.
In production that object is `SystemClock`. In tests it is `FakeClock`: time only moves when the test says so
(or when code "sleeps" on it), so a test of a 10-second camera move runs in a millisecond and gives the same
result every time.
"""
from __future__ import annotations

import threading
import time
from typing import Protocol


class Clock(Protocol):
    """What `zone` needs from a clock."""

    def monotonic(self) -> float:
        """Seconds from an arbitrary start; never goes backwards. Use it for durations and deadlines."""

    def wall(self) -> float:
        """Wall-clock time (Unix seconds), for log lines a human will line up with other recordings."""

    def sleep(self, seconds: float) -> None:
        """Wait `seconds` (a fake clock just moves its time forward)."""


class SystemClock:
    """The real clock of this computer."""

    def monotonic(self) -> float:
        return time.monotonic()

    def wall(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class FakeClock:
    """A clock that only moves when told to: `advance(seconds)` or `sleep(seconds)`.

    `wall()` is `wall_start + monotonic()`, so log lines still carry a plausible date.
    Thread-safe, so a driver's background thread can read it while a test advances it.
    """

    def __init__(self, start: float = 0.0, wall_start: float = 1_800_000_000.0) -> None:
        self._now = float(start)
        self._wall_offset = wall_start - start
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        with self._lock:
            return self._now

    def wall(self) -> float:
        with self._lock:
            return self._now + self._wall_offset

    def advance(self, seconds: float) -> None:
        """Move time forward by `seconds` (negative values are refused: time never goes back)."""
        if seconds < 0:
            raise ValueError("a clock cannot go backwards")
        with self._lock:
            self._now += seconds

    def sleep(self, seconds: float) -> None:
        """A fake sleep returns at once, with the clock `seconds` later."""
        if seconds > 0:
            self.advance(seconds)
