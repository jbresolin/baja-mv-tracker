"""Exponential backoff and an injectable clock, shared by the camera reconnect loop and the launcher.

Both helpers exist so that tests can drive time by hand (FakeClock) instead of sleeping.
"""
import time


class Backoff:
    """Wait times that double after each failure, up to a maximum: 5, 10, 20, 40, 60, 60, ... with the defaults
    of the launcher. reset() goes back to the first value (call it once things work again)."""

    def __init__(self, first_s, factor=2.0, max_s=60.0):
        self.first, self.factor, self.max = float(first_s), float(factor), float(max_s)
        self.current = None

    def next(self):
        """The wait before the next attempt."""
        self.current = self.first if self.current is None else min(self.max, self.current * self.factor)
        return self.current

    def reset(self):
        self.current = None


class SystemClock:
    """The real clocks. wall() = time.time() (epoch seconds, can jump when NTP corrects it); mono() =
    time.monotonic() (never jumps, only good for durations)."""

    def wall(self):
        return time.time()

    def mono(self):
        return time.monotonic()


class FakeClock:
    """A clock for tests: it only moves when advance() is called. wall and mono start at different values on
    purpose, so a test notices if code mixes them up."""

    def __init__(self, wall=1_800_000_000.0, mono=1000.0):
        self.t_wall, self.t_mono = float(wall), float(mono)

    def wall(self):
        return self.t_wall

    def mono(self):
        return self.t_mono

    def advance(self, dt):
        self.t_wall += dt
        self.t_mono += dt
