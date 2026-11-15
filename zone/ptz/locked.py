"""`LockedPtz`: makes a PTZ that isn't thread-safe (the simulator) safe to call from several threads.

The real driver (`ViscaPtz`) already puts one request on the wire at a time behind its own lock. `SimulatedPtz` has no
lock: it was written for one caller on a fake clock. In the running app several threads call the PTZ (the control
loop, the health check, the teach page, and the studio's agent API), so the app wraps the simulator in this:
every method call holds one lock, so two calls never interleave inside the simulator.
"""
import functools
import threading


class LockedPtz:
    """Forwards every attribute to `inner`; methods run one at a time."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._lock = threading.RLock()

    @property
    def inner(self):
        return self._inner

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        @functools.wraps(attr)
        def locked(*args, **kwargs):
            with self._lock:
                return attr(*args, **kwargs)
        return locked
