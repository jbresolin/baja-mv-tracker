"""Tests for zone/ptz/locked.py: `LockedPtz` lets several threads share the (not thread-safe) simulated PTZ.

For a first-time reader: the running program calls the camera from several threads at once (the control loop, the
health check, the page, the studio's agent API). The real VISCA driver queues those calls itself; the simulator
doesn't, so the app wraps it in `LockedPtz`, which lets one call in at a time and otherwise behaves exactly like the
simulator."""
import threading

from zone.clock import FakeClock
from zone.ptz.locked import LockedPtz
from zone.ptz.sim import SimulatedPtz


def test_it_behaves_like_the_ptz_it_wraps():
    clock = FakeClock()
    ptz = LockedPtz(SimulatedPtz(clock=clock))
    assert hasattr(ptz, "tick") and hasattr(ptz, "define_preset")     # the app checks for these
    ptz.pan_tilt_velocity(10, 0, ttl=1.0)
    clock.advance(0.5)
    assert ptz.is_moving() and ptz.inner.is_moving()
    assert ptz.model is ptz.inner.model                                 # plain attributes pass through too


def test_two_threads_never_run_inside_the_simulator_at_once():
    # A fake "simulator" whose method notices if a second caller comes in while the first is still inside it.
    class Slow:
        def __init__(self):
            self.inside, self.overlaps = 0, 0

        def call(self):
            self.inside += 1
            if self.inside > 1:
                self.overlaps += 1
            for _ in range(20000):       # some work, during which the other thread would get in without the lock
                pass
            self.inside -= 1

    slow = Slow()
    ptz = LockedPtz(slow)
    threads = [threading.Thread(target=lambda: [ptz.call() for _ in range(20)]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert slow.overlaps == 0
