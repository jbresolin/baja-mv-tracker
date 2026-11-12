"""Tests for what happens when something gets stuck: a detector or a PTZ call that never returns.

The danger being tested: the control loop holds the runtime's lock while it works. If a call inside it hangs, then
anything else that wants that lock (the big STOP button, the watchdog's shutdown) would wait forever, and the
process would never die, so the launcher would never restart it. The fixes:

* the big STOP waits for the lock only a short time; if it can't get it, it sends the stop straight to the PTZ
  driver and leaves a note ("stop requested") that the loop obeys as soon as it wakes up;
* the watchdog runs the shutdown in a helper thread and, if it isn't done after a few seconds, kills the process
  (os._exit). In these tests "killing the process" is a fake function that only records the exit code.

How a "hung" call is faked: the fake waits on a `threading.Event` (a flag another thread can raise). The test
raises it at the end, so the stuck thread finishes and nothing is left running. No real sleeps: the tests wait for
events ("the fake has entered the hung call"), never for a fixed time.
"""
import threading

import pytest

pytest.importorskip("fastapi")

from zone import config as C  # noqa: E402
from zone import director as D  # noqa: E402
from zone.app import ZoneApp  # noqa: E402
from zone.clock import FakeClock  # noqa: E402
from zone.ptz.sim import PtzModel, SimulatedPtz  # noqa: E402
from zone.runtime import ZoneRuntime  # noqa: E402
from zone.source import ZoneFrame  # noqa: E402
from zone.tests.z2_fixtures import settings, straight_zone  # noqa: E402


class RecordingPtz:
    """Wraps the simulated PTZ and remembers the name of every call, in order. With `hang_on="stop"` (or any other
    method name) that call blocks until `release` is set: a camera / network that stopped answering."""

    def __init__(self, inner, hang_on=None):
        self.inner = inner
        self.calls = []
        self.hang_on = hang_on
        self.entered = threading.Event()   # raised when a call starts hanging (so the test knows it's stuck now)
        self.release = threading.Event()   # the test raises this to let the hung call finish

    def __getattr__(self, name):
        fn = getattr(self.inner, name)
        if not callable(fn):
            return fn

        def wrapped(*a, **kw):
            self.calls.append(name)
            if name == self.hang_on:
                self.entered.set()
                self.release.wait(10)        # "forever", as far as the test is concerned
            return fn(*a, **kw)
        return wrapped


def hold_lock_in_another_thread(lock):
    """Simulates a control cycle that got stuck while holding the runtime lock. Returns (thread, release event)."""
    got, release = threading.Event(), threading.Event()

    def stuck_cycle():
        with lock:
            got.set()
            release.wait(10)
    th = threading.Thread(target=stuck_cycle, daemon=True)
    th.start()
    assert got.wait(5)
    return th, release


# ======================================================================================================= big STOP
def test_big_stop_does_not_wait_forever_for_a_stuck_loop():
    clock = FakeClock(0.0)
    ptz = RecordingPtz(SimulatedPtz(PtzModel(), clock))
    rt = ZoneRuntime(settings("B"), straight_zone(), ptz, clock)
    rt.set_mode(D.AUTO)
    ptz.calls.clear()
    th, release = hold_lock_in_another_thread(rt.lock)
    try:
        # the button is pressed while the loop is stuck: it must come back at once (timeout 0 here)...
        assert rt.big_stop(lock_timeout_s=0) is False
        # ...having sent the stop straight to the driver (pan/tilt and zoom)
        assert ptz.calls == ["stop", "zoom_stop"]
        # the director couldn't be touched (the loop owns it), so a note is left for the loop
        assert rt.stop_requested is True
    finally:
        release.set()
        th.join(5)
    # the loop wakes up: its very next cycle switches to STOPPED before doing anything else
    rt.cycle(ZoneFrame(None, clock.monotonic(), clock.wall(), 0), tracker_out=[])
    assert rt.director.mode == D.STOPPED
    assert rt.stop_requested is False


def test_big_stop_with_a_free_lock_switches_to_stopped_at_once():
    clock = FakeClock(0.0)
    ptz = RecordingPtz(SimulatedPtz(PtzModel(), clock))
    rt = ZoneRuntime(settings("B"), straight_zone(), ptz, clock)
    rt.set_mode(D.AUTO)
    assert rt.big_stop(lock_timeout_s=0) is True
    assert rt.director.mode == D.STOPPED and rt.stop_requested is False
    assert "stop" in ptz.calls


# ======================================================================================================= watchdog
def make_app(tmp_path, ptz, clock, wait_event):
    cfg = C.load(None, ["ptz.driver=sim", "health.no_frame_exit_s=20", "health.shutdown_grace_s=5"])
    exits = []                                     # the fake os._exit writes the exit code here
    app = ZoneApp(cfg, C.build(cfg), ptz, clock, straight_zone(), tmp_path / "zone.json", tmp_path / "out",
                  log=lambda m: None, hard_exit=exits.append, wait_event=wait_event)
    return app, exits


def test_watchdog_kills_the_process_when_the_shutdown_hangs(tmp_path):
    clock = FakeClock(0.0)
    # a PTZ whose stop never returns: the shutdown (which stops the camera) gets stuck in it
    ptz = RecordingPtz(SimulatedPtz(PtzModel(), clock), hang_on="stop")
    # the fake wait doesn't wait at all: "has the shutdown finished?" is checked once, right now. The shutdown
    # thread is blocked in stop(), so the answer is no, exactly as it would be after the real 5 s.
    app, exits = make_app(tmp_path, ptz, clock, wait_event=lambda ev, timeout: ev.is_set())
    try:
        assert app.check_frames() is False         # frames are fresh: nothing happens
        clock.advance(21.0)                        # 21 s without a spotter frame (the limit is 20)
        assert app.check_frames() is True
        assert ptz.entered.wait(5)                 # the shutdown really is stuck in the PTZ stop
        assert exits == [3]                        # -> the process was killed with exit code 3
    finally:
        ptz.release.set()


def test_watchdog_exits_cleanly_when_the_shutdown_finishes(tmp_path):
    clock = FakeClock(0.0)
    ptz = RecordingPtz(SimulatedPtz(PtzModel(), clock))
    # this wait really waits for the event, but returns the moment it is set (the shutdown is quick here)
    app, exits = make_app(tmp_path, ptz, clock, wait_event=lambda ev, timeout: ev.wait(5))
    app.exited_ev.set()                            # as main() does when it returns normally
    clock.advance(21.0)
    assert app.check_frames() is True
    assert exits == []                             # nobody had to be killed
    assert app.exit_code == 3                      # the launcher still sees "restart me"
    assert app.stop_ev.is_set()
    assert "stop" in ptz.calls                     # and the camera was told to stop on the way out


def test_watchdog_kills_the_process_if_main_never_returns(tmp_path):
    clock = FakeClock(0.0)
    ptz = RecordingPtz(SimulatedPtz(PtzModel(), clock))
    calls = []

    def fake_wait(ev, timeout):
        calls.append(ev)
        return ev.wait(5) if len(calls) == 1 else ev.is_set()   # the shutdown finishes; main() never does
    app, exits = make_app(tmp_path, ptz, clock, wait_event=fake_wait)
    clock.advance(21.0)
    app.check_frames()
    assert exits == [3]


# ======================================================================================= STOP doesn't wait for vision
class HungSpotter:
    """A spotter (detector + tracker) whose detection never finishes until the test says so."""

    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.last_dets, self.last_ms = ([],), 0.0

    def set_roi(self, roi):
        pass

    def process(self, img):
        self.entered.set()
        self.release.wait(10)
        return []


def test_big_stop_is_immediate_while_a_detection_is_running():
    # Detection is the slow part of a cycle (50-150 ms, more on a busy laptop). It now runs outside the control lock,
    # so the big STOP gets the lock at once (timeout 0 here) even while a detection is in progress.
    import numpy as np
    clock = FakeClock(0.0)
    ptz = RecordingPtz(SimulatedPtz(PtzModel(), clock))
    spotter = HungSpotter()
    rt = ZoneRuntime(settings("B"), straight_zone(), ptz, clock, spotter=spotter)
    rt.set_mode(D.AUTO)
    ptz.calls.clear()
    frame = ZoneFrame(np.zeros((1080, 1920, 3), np.uint8), clock.monotonic(), clock.wall(), 0)
    th = threading.Thread(target=rt.cycle, args=(frame,), daemon=True)
    th.start()
    try:
        assert spotter.entered.wait(5)              # the detection is running (and stuck) now
        assert rt.big_stop(lock_timeout_s=0) is True
        assert rt.director.mode == D.STOPPED        # switched right away, without waiting for the detection
        assert ptz.calls[:2] == ["stop", "zoom_stop"]   # and the driver's stop was the very first thing sent
    finally:
        spotter.release.set()
        th.join(5)
    # the cycle that was running finishes in STOPPED: it sends no movement command afterwards
    assert "pan_tilt_velocity" not in ptz.calls and "preset_recall" not in ptz.calls
