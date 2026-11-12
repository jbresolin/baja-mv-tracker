"""Tests for zone/controller.py, against the simulated PTZ on a fake clock (no camera, no real time).

The zone used here: keyframes at s = 0, 0.5, 1 with pan 0, 400, 800 (raw units), so the camera must turn 800 units
while the car crosses the zone. A car at v = 0.25 /s needs pan rate 800 * 0.25 = 200 units/s."""
import pytest

from zone.controller import Controller
from zone.path import Pose, PoseCurve
from zone.ptz.sim import PtzModel, SimulatedPtz
from zone.clock import FakeClock
from zone.tests.z2_fixtures import settings

KEYS = [Pose(0, 0, 2000), Pose(400, 40, 3000), Pose(800, 0, 4000)]
CURVE = PoseCurve([0.0, 0.5, 1.0], KEYS, "pchip")


class Spy:
    """Wraps the simulator and counts what the controller asks of it."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def __getattr__(self, name):
        fn = getattr(self.inner, name)
        if not callable(fn):
            return fn

        def wrapped(*a, **kw):
            self.calls.append((name, a, kw))
            return fn(*a, **kw)
        return wrapped

    def count(self, name):
        return sum(1 for c in self.calls if c[0] == name)


def setup(branch, inquiry_mode=None, pan_sign=1, sets=()):
    model = PtzModel(inquiry_mode=inquiry_mode or ("always" if branch == "A" else "never"), pan_sign=pan_sign)
    st = settings(branch, model=model, sets=sets)
    clock = FakeClock(0.0)
    sim = SimulatedPtz(model, clock)
    sim.define_preset(10, 0, 0, 2000)          # the entry keyframe's preset, as the teach stored it
    spy = Spy(sim)
    ctrl = Controller(spy, clock, CURVE, st.pan_map, st.tilt_map, st.controller)
    # start like the director does: recall the entry preset and wait for it to settle
    ctrl.recall(10, KEYS[0], settle_s=st.director.settle_s)
    clock.advance(st.director.settle_s + 0.1)
    sim.tick()
    return ctrl, sim, spy, clock, st


def follow_car(ctrl, sim, clock, v=0.25, s0=-0.05, t_end=4.0, dt=0.1, proc=0.05, until_s=1.0):
    """A car at constant speed: each cycle the controller gets s, v measured at the frame's grab time.
    Returns the pan error (raw units) to pose(s_true) at every cycle once the car is inside the zone."""
    errs = []                     # (s_true, pan error)
    t0 = clock.monotonic()
    k = 0
    while True:
        t_grab = t0 + k * dt
        s = s0 + v * (t_grab - t0)
        if s > until_s or t_grab - t0 > t_end:
            break
        clock.advance(max(0.0, t_grab + proc - clock.monotonic()))
        cmd = ctrl.follow(s, v, t_grab)
        now = clock.monotonic()
        s_now = s0 + v * (now - t0)
        if 0.15 < s_now < 0.95:
            errs.append((s_now, sim.true_pan_tilt()[0] - CURVE.pose(s_now).pan))
        k += 1
    return errs, cmd


@pytest.mark.parametrize("branch", ["A", "B"])
def test_follows_a_constant_speed_car(branch):
    ctrl, sim, spy, clock, st = setup(branch)
    errs, _ = follow_car(ctrl, sim, clock)
    # 14.4 units ~ 1 degree here. The car crosses s = 0 at full speed while the camera is still parked at the entry,
    # so the camera starts a little behind (the "entry lag", ~3-4 degrees); it then catches up to within ~2 degrees
    # (the rest is the speed-step hysteresis: smooth motion is preferred over chasing the last degree).
    assert max(abs(e) for _, e in errs) < 60, errs
    assert max(abs(e) for s, e in errs if s > 0.5) < 35


def test_every_velocity_command_carries_the_ttl():
    ctrl, sim, spy, clock, st = setup("B")
    follow_car(ctrl, sim, clock, t_end=1.0)
    ttls = [kw.get("ttl") for name, a, kw in spy.calls if name == "pan_tilt_velocity"]
    assert ttls and all(t == pytest.approx(0.3) for t in ttls)


def test_a_stalled_loop_stops_the_camera_by_itself():
    ctrl, sim, spy, clock, st = setup("B")
    follow_car(ctrl, sim, clock, t_end=1.5)
    assert sim.is_moving()
    # the loop dies here: no more commands. After the TTL (0.3 s) + stop latency + ramp, the camera must be still.
    clock.advance(1.0)
    pan_a = sim.true_pan_tilt()[0]
    clock.advance(2.0)
    assert sim.true_pan_tilt()[0] == pytest.approx(pan_a)
    assert not sim.is_moving()


def test_branch_a_polls_inquiry_and_uses_feedback():
    ctrl, sim, spy, clock, st = setup("A")
    _, cmd = follow_car(ctrl, sim, clock, t_end=2.0)
    assert spy.count("inquire_pan_tilt") >= 10
    assert cmd.pose_now.source == "inquiry" and cmd.pose_now.trusted


def test_branch_a_poll_rate_is_capped():
    ctrl, sim, spy, clock, st = setup("A", sets=["controller.inquiry_hz=5"])
    follow_car(ctrl, sim, clock, t_end=2.0)
    assert spy.count("inquire_pan_tilt") <= 11      # 5 Hz over ~2 s (+1)


def test_branch_b_never_asks_the_camera_and_trusts_dead_reckoning_only_after_a_recall():
    ctrl, sim, spy, clock, st = setup("B")
    trusted = []
    t0 = clock.monotonic()
    for k in range(60):                      # 6 s of a slow car (v = 0.15 /s)
        t = t0 + k * 0.1
        clock.advance(max(0.0, t + 0.05 - clock.monotonic()))
        cmd = ctrl.follow(-0.05 + 0.15 * k * 0.1, 0.15, t)
        trusted.append((cmd.pose_now.trusted, cmd.fb_pan))
    assert spy.count("inquire_pan_tilt") == 0
    # right after the recall the pose is known: feedback allowed...
    assert trusted[0][0] is True
    # ...but after more than dead_reckoning_trust_s (4 s) of motion it is not, and the feedback term is exactly 0
    assert trusted[-1] == (False, 0.0)


def test_branch_b_dead_reckoning_matches_the_camera():
    ctrl, sim, spy, clock, st = setup("B")
    _, cmd = follow_car(ctrl, sim, clock, t_end=2.0)
    assert cmd.pose_now.pan == pytest.approx(sim.true_pan_tilt()[0], abs=15)


def test_camera_whose_pan_counts_the_other_way():
    # pan_sign = -1: the camera reports pan DEcreasing when it turns right. Keyframes were taught in reported units,
    # so the controller must send negative steps to make the reported pan grow.
    ctrl, sim, spy, clock, st = setup("B", pan_sign=-1)
    errs, cmd = follow_car(ctrl, sim, clock)
    assert cmd.pan_step < 0
    assert max(abs(e) for _, e in errs) < 60
    assert max(abs(e) for s, e in errs if s > 0.5) < 35


def test_zoom_is_rate_limited_loosened_and_thresholded():
    ctrl, sim, spy, clock, st = setup("B", sets=["controller.zoom_loosen=0.5", "controller.zoom_min_interval_s=0.5",
                                                 "controller.zoom_threshold=100"])
    follow_car(ctrl, sim, clock, t_end=4.0)
    zooms = [(c[1][0]) for c in spy.calls if c[0] == "zoom_direct"]
    times = [t for t, name, args in sim.commands if name == "zoom_direct"]
    assert zooms
    # at most 2 per second
    assert all(b - a >= 0.5 - 1e-9 for a, b in zip(times, times[1:]))
    # looser than taught: taught zoom is 2000..4000, so with loosen 0.5 (wide = 0) it stays within 1000..2000
    assert all(1000 <= z <= 2000 for z in zooms)
    # and only sent when it changed by more than the threshold
    assert all(abs(b - a) > 100 for a, b in zip(zooms, zooms[1:]))


def test_stop_halts_the_camera():
    ctrl, sim, spy, clock, st = setup("B")
    follow_car(ctrl, sim, clock, t_end=1.5)
    ctrl.stop()
    clock.advance(0.6)
    pan = sim.true_pan_tilt()[0]
    clock.advance(1.0)
    assert sim.true_pan_tilt()[0] == pytest.approx(pan)
    assert ctrl.pan_step == 0


# ================================================================================= did the camera answer? (health)
class TimeoutPtz:
    """Every command "times out": the driver got no reply, as with an unplugged camera."""

    def __getattr__(self, name):
        from zone.ptz.interface import CommandResult
        return lambda *a, **kw: CommandResult("timeout")


def test_ptz_ok_is_unknown_until_the_first_command_then_follows_the_answers():
    model = PtzModel(inquiry_mode="never")
    st = settings("B", model=model)
    clock = FakeClock(0.0)
    sim = SimulatedPtz(model, clock)
    ctrl = Controller(sim, clock, CURVE, st.pan_map, st.tilt_map, st.controller)
    # nothing sent yet: we don't know whether the camera is there (None, not True)
    assert ctrl.ptz_ok is None
    ctrl.stop()                                  # the simulator answers every command
    assert ctrl.ptz_ok is True
    # the same controller talking to a camera that never answers
    ctrl.ptz = TimeoutPtz()
    ctrl.follow(0.2, 0.25, clock.monotonic())
    assert ctrl.ptz_ok is False


# ================================================================================= divergence guard
def test_divergence_guard_needs_more_than_n_cycles_in_a_row():
    # The guard compares the target with where the controller believes the camera is (dead reckoning in Branch B).
    # Here we make that belief absurd on purpose: every cycle we tell the estimator the camera is at pan -1000,
    # while the target is near 0..800. The limit is 300 units and 5 cycles, so cycles 1-5 are tolerated (a short
    # glitch must not drop a car) and cycle 6 reports "diverged".
    ctrl, sim, spy, clock, st = setup("B")
    ctrl.cfg.divergence_pan_units, ctrl.cfg.divergence_cycles = 300.0, 5
    flags = []
    for k in range(6):
        clock.advance(0.1)
        ctrl.pose.set_known(-1000, 0)
        cmd = ctrl.follow(0.1 + 0.02 * k, 0.2, clock.monotonic() - 0.05)
        flags.append(cmd.diverged)
    assert flags == [False] * 5 + [True]
    assert cmd.gap_pan > 1000                    # the logged gap: target minus believed camera position
    # one good cycle (the camera believed to be right on the target) resets the count
    ctrl.pose.set_known(CURVE.pose(0.22).pan, 0)
    assert ctrl.follow(0.22, 0.0, clock.monotonic()).diverged is False and ctrl.diverged_cycles == 0


def test_normal_following_never_trips_the_divergence_guard():
    # the default limit (15 degrees, 5 cycles) on an ordinary pass
    ctrl, sim, spy, clock, st = setup("B")
    assert st.controller.divergence_pan_units > 0
    t0 = clock.monotonic()
    for k in range(40):
        t = t0 + k * 0.1
        clock.advance(max(0.0, t + 0.05 - clock.monotonic()))
        assert not ctrl.follow(-0.05 + 0.25 * k * 0.1, 0.25, t).diverged
