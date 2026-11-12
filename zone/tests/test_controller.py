"""Tests for zone/controller.py, against the simulated PTZ on a fake clock (no camera, no real time).

Words used below, for a first-time reader:
* `s` is how far the car is through the zone: 0 = at the entry keyframe, 1 = at the exit keyframe. `v` is how fast
  s grows, per second (v = 0.25 means the car crosses the zone in 4 s).
* A "pose" is where the camera points: pan (left/right), tilt (up/down), zoom, in the camera's raw units.
* The controller turns "the car is at s, moving at v" into speed commands for the camera ("speed steps": whole
  numbers, the camera only has a few dozen speeds).
* "Branch A" cameras tell us where they are while moving (an "inquiry"); "Branch B" cameras don't, so the
  controller estimates it from its own commands ("dead reckoning").
* `SimulatedPtz` is a model of the camera; `FakeClock` is a clock that only moves when the test calls
  `clock.advance(seconds)`, so every run gives the same numbers and takes no real time.

The zone used here: keyframes at s = 0, 0.5, 1 with pan 0, 400, 800 (raw units), so the camera must turn 800 units
while the car crosses the zone. A car at v = 0.25 /s needs pan rate 800 * 0.25 = 200 units/s. 14.4 raw units are
about 1 degree."""
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
    """A controller driving a simulated camera, parked at the entry like at the start of a pass.
    -> (controller, simulator, spy that records the calls, clock, settings). `sets` changes config values, as
    `--set` does on the command line (e.g. "controller.k_pan=0.5")."""
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
    # TTL = "time to live": every speed command tells the camera "stop by yourself if you hear nothing more from me
    # within 0.3 s". That's what stops the camera if the program dies. Every single command must carry it.
    ctrl, sim, spy, clock, st = setup("B")
    follow_car(ctrl, sim, clock, t_end=1.0)
    ttls = [kw.get("ttl") for name, a, kw in spy.calls if name == "pan_tilt_velocity"]
    assert ttls and all(t == pytest.approx(0.3) for t in ttls)


def test_a_stalled_loop_stops_the_camera_by_itself():
    # the same safety net seen from the camera: follow a car, then send nothing more (as if the program froze)
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
    # Branch A: the controller asks the camera where it is (inquire_pan_tilt) and corrects with that answer
    ctrl, sim, spy, clock, st = setup("A")
    _, cmd = follow_car(ctrl, sim, clock, t_end=2.0)
    assert spy.count("inquire_pan_tilt") >= 10
    assert cmd.pose_now.source == "inquiry" and cmd.pose_now.trusted


def test_branch_a_poll_rate_is_capped():
    # ...but not more often than configured: too many questions would crowd out the speed commands
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
    # Branch B never asks the camera; its own estimate of where the camera is (from the commands it sent and the
    # camera's measured speeds) must stay close to the truth, which only the simulator knows (true_pan_tilt)
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
    # Zoom is sent as an absolute position, not a speed. Three rules: at most every 0.5 s, framed looser than
    # taught (loosen 0.5 = halfway between fully wide and the taught zoom), and only when it changed enough.
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
    # stop() (big STOP, end of a pass): after a short braking time the camera doesn't move any more
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


# ================================================================================= quantization integral, zoom support
def test_quantization_integral_cannot_wind_up():
    # The quantization compensation adds up "wanted rate minus commanded rate" over time (an integral). If the camera
    # can't give what is wanted for a long time (here: 2400 units/s wanted, the top speed is ~1070, for 20 s), that
    # sum would grow without end, and keep pushing long after the reason was gone. It is now kept within what its
    # output can use: max_feedback / k_quant (150 / 1.5 = 100 units of pan).
    ctrl, sim, spy, clock, st = setup("B", sets=["controller.dead_reckoning_trust_s=0"])
    t0 = clock.monotonic()
    for k in range(200):
        clock.advance(0.1)
        # every cycle the same car: s = -0.5, v = 3 zones per second -> the camera should turn at 800 * 3 units/s
        ctrl.follow(-0.5, 3.0, t0 + k * 0.1)
    limit = st.controller.max_feedback_pan / st.controller.k_quant
    assert abs(ctrl.q_pan) <= limit + 1e-9
    assert abs(ctrl.q_pan) == pytest.approx(limit)     # it did reach the cap (so the test really tested something)


def test_no_zoom_commands_to_a_camera_without_zoom_direct():
    # the bench found the camera can't take an absolute zoom position: the controller must not send any (the zoom
    # comes only from the presets, recalled at the entry)
    model = PtzModel(inquiry_mode="never", supports_zoom_direct=False)
    st = settings("B", model=model)
    assert st.controller.zoom_direct_supported is False
    clock = FakeClock(0.0)
    sim = SimulatedPtz(model, clock)
    sim.define_preset(10, 0, 0, 2000)
    spy = Spy(sim)
    ctrl = Controller(spy, clock, CURVE, st.pan_map, st.tilt_map, st.controller)
    follow_car(ctrl, sim, clock, t_end=3.0)
    assert spy.count("zoom_direct") == 0
    assert spy.count("pan_tilt_velocity") > 0          # pan/tilt following is unaffected
