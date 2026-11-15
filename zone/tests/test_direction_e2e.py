"""Does the PTZ turn TOWARD the car? End to end over the real VISCA path.

Why this test exists: on the bench (2026-11-15) the owner saw the PTZ "start moving in a completely wrong direction"
right when it locked onto a car. Everything in between the detector and the camera's motor can flip a sign: the VISCA
direction byte, the way the camera reports its position (some cameras count pan DOWN when turning right), the sign the
zone believes in (`pan_sign`), the pose source used while teaching, the feedback term... The in-process simulator
tests never send a single VISCA byte, so this test runs the whole real path:

    page API (teach: jog, add keyframe)  ->  ZoneApp  ->  ViscaPtz (the real driver)  ->  UDP on 127.0.0.1
        ->  FakeViscaUdpServer  ->  FakeViscaCamera  ->  SimulatedPtz (the "motor")

How it works, for a first-time reader:
* Time is a `FakeClock`: it stands still until the test calls `clock.advance(seconds)`. The camera's motor
  (SimulatedPtz) moves only when time moves, so the test is exact and instant. The UDP packets are real, though.
* `Scene` (zone/testing.py) is a small fake race track: a racing line in metres, a fixed spotter camera that sees it
  (a mapping to image pixels) and a PTZ standing in front of it. For any spot on the track it knows the PHYSICAL pan
  (degrees to the right of the PTZ, as raw units) that points at it. A car drives left to right, so the PTZ must turn
  RIGHT to follow it.
* The "operator" teaches the zone exactly as the owner does on the page: jog the PTZ (arrow buttons) until it points
  at a spot, click "Adicionar keyframe", click that spot in the spotter image. He looks at the picture, so he jogs in
  PHYSICAL directions: the right arrow means "turn right", whatever numbers the camera reports.
* `pan_sign = -1` in the camera model makes the fake camera report pan values that DECREASE when it turns right.
  Real cameras do this (and some report tilt upside down). The zone must follow the car either way.
"""
import math

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from rt.tracker import ByteTracker  # noqa: E402
from zone import config as C  # noqa: E402
from zone import director as D  # noqa: E402
from zone.app import ZoneApp, create_web  # noqa: E402
from zone.clock import FakeClock  # noqa: E402
from zone.ptz.fake_camera import FakeViscaCamera, FakeViscaUdpServer  # noqa: E402
from zone.ptz.sim import PtzModel, SimulatedPtz  # noqa: E402
from zone.ptz.visca import ViscaPtz  # noqa: E402
from zone.source import ZoneFrame  # noqa: E402
from zone.testing import UNITS_PER_DEGREE, Car, Scene, boxes_at  # noqa: E402
from zone.tools.characterize import BenchOptions, Characterizer, ScriptedUI  # noqa: E402
from zone.zonefile import Zone  # noqa: E402


# ------------------------------------------------------------------------------------------------- the world
def camera_model(sign, inquiry_mode="always"):
    """The fake camera's "truth". sign = +1: reported pan/tilt grow when turning right/up; -1: they shrink."""
    return PtzModel(inquiry_mode=inquiry_mode, pan_sign=sign, tilt_sign=sign,
                    pan_units_per_degree=UNITS_PER_DEGREE, tilt_units_per_degree=UNITS_PER_DEGREE)


def bench_report(tmp_path, model):
    """Run the real bench script (zone/tools/characterize.py, --quick) on a camera like `model` and return the path
    of the report.json it writes. This is the report the owner would get from his own bench run."""
    clock = FakeClock()
    out = tmp_path / "bench"
    out.mkdir()
    Characterizer(SimulatedPtz(model, clock), clock, ScriptedUI(), BenchOptions(quick=True), out,
                  {"ip": "sim", "port": 0}).run()
    return out / "report.json"


class Rig:
    """The zone app with the real VISCA driver talking UDP to the fake camera. Use as a context manager."""

    def __init__(self, tmp_path, model, sets=(), report_path=""):
        self.clock = FakeClock(0.0)
        self.sim = SimulatedPtz(model, self.clock)          # the camera's motor
        self.server = FakeViscaUdpServer(FakeViscaCamera(self.sim)).start()
        host, port = self.server.address
        # the real driver. watchdog=False: no background thread (the test moves time itself); a generous timeout
        # because a slow CI machine can take a while to answer even on 127.0.0.1
        self.ptz = ViscaPtz(host, port, timeout=2.0, clock=self.clock, watchdog=False)
        cfg = C.load(None, ["ptz.driver=visca", "ptz.report=%s" % report_path] + list(sets))
        self.settings = C.build(cfg)
        self.app = ZoneApp(cfg, self.settings, self.ptz, self.clock, Zone(), tmp_path / "zone.json",
                           tmp_path / "out", log=lambda m: None, cycle_log=False)
        self.client = TestClient(create_web(self.app))
        self.scene = Scene()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.ptz.close()
        self.server.stop()

    # ---- what the operator does on the page
    def physical_pan_tilt(self):
        """Where the camera REALLY points (physical frame: + = right / up), whatever it reports."""
        return self.sim.pan.position, self.sim.tilt.position

    def jog_to(self, pan_target, tilt_target):
        """Hold the arrow buttons until the camera points at (pan_target, tilt_target), in the PHYSICAL frame, as a
        person looking at the picture would: right arrow while the target is to the right, and so on."""
        post = self.client.post
        for _ in range(400):
            pan, tilt = self.physical_pan_tilt()
            dp, dt = pan_target - pan, tilt_target - tilt
            if abs(dp) < 4 and abs(dt) < 4:
                break
            # fast buttons when far, slow when close (the page's jog speeds are 2 / 6 / 12)
            sp = 0 if abs(dp) < 4 else (12 if abs(dp) > 200 else 6 if abs(dp) > 40 else 2)
            st = 0 if abs(dt) < 4 else (12 if abs(dt) > 200 else 6 if abs(dt) > 40 else 2)
            r = post("/api/teach/jog", json={"pan": int(math.copysign(sp, dp)) if sp else 0,
                                             "tilt": int(math.copysign(st, dt)) if st else 0})
            assert r.status_code == 200, r.text
            self.clock.advance(0.15)                 # the page repeats the jog every 150 ms while held
        post("/api/teach/jog_stop")
        self.clock.advance(1.0)                      # let it come to rest before the click

    def teach(self, n_keyframes=3):
        """The whole teach, through the page API: path points along the line, keyframes where the camera was jogged
        to frame that spot. Then AUTO."""
        sc = self.scene
        post = self.client.post
        assert post("/api/mode", json={"mode": "PASSIVE"}).status_code == 200
        assert post("/api/teach/mode", json={"on": True}).json()["result"] is True
        kf_arcs = [sc.arc_entry + (sc.arc_exit - sc.arc_entry) * k / (n_keyframes - 1) for k in range(n_keyframes)]
        arcs = sorted(set([a for a in range(0, int(sc.length), 5)] + kf_arcs))
        for a in arcs:
            x, y = sc.image_of(sc.world_at(a))
            if any(abs(a - k) < 1e-6 for k in kf_arcs):
                pan, tilt, _ = sc.true_pose(sc.world_at(a))       # the physical framing of that spot
                self.jog_to(pan, tilt)
                r = post("/api/teach/point", json={"x": x, "y": y, "keyframe": True})
            else:
                r = post("/api/teach/point", json={"x": x, "y": y})
            assert r.status_code == 200, r.text

    def nudge_to(self, pan_target, tilt_target, true_sign):
        """Teaching with the "commanded" pose source (absolute nudges, no inquiry): the page has nudge buttons
        that add raw units to the pose. On a camera that counts the other way, the right button turns it LEFT;
        the operator notices and presses the other one. `true_sign` is that learned habit."""
        for _ in range(400):
            pan, tilt = self.physical_pan_tilt()
            dp, dt = pan_target - pan, tilt_target - tilt
            if abs(dp) < 10 and abs(dt) < 10:
                break
            # the page's nudge sizes are 8 / 32 / 128 raw units
            up = 0 if abs(dp) < 10 else (128 if abs(dp) > 200 else 32 if abs(dp) > 40 else 8)
            ut = 0 if abs(dt) < 10 else (128 if abs(dt) > 200 else 32 if abs(dt) > 40 else 8)
            r = self.client.post("/api/teach/nudge", json={"pan": int(math.copysign(up, dp)) * true_sign if up else 0,
                                                           "tilt": int(math.copysign(ut, dt)) * true_sign if ut else 0,
                                                           "speed": 12})
            assert r.status_code == 200, r.text
            self.clock.advance(1.0)


    def run_car(self, seconds=14.0, car_start=3.0):
        """AUTO, then one car drives through. -> list of (time, physical pan of the camera, physical pan that would
        frame the car, pan step commanded, director sub-state)."""
        assert self.client.post("/api/mode", json={"mode": "AUTO"}).status_code == 200
        sc = self.scene
        trk = ByteTracker()
        rows = []
        t0 = self.clock.monotonic()          # the teach took a while of (fake) time: the race starts now
        car = Car(sc, t0 + car_start)
        for k in range(int(seconds * 10)):
            t = t0 + k * 0.1
            self.clock.advance(max(0.0, t + 0.05 - self.clock.monotonic()))   # 50 ms of "detection"
            b, s = boxes_at(sc, [car], t)
            res = self.app.runtime.cycle(ZoneFrame(None, t, 1.8e9 + t, k), trk.update(b, s))
            arc = car.arc_at(self.clock.monotonic())
            want = sc.true_pose(sc.world_at(arc))[0] if arc is not None else None
            step = res.command.pan_step if res.command else None
            rows.append((t, self.physical_pan_tilt()[0], want, step, self.app.runtime.director.state()["sub"]))
        return rows


def first_moves(rows, n=5):
    """The first n nonzero pan steps commanded while following."""
    return [r[3] for r in rows if r[3]][:n]


def following_errors_deg(rows):
    return [abs(r[1] - r[2]) / UNITS_PER_DEGREE for r in rows if r[4] == D.TRACKING and r[2] is not None]


def teach_and_run(tmp_path, sign, sets=(), report_path=""):
    with Rig(tmp_path, camera_model(sign), sets, report_path) as rig:
        rig.teach()
        rows = rig.run_car()
        return rig, rows


# ================================================================================================== the tests
@pytest.mark.parametrize("branch", ["A", "B"])
@pytest.mark.parametrize("sign", [1, -1])
def test_without_a_report_the_teach_measures_the_sign_and_the_ptz_turns_toward_the_car(tmp_path, sign, branch):
    # No bench report at all (the zone's defaults), the owner teaches with the jog buttons, then one car passes.
    # sign = -1 is the camera that counts pan DOWN when turning right: before the fix the zone assumed +1 and the PTZ
    # turned away from the car from its very first command (Branch B: for the whole pass; Branch A: until the
    # divergence guard dropped the car).
    rig, rows = teach_and_run(tmp_path, sign, ["controller.branch=%s" % branch])
    zone = rig.app.teach.zone
    # 1. the jogs measured the camera's sign, it was saved with the zone, and the controller uses it
    assert zone.ptz_signs["pan"] == sign and zone.ptz_signs["tilt"] == sign
    assert rig.settings.controller.pan_sign == sign and rig.settings.sign_how["pan"] == "teach"
    # 2. the very first commands turn the camera RIGHT (positive steps), the way the car goes
    moves = first_moves(rows)
    assert moves and all(m > 0 for m in moves), moves
    # 3. it follows the car through the whole zone, never more than 6 degrees off, and lets it go at the exit
    errors = following_errors_deg(rows)
    assert max(errors) < 6.0, max(errors)
    assert rig.app.runtime.director.passes_followed == 1
    assert rig.app.runtime.director.last_drop_reason == "exit"


@pytest.mark.parametrize("branch", ["A", "B"])
def test_with_the_bench_report_of_a_reversed_camera(tmp_path, branch):
    # The owner ran the bench script first (it measures the sign while measuring the speed table) and gave its
    # report to the zone. The teach measures the same sign: the two agree, no warning.
    report = bench_report(tmp_path, camera_model(-1))
    rig, rows = teach_and_run(tmp_path, -1, ["controller.branch=%s" % branch], str(report))
    assert rig.settings.sign_base_how["pan"] == "report" and rig.settings.sign_base["pan"] == -1
    assert rig.app.runtime.sign_warnings == []
    assert all(m > 0 for m in first_moves(rows))
    assert max(following_errors_deg(rows)) < 8.0
    assert rig.app.runtime.director.last_drop_reason == "exit"


def test_a_wrong_sign_in_branch_b_turns_the_ptz_away_for_the_whole_pass(tmp_path):
    # What the owner saw, reproduced on purpose: the config forces pan_sign = +1 on a camera that counts the other
    # way. Branch B has no position feedback, so it believes its own (wrong) bookkeeping: the PTZ turns LEFT while
    # the car goes right. The page shows a red warning that the config contradicts what the teach measured.
    rig, rows = teach_and_run(tmp_path, -1, ["controller.branch=B", "ptz.pan_sign=1", "ptz.tilt_sign=1"])
    assert all(m < 0 for m in first_moves(rows))                       # the wrong way, from the first command
    assert max(following_errors_deg(rows)) > 45.0
    assert any("força +1" in w for w in rig.app.runtime.sign_warnings)
    health = rig.client.get("/api/state").json()["health"]
    assert health["ptz_signs"]["pan"] == {"sign": 1, "how": "config"} and health["sign_warnings"]


def test_a_wrong_sign_in_branch_a_is_caught_within_a_second(tmp_path):
    # Branch A asks the camera where it is: when it moves against the sign we use, the car is dropped with the
    # reason "wrong_direction" (not the vague "diverged"), and the event says which key to change.
    rig, rows = teach_and_run(tmp_path, -1, ["controller.branch=A", "ptz.pan_sign=1", "ptz.tilt_sign=1"])
    events = rig.app.events
    lock = next(e for e in events if e.get("event") == "lock")
    mismatch = next(e for e in events if e.get("type") == "direction_mismatch")
    assert mismatch["axis"] in ("pan", "tilt") and mismatch["observed"] == -1 and "sign = -1" in mismatch["msg"]
    release = next(e for e in events if e.get("event") == "release")
    assert release["reason"] == "wrong_direction" and release["t"] - lock["t"] < 1.0
    # the page header shows it
    last = rig.client.get("/api/state").json()["runtime"]["director"]["last_drop"]
    assert last["reason"] == "wrong_direction"


def test_commanded_pose_source_cannot_measure_the_sign_so_the_page_warns(tmp_path):
    # A camera whose position can't be read (e.g. a bench report that found no inquiry): the zone teaches with
    # absolute "nudges". Nothing can measure the sign then, so the page says so in red, and the fix is one config
    # key. Here the camera counts the other way: without the key it turns away, with ptz.pan_sign = -1 it follows.
    def run(extra):
        sub = tmp_path / ("run%d" % len(extra))
        sub.mkdir()
        with Rig(sub, camera_model(-1), ["controller.branch=B", "teach.pose_source=commanded"] + extra) as rig:
            post = rig.client.post
            post("/api/mode", json={"mode": "PASSIVE"})
            post("/api/teach/mode", json={"on": True})
            assert post("/api/teach/home").status_code == 200
            rig.clock.advance(3.0)
            sc = rig.scene
            kf_arcs = [sc.arc_entry, (sc.arc_entry + sc.arc_exit) / 2, sc.arc_exit]
            for a in sorted(set(list(range(0, int(sc.length), 5)) + kf_arcs)):
                x, y = sc.image_of(sc.world_at(a))
                if a in kf_arcs:
                    pan, tilt, _ = sc.true_pose(sc.world_at(a))
                    rig.nudge_to(pan, tilt, true_sign=-1)
                assert post("/api/teach/point", json={"x": x, "y": y, "keyframe": a in kf_arcs}).status_code == 200
            rows = rig.run_car()
            return rig, rows

    rig, rows = run([])
    assert rig.app.teach.zone.ptz_signs == {}                          # nothing could measure it
    assert any("NÃO MEDIDO" in w for w in rig.app.runtime.sign_warnings)
    assert all(m < 0 for m in first_moves(rows))                       # so it turns the wrong way
    rig, rows = run(["ptz.pan_sign=-1", "ptz.tilt_sign=-1"])
    assert rig.app.runtime.sign_warnings == []
    assert all(m > 0 for m in first_moves(rows))
    assert rig.app.runtime.director.last_drop_reason == "exit"
