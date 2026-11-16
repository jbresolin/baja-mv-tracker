"""Detection dropouts: the followed car is not detected for a while (dust, another car in front of it, a few bad
frames). What should happen, and what these tests check:

* While it is unseen, its position is PREDICTED: it keeps the speed it had at the last detection (frozen: the
  prediction never makes it faster or slower by itself), clamped to a sane band around its recent mean speed. Once
  the speed profile has learned how fast cars usually go at each spot, the predicted speed slides from the frozen
  value to that typical speed.
* The prediction never runs more than 0.25 of the zone past the last place the car was really seen: there it HOLDS
  (speed 0), so the camera waits for the car instead of running off ahead of it.
* The director waits up to `lost_timeout_s` (3 s at the race, 1.5 s before) before giving the car up, and loosens the
  zoom a little during a long dropout so the car is still in the picture when it reappears a bit off the prediction.
* When the car reappears near the prediction it is picked up again ("reacquired"), even when the tracker (ByteTrack)
  gives it a NEW id, or doesn't report it at all yet (a single raw detection: ByteTrack only reports a new track
  from its second detection). A car that was already there before the dropout is another car: never adopted.

Background for a first-time reader:
* `s` is the progress along the path drawn on the spotter image: 0 at the first keyframe, 1 at the last one. `v` is
  in "zone per second" (0.1 = the car crosses the zone in 10 s).
* There is no real camera or clock here. Times are plain numbers (seconds), one "frame" every 0.1 s (10 fps), and a
  car is just a 40 x 40 pixel box whose bottom centre slides along a straight path at y = 500.
"""
import numpy as np
import pytest

from rt.tracker import ByteTracker
from zone import config as C
from zone import director as D
from zone.clock import FakeClock
from zone.controller import Controller
from zone.director import Director, DirectorSettings
from zone.path import PathModel, PoseCurve
from zone.progress import ProgressTracker, SpeedProfile
from zone.ptz.sim import PtzModel, SimulatedPtz
from zone.runtime import ZoneRuntime
from zone.source import ZoneFrame
from zone.tests.z2_fixtures import settings, straight_zone
from zone.zonefile import Keyframe, PathPoint, Zone

FPS = 10.0
DT = 1.0 / FPS


# =================================================================================================== helpers
def box_at(x, y=500.0):
    """A 40 x 40 box whose bottom centre (the car's 'ground point') is at (x, y)."""
    return (x - 20.0, y - 40.0, x + 20.0, y)


class Rig:
    """The progress tracker + the director, glued together the way zone/runtime.py glues them, with the director
    already in AUTO and waiting at the entry. The path is straight along y = 500: the entry keyframe at x = 200
    (s = 0), the exit keyframe at x = 200 + length_px (s = 1). So with the default 1000 px, s = (x - 200) / 1000 and
    a car doing 100 px/s has v = 0.1."""

    def __init__(self, length_px=1000.0, profile=None, sets=()):
        zone = Zone(points=[PathPoint(100, 500), PathPoint(200, 500, Keyframe(0, 0, 0, 10)),
                            PathPoint(200 + length_px, 500, Keyframe(500, 0, 0, 11))], corridor_half_width_px=60)
        self.st = C.build(C.load(None, list(sets)), C.Report(PtzModel(), "B", "", "", {}))   # default config
        self.path = PathModel(zone)
        self.prog = ProgressTracker(self.path, self.st.progress, profile or SpeedProfile())
        self.events = []                                   # everything the director logged
        self.director = Director(self.st.director, on_event=self.events.append)
        self.director.set_mode(D.AUTO, 0.0)
        self.director.step(0.0, {})                        # the entry recall
        self.director.step(self.st.director.settle_s + 0.1, {})   # settled: waiting for a car
        assert self.director.sub == D.WAITING
        self.releases = []                                 # (time, reason) of every release
        self.decisions = []                                # (time, Decision) of every cycle

    def cycle(self, t, tracker_out, raw_boxes=None):
        """One frame: progress (with the followed car as its focus), then the director. -> the views."""
        d = self.director
        focus = d.locked_id if d.mode == D.AUTO and d.sub == D.TRACKING else None
        views = self.prog.update(t, tracker_out, focus, raw_boxes)
        for ev in self.prog.reacquired:
            d.reacquired(t, ev)
        dec = d.step(t, views)
        self.decisions.append((t, dec))
        if dec.action == D.RECALL_ENTRY:
            self.releases.append((round(t, 2), dec.reason))
        return views

    def events_of(self, kind):
        return [e for e in self.events if e.get("event") == kind]


def drive(rig, cars, seconds, t0=5.0):
    """Run `seconds` of frames from t0. `cars` = list of functions f(t) -> (track_id, x) or None (not detected in
    this frame); t is counted from t0. -> list of (t from t0, views) per frame."""
    out = []
    for k in range(int(round(seconds * FPS))):
        t = t0 + k * DT
        tracker_out = []
        for car in cars:
            seen = car(t - t0)
            if seen is not None:
                tid, x = seen
                tracker_out.append((tid, box_at(x), 0.9))
        out.append((round(t - t0, 2), rig.cycle(t, tracker_out)))
    return out


def steady_car(speed_px_s, hidden=lambda t: False, track_id=1, x0=100.0, id_after=None):
    """A car driving at constant speed from x0. `hidden(t)` -> True while it is not detected. `id_after` =
    (t_switch, new_id): from t_switch on, the tracker reports it under new_id (ByteTrack started a new track)."""
    def car(t):
        if hidden(t):
            return None
        tid = id_after[1] if id_after and t >= id_after[0] else track_id
        return tid, x0 + speed_px_s * t
    return car


# =================================================================================================== the bench case
@pytest.mark.parametrize("speed_px_s", [80.0, 200.0])
def test_a_2_5_s_dropout_at_steady_speed_is_followed_through_and_reacquired(speed_px_s):
    # A car at steady speed (0.08 or 0.2 of the zone per second) vanishes 2 s after it starts, for 2.5 s, then comes
    # back under the same tracker id. The default rules (lost_timeout_s = 3) keep following it through the gap, pick
    # it up again (a "reacquired" event that says the gap lasted ~2.5 s), and let it go only at the exit.
    rig = Rig()
    hidden = lambda t: 2.0 <= t < 4.5
    drive(rig, [steady_car(speed_px_s, hidden)], seconds=1300.0 / speed_px_s)
    assert rig.events_of("lock"), "the car was never locked"
    assert rig.releases and rig.releases[0][1] == "exit", rig.releases
    re = rig.events_of("reacquired")
    assert len(re) == 1 and re[0]["how"] == "same_id"
    assert re[0]["dropout_s"] == pytest.approx(2.6, abs=0.15)   # last seen at 1.9, back at 4.5


def test_the_old_1_5_s_rule_dropped_that_car():
    # The same 80 px/s car with the old timeout (1.5 s): it is given up during the gap, as on the bench
    rig = Rig(sets=["director.lost_timeout_s=1.5"])
    drive(rig, [steady_car(80.0, lambda t: 2.0 <= t < 4.5)], seconds=16.0)
    assert rig.releases[0][1] == "lost", rig.releases
    assert rig.releases[0][0] - 5.0 == pytest.approx(1.9 + 1.5, abs=0.15)   # 1.5 s after the last detection


def test_a_car_that_is_really_gone_is_still_released_after_lost_timeout_s():
    # The car disappears for good at t = 2 s. It must be let go as "lost" right after lost_timeout_s (3 s) without
    # a detection, not earlier and not much later, and not as "stopped" (its held prediction has v = 0)
    rig = Rig()
    drive(rig, [steady_car(100.0, lambda t: t >= 2.0)], seconds=8.0)
    assert len(rig.releases) == 1 and rig.releases[0][1] == "lost", rig.releases
    last_seen = 5.0 + 1.9
    assert last_seen + 3.0 < rig.releases[0][0] <= last_seen + 3.0 + 0.15


def test_the_2026_11_15_bench_sequence():
    # What the bench log showed (a printed picture moved by hand at ~0.09 of the zone per second): detected every
    # frame for 1 s after the lock, then nothing except ONE detection 0.2 s later (which ByteTrack never reports:
    # a lone detection), then detections again 1.7 s after the last one, under a NEW tracker id. The old rules lost
    # it at 1.5 s. Now: the lone detection updates the followed car, the new id is adopted, and the picture is
    # followed to the exit.
    rig = Rig()
    lock_t = None
    for k in range(int(16.0 * FPS)):
        t = 5.0 + k * DT
        x = 100.0 + 90.0 * (t - 5.0)
        locked = rig.director.sub == D.TRACKING
        if locked and lock_t is None:
            lock_t = t
        since = None if lock_t is None else t - lock_t
        if since is None or since < 1.0:
            rig.cycle(t, [(1, box_at(x), 0.9)])             # before the lock and 1 s after it: every frame
        elif abs(since - 1.2) < 1e-6:
            rig.cycle(t, [], raw_boxes=[box_at(x)])         # the lone detection: raw only
        elif since >= 2.9:
            rig.cycle(t, [(5, box_at(x), 0.9)])             # back, as a new ByteTrack track
        else:
            rig.cycle(t, [])                                # nothing
    hows = [e["how"] for e in rig.events_of("reacquired")]
    assert hows == ["raw_detection", "new_id"], hows
    assert rig.releases and rig.releases[0][1] == "exit", rig.releases


# =================================================================================================== prediction
def test_the_speed_does_not_grow_while_predicting():
    # Before the profile has learned anything, the prediction keeps the last measured speed. We record the speed
    # the director is given at every frame of a 1 s dropout: it must be exactly the same number every time (and the
    # filter's own stored speed must not change either: predicting never writes into the filter).
    rig = Rig()
    frames = drive(rig, [steady_car(100.0, lambda t: t >= 2.0)], seconds=3.0)
    tp = rig.prog.tracks[1]
    v_filter = tp.filt.v
    speeds = [views[1].v for t, views in frames if 2.05 < t < 3.0]
    assert len(speeds) >= 9
    assert max(speeds) - min(speeds) < 1e-12               # frozen: not a hair faster or slower
    assert speeds[0] == pytest.approx(0.1, abs=0.01)       # ... at the car's real speed (100 px/s on 1000 px)
    assert tp.filt.v == v_filter


def test_a_speed_that_jumped_on_the_last_detection_is_clamped():
    # The car drives at 100 px/s (v = 0.1), but its LAST detection is 100 px too far ahead (a bad box: the hand or a
    # cloud of dust). The filter's speed jumps up on it; the prediction must not fly off with that speed: it is
    # clamped to at most 1.5 x the mean speed of the last second (~0.1 -> at most ~0.15).
    rig = Rig()

    def car(t):
        if t >= 2.0:
            return None
        return 1, 100.0 + 100.0 * t + (100.0 if t >= 1.85 else 0.0)

    drive(rig, [car], seconds=2.0)
    tp = rig.prog.tracks[1]
    assert tp.filt.v > 0.17                                # the jump did fool the filter
    views = rig.prog.update(5.0 + 2.5, [])                 # 0.6 s into the dropout
    mean_v = 0.1 + 0.1 / 1.0                               # what the last second looks like with the bad box in it
    assert views[1].v <= 1.5 * mean_v + 1e-6
    assert views[1].v < tp.filt.v


def test_the_prediction_holds_at_the_cap_and_so_does_the_camera_target():
    # A fast car (300 px/s = 0.3 /s) vanishes. Constant speed would carry the prediction 0.6 of the zone ahead in
    # 2 s; it may only go 0.25 past the last measured s, and then it holds there with v = 0.
    rig = Rig()
    frames = drive(rig, [steady_car(300.0, lambda t: t >= 1.0)], seconds=3.0)
    tp = rig.prog.tracks[1]
    s_last = tp.filt.s                                     # where it was really last seen (filtered)
    dropout = [views[1] for t, views in frames if t >= 1.0]
    assert all(v.s <= s_last + 0.25 + 1e-9 for v in dropout)
    held = [v for v in dropout if v.held]
    assert held, "the cap was never reached"
    assert all(v.v == 0.0 and v.s == pytest.approx(s_last + 0.25) for v in held)
    # The controller, fed the held (s, v), aims at the same place every cycle and has no feed-forward speed: the
    # camera waits at that framing.
    zone = straight_zone()
    path = PathModel(zone)
    st = settings("B")
    clock = FakeClock(0.0)
    ctl = Controller(SimulatedPtz(PtzModel(), clock), clock, PoseCurve.from_zone(zone, path, "pchip", 0.0),
                     st.pan_map, st.tilt_map, st.controller)
    cmds = []
    for v in held[:5]:                                     # five held cycles, 0.1 s apart
        clock.advance(DT)
        cmds.append(ctl.follow(v.s, v.v, clock.monotonic()))
    assert len({(c.target.pan, c.target.tilt) for c in cmds}) == 1
    assert all(c.ff_pan == 0.0 and c.ff_tilt == 0.0 for c in cmds)


def test_once_learned_the_prediction_blends_to_the_typical_speed():
    # The profile learned that cars do 0.2 everywhere. This car was measured at 0.1. During a dropout its predicted
    # speed slides linearly from 0.1 to 0.2 over typ_blend_s (1 s): ~0.15 after half a second, 0.2 from 1 s on.
    profile = SpeedProfile()
    profile.add_pass([(s / 100, 0.2) for s in range(-40, 120)])
    rig = Rig(profile=profile)
    drive(rig, [steady_car(100.0, lambda t: t >= 2.0)], seconds=2.0)
    tp = rig.prog.tracks[1]
    t_last = tp.last_update_t
    v0 = rig.prog.frozen_speed(tp)
    assert v0 == pytest.approx(0.1, abs=0.01)
    assert rig.prog.prediction(tp, t_last + 0.5).v_free == pytest.approx((v0 + 0.2) / 2, abs=0.01)
    assert rig.prog.prediction(tp, t_last + 1.0).v_free == pytest.approx(0.2, abs=1e-6)
    assert rig.prog.prediction(tp, t_last + 1.5).v_free == pytest.approx(0.2, abs=1e-6)


def test_before_the_profile_has_learned_the_prediction_keeps_the_frozen_speed():
    # Same car, nothing learned: no typical speed to blend to, it stays at its own 0.1
    rig = Rig()
    drive(rig, [steady_car(100.0, lambda t: t >= 2.0)], seconds=2.0)
    tp = rig.prog.tracks[1]
    for dt in (0.3, 1.0, 1.5):
        assert rig.prog.prediction(tp, tp.last_update_t + dt).v_free == pytest.approx(0.1, abs=0.01)


# =================================================================================================== zoom
def tv(s=0.3, age=0.0):
    """A track as progress would report it, seen `age` seconds ago (age > 0 = predicted)."""
    from zone.progress import TrackView
    return TrackView(track_id=1, s=s, v=0.2, s_std=0.01, v_std=0.01, lateral_px=0.0, n_updates=10,
                     since_update_s=age, off_corridor_frames=0, predicted=age > 0, box=(), ground=())


def test_the_zoom_loosens_during_a_dropout_and_comes_back_smoothly():
    d = Director(DirectorSettings(settle_s=1.0))           # defaults: loosen after 0.5 s, up to 0.3 at 3 s
    d.set_mode(D.AUTO, 0.0)
    d.step(0.0, {})
    d.step(2.0, {})
    assert d.step(10.0, {1: tv(s=0.0)}).reason == "lock"
    extra = {}
    t = 10.0
    # 2.9 s of dropout, one cycle every 0.1 s
    for k in range(1, 30):
        t = 10.0 + k * DT
        extra[round(k * DT, 1)] = d.step(t, {1: tv(age=k * DT)}).zoom_extra
    assert extra[0.3] == 0.0 and extra[0.5] == 0.0         # short gaps: framing unchanged
    assert 0.0 < extra[1.0] < extra[2.0] < extra[2.9]      # then wider and wider ...
    assert extra[2.9] == pytest.approx(0.3 * (2.9 - 0.5) / 2.5, abs=1e-6)   # ... towards 0.3 at 3 s
    # the car is seen again: the extra zoom-out goes back down step by step (0.3 per second), not in one jump
    back = []
    for k in range(1, 15):
        back.append(d.step(t + k * DT, {1: tv(age=0.0)}).zoom_extra)
    assert back[0] == pytest.approx(extra[2.9] - 0.03, abs=1e-6)
    assert all(b2 <= b1 for b1, b2 in zip(back, back[1:]))
    assert back[-1] == 0.0
    # and the controller turns it into a wider zoom (a smaller raw zoom value)
    ctl = Controller(None, FakeClock(0.0), None, None, None)
    assert ctl.zoom_for(10000, 0.3) < ctl.zoom_for(10000, 0.0)
    assert ctl.zoom_for(10000, 0.0) == ctl.zoom_for(10000)


# =================================================================================================== re-acquisition
def test_a_new_tracker_id_near_the_prediction_is_re_adopted():
    # The car (id 1) vanishes at 2 s and comes back at 3.5 s, but ByteTrack calls it id 7 now. Progress re-attaches
    # it to the followed track (the track takes the id 7 and keeps everything it knew), the director follows id 7,
    # and logs "reacquired" with how = "new_id". It is then followed to the exit.
    rig = Rig()
    car = steady_car(100.0, lambda t: 2.0 <= t < 3.5, id_after=(3.5, 7))
    drive(rig, [car], seconds=15.0)
    re = rig.events_of("reacquired")
    assert len(re) == 1 and re[0]["how"] == "new_id" and re[0]["old_id"] == 1 and re[0]["track_id"] == 7
    assert re[0]["dropout_s"] == pytest.approx(1.6, abs=0.15)
    assert rig.releases and rig.releases[0][1] == "exit", rig.releases
    # one track all along, not a fresh one: at the exit it has more detections than id 7 alone could have
    # (id 7 was seen for ~8 s = ~80 frames; the followed track also has the ~20 frames of id 1 before the gap)
    n_before = [dec.track.n_updates for t, dec in rig.decisions if dec.track is not None and t < 5.0 + 2.0][-1]
    exit_view = [dec.track for t, dec in rig.decisions if dec.reason == "exit"][0]
    t_exit = [t for t, dec in rig.decisions if dec.reason == "exit"][0]
    seen_as_7 = len([1 for t, dec in rig.decisions if 5.0 + 3.5 - 1e-9 <= t <= t_exit])
    assert exit_view.track_id == 7
    assert exit_view.n_updates >= n_before + seen_as_7 - 1


def test_a_car_that_was_already_there_is_not_adopted():
    # Two cars: id 1 (followed) and id 2, 60 px behind it, both visible from the start. Id 1 vanishes for good.
    # Id 2 passes right through the gate around the prediction, but it existed before the dropout: it is another
    # car, never adopted. Id 1 is let go as lost.
    rig = Rig()
    drive(rig, [steady_car(100.0, lambda t: t >= 2.0, track_id=1),
                steady_car(100.0, track_id=2, x0=40.0)], seconds=6.0)
    assert not rig.events_of("reacquired")
    assert rig.releases and rig.releases[0][1] == "lost", rig.releases
    assert rig.events_of("lock")[0]["track_id"] == 1


def test_a_new_id_far_from_the_prediction_is_not_adopted():
    # The followed car vanishes; a NEW id shows up 0.5 of the zone further on (another car, or a false detection
    # deep in the zone): far outside the gate. Not adopted; the followed car is let go as lost.
    rig = Rig()
    drive(rig, [steady_car(100.0, lambda t: t >= 2.0),
                lambda t: (9, 100.0 + 100.0 * t + 500.0) if t >= 2.5 else None], seconds=6.0)
    assert not rig.events_of("reacquired")
    assert rig.releases[0][1] == "lost"


def test_a_lone_raw_detection_near_the_prediction_updates_the_followed_car():
    # During the dropout the detector sees the car ONCE (as on the bench at 1.2 s). ByteTrack doesn't report a track
    # from a single detection, so the tracker output has no box for it; the raw detection is used directly. A second
    # car (id 4, tracked all along, 100 px behind) is also in the raw detections: its raw box is the one the tracker
    # reports for id 4, so it is recognised as that car and ignored, although it is inside the gate too.
    rig = Rig()
    car1 = steady_car(100.0, lambda t: t >= 2.0)
    car4 = steady_car(100.0, track_id=4, x0=0.0)
    drive(rig, [car1, car4], seconds=2.0)
    tp = rig.prog.tracks[1]
    t = 5.0 + 3.0                                          # 1.1 s after the last detection of car 1
    x_true = 100.0 + 100.0 * 3.0
    other = box_at(0.0 + 100.0 * 3.0)
    rig.cycle(t, [(4, other, 0.9)], raw_boxes=np.array([other, box_at(x_true)]))
    assert tp.last_update_t == t
    assert tp.samples[-1][1] == pytest.approx((x_true - 200.0) / 1000.0, abs=0.005)
    re = rig.events_of("reacquired")
    assert len(re) == 1 and re[0]["how"] == "raw_detection" and re[0]["track_id"] == 1


def test_a_raw_detection_far_from_the_prediction_is_ignored():
    rig = Rig()
    drive(rig, [steady_car(100.0, lambda t: t >= 2.0)], seconds=2.0)
    tp = rig.prog.tracks[1]
    last = tp.last_update_t
    rig.cycle(5.0 + 2.5, [], raw_boxes=np.array([box_at(100.0 + 100.0 * 2.5 + 450.0)]))   # 0.45 ahead of the car
    assert tp.last_update_t == last
    assert not rig.events_of("reacquired")


# =================================================================================================== whole runtime
class ScriptedSpotter:
    """Stands in for zone/spotter.py: no image, the boxes of each frame are set by the test, then go through rt's
    real ByteTracker. Keeps `last_dets` (the raw detections) like the real one."""

    def __init__(self):
        self.tracker = ByteTracker()
        self.boxes = []
        self.last_dets = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
        self.last_ms = 0.0

    def set_roi(self, roi):
        pass

    def process(self, img):
        b = np.array(self.boxes, np.float32).reshape(-1, 4)
        s = np.full(len(b), 0.9, np.float32)
        self.last_dets = (b, s)
        return self.tracker.update(b, s)


def run_runtime(sets=(), dropout=(5.0, 7.5), seconds=13.0):
    """The real ZoneRuntime on a fake clock, with the simulated PTZ and the real ByteTracker. One car at 120 px/s
    (0.15 /s on this 800 px path) from t = 3 s, undetected during `dropout`. -> (runtime, events)."""
    zone = straight_zone()                                 # s = (x - 200) / 800
    st = settings("B", sets=sets)
    clock = FakeClock(0.0)
    ptz = SimulatedPtz(PtzModel(), clock)
    for k in zone.keyframes():
        ptz.define_preset(k.preset, k.pan, k.tilt, k.zoom)
    events = []
    spotter = ScriptedSpotter()
    rt = ZoneRuntime(st, zone, ptz, clock, spotter=spotter, on_event=events.append)
    rt.set_mode(D.AUTO)
    for k in range(int(seconds * FPS)):
        t = k * DT
        clock.advance(max(0.0, t + 0.05 - clock.monotonic()))
        spotter.boxes = [] if (t < 3.0 or dropout[0] <= t < dropout[1]) else [box_at(100.0 + 120.0 * (t - 3.0))]
        rt.cycle(ZoneFrame(None, t, 1.8e9 + t, k))
    return rt, events


def test_the_runtime_follows_a_car_through_a_2_5_s_dropout_with_the_real_tracker():
    rt, events = run_runtime()
    director = [e for e in events if e.get("type") == "director"]
    releases = [e for e in director if e["event"] == "release"]
    assert releases and releases[0]["reason"] == "exit", releases
    re = [e for e in director if e["event"] == "reacquired"]
    assert len(re) == 1 and re[0]["dropout_s"] == pytest.approx(2.6, abs=0.15)
    # whichever id ByteTrack gave it when it came back, the director now follows that one
    assert re[0]["how"] in ("same_id", "new_id")


def test_the_runtime_with_the_old_timeout_loses_it():
    _rt, events = run_runtime(sets=["director.lost_timeout_s=1.5"])
    releases = [e for e in events if e.get("type") == "director" and e["event"] == "release"]
    assert releases[0]["reason"] == "lost"


# =================================================================================================== config
def test_config_defaults_and_the_bench_hand_profile():
    st = C.build(C.load(None, []), C.Report(PtzModel(), "B", "", "", {}))
    assert st.director.lost_timeout_s == 3.0
    assert st.director.loosen_after_s == 0.5 and st.director.loosen_max == 0.3
    assert st.progress.max_predict_s_ahead == 0.25
    # progress must not forget the followed car before the director gives it up
    assert st.progress.finish_after_s > st.director.lost_timeout_s
    bench = C.build(C.load(None, ["profile=bench_hand"]), C.Report(PtzModel(), "B", "", "", {}))
    assert bench.director.lost_timeout_s >= 4.0
    assert bench.progress.finish_after_s > bench.director.lost_timeout_s
    assert bench.progress.max_prediction_s >= bench.director.lost_timeout_s
