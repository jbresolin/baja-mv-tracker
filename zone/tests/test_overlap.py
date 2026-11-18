"""Overlapping track sections: choosing the right stretch of path for a NEW car by its apparent size.

The situation: the taught path passes near (or across) itself in the spotter image. A car detected there for the
first time is inside the corridor of two stretches of path that are far apart along it (here: just after the entry,
s ~ 0.02, and just before the exit, s ~ 0.92). Which one is it on? The closest point can't tell (a car a little off
its line is often closer to the other stretch).

What helps: the two stretches are at different distances from the spotter, so cars look different sizes there. In
PASSIVE / AUTO the tracker learns the typical box HEIGHT at each s (the "size profile", zone.size.json, next to the
speed profile), from cars whose track started where there was no doubt. For an ambiguous new detection it then picks
the stretch whose expected height matches (height / expected closest to 1, within progress.size_tolerance). If the
profile hasn't learned those places yet, or nothing matches, no track is started from that detection: the tracker
waits for the next frame (the car soon leaves the overlap, and is then unambiguous).

Tracks that already exist are not affected: they keep searching only near where they were (the "projection window").

Background for a first-time reader:
* The zone used here is `crossing_zone()` (zone/tests/z2_fixtures.py): approach -> entry (s = 0) -> right -> up ->
  left -> down -> a diagonal that crosses the entry stretch at (350, 500) -> exit (s = 1).
* A detection is a box; its "ground point" is the bottom centre. `box_from_ground(x, y, height)` builds one.
* These tests run on a fake clock: one frame every 0.1 s, with the tracker output written by hand.
"""
import json

import pytest

from zone import director as D
from zone.path import PathModel
from zone.progress import ProgressSettings, ProgressTracker, SizeProfile, size_profile_path_for
from zone.runtime import ZoneRuntime
from zone.source import ZoneFrame
from zone.tests.z2_fixtures import box_from_ground, crossing_zone, settings, sim, straight_zone

DT = 0.1
NEAR_H, FAR_H = 80.0, 30.0       # box heights: the entry half of the path is near the spotter, the rest far


def height_at(s):
    """The 'true' apparent height of a car at s in this made-up world: big (near) before s = 0.5, small after."""
    return NEAR_H if s < 0.5 else FAR_H


# =================================================================================================== geometry
def test_candidates_at_the_crossing_are_both_stretches():
    path = PathModel(crossing_zone())
    cands = path.candidates(350.0, 500.0)
    assert len(cands) == 2
    assert cands[0].projection.s == pytest.approx(0.0187, abs=0.002)    # the entry stretch
    assert cands[1].projection.s == pytest.approx(0.9207, abs=0.002)    # the diagonal before the exit
    assert path.candidates(800.0, 500.0)[0].projection.s == pytest.approx(path.project(800.0, 500.0).s)
    assert len(path.candidates(800.0, 500.0)) == 1                       # elsewhere: one stretch only
    assert path.candidates(800.0, 900.0) == []                           # outside every corridor


def test_a_sharp_corner_is_not_two_stretches():
    # Next to the corner at (1100, 500) a point is inside the corridor of the segment before and the one after,
    # but they are the same place along the path (less than overlap_min_sep_px apart): one candidate.
    path = PathModel(crossing_zone())
    assert len(path.candidates(1080.0, 480.0)) == 1


def test_candidates_use_the_per_point_width():
    # Narrow the corridor to 35 px (on the diagonal's two points; the points before them hold the nearest value, so
    # the whole path is 35 px here): a detection 50 px from the diagonal and 15 px from the entry stretch is then
    # only on the entry stretch. With the global 100 px it is on both.
    zone = crossing_zone()
    zone.points[5].half_width_px = 35.0
    zone.points[6].half_width_px = 35.0
    path = PathModel(zone)
    gy = 485.0                                         # 15 px above the entry stretch (y = 500)
    gx = 350.0 - 15.0 + 50.0 * 2 ** 0.5                # 50 px to the right of the diagonal (which runs at 45 degrees)
    assert len(PathModel(crossing_zone()).candidates(gx, gy)) == 2
    assert len(path.candidates(gx, gy)) == 1


# =================================================================================================== the size profile
def test_size_profile_bins_and_file(tmp_path):
    p = SizeProfile(s_lo=-0.5, s_hi=1.2, bin_width=0.05)
    assert p.expected(0.02) is None                          # nothing learned
    for _ in range(3):
        p.add_pass([(0.02, 79.0), (0.03, 81.0), (0.92, 30.0)])
    assert p.expected(0.02) == pytest.approx(80.0)            # median of the per-pass medians
    assert p.expected(0.02, min_passes=4) is None             # only 3 passes there
    assert p.expected(0.7) is None                            # no borrowing from other bins
    f = tmp_path / "zone.size.json"
    p.save(f)
    q = SizeProfile.load(f)
    assert q.expected(0.92) == pytest.approx(30.0) and q.n_passes == 3
    assert size_profile_path_for(tmp_path / "zone.json") == f


# =================================================================================================== progress alone
def learned_tracker(passes=3, **cfg):
    """A progress tracker on the crossing zone whose size profile has learned NEAR_H / FAR_H along the path."""
    path = PathModel(crossing_zone())
    prog = ProgressTracker(path, ProgressSettings(**cfg))
    s = path.s_min
    samples = []
    while s < path.s_max:
        samples.append((s, height_at(s)))
        s += 0.01
    for _ in range(passes):
        prog.size_profile.add_pass(samples)
    return prog


def test_unlearned_profile_no_track_in_the_overlap():
    prog = ProgressTracker(PathModel(crossing_zone()), ProgressSettings())
    views = prog.update(1.0, [(1, box_from_ground(350.0, 500.0, FAR_H), 0.9)])
    assert views == {}                                         # waits: no track from this detection
    assert prog.ambiguity == dict(ambiguous=1, resolved=0, waited_unlearned=1, waited_no_match=0)
    # the same car one frame later, off the crossing on its diagonal: unambiguous, the track starts
    views = prog.update(1.1, [(1, box_from_ground(480.0, 630.0, FAR_H), 0.9)])
    assert views[1].s > 0.9 and views[1].born_how == "unique"


def test_learned_profile_picks_the_far_stretch_for_a_small_car():
    prog = learned_tracker()
    views = prog.update(1.0, [(1, box_from_ground(350.0, 500.0, 31.0), 0.9)])
    assert views[1].s == pytest.approx(0.92, abs=0.01)
    assert views[1].born_how == "size"
    assert prog.ambiguity["resolved"] == 1
    assert prog.overlap_events and prog.overlap_events[0]["track_id"] == 1


def test_learned_profile_picks_the_near_stretch_for_a_big_car():
    prog = learned_tracker()
    views = prog.update(1.0, [(1, box_from_ground(350.0, 500.0, 77.0), 0.9)])
    assert views[1].s == pytest.approx(0.02, abs=0.01)


def test_no_match_within_tolerance_waits():
    # 50 px: 0.62 x the near height and 1.67 x the far one, both outside +-35 %: no track
    prog = learned_tracker()
    assert prog.update(1.0, [(1, box_from_ground(350.0, 500.0, 50.0), 0.9)]) == {}
    assert prog.ambiguity["waited_no_match"] == 1


def test_two_stretches_of_similar_size_cant_be_told_apart():
    # If both stretches look the same size (the profile says 40 px at both), a 40 px car could be on either: wait.
    path = PathModel(crossing_zone())
    prog = ProgressTracker(path, ProgressSettings())
    for _ in range(3):
        prog.size_profile.add_pass([(0.0187, 40.0), (0.9207, 42.0)])
    assert prog.update(1.0, [(1, box_from_ground(350.0, 500.0, 40.0), 0.9)]) == {}
    assert prog.ambiguity["waited_no_match"] == 1


def test_non_ambiguous_detections_are_unaffected():
    # Away from the crossing a new car starts a track at once, learned profile or not, exactly where the old
    # closest-point rule put it.
    prog = ProgressTracker(PathModel(crossing_zone()), ProgressSettings())
    views = prog.update(1.0, [(1, box_from_ground(800.0, 500.0, 50.0), 0.9)])
    assert views[1].s == pytest.approx(prog.path.project(800.0, 500.0).s)
    assert prog.ambiguity["ambiguous"] == 0


def test_switch_off_gives_the_old_closest_point_rule():
    prog = ProgressTracker(PathModel(crossing_zone()), ProgressSettings(overlap_size_check=False))
    views = prog.update(1.0, [(1, box_from_ground(355.0, 500.0, FAR_H), 0.9)])
    assert views[1].s == pytest.approx(0.0206, abs=0.002)       # on the entry stretch: the closest point
    assert prog.ambiguity["ambiguous"] == 0


def test_an_existing_track_keeps_its_stretch_through_the_crossing():
    # A car that came through the entry drives over the crossing: it stays on the entry stretch (the projection
    # window), whatever its size, even with nothing learned.
    prog = ProgressTracker(PathModel(crossing_zone()), ProgressSettings())
    x = 150.0                                                   # in the approach: only one stretch near it
    for k in range(9):
        views = prog.update(1.0 + k * DT, [(1, box_from_ground(x, 500.0, FAR_H), 0.9)])
        x += 30.0
    assert 0.0 < views[1].s < 0.1                               # crossed (350, 500) at s ~ 0.02, still on its stretch
    assert prog.ambiguity["ambiguous"] == 0                     # only NEW detections are checked


# =================================================================================================== the runtime
class Run:
    """The real runtime on a fake clock (as in test_born_upstream.py), with the profiles saved in tmp_path."""

    def __init__(self, tmp_path, mode=D.PASSIVE, sets=()):
        self.zone = crossing_zone()
        self.path = PathModel(self.zone)
        self.ptz, self.clock = sim()
        for k in self.zone.keyframes():
            self.ptz.define_preset(k.preset, k.pan, k.tilt, k.zoom)
        self.events = []
        self.size_file = tmp_path / "zone.size.json"
        self.rt = ZoneRuntime(settings("B", sets=sets), self.zone, self.ptz, self.clock,
                              profile_path=tmp_path / "zone.speed.json", size_profile_path=self.size_file,
                              on_event=self.events.append)
        assert self.rt.set_mode(mode)[0]
        self.k = 0

    def frame(self, t, tracker_out):
        self.clock.advance(max(0.0, t - self.clock.monotonic()))
        self.k += 1
        return self.rt.cycle(ZoneFrame(None, t, 1.8e9 + t, self.k), tracker_out)

    def drive_full_pass(self, track_id, t0, speed=0.5):
        """A car along the whole path, from the start of the approach to the end, `speed` zone per second, with the
        true apparent height at each place; then 4 s of nothing so its track finishes (and teaches the profiles).
        -> the time after it."""
        t = t0
        s = self.path.s_min
        while s < self.path.s_max - 0.01:
            gx, gy = self.path.point_at(s)
            self.frame(t, [(track_id, box_from_ground(gx, gy, height_at(s)), 0.9)])
            t += DT
            s += speed * DT
        for _ in range(40):
            self.frame(t, [])
            t += DT
        return t


def test_the_runtime_learns_the_size_profile_in_passive_and_saves_it(tmp_path):
    run = Run(tmp_path)
    t = 1.0
    for tid in (1, 2, 3):
        t = run.drive_full_pass(tid, t)
    sz = run.rt.size_profile
    assert sz.n_passes == 3
    assert sz.expected(0.02) == pytest.approx(NEAR_H) and sz.expected(0.92) == pytest.approx(FAR_H)
    saved = json.loads(run.size_file.read_text())
    assert saved["n_passes"] == 3                               # zone.size.json, next to zone.speed.json
    # a new runtime (a restart) loads it
    run2 = Run(tmp_path)
    assert run2.rt.size_profile.expected(0.92) == pytest.approx(FAR_H)
    # then an ambiguous small car at the crossing gets the far stretch at once
    res = run2.frame(1.0, [(9, box_from_ground(352.0, 500.0, 29.0), 0.9)])
    assert res.views[9].s == pytest.approx(0.92, abs=0.01)
    assert any(e.get("type") == "overlap" for e in run2.events)     # logged (and shown in the page's events)
    snap = run2.rt.snapshot()
    assert snap["overlap"]["counts"]["resolved"] == 1 and snap["overlap"]["size_bins"] > 10


def test_the_size_profile_does_not_learn_in_stopped(tmp_path):
    run = Run(tmp_path, mode=D.STOPPED)
    run.drive_full_pass(1, 1.0)
    assert run.rt.size_profile.n_passes == 0
    assert run.rt.profile.n_passes == 1                         # the speed profile learns as before


def test_a_car_born_by_a_size_decision_does_not_teach_the_size_profile(tmp_path):
    # Only tracks that started without any doubt teach the profile (a wrong guess must not teach itself).
    run = Run(tmp_path)
    for _ in range(3):
        run.rt.size_profile.add_pass([(0.0187, NEAR_H), (0.9207, FAR_H)])
    t = 1.0
    # born at the crossing by size, then drives on along the diagonal
    for k in range(8):
        gx, gy = 350.0 + 20.0 * k, 500.0 + 20.0 * k
        run.frame(t, [(5, box_from_ground(gx, gy, FAR_H), 0.9)])
        t += DT
    assert run.rt.last.views[5].born_how == "size"
    for _ in range(40):
        run.frame(t, [])
        t += DT
    assert run.rt.size_profile.n_passes == 3


def test_the_page_shows_the_ambiguity_counters(tmp_path):
    from zone.tests.test_app import make
    client, app, _, _ = make(tmp_path, crossing_zone())
    app.runtime.set_mode(D.PASSIVE)
    app.runtime.cycle(ZoneFrame(None, 1.0, 1.8e9, 1), [(1, box_from_ground(350.0, 500.0, 30.0), 0.9)])
    ov = client.get("/api/state").json()["runtime"]["overlap"]
    assert ov["enabled"] is True and ov["counts"]["waited_unlearned"] == 1
    assert "trechos sobrepostos" in client.get("/app.js").text


def test_the_control_log_has_the_counters(tmp_path):
    from zone.runtime import ControlLog
    ptz, clock = sim()
    log = ControlLog(tmp_path / "log.jsonl", clock, flush_every=1)
    rt = ZoneRuntime(settings(), straight_zone(), ptz, clock, log=log)
    rt.cycle(ZoneFrame(None, 0.0, 1.8e9, 0), [])
    log.close()
    lines = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    cycles = [x for x in lines if x.get("type") == "cycle"]
    assert cycles[-1]["amb"] == dict(ambiguous=0, resolved=0, waited_unlearned=0, waited_no_match=0)
