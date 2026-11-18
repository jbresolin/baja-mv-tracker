"""Only lock cars that came in through the entry (director.require_born_upstream).

The problem: where the taught path loops back and passes near (or across) its own entry, a car on the LATER stretch
can be detected for the first time right at the entry. Its first position is then taken on the entry stretch, it
looks like a car that just crossed the entry, and AUTO locks it: the PTZ follows a car that isn't where the camera
thinks it is.

The rule now: a track may be locked only if
  * it was BORN (first detected) upstream of the entry: before director.s_lock, in the approach (or, for a zone with
    no approach, within director.born_window of the path's start), and
  * it has been followed continuously since (a dropout up to director.lost_timeout_s is fine, also when the tracker
    gives the car a new id after it).
A car born anywhere else is never locked, and the page says why ("nasceu fora da entrada").

Background for a first-time reader:
* `s` is the progress along the path: 0 at the first keyframe (the entry), 1 at the last one, negative in the
  approach. `v` is in "zone per second".
* These tests run the real ZoneRuntime (progress tracker + director + controller + simulated PTZ) on a FakeClock:
  time only moves when the test moves it, one frame every 0.1 s. The "tracker output" is written by hand: a list of
  (track id, box, score), as ByteTrack would return it.
"""
import pytest

from zone import director as D
from zone.director import Director, DirectorSettings
from zone.progress import TrackView
from zone.runtime import ZoneRuntime
from zone.source import ZoneFrame
from zone.tests.z2_fixtures import box_from_ground, crossing_zone, settings, sim, straight_zone

FPS = 10.0
DT = 1.0 / FPS


class Run:
    """The real runtime on a fake clock. `frame(t, tracker_out)` runs one control cycle at time t."""

    def __init__(self, zone, sets=(), mode=D.AUTO):
        self.ptz, self.clock = sim()
        for k in zone.keyframes():
            self.ptz.define_preset(k.preset, k.pan, k.tilt, k.zoom)
        self.events = []
        self.rt = ZoneRuntime(settings("B", sets=sets), zone, self.ptz, self.clock, on_event=self.events.append)
        ok, _ = self.rt.set_mode(mode)
        assert ok
        self.k = 0

    def frame(self, t, tracker_out):
        self.clock.advance(max(0.0, t - self.clock.monotonic()))
        self.k += 1
        return self.rt.cycle(ZoneFrame(None, t, 1.8e9 + t, self.k), tracker_out)

    def locks(self):
        """Track ids the director locked, in order."""
        return [e["track_id"] for e in self.events if e.get("type") == "director" and e.get("event") == "lock"]


# =================================================================================================== the rule
def test_born_limit_is_s_lock_or_the_start_of_a_path_without_approach():
    cfg = DirectorSettings(s_lock=-0.08, born_window=0.05)
    assert cfg.born_limit(-0.5) == pytest.approx(-0.08)     # a long approach: born before s_lock
    assert cfg.born_limit(0.0) == pytest.approx(0.05)       # no approach (path starts at the entry keyframe)
    assert cfg.born_limit(-0.1) == pytest.approx(-0.05)     # a short approach: whichever is further along


def view(tid, s, born_s=None, up=None, cont=True):
    """A track as the progress tracker would report it, 5 detections, moving forward."""
    return TrackView(track_id=tid, s=s, v=0.25, s_std=0.01, v_std=0.01, lateral_px=0.0, n_updates=5,
                     since_update_s=0.0, off_corridor_frames=0, predicted=False, box=(), ground=(),
                     born_s=born_s, born_upstream=up, continuous=cont)


def auto_waiting(**kw):
    d = Director(DirectorSettings(settle_s=1.0, s_lock=-0.08, **kw))
    d.set_mode(D.AUTO, 0.0)
    d.step(0.0, {})
    d.step(5.0, {})
    assert d.sub == D.WAITING
    return d


def test_the_director_skips_a_car_born_elsewhere_and_says_why():
    d = auto_waiting()
    outsider = view(1, 0.02, born_s=0.02, up=False)
    assert d.lock_block_reason(outsider) == "nasceu fora da entrada (s=0.02)"
    assert d.step(5.1, {1: outsider}).action == D.NONE
    # a car that came through the approach is locked
    dec = d.step(5.2, {1: outsider, 2: view(2, 0.01, born_s=-0.3, up=True)})
    assert dec.action == D.FOLLOW and dec.track.track_id == 2


def test_a_car_followed_with_a_gap_is_not_lockable():
    d = auto_waiting()
    gap = view(3, 0.02, born_s=-0.3, up=True, cont=False)
    assert d.lock_block_reason(gap) == "perdeu a continuidade desde a entrada"
    assert d.step(5.1, {3: gap}).action == D.NONE


def test_the_switch_turns_the_rule_off():
    d = auto_waiting(require_born_upstream=False)
    assert d.step(5.1, {1: view(1, 0.02, born_s=0.02, up=False)}).action == D.FOLLOW


def test_hand_made_tracks_without_birth_information_still_lock():
    # TrackViews that don't say where they were born (born_upstream None) are not blocked: no rule applies
    d = auto_waiting()
    assert d.step(5.1, {1: view(1, 0.02)}).action == D.FOLLOW


# =================================================================================================== straight path
def test_a_car_through_the_approach_is_locked():
    # straight_zone: path from x = 100 (s = -0.125) to the entry at x = 200, s = (x - 200) / 800.
    run = Run(straight_zone())
    for k in range(30):
        t = 2.0 + k * DT
        run.frame(t, [(1, box_from_ground(110.0 + 120.0 * k * DT, 500.0), 0.9)])
    assert run.locks() == [1]


def test_a_car_that_appears_in_the_middle_of_the_lock_window_is_not_locked():
    # The same car, but first detected at x = 205 (s = 0.006, past s_lock = -0.08): not born upstream
    run = Run(straight_zone())
    for k in range(10):
        t = 2.0 + k * DT
        run.frame(t, [(1, box_from_ground(205.0 + 120.0 * k * DT, 500.0), 0.9)])
    assert run.locks() == []
    snap = run.rt.snapshot()
    assert snap["tracks"][0]["why_not"].startswith("nasceu fora da entrada")
    assert snap["tracks"][0]["up"] is False
    # ... unless the switch is off (the old behaviour)
    run = Run(straight_zone(), sets=["director.require_born_upstream=false"])
    for k in range(10):
        t = 2.0 + k * DT
        run.frame(t, [(1, box_from_ground(205.0 + 120.0 * k * DT, 500.0), 0.9)])
    assert run.locks() == [1]


def test_a_zone_without_approach_uses_the_born_window():
    # The path starts AT the entry keyframe (no approach point). A car first seen in the first 0.05 of the zone
    # came through the entry; one first seen at s = 0.06 did not.
    zone = straight_zone()
    del zone.points[0]                       # drop the approach point: now s starts at 0
    run = Run(zone)
    assert run.rt.progress.born_limit_s == pytest.approx(0.05)
    for k in range(8):
        t = 2.0 + k * DT
        run.frame(t, [(1, box_from_ground(210.0 + 120.0 * k * DT, 500.0), 0.9),    # born s = 0.0125
                      (2, box_from_ground(250.0 + 120.0 * k * DT, 500.0), 0.9)])   # born s = 0.0625
    views = run.rt.last.views
    assert views[1].born_upstream is True and views[2].born_upstream is False
    assert run.locks() == [1]


def test_a_new_tracker_id_after_a_dropout_in_the_approach_keeps_the_car_lockable():
    # ByteTrack often renumbers a car that was unseen for a few frames. Car 1 is born in the approach (3 detections,
    # not locked yet: still before s_lock), unseen for 0.5 s, and comes back as id 7 at the entry (where nothing
    # could be born upstream): id 7 inherits "born upstream" from car 1 and is locked.
    run = Run(straight_zone())

    def x_at(t):
        return 100.0 + 120.0 * (t - 2.0)
    for k in range(25):
        t = 2.0 + k * DT
        if t < 2.25:
            out = [(1, box_from_ground(x_at(t), 500.0), 0.9)]
        elif t < 2.75:
            out = []                                   # dropout
        else:
            out = [(7, box_from_ground(x_at(t), 500.0), 0.9)]
        run.frame(t, out)
    assert run.locks() == [7]
    v7 = run.rt.last.views[7]
    assert v7.born_upstream is True and v7.born_s < -0.08


def test_a_new_id_far_from_the_lost_car_does_not_inherit():
    # Car 1 vanishes in the approach; a car with a new id appears 300 px further on (not where car 1 can be): it
    # is another car, born outside the entry, never locked.
    run = Run(straight_zone())
    for k in range(20):
        t = 2.0 + k * DT
        if t < 2.25:
            out = [(1, box_from_ground(100.0 + 120.0 * (t - 2.0), 500.0), 0.9)]
        elif t < 2.75:
            out = []
        else:
            out = [(7, box_from_ground(500.0 + 120.0 * (t - 2.75), 500.0), 0.9)]
        run.frame(t, out)
    assert run.locks() == []
    assert run.rt.last.views[7].born_upstream is False


def test_a_long_gap_under_the_same_id_breaks_continuity():
    # Same id, but unseen for longer than lost_timeout_s (3 s): maybe not the same car any more.
    run = Run(straight_zone(), sets=["director.lost_timeout_s=1.0"])
    run.frame(2.0, [(1, box_from_ground(110.0, 500.0), 0.9)])
    run.frame(2.1, [(1, box_from_ground(122.0, 500.0), 0.9)])
    for k in range(6):
        run.frame(3.5 + k * DT, [(1, box_from_ground(190.0 + 12.0 * k, 500.0), 0.9)])   # 1.4 s later
    assert run.rt.last.views[1].continuous is False
    assert run.locks() == []
    assert run.rt.snapshot()["tracks"][0]["why_not"] == "perdeu a continuidade desde a entrada"


def test_editing_the_zone_makes_cars_already_inside_unlockable():
    # After an edit the tracks restart where the cars are: none of them came "through the entry" any more.
    run = Run(straight_zone(), mode=D.PASSIVE)
    run.frame(2.0, [(1, box_from_ground(110.0, 500.0), 0.9)])
    assert run.rt.last.views[1].born_upstream is True
    run.rt.set_zone(straight_zone(half_width=70.0))
    run.frame(2.1, [(1, box_from_ground(205.0, 500.0), 0.9)])
    assert run.rt.last.views[1].born_upstream is False


# =================================================================================================== a looped path
def shifted_diagonal_ground(t, t0, speed_px_s=300.0):
    """A car on the crossing zone's LAST stretch (the diagonal from (200,350) to (500,650)), running 40 px to the
    right of the taught line (real cars don't drive exactly on it). At t0 its ground point is exactly on the ENTRY
    stretch (y = 500), at (390,500): closer to the entry stretch than to its own diagonal."""
    d = speed_px_s * (t - t0) / 2 ** 0.5           # moves down-right at 45 degrees
    return 390.0 + d, 500.0 + d


def run_the_crossing_car(sets=()):
    """A car on the looped path's last stretch, first detected right where it crosses the entry (it was hidden
    before: dust, the detector, another car). -> the Run."""
    run = Run(crossing_zone(), sets=sets)
    for k in range(12):
        t = 2.0 + k * DT
        gx, gy = shifted_diagonal_ground(t, 2.0)
        run.frame(t, [(4, box_from_ground(gx, gy), 0.9)])
    return run


def test_old_rules_lock_a_car_from_the_later_stretch_crossing_the_entry():
    # This is the failure: with the closest-point birth (overlap check off) and no birth rule, the crossing car is
    # taken for a car on the entry stretch at s ~ 0.03 that moves forward, and it is locked.
    run = run_the_crossing_car(sets=["director.require_born_upstream=false", "progress.overlap_size_check=false"])
    assert run.locks() == [4]


def test_the_birth_rule_alone_refuses_it():
    # Same closest-point birth (overlap check off), birth rule on: the track is born at s ~ 0.03, after s_lock: never
    # locked, and the page says why.
    run = run_the_crossing_car(sets=["progress.overlap_size_check=false"])
    assert run.locks() == []
    why = [t["why_not"] for t in run.rt.snapshot()["tracks"] if t["id"] == 4]
    assert why and why[0].startswith("nasceu fora da entrada")


def test_with_the_defaults_the_crossing_car_is_never_locked():
    # Defaults: the first detection is near two stretches (ambiguous) and no size profile is learned yet, so no
    # track starts there; once the car is off the crossing it is born on its own (diagonal) stretch at s ~ 0.95.
    run = run_the_crossing_car()
    assert run.locks() == []
    assert run.rt.progress.ambiguity["waited_unlearned"] >= 1
    v = run.rt.last.views.get(4)
    assert v is not None and v.s > 0.9 and v.born_upstream is False


def test_a_normal_car_on_the_looped_path_is_still_locked():
    # A car that drives the approach of the looped path is born upstream and locked as usual, and keeps its own
    # stretch through the crossing (existing tracks follow the projection window).
    zone = crossing_zone()
    run = Run(zone)
    from zone.path import PathModel
    path = PathModel(zone)
    for k in range(20):
        t = 2.0 + k * DT
        gx, gy = path.point_at(path.s_min + 0.3 * (t - 2.0))    # 0.3 of the zone per second
        run.frame(t, [(2, box_from_ground(gx, gy), 0.9)])
    assert run.locks() == [2]
    assert run.rt.last.views[2].s < 0.6


# =================================================================================================== config
def test_the_new_config_keys_reach_the_settings():
    # zone.example.toml has the defaults; --set (or site.toml) changes them
    st = settings("B")
    assert st.director.require_born_upstream is True and st.director.born_window == pytest.approx(0.05)
    assert st.progress.overlap_size_check is True and st.progress.size_tolerance == pytest.approx(0.35)
    assert st.progress.continuity_gap_s == pytest.approx(st.director.lost_timeout_s)   # follows the timeout
    st = settings("B", sets=["director.require_born_upstream=false", "director.born_window=0.1",
                             "progress.overlap_size_check=false", "progress.overlap_min_sep_px=300",
                             "progress.size_tolerance=0.2", "progress.size_min_contrast=1.5",
                             "progress.size_min_passes=5", "director.lost_timeout_s=4.5"])
    assert st.director.require_born_upstream is False and st.director.born_window == pytest.approx(0.1)
    p = st.progress
    assert (p.overlap_size_check, p.overlap_min_sep_px, p.size_tolerance, p.size_min_contrast, p.size_min_passes) == (
        False, 300.0, 0.2, 1.5, 5)
    assert p.continuity_gap_s == pytest.approx(4.5)
