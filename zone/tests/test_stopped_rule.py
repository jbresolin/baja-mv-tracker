"""When is a followed car "stopped"? (zone/director.py, the stopped rule; zone/progress.py, ground_px_s)

Background for a first-time reader: a car's progress `s` is measured along the path drawn on the SPOTTER IMAGE, in
pixels. Where the track runs toward (or away from) the spotter, many metres of track fit in a few pixels, so a car at
racing speed gains very little `s` per second there. The old rule ("|v| < 0.02 s/s for 3 s = stopped") dropped such a
car although it was flying. The rule now needs BOTH:

  (a) slow along the path: below 15 % of the typical speed learned at that spot (or below 0.02 s/s before anything
      is learned), and
  (b) still in the image: the bottom centre of its box moves less than 8 pixels per second.

These tests drive a fake car (just a box, moving in pixels) through the real progress tracker and director, with the
default config values, ten frames a second. No camera, no PTZ: only the decisions.
"""
import math

from zone import config as C
from zone import director as D
from zone.director import Director
from zone.path import PathModel
from zone.progress import ProgressTracker, SpeedProfile
from zone.ptz.sim import PtzModel
from zone.zonefile import Keyframe, PathPoint, Zone

FPS = 10.0


def make(length_px=1000.0, profile=None, sets=()):
    """A straight path along y = 500: an approach point at x = 100, the entry keyframe at x = 200 (s = 0) and the
    exit keyframe `length_px` further (s = 1). -> (progress tracker, director in AUTO waiting at the entry, settings)."""
    zone = Zone(points=[PathPoint(100, 500), PathPoint(200, 500, Keyframe(0, 0, 0, 10)),
                        PathPoint(200 + length_px, 500, Keyframe(500, 0, 0, 11))], corridor_half_width_px=60)
    st = C.build(C.load(None, list(sets)), C.Report(PtzModel(), "B", "", "", {}))   # the config (default), no report
    path = PathModel(zone)
    prog = ProgressTracker(path, st.progress, profile or SpeedProfile())
    director = Director(st.director)
    director.set_mode(D.AUTO, 0.0)
    director.step(0.0, {})                        # the entry recall
    director.step(st.director.settle_s + 0.1, {})  # settled: waiting for a car
    assert director.sub == D.WAITING
    return prog, director, st


def drive(prog, director, x_of_t, seconds, t0=5.0, jitter_px=0.0):
    """Move a 40x40 box whose bottom centre is at (x_of_t(t), 500) for `seconds`. `jitter_px`: the box's bottom edge
    jumps up and down by that much every frame, as a noisy detection does. -> the director's releases
    [(seconds since the start, reason)], and whether it was locked at some point."""
    out, locked = [], False
    for k in range(int(seconds * FPS)):
        t = t0 + k / FPS
        x = x_of_t(t - t0)
        y = 500 + (jitter_px if k % 2 else -jitter_px)
        views = prog.update(t, [(1, (x - 20, y - 40, x + 20, y), 0.9)])
        decision = director.step(t, views)
        locked = locked or decision.reason == "lock"
        if decision.action == D.RECALL_ENTRY:
            out.append((round(t - t0, 1), decision.reason))
    return out, locked


def piecewise(*legs):
    """x(t) for a car that drives legs of (seconds, px per second), starting at x = 100 (on the approach)."""
    def x(t):
        pos, left = 100.0, t
        for dur, speed in legs:
            step = min(left, dur)
            pos += step * speed
            left -= step
            if left <= 0:
                break
        return pos
    return x


def test_a_fast_car_on_a_head_on_stretch_is_not_dropped():
    # 400 px/s on the normal part of the path; on the head-on stretch the box moves only 12 px/s in the image
    # (0.012 s/s: below the old 0.02 limit) for 8 seconds; then it comes out and reaches the exit.
    prog, director, _ = make()
    drops, locked = drive(prog, director, piecewise((1.0, 400), (8.0, 12), (10.0, 400)), 14)
    assert locked
    assert drops == [(drops[0][0], "exit")], drops       # let go at the exit, never "stopped"


def test_a_car_that_really_stops_is_still_dropped():
    # the same car stops dead on the stretch (its box jitters by 4 pixels each frame, like a noisy detection): after
    # stopped_timeout_s (3 s) it is dropped as "stopped"
    prog, director, st = make()
    drops, _ = drive(prog, director, piecewise((1.0, 400), (0.5, 12)), 10, jitter_px=4.0)
    assert len(drops) == 1 and drops[0][1] == "stopped", drops
    # it is slow from t = 1.0 (12 px/s is already below 0.02 s/s) and still from t = 1.5: dropped stopped_timeout_s
    # after that, give or take the filter's reaction time
    assert 1.0 + st.director.stopped_timeout_s <= drops[0][0] <= 1.5 + st.director.stopped_timeout_s + 2.0


def creep(profile):
    """A car that, after entering, crawls at 6 px/s (below the 8 px/s pixel limit) on a 500 px path: 0.012 s/s."""
    prog, director, _ = make(length_px=500, profile=profile)
    return drive(prog, director, piecewise((1.0, 400), (12.0, 6)), 10)[0]


def test_before_the_profile_is_learned_the_absolute_limit_decides():
    # nothing learned: 0.012 s/s is below stopped_v (0.02), and 6 px/s is below 8: dropped as stopped
    assert [r for _, r in creep(SpeedProfile())] == ["stopped"]


def test_once_learned_a_car_at_the_usual_speed_of_that_spot_is_not_stopped():
    # the profile learned that cars usually do ~0.012 s/s around s = 0.5..1 (a head-on stretch on this path), so
    # 0.012 there is normal speed, not "slow": the car is kept even though it barely moves in the image
    profile = SpeedProfile()
    for _ in range(3):
        profile.add_pass([(s / 100, 0.8 if s < 50 else 0.012) for s in range(-20, 120)])
    assert creep(profile) == []


def test_the_learned_rule_also_catches_a_crawl_that_the_old_limit_missed():
    # the opposite case: cars usually do 0.8 s/s there; a car crawling at 0.03 s/s (above the old 0.02, so the
    # old rule would follow it forever) and moving only 3 px/s in the image is stopped
    profile = SpeedProfile()
    profile.add_pass([(s / 100, 0.8) for s in range(-20, 120)])
    prog, director, _ = make(length_px=100, profile=profile)
    drops, _ = drive(prog, director, piecewise((0.25, 400), (12.0, 3)), 10)
    assert [r for _, r in drops] == ["stopped"]


def test_lock_also_accepts_a_head_on_entry_that_visibly_moves():
    # v_min is 0.05 s/s. An entry seen head-on: the car advances only 0.03 s/s (30 px/s on this 1000 px path) but its
    # box clearly moves forward in the image (>= lock_px_s = 15 px/s): it is locked
    prog, director, _ = make()
    _, locked = drive(prog, director, piecewise((30.0, 30)), 8)
    assert locked
    # with the pixel rule off (lock_px_s = 0) the same car is never locked
    prog, director, _ = make()
    director.cfg.lock_px_s = 0.0
    _, locked = drive(prog, director, piecewise((30.0, 30)), 8)
    assert not locked


# ------------------------------------------------------------------------------------------- the hand-held bench test
def hand(prog, director, seconds=30, pause_s=0.0, wobble_px=0.0, dropout=None):
    """A printed picture moved by hand along a 600 px path: 1 s at 150 px/s, then a pause of `pause_s`, again and
    again. `wobble_px`: the hand goes up and down that much (the picture's bottom edge leaves the taught line).
    `dropout` = (from, to) seconds: the detector loses the picture (blur, the hand covers it). -> first release reason."""
    x = 100.0
    for k in range(int(seconds * FPS)):
        t = 5.0 + k / FPS
        if (k / FPS) % (1.0 + pause_s) < 1.0:
            x += 150 / FPS
        y = 500 + wobble_px * math.sin(k / 3)
        seen = not (dropout and dropout[0] <= k / FPS < dropout[1])
        decision = director.step(t, prog.update(t, [(1, (x - 30, y - 60, x + 30, y), 0.9)] if seen else []))
        if decision.action == D.RECALL_ENTRY:
            return decision.reason
    return None


def test_a_hand_moved_picture_trips_the_race_rules_but_not_the_bench_hand_profile():
    # What drops a picture moved by hand first, with the race defaults, and that `--set profile=bench_hand` keeps it
    # to the exit. Each case is one way a hand differs from a car.
    cases = [(dict(wobble_px=80), "off_corridor"),      # the bottom edge wanders 80 px off the line (corridor 60)
             (dict(dropout=(3, 5)), "lost"),            # 2 s without a detection (race limit 1.5 s)
             (dict(pause_s=3.5), "stopped")]            # the hand stops for 3.5 s (race limit 3 s)
    for kwargs, race_reason in cases:
        prog, director, _ = make(length_px=600)
        assert hand(prog, director, **kwargs) == race_reason, kwargs
        prog, director, _ = make(length_px=600, sets=["profile=bench_hand"])
        assert hand(prog, director, **kwargs) == "exit", kwargs
