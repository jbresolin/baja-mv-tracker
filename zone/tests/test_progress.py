"""Tests for zone/progress.py: the s/v filter, the track manager and the speed profile.

All times are made up (no real clock): a "frame" every 0.1 s, as at 10 fps."""
import math

import numpy as np
import pytest

from zone.path import PathModel
from zone.progress import ProgressSettings, ProgressTracker, SpeedProfile, SVFilter, profile_path_for
from zone.tests.z2_fixtures import straight_zone


# ------------------------------------------------------------------------------------------------ the filter
def run_filter(s_of_t, ts, noise=0.0, seed=0, **kw):
    rng = np.random.default_rng(seed)
    f = SVFilter(ts[0], s_of_t(ts[0]), meas_sigma=kw.get("meas_sigma", 0.01), accel_sigma=kw.get("accel_sigma", 0.4))
    out = []
    for t in ts[1:]:
        f.update(t, s_of_t(t) + rng.normal(0, noise))
        out.append((t, f.s, f.v))
    return f, out


def test_constant_speed_is_learned():
    # a car at 0.25 s-per-second (crosses the zone in 4 s), measured with a little noise
    ts = np.arange(0, 3, 0.1)
    f, out = run_filter(lambda t: -0.2 + 0.25 * t, ts, noise=0.005)
    assert f.v == pytest.approx(0.25, abs=0.02)
    assert f.s == pytest.approx(-0.2 + 0.25 * ts[-1], abs=0.01)
    # it also says how sure it is: the speed uncertainty shrank from the initial 0.5
    assert math.sqrt(f.P[1, 1]) < 0.2


def test_braking_is_followed():
    # 0.3 /s, then brakes hard to 0.1 /s at t = 1.5 s
    def s_of_t(t):
        return 0.3 * t if t < 1.5 else 0.45 + 0.1 * (t - 1.5)
    ts = np.arange(0, 3.5, 0.1)
    _, out = run_filter(s_of_t, ts, accel_sigma=1.0)
    v_end = out[-1][2]
    assert v_end == pytest.approx(0.1, abs=0.03)
    # one second after braking the estimate has mostly caught up
    v_after_1s = [v for t, _, v in out if abs(t - 2.5) < 1e-6][0]
    assert v_after_1s < 0.17


def test_prediction_during_a_dropout_keeps_constant_speed():
    ts = np.arange(0, 2, 0.1)
    f, _ = run_filter(lambda t: 0.2 * t, ts)
    s, v, var_s, _ = f.predict(ts[-1] + 0.5)          # half a second without detections
    assert s == pytest.approx(0.2 * (ts[-1] + 0.5), abs=0.01)
    assert v == pytest.approx(0.2, abs=0.01)
    assert var_s > f.P[0, 0]                          # less sure than right after the last detection


def test_old_measurements_are_ignored():
    f = SVFilter(1.0, 0.0, 0.01, 0.4)
    assert f.update(0.5, 0.3) is False
    assert f.s == 0.0


# ------------------------------------------------------------------------------------------------ the track manager
def box_at(x, y=500, w=80, h=50):
    """A box whose bottom centre (the ground point) is at (x, y)."""
    return (x - w / 2, y - h, x + w / 2, y)


def tracker():
    path = PathModel(straight_zone())              # x 100..1000, s = (x - 200) / 800
    return ProgressTracker(path, ProgressSettings(finish_after_s=1.0, profile_min_span=0.3, profile_min_updates=5))


def drive(pt, track_id, x0, px_per_s, t0, t1, dt=0.1, skip=lambda t: False, y=500):
    views = {}
    t = t0
    while t <= t1 + 1e-9:
        out = [] if skip(t) else [(track_id, box_at(x0 + px_per_s * (t - t0), y), 0.9)]
        views = pt.update(t, out)
        t = round(t + dt, 6)
    return views


def test_tracks_get_s_and_v():
    pt = tracker()
    # 200 px/s on an 800 px zone = 0.25 /s
    views = drive(pt, 7, 120, 200, 0.0, 2.0)
    v = views[7]
    assert v.s == pytest.approx((120 + 400 - 200) / 800, abs=0.01)
    assert v.v == pytest.approx(0.25, abs=0.02)
    assert v.n_updates == 21 and not v.predicted


def test_detection_outside_the_corridor_does_not_update():
    pt = tracker()
    drive(pt, 3, 300, 200, 0.0, 1.0)
    before = pt.tracks[3].n_updates
    views = pt.update(1.1, [(3, box_at(520, y=700), 0.9)])  # 200 px below the path: off the racing line
    assert pt.tracks[3].n_updates == before
    assert views[3].off_corridor_frames == 1
    assert views[3].predicted                              # s/v come from the prediction, not the bad point


def test_a_track_that_never_enters_the_corridor_is_ignored():
    pt = tracker()
    views = pt.update(0.0, [(9, box_at(500, y=900), 0.9)])
    assert views == {}


def test_dropout_then_recovery():
    pt = tracker()
    views = drive(pt, 1, 200, 200, 0.0, 2.0, skip=lambda t: 0.8 < t < 1.3)
    assert views[1].v == pytest.approx(0.25, abs=0.03)
    assert views[1].n_updates == 21 - 4


def test_finished_tracks_teach_the_speed_profile():
    pt = tracker()
    finished = []
    pt.on_finished = finished.append
    drive(pt, 1, 150, 200, 0.0, 4.5)               # x 150 -> 1050: s from -0.06 to 1.06
    pt.update(10.0, [])                            # long after: the track is finished
    assert len(finished) == 1 and pt.profile.n_passes == 1
    assert pt.profile.v_typ(0.5) == pytest.approx(0.25, abs=0.03)


def test_short_tracks_do_not_teach_the_profile():
    pt = tracker()
    drive(pt, 1, 500, 200, 0.0, 0.5)               # only 0.125 of s
    pt.update(10.0, [])
    assert pt.profile.n_passes == 0


def test_prediction_rides_the_learned_profile():
    # profile: fast (0.4) before s = 0.5, slow (0.1) after it (cars brake there)
    path = PathModel(straight_zone())
    prof = SpeedProfile(-0.5, 1.5, 0.05, default_v=0.25)
    prof.add_pass([(s, 0.4 if s < 0.5 else 0.1) for s in np.arange(-0.4, 1.4, 0.01)])
    pt = ProgressTracker(path, ProgressSettings(), prof)
    drive(pt, 1, 360, 320, 0.0, 0.5)               # 320 px/s = 0.4 /s, ends near s = 0.4
    tp = pt.tracks[1]
    s_pred, v_pred, _, _ = pt.predict_s_v(tp, 1.5)  # one second with no detection
    # constant speed would put it at ~0.8; riding the profile it slows down past 0.5
    assert v_pred == pytest.approx(0.1, abs=0.03)
    assert 0.5 < s_pred < 0.65


# ------------------------------------------------------------------------------------------------ speed profile
def test_profile_is_a_robust_median():
    p = SpeedProfile(0.0, 1.0, 0.1)
    for v in (0.2, 0.21, 0.19, 0.2, 5.0):           # one crazy pass (a bad track)
        p.add_pass([(0.55, v)])
    assert p.v_typ(0.55) == pytest.approx(0.2)


def test_profile_default_and_nearest_bin():
    p = SpeedProfile(0.0, 1.0, 0.1, default_v=0.3)
    assert p.v_typ(0.5) == 0.3                     # nothing learned yet
    p.add_pass([(0.15, 0.2)])
    assert p.v_typ(0.9) == pytest.approx(0.2)      # empty bins borrow from the nearest learned one


def test_profile_keeps_only_the_last_passes():
    p = SpeedProfile(0.0, 1.0, 0.5, max_passes=3)
    for v in (1, 1, 1, 2, 2, 2):
        p.add_pass([(0.2, v)])
    assert p.v_typ(0.2) == pytest.approx(2)


def test_profile_save_and_load(tmp_path):
    p = SpeedProfile(-0.2, 1.2, 0.1)
    p.add_pass([(s, 0.2 + s / 10) for s in np.arange(-0.1, 1.1, 0.05)])
    f = profile_path_for(tmp_path / "zone.json")
    assert f.name == "zone.speed.json"
    p.save(f)
    q = SpeedProfile.load(f)
    assert q.n_passes == 1
    for s in (0.0, 0.33, 0.9):
        assert q.v_typ(s) == pytest.approx(p.v_typ(s))


@pytest.mark.parametrize("sigmas, found", [(3.0, True), (0.0, False)])
def test_a_car_that_braked_while_unseen_is_still_found(sigmas, found):
    # The car drives at 200 px/s, then disappears for 2 s (hidden by a tree, say) and brakes while hidden: it comes
    # back at x = 420, only 20 px further than where it was last seen. The filter, knowing nothing better, predicted
    # it ~400 px further on (x ~ 800). The search window is centred on that prediction and normally reaches only
    # 40 px back: the real position would be out of reach, and the car's s would be stuck at the window's edge.
    # With the window widened by the prediction's uncertainty (3 standard deviations, the default), it is found.
    path = PathModel(straight_zone())                       # x 100..1000, s = (x - 200) / 800
    pt = ProgressTracker(path, ProgressSettings(finish_after_s=5.0, window_sigmas=sigmas))
    drive(pt, 4, 200, 200, 0.0, 1.0)                        # seen until x = 400 at t = 1.0
    views = pt.update(3.0, [(4, box_at(420), 0.9)])          # back at t = 3.0, at x = 420
    s_raw = pt.tracks[4].samples[-1][1]                     # the projected s of that detection
    true_s = (420 - 200) / 800
    assert (abs(s_raw - true_s) < 0.02) is found
    assert 4 in views
