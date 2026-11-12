"""Tests for zone/path.py: progress s along the taught path, projection of a car onto it, and the pose curve.

Reminder: s = 0 at the first keyframe, s = 1 at the last keyframe, negative before (the approach)."""
import math

import pytest

from zone.path import PathModel, Pose, PoseCurve
from zone.tests.z2_fixtures import straight_zone
from zone.zonefile import Keyframe, PathPoint, Zone


def test_s_is_zero_at_first_keyframe_and_one_at_last():
    # straight path: x from 100 to 1000; first keyframe at x=200, last at x=1000
    path = PathModel(straight_zone())
    assert path.project(200, 500).s == pytest.approx(0.0)
    assert path.project(1000, 500).s == pytest.approx(1.0)
    assert path.project(600, 500).s == pytest.approx(0.5)


def test_upstream_points_get_negative_s():
    # the path starts 100 px before the first keyframe, and the zone is 800 px long -> s_min = -100 / 800
    path = PathModel(straight_zone())
    assert path.s_min == pytest.approx(-0.125)
    assert path.project(150, 500).s == pytest.approx(-0.0625)


def test_lateral_offset_and_corridor():
    path = PathModel(straight_zone(half_width=60))
    near = path.project(500, 530)      # 30 px below the path
    far = path.project(500, 600)       # 100 px below the path
    assert near.offset_px == pytest.approx(30)
    assert path.in_corridor(near)
    assert not path.in_corridor(far)
    # the sign tells the side: image y grows downwards, so "below" is to the right when travelling to the right
    assert near.lateral_px < 0 < path.project(500, 470).lateral_px


def test_needs_two_keyframes():
    z = straight_zone()
    for p in z.points[2:]:
        p.keyframe = None             # only one keyframe left
    assert PathModel.build(z) is None
    with pytest.raises(ValueError):
        PathModel(z)


def hairpin_zone():
    """A hairpin seen from the side: out along y=500 to the right, back along y=560 to the left. The two legs are only
    60 px apart, so a car on the return leg is also close to the outward leg."""
    pts = [PathPoint(100, 500, Keyframe(0, 0, 0, 10)), PathPoint(900, 500), PathPoint(900, 560),
           PathPoint(100, 560, Keyframe(100, 0, 0, 11))]
    return Zone(points=pts, corridor_half_width_px=40)


def test_path_passing_near_itself_does_not_jump_with_a_monotonic_window():
    path = PathModel(hairpin_zone())
    # A car really on the RETURN leg (y=560), but its ground point came out a bit high: y=528. That's 28 px from the
    # outward leg (y=500) and 32 px from the return leg, so the closest point overall is on the WRONG leg.
    x, y = 500, 528
    glob = path.project(x, y)                       # no history: the whole path is searched
    assert glob.s < 0.5                             # -> it lands on the outward leg (first half of the path)
    # but we know the car was on the return leg a moment ago: only search near there (40 px back, 400 ahead)
    s_prev = path.project(520, 560).s
    windowed = path.project(x, y, s_prev=s_prev, back_px=40, ahead_px=400)
    assert windowed.s > 0.5                         # -> it stays on the return leg
    assert windowed.s == pytest.approx(path.project(500, 560).s, abs=0.01)


def test_point_at_inverts_projection():
    path = PathModel(straight_zone())
    for s in (-0.1, 0.0, 0.3, 1.0):
        x, y = path.point_at(s)
        assert path.project(x, y).s == pytest.approx(s, abs=1e-9)


def test_corridor_bbox_is_clipped_to_the_image():
    path = PathModel(straight_zone(half_width=60))
    x0, y0, x1, y1 = path.corridor_bbox(margin_px=20, image_size=(1920, 1080))
    assert (x0, y0, x1, y1) == (20, 420, 1080, 580)
    assert path.corridor_bbox(margin_px=200, image_size=(1000, 1000))[0] == 0


# ------------------------------------------------------------------------------------------------ pose curve
def curve(kind):
    s = [0.0, 0.3, 0.6, 1.0]
    poses = [Pose(0, 0, 1000), Pose(300, 50, 2000), Pose(400, 60, 2500), Pose(1000, 0, 4000)]
    return PoseCurve(s, poses, kind)


@pytest.mark.parametrize("kind", ["pchip", "linear"])
def test_pose_curve_passes_through_keyframes(kind):
    c = curve(kind)
    assert c.pose(0.3).pan == pytest.approx(300)
    assert c.pose(0.6).tilt == pytest.approx(60)
    assert c.pose(1.0).zoom == pytest.approx(4000)


@pytest.mark.parametrize("kind", ["pchip", "linear"])
def test_pose_is_held_outside_zero_one(kind):
    # before the entry keyframe the camera should sit at the entry pose; after the exit, at the exit pose
    c = curve(kind)
    assert c.pose(-0.5).pan == pytest.approx(0)
    assert c.pose(1.7).pan == pytest.approx(1000)
    assert c.dpose_ds(-0.5).pan == 0.0
    assert c.dpose_ds(1.7).pan == 0.0


@pytest.mark.parametrize("kind", ["pchip", "linear"])
def test_derivative_matches_finite_differences(kind):
    c = curve(kind)
    h = 1e-6
    for s in (0.05, 0.2, 0.45, 0.8, 0.95):
        numeric = (c.pose(s + h).pan - c.pose(s - h).pan) / (2 * h)
        assert c.dpose_ds(s).pan == pytest.approx(numeric, rel=1e-4, abs=1e-3)


def test_pchip_never_overshoots_between_keyframes():
    # tilt goes 0 -> 50 -> 60 -> 0: a plain cubic spline would bulge above 60; pchip must stay within the keyframes
    c = curve("pchip")
    vals = [c.pose(i / 1000).tilt for i in range(1001)]
    assert max(vals) <= 60 + 1e-9
    assert min(vals) >= 0 - 1e-9
    # and between two keyframes with increasing pan, pan keeps increasing (monotone)
    pans = [c.pose(i / 1000).pan for i in range(1001)]
    assert all(b >= a - 1e-9 for a, b in zip(pans, pans[1:]))


def test_pose_curve_rejects_keyframes_out_of_order():
    with pytest.raises(ValueError):
        PoseCurve([0.0, 0.5, 0.5], [Pose(0, 0, 0)] * 3)


def test_pose_curve_from_zone_uses_keyframe_positions():
    z = straight_zone(n_keyframes=3)
    c = PoseCurve.from_zone(z, PathModel(z), "linear")
    assert c.pose(0.5).pan == pytest.approx(100)      # middle keyframe has pan 100
    assert c.dpose_ds(0.25).pan == pytest.approx(200)  # 100 units of pan per 0.5 of s
    assert not math.isnan(c.pose(0.25).zoom)


# ------------------------------------------------------------------------------------------------ ease-in
def test_ease_in_starts_flat_before_the_entry_and_joins_the_curve():
    # pan 0 at s = 0 rising to 300 at s = 0.3: slope 1000 per unit of s. With ease_in = 0.1 the camera starts
    # turning at s = -0.1 (zero slope there) and is back on the taught curve at s = +0.1.
    plain, eased = curve("linear"), PoseCurve([0.0, 0.3, 0.6, 1.0], [Pose(0, 0, 1000), Pose(300, 50, 2000),
                                                                         Pose(400, 60, 2500), Pose(1000, 0, 4000)],
                                                 "linear", ease_in=0.1)
    assert eased.pose(-0.2).pan == pytest.approx(0)            # still parked at the entry
    assert eased.pose(-0.1).pan == pytest.approx(0)
    assert eased.dpose_ds(-0.1).pan == pytest.approx(0)        # starts gently
    assert eased.pose(0.0).pan > 0                             # already moving when the car reaches the entry
    assert eased.pose(0.1).pan == pytest.approx(plain.pose(0.1).pan)      # joined the taught curve
    assert eased.dpose_ds(0.1).pan == pytest.approx(plain.dpose_ds(0.1).pan)
    assert eased.pose(0.5).pan == pytest.approx(plain.pose(0.5).pan)      # untouched further on
    # no jumps anywhere
    vals = [eased.pose(-0.2 + i / 1000).pan for i in range(500)]
    assert max(abs(b - a) for a, b in zip(vals, vals[1:])) < 2.0


def test_config_moves_s_lock_before_the_ease_in():
    from zone.tests.z2_fixtures import settings
    assert settings(sets=["zone.ease_in=0.1", "director.s_lock=0"]).director.s_lock == pytest.approx(-0.1)
    assert settings(sets=["zone.ease_in=0", "director.s_lock=0"]).director.s_lock == 0
