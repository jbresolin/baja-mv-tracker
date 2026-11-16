"""Tests for zone/keyframe_check.py: the sanity check that flags keyframes whose stored pose looks wrong.

For a first-time reader:
* A zone is a list of points clicked along the track, in the order a car drives past. Some points are keyframes:
  they also store where the PTZ pointed (pan, tilt, zoom, in the camera's raw units) and a preset number.
* The check never blocks anything. It only returns, for each suspicious keyframe, the reasons (in Portuguese, for
  the page), so the operator can re-teach it.
* `zone(...)` below builds a zone from a short list of (pan, tilt, zoom) tuples, presets numbered from 10.
"""
from zone.keyframe_check import check_keyframes, longest_one_way_run, summary
from zone.zonefile import Keyframe, PathPoint


def zone(*poses, path_points_between=False):
    """Points with one keyframe per pose (presets 10, 11, ...). path_points_between: put a plain path point (no
    keyframe) between keyframes, as real zones have, to check the point numbering."""
    points = []
    for n, (pan, tilt, zoom) in enumerate(poses):
        if path_points_between and n:
            points.append(PathPoint(n * 100 - 50, 300))
        points.append(PathPoint(n * 100, 300, Keyframe(pan, tilt, zoom, 10 + n)))
    return points


def kinds(warnings, point_index):
    return sorted(w["kind"] for w in warnings.get(point_index, []))


# The four keyframes of the 2026-11-15 bench, in path order. P11 is the bad one.
BENCH = [(1199, 136, 8724), (1764, -4, 0), (1448, 176, 6772), (1636, 100, 9399)]


def test_the_bench_keyframe_p11_is_flagged_for_all_three_reasons():
    points = zone(*BENCH)
    warnings = check_keyframes(points)
    assert kinds(warnings, 1) == ["jump", "order_pan", "zoom_zero"]   # P11: out of order, zoom 0, big jump
    assert 0 not in warnings and 3 not in warnings                    # P10 and P13 are fine
    # P12's tilt (176 between 136 and 100) goes back: a soft warning, as a track can climb then dip
    assert kinds(warnings, 2) == ["order_tilt"]
    assert "pode ser normal" in warnings[2][0]["text"]
    lines = summary(points, warnings)
    assert lines[0].startswith("P11 (ponto 1): pan fora de ordem")


def test_a_clean_zone_has_no_warnings():
    # pan grows steadily, tilt grows, zoom changes smoothly: nothing to say
    points = zone((1199, 136, 8724), (1320, 150, 8000), (1448, 160, 6772), (1636, 170, 9399))
    assert check_keyframes(points) == {}


def test_a_zone_taught_right_to_left_is_fine_too():
    # the order rule doesn't care which way, only that it is one way
    points = zone((1636, 100, 9000), (1448, 90, 8000), (1300, 80, 7000), (1199, 70, 6000))
    assert check_keyframes(points) == {}


def test_tiny_steps_back_are_reading_noise_not_a_warning():
    # pan 1000, 1203, 1200 (3 units back), 1400: within the tolerance
    points = zone((1000, 0, 5000), (1203, 0, 5000), (1200, 0, 5000), (1400, 0, 5000))
    assert check_keyframes(points) == {}


def test_point_indices_count_path_points_too():
    # the warning is attached to the index of the point in the whole list (the page's row number)
    points = zone(*BENCH, path_points_between=True)
    warnings = check_keyframes(points)
    assert "order_pan" in kinds(warnings, 2)          # P11 is point 2 (path point 1 sits between P10 and P11)


def test_zoom_zero_is_fine_when_a_neighbour_is_also_wide():
    # a zone that really starts fully wide: zoom 0 then 300 (near 0 too) is not suspicious
    points = zone((1000, 0, 0), (1100, 0, 300), (1200, 0, 5000))
    assert "zoom_zero" not in kinds(check_keyframes(points), 0)


def test_with_two_keyframes_only_the_zoom_rule_applies():
    # with two keyframes there is no "order" to break and the only step is always 100 % of the range
    points = zone((1000, 0, 8000), (2000, 0, 0))
    assert kinds(check_keyframes(points), 1) == ["zoom_zero"]


def test_a_confirmed_keyframe_has_no_warnings():
    # the operator tested P12 and the track really does dip: "conferido" hides its warnings
    points = zone(*BENCH)
    points[2].keyframe.confirmed = True
    warnings = check_keyframes(points)
    assert 2 not in warnings and 1 in warnings        # P11 is still flagged


def test_longest_one_way_run():
    assert longest_one_way_run([1199, 1764, 1448, 1636]) == (1, [0, 2, 3])
    assert longest_one_way_run([100, 80, 60, 90, 40]) == (-1, [0, 1, 2, 4])   # 90 goes back up
    assert longest_one_way_run([]) == (1, [])
