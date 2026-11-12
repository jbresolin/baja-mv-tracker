"""Tests for zone/speedmap.py: turning a wanted rate into a PTZ speed step without chatter.

The table used here: step k moves at 10 * k units/s (step 1 = 10 u/s ... step 10 = 100 u/s)."""
import pytest

from zone.speedmap import AxisSpeedMap

TABLE = {k: 10.0 * k for k in range(1, 11)}


def smap(**kw):
    kw.setdefault("max_step_change", 10)    # no ramp unless a test wants one
    return AxisSpeedMap(TABLE, **kw)


def test_rate_of_and_inverse():
    m = smap()
    assert m.rate_of(3) == 30 and m.rate_of(-3) == -30 and m.rate_of(0) == 0
    assert m.ideal_step(45) == pytest.approx(4.5)
    assert m.ideal_step(-45) == pytest.approx(-4.5)
    assert m.ideal_step(1000) == pytest.approx(10)     # clamped to the fastest step


def test_nonmonotone_measured_table_is_made_monotone():
    m = AxisSpeedMap({1: 10, 2: 25, 3: 24, 4: 40})     # step 3 measured a bit slower than step 2
    assert m.rate_of(3) == 25
    assert 1.999 <= m.ideal_step(25) <= 3        # any step that gives 25 u/s is fine


def test_asymmetric_camera_uses_the_negative_table():
    m = AxisSpeedMap(TABLE, negative_table={k: 20.0 * k for k in range(1, 11)})
    assert m.rate_of(2) == 20 and m.rate_of(-2) == -40


def test_picks_nearest_step_from_rest():
    m = smap()
    assert m.next_step(41, 0) == 4
    assert m.next_step(-68, 0) == -7


def test_hysteresis_keeps_the_current_step():
    m = smap(hysteresis=0.3)
    # currently at step 4 (40 u/s). Wanting 47 u/s (ideal 4.7) is within 0.5 + 0.3 of 4: keep 4
    assert m.next_step(47, 4) == 4
    # wanting 49 u/s (ideal 4.9) is 0.9 away: switch to 5
    assert m.next_step(49, 4) == 5
    # without hysteresis the same 47 would round to 5
    assert smap(hysteresis=0.0).next_step(47, 4) == 5


def test_rate_wobbling_between_steps_does_not_chatter():
    m = smap(hysteresis=0.3)
    step, seen = 4, []
    for rate in [44, 46, 44, 46, 44, 46, 45, 44]:
        step = m.next_step(rate, step)
        seen.append(step)
    assert set(seen) == {4}


def test_ramp_limits_the_change_per_cycle():
    m = smap(max_step_change=2)
    steps, s = [], 2
    for _ in range(5):
        s = m.next_step(100, s)      # wants step 10
        steps.append(s)
    assert steps == [4, 6, 8, 10, 10]


def test_reversal_ramps_down_to_a_stop_first():
    m = smap(max_step_change=3)
    s, seq = 7, []
    for _ in range(5):
        s = m.next_step(-50, s)
        seq.append(s)
    assert seq[:3] == [4, 1, 0]      # down through the same direction, stop...
    assert seq[3] < 0                # ...then go the other way


def test_minimum_step_rule_hold_or_stop_whichever_error_is_smaller():
    m = smap(min_smooth_step=3, start_margin=0.0)    # slowest smooth step = 3 = 30 u/s; half of it = 15 u/s
    assert m.next_step(20, 0) == 3       # 20 u/s: holding 30 is 10 off, stopping is 20 off -> hold the minimum
    assert m.next_step(10, 0) == 0       # 10 u/s: stopping is 10 off, holding 30 is 20 off -> stop
    assert m.next_step(50, 0) == 5       # normal speeds are unaffected


def test_never_alternates_stop_and_start_around_the_threshold():
    # the wanted rate hovers around half the minimum rate (15 u/s): with the margin band the camera picks one
    # state and stays in it, instead of stopping and starting every cycle
    m = smap(min_smooth_step=3, start_margin=0.15)
    s, seq = 0, []
    for rate in [14, 16, 14, 16, 15, 14, 16, 18, 12, 14]:
        s = m.next_step(rate, s)
        seq.append(s)
    changes = sum(1 for a, b in zip(seq, seq[1:]) if (a == 0) != (b == 0))
    assert changes <= 1
    # once moving, it keeps the minimum step down to (0.5 - 0.15) * 30 = 10.5 u/s
    assert m.next_step(11, 3) == 3
    assert m.next_step(10, 3) == 0
    # and from rest it only starts above (0.5 + 0.15) * 30 = 19.5 u/s
    assert m.next_step(19, 0) == 0
    assert m.next_step(20, 0) == 3


def test_starting_jumps_straight_to_the_minimum_step():
    m = smap(min_smooth_step=3, max_step_change=1)
    assert m.next_step(100, 0) == 3      # anything below 3 wouldn't move smoothly, even if the ramp says 1
