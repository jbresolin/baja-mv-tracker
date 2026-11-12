"""Number crunching for the bench script: rates, latencies, smoothness, the verdict. Pure functions, no camera.

A "sample" is `(t, value)`: the time (seconds, monotonic) at which the camera reported raw position `value`. The
bench script uses the middle of the inquiry's round trip as `t`, the best guess of when the camera read its encoder.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Sequence

Sample = tuple[float, float]

BRANCH_A_MIN_HZ = 10.0  # the brief: inquiry while moving at >= 10 Hz gives Branch A


def percentile(values: Sequence[float], q: float) -> float | None:
    """The q-th percentile (0..100) by linear interpolation, or None for no values."""
    data = sorted(values)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * q / 100
    low = math.floor(position)
    high = min(low + 1, len(data) - 1)
    return data[low] + (data[high] - data[low]) * (position - low)


def fit_rate(samples: Sequence[Sample]) -> float | None:
    """Least-squares slope of value against time (units/s), or None with fewer than 3 samples or no time spread."""
    if len(samples) < 3:
        return None
    ts = [t for t, _ in samples]
    vs = [v for _, v in samples]
    t_mean = statistics.fmean(ts)
    v_mean = statistics.fmean(vs)
    spread = sum((t - t_mean) ** 2 for t in ts)
    if spread <= 0:
        return None
    return sum((t - t_mean) * (v - v_mean) for t, v in zip(ts, vs)) / spread


def interval_rates(samples: Sequence[Sample]) -> list[float]:
    """Rate between each pair of consecutive samples (units/s)."""
    rates = []
    for (t0, v0), (t1, v1) in zip(samples, samples[1:]):
        if t1 > t0:
            rates.append((v1 - v0) / (t1 - t0))
    return rates


def smoothness_cv(samples: Sequence[Sample]) -> float | None:
    """How uneven the motion is: standard deviation / mean of the interval rates (0 = perfectly even).

    A camera that moves in steps, or whose position report freezes and then jumps, gives a large value.
    """
    rates = interval_rates(samples)
    if len(rates) < 3:
        return None
    mean = statistics.fmean(rates)
    if abs(mean) < 1e-9:
        return None
    return statistics.pstdev(rates) / abs(mean)


def count_jumps(samples: Sequence[Sample], factor: float = 3.0) -> int:
    """How many interval rates are more than `factor` x the median absolute rate (a freeze followed by a jump)."""
    rates = [abs(r) for r in interval_rates(samples)]
    if len(rates) < 3:
        return 0
    median = statistics.median(rates)
    if median <= 0:
        return sum(1 for r in rates if r > 0)
    return sum(1 for r in rates if r > factor * median)


def first_change(samples: Sequence[Sample], baseline: float, after: float, threshold: float = 1.0) -> float | None:
    """Time of the first sample after `after` that differs from `baseline` by at least `threshold`."""
    for t, v in samples:
        if t >= after and abs(v - baseline) >= threshold:
            return t
    return None


def last_change(samples: Sequence[Sample], after: float) -> float | None:
    """Time of the last sample after `after` whose value differs from the sample before it."""
    result = None
    for (_, v0), (t1, v1) in zip(samples, samples[1:]):
        if t1 >= after and v1 != v0:
            result = t1
    return result


def time_to_reach_rate(samples: Sequence[Sample], start: float, steady: float, fraction: float = 0.9) -> float | None:
    """First time after `start` at which the rate over 3 consecutive samples reaches `fraction` x `steady`."""
    later = [s for s in samples if s[0] >= start]
    for i in range(len(later) - 2):
        window = later[i:i + 3]
        rate = (window[-1][1] - window[0][1]) / (window[-1][0] - window[0][0]) if window[-1][0] > window[0][0] else 0
        if steady != 0 and rate / steady >= fraction:
            return window[-1][0]
    return None


def time_rate_drops(samples: Sequence[Sample], start: float, steady: float, fraction: float = 0.9) -> float | None:
    """First time after `start` at which the rate over 2 consecutive samples falls below `fraction` x `steady`."""
    later = [s for s in samples if s[0] >= start]
    for (t0, v0), (t1, v1) in zip(later, later[1:]):
        if t1 > t0 and steady != 0 and ((v1 - v0) / (t1 - t0)) / steady < fraction:
            return t1
    return None


def edge_time(polls: Sequence[Any], after: float, answered_before: bool) -> float | None:
    """When the answers to a series of polls switch between answered and unanswered, after time `after`.

    `polls` have `.sent`, `.done` and `.value` (None = no answer). With `answered_before=True`, it finds the first
    unanswered poll after `after` that follows an answered one ("went silent"). With False, the first answered
    poll after an unanswered one ("answers again"). The edge is placed halfway between the two polls.

    Each poll is placed where the camera most likely handled it: an answered poll at the middle of its round trip,
    an unanswered one at its send time (its "round trip" is just the driver's timeout, which says nothing about
    when the camera saw it).
    """
    def seen_at(p: Any) -> float:
        return (p.sent + p.done) / 2 if p.value is not None else p.sent

    later = [p for p in polls if p.sent >= after]
    for previous, current in zip(later, later[1:]):
        was = previous.value is not None
        now = current.value is not None
        if was == answered_before and now != answered_before:
            return (seen_at(previous) + seen_at(current)) / 2
    return None


def classify_feedback(answered_fraction: float, changed: bool) -> str:
    """While-moving behaviour of the position inquiry: "yes", "frozen" (answers, never changes) or "no"."""
    if answered_fraction < 0.2:
        return "no"
    return "yes" if changed else "frozen"


def verdict(while_moving: str | None, max_rate_hz: float | None) -> tuple[str, str]:
    """("A" or "B", one-line reason), following the brief's rule."""
    if while_moving == "yes" and max_rate_hz is not None and max_rate_hz >= BRANCH_A_MIN_HZ:
        return "A", (f"pan/tilt inquiry works while moving at {max_rate_hz:g} Hz (>= {BRANCH_A_MIN_HZ:g} Hz), "
                     "so use feed-forward + feedback")
    if while_moving == "yes":
        why = f"pan/tilt inquiry works while moving, but only up to {max_rate_hz or 0:g} Hz"
    elif while_moving == "frozen":
        why = "pan/tilt inquiry answers while moving, but the value doesn't change until the camera stops"
    elif while_moving == "no":
        why = "pan/tilt inquiry doesn't answer while moving"
    else:
        why = "pan/tilt inquiry doesn't work"
    return "B", why + ", so use feed-forward from the speed table and reset to the entry preset after each car"
