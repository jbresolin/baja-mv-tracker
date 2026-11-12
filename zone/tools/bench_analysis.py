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
# The position must also *change* at >= 10 Hz. Counting changes over a 1.5 s window is exact only to about one
# change, so 5% below the line still counts (a live camera polled at exactly 10 Hz measures 9.6-10.0 Hz).
UPDATE_RATE_TOLERANCE = 0.95

# Rules for "this poll rate gives live feedback" (see `judge_feedback_rate`).
MIN_ANSWERED_FRACTION = 0.9  # at least 90% of the inquiries answered...
MIN_ANSWER_RATE_FRACTION = 0.8  # ...arriving at >= 80% of the asked rate...
MIN_CHANGE_FRACTION = 0.7  # ...and the value changes on >= 70% of the answers where it should have
SMOOTH_CV_LIMIT = 0.35  # interval-rate std/mean below this counts as smooth
QUANTIZATION_UNITS = 1.0  # positions are whole raw units: a perfectly smooth move is off a straight line by <= 0.5
JUMP_FACTOR = 3.0  # an interval that moved > 3x what the fitted speed predicts (plus quantization) is a jump


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


def fit_residual_max(samples: Sequence[Sample]) -> float | None:
    """The largest distance (raw units) of any sample from the least-squares straight line through them all.

    Why: positions are whole numbers. At a slow speed a perfectly smooth move reads 0, 0, 1, 1, 1, 2... so the
    interval rates swing between 0 and "a lot" and `smoothness_cv` calls it jerky. Against a straight line, though,
    those values are never more than half a unit off (plus a little timing noise). A camera that really moves in
    steps, or whose report freezes and then jumps, is many units off the line.
    """
    slope = fit_rate(samples)
    if slope is None:
        return None
    t_mean = statistics.fmean(t for t, _ in samples)
    v_mean = statistics.fmean(v for _, v in samples)
    return max(abs(v - (v_mean + slope * (t - t_mean))) for t, v in samples)


def is_smooth(samples: Sequence[Sample], cv_limit: float = SMOOTH_CV_LIMIT,
              quantization: float = QUANTIZATION_UNITS) -> bool | None:
    """True if the motion is smooth at this sampling: either the interval rates are even (cv below `cv_limit`), or
    every sample is within `quantization` raw units of a straight line (only rounding to whole units is left).
    None if there are too few samples to tell."""
    cv = smoothness_cv(samples)
    residual = fit_residual_max(samples)
    if cv is None and residual is None:
        return None
    return (cv is not None and cv <= cv_limit) or (residual is not None and residual <= quantization)


def value_changes(samples: Sequence[Sample]) -> int:
    """How many times the value differs from the sample before it."""
    return sum(1 for (_, v0), (_, v1) in zip(samples, samples[1:]) if v1 != v0)


def update_rate_hz(samples: Sequence[Sample]) -> float:
    """How often the reported value actually changes (changes per second over the samples' time span).

    This is the camera's real refresh rate as seen from here, which can be far below the rate it *answers* at:
    a camera answering 50 times a second from a position cache refreshed twice a second gives ~2 Hz.
    """
    if len(samples) < 2 or samples[-1][0] <= samples[0][0]:
        return 0.0
    change_times = [t1 for (_, v0), (t1, v1) in zip(samples, samples[1:]) if v1 != v0]
    if len(change_times) >= 2 and change_times[-1] > change_times[0]:
        # Count from the first change to the last: N changes span N-1 refresh periods. Counting over the whole
        # window instead would depend on where the window happens to start and end relative to the refreshes.
        return (len(change_times) - 1) / (change_times[-1] - change_times[0])
    return len(change_times) / (samples[-1][0] - samples[0][0])


def expected_change_fraction(samples: Sequence[Sample]) -> float | None:
    """The fraction of consecutive answers that *should* differ if the camera reports live whole-unit positions.

    If the camera moves 0.5 units between two polls, a live report changes only about every other poll, so a
    change fraction of 0.5 is perfect there. From one unit per poll upwards, every answer should differ.
    """
    slope = fit_rate(samples)
    if slope is None:
        return None
    gaps = [t1 - t0 for (t0, _), (t1, _) in zip(samples, samples[1:])]
    return min(1.0, abs(slope) * statistics.fmean(gaps) / QUANTIZATION_UNITS)


def count_jumps_vs_fit(samples: Sequence[Sample], factor: float = JUMP_FACTOR,
                       quantization: float = QUANTIZATION_UNITS) -> int:
    """How many intervals moved far more than the fitted speed predicts for their length.

    A stale-cache camera repeats a value and then leaps to catch up: that leap is a jump. Unlike `count_jumps`,
    this doesn't compare against the median interval (which is 0 for a slow move or a stale cache), so a slow but
    live camera doesn't count every +1 as a jump.
    """
    slope = fit_rate(samples)
    if slope is None:
        return 0
    jumps = 0
    for (t0, v0), (t1, v1) in zip(samples, samples[1:]):
        predicted = abs(slope) * (t1 - t0)
        if abs(v1 - v0) > factor * predicted + quantization:
            jumps += 1
    return jumps


def judge_feedback_rate(asked_hz: float, sent: int, samples: Sequence[Sample]) -> dict[str, Any]:
    """Decide whether polling at `asked_hz` during a constant-speed move gives *live* position feedback.

    `sent` is how many inquiries went out, `samples` the (t, value) of those answered. Passing needs all of:
      - answers: >= 90% answered, arriving at >= 80% of the asked rate;
      - updates: the value changes on >= 70% of the answers where a live whole-unit report would (a camera that
        answers fast from a slowly refreshed cache fails here);
      - smooth: even interval rates, or within quantization of a straight line (`is_smooth`);
      - no jumps (`count_jumps_vs_fit`).
    Returns all the numbers, `ok`, and `why`: "passed", or the list of failed rules in words.
    """
    n = len(samples)
    span = samples[-1][0] - samples[0][0] if n > 1 else 0.0
    answers_hz = (n - 1) / span if span > 0 else 0.0
    answered_fraction = n / sent if sent else 0.0
    changes = value_changes(samples)
    change_fraction = changes / (n - 1) if n > 1 else 0.0
    expected = expected_change_fraction(samples)
    smooth = is_smooth(samples)
    jumps = count_jumps_vs_fit(samples)
    failed = []
    if answered_fraction < MIN_ANSWERED_FRACTION:
        failed.append(f"only {answered_fraction:.0%} of inquiries answered")
    if answers_hz < MIN_ANSWER_RATE_FRACTION * asked_hz:
        failed.append(f"answers at {answers_hz:.1f} Hz, below {MIN_ANSWER_RATE_FRACTION:.0%} of {asked_hz:g} Hz")
    if changes == 0:
        failed.append("value never changed")
    elif not expected or change_fraction < MIN_CHANGE_FRACTION * expected:
        failed.append(f"value changed on only {change_fraction:.0%} of answers (stale position cache?)")
    if smooth is False:
        failed.append("not smooth")
    elif smooth is None:
        failed.append("too few answers to judge smoothness")
    if jumps:
        failed.append(f"{jumps} jump(s)")
    return {
        "answered_fraction": round(answered_fraction, 3),
        "answers_hz": round(answers_hz, 1),
        "updates_hz": round(update_rate_hz(samples), 1),
        "change_fraction": round(change_fraction, 3),
        "expected_change_fraction": None if expected is None else round(expected, 3),
        "smoothness_cv": None if smoothness_cv(samples) is None else round(smoothness_cv(samples), 3),
        "fit_residual_max": None if fit_residual_max(samples) is None else round(fit_residual_max(samples), 2),
        "jumps": jumps,
        "ok": not failed,
        "why": "passed" if not failed else "; ".join(failed),
    }


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


def verdict(while_moving: str | None, max_rate_hz: float | None, update_rate_hz: float | None = None,
            refresh_hz: float | None = None) -> tuple[str, str]:
    """("A" or "B", one-line reason), following the brief's rule.

    `max_rate_hz` is the fastest poll rate that passed `judge_feedback_rate` (None: none did). `update_rate_hz` is
    how often the value really changed while polling at that rate; when given, Branch A also needs it >= 10 Hz
    (with `UPDATE_RATE_TOLERANCE`). `refresh_hz` (the most updates/s seen at any rate) only improves the wording.
    Callers without these numbers may leave them out.
    """
    updates_ok = update_rate_hz is None or update_rate_hz >= BRANCH_A_MIN_HZ * UPDATE_RATE_TOLERANCE
    refresh = f" (the position refreshes at most {refresh_hz:g} times/s)" if refresh_hz is not None else ""
    if while_moving == "yes" and max_rate_hz is not None and max_rate_hz >= BRANCH_A_MIN_HZ and updates_ok:
        updates = f", position updates at {update_rate_hz:g} Hz" if update_rate_hz is not None else ""
        return "A", (f"pan/tilt inquiry works while moving at {max_rate_hz:g} Hz{updates} (>= {BRANCH_A_MIN_HZ:g} Hz), "
                     "so use feed-forward + feedback")
    if while_moving == "yes" and max_rate_hz is None:
        why = "pan/tilt inquiry answers while moving, but no poll rate gave live, smooth positions" + refresh
    elif while_moving == "yes" and max_rate_hz < BRANCH_A_MIN_HZ:
        why = f"pan/tilt inquiry gives live, smooth positions only when polled at up to {max_rate_hz:g} Hz" + refresh
    elif while_moving == "yes":
        why = (f"pan/tilt inquiry answers at {max_rate_hz:g} Hz while moving, but the position only updates at "
               f"{update_rate_hz:g} Hz (< {BRANCH_A_MIN_HZ:g} Hz)")
    elif while_moving == "frozen":
        why = "pan/tilt inquiry answers while moving, but the value doesn't change until the camera stops"
    elif while_moving == "no":
        why = "pan/tilt inquiry doesn't answer while moving"
    else:
        why = "pan/tilt inquiry doesn't work"
    return "B", why + ", so use feed-forward from the speed table and reset to the entry preset after each car"
