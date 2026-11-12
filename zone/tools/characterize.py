"""PTZ bench characterization: run it against a real camera; it writes a JSON report and a short summary.

    python -m zone.tools.characterize --ip 192.168.1.50 --out reports/smtav-a/

The camera moves by itself. The script asks before the first move, then starts from the home position: `--home-preset N`
if given, otherwise the camera's VISCA "home" (pan 0, tilt 0). Put the camera on the bench with free space around
it. A full run takes about 15-25 minutes. `--quick` measures fewer speed steps (~8 minutes).

What it answers (the WP-Z1 brief, section 3):
 1. feedback        does position inquiry answer, while moving, at what rate, smoothly?
 2. speed_table     speed step -> raw units/s, both directions, pan and tilt; the minimum smooth step
 3. latency         command -> first motion, ramp-up, stop -> standstill
 4. absolute        does absolute positioning work, how long a move takes, smooth or stepped
 5. zoom            zoom inquiry (also while zooming), range, zoom_direct accuracy/time, FOV screenshot stops
 6. repeatability   10 preset recalls from different starting points: spread of the arrived position
 7. units           raw units per degree (from --pan-units-per-degree, or two marks you aim at)
Then the verdict: Branch A (inquiry works while moving at >= 10 Hz) or Branch B.

Outputs in --out: `report.json` (all measurements, plus a "model" section that `zone.ptz.sim.PtzModel.from_report`
reads), `summary.txt`, and `visca_log.jsonl` (every datagram, both clocks). The report is rewritten after every
section, so an interrupted run still leaves what was measured. A camera that refuses or ignores an inquiry is
recorded as "unsupported" and the run goes on.

The measuring code only uses the `Ptz` interface and a clock, so the tests run it end to end against
`SimulatedPtz` with a fake clock.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from zone.clock import Clock, SystemClock
from zone.ptz.interface import Ptz
from zone.ptz.sim import PtzModel
from zone.tools import bench_analysis as ba

REPORT_SCHEMA = "zone.ptz.characterization/1"
FEEDBACK_RATES_HZ = (5, 10, 20, 50)
QUICK_STEPS_PAN = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24)
QUICK_STEPS_TILT = (1, 2, 3, 4, 6, 8, 12, 16, 20, 23)
SMOOTH_CV_LIMIT = 0.35  # interval-rate std/mean below this counts as smooth
SETTLE_STABLE_S = 0.3  # position unchanged this long = at rest
FIXED_WAIT_S = 5.0  # wait used after a move when position can't be read
EPSILON_S = 1e-6  # times this close count as equal (float sums like 0.1 + 0.2 are never exact)


# --- talking to the operator --------------------------------------------------------------------------------------


class ConsoleUI:
    """Prompts on the terminal."""

    def info(self, message: str) -> None:
        print(message, flush=True)

    def confirm(self, message: str) -> bool:
        return input(f"{message} [y/N] ").strip().lower() in ("y", "yes", "s", "sim")

    def pause(self, message: str) -> None:
        input(f"{message} (Enter to continue) ")

    def ask_float(self, message: str) -> float | None:
        """A number, or None if the answer is empty (skip)."""
        while True:
            text = input(f"{message} (empty to skip) ").strip().replace(",", ".")
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                print("  not a number, try again")


class ScriptedUI:
    """Canned answers, for tests and `--yes` runs: confirms, skips every question, records what was said."""

    def __init__(self, confirm: bool = True, numbers: Sequence[float | None] = (), on_pause: Callable[[str], None] | None = None) -> None:
        self._confirm = confirm
        self._numbers = list(numbers)
        self._on_pause = on_pause
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def confirm(self, message: str) -> bool:
        self.messages.append(message)
        return self._confirm

    def pause(self, message: str) -> None:
        self.messages.append(message)
        if self._on_pause is not None:
            self._on_pause(message)

    def ask_float(self, message: str) -> float | None:
        self.messages.append(message)
        return self._numbers.pop(0) if self._numbers else None


# --- options ------------------------------------------------------------------------------------------------------


@dataclass
class BenchOptions:
    home_preset: int | None = None  # recall this preset as home (None: the VISCA home command)
    repeat_preset: int | None = None  # preset used for repeatability (None: same as home)
    pan_max_step: int | None = None  # None: ask the camera, else the VISCA spec (24)
    tilt_max_step: int | None = None  # None: ask the camera, else 23
    quick: bool = False
    pan_travel: float = 1200.0  # raw units one speed measurement may travel (~90 deg at 0.075 deg/unit)
    tilt_travel: float = 400.0  # ~30 deg
    absolute_offset: tuple[int, int] = (300, 100)  # raw (pan, tilt) offset for the absolute-move test
    fov_pauses: bool = True
    pan_units_per_degree: float | None = None
    tilt_units_per_degree: float | None = None
    skip: tuple[str, ...] = ()


@dataclass
class Poll:
    """One inquiry: when it was sent and answered, and the answer (None: no answer)."""

    sent: float
    done: float
    value: Any

    @property
    def t(self) -> float:
        return (self.sent + self.done) / 2


def _axis_samples(polls: Sequence[Poll], axis: int | None) -> list[ba.Sample]:
    """(t, value) for answered polls; `axis` picks pan (0) or tilt (1) from a pan/tilt answer, None for zoom."""
    out = []
    for p in polls:
        if p.value is not None:
            out.append((p.t, float(p.value if axis is None else p.value[axis])))
    return out


def _r(x: float | None, digits: int = 3) -> float | None:
    return None if x is None else round(x, digits)


# --- the bench --------------------------------------------------------------------------------------------------


class Characterizer:
    """Runs the bench sections against `ptz` and fills `self.report`."""

    def __init__(self, ptz: Ptz, clock: Clock, ui: Any, options: BenchOptions | None = None,
                 out_dir: Path | None = None, camera_info: dict[str, Any] | None = None) -> None:
        self.ptz = ptz
        self.clock = clock
        self.ui = ui
        self.o = options or BenchOptions()
        self.out_dir = out_dir
        self.report: dict[str, Any] = {"schema": REPORT_SCHEMA, "created_wall": clock.wall(),
                                       "camera": camera_info or {}, "options": _options_dict(self.o)}
        # Learned along the way, used by later sections.
        self.pt_inquiry = False  # pan/tilt inquiry answers at rest
        self.while_moving: str | None = None  # "yes" / "frozen" / "no" (None: inquiry doesn't work at all)
        self.max_rate_hz: float | None = None  # fastest poll rate giving live, smooth feedback (None: none)
        self.update_rate_hz: float | None = None  # most updates/s seen while moving (the camera's refresh rate)
        self.updates_at_max_rate_hz: float | None = None  # updates/s while polling at max_rate_hz
        self.refresh_limited = False  # answers fast, but from a position cache refreshed more slowly
        self.settle_stable_s = SETTLE_STABLE_S  # unchanged this long = at rest (longer for a slow-refresh camera)
        self.pan_max = 24
        self.tilt_max = 23
        self.rates: dict[str, dict[int, float]] = {}  # "pan+" -> {step: units/s (signed)}

    # --- plumbing --------------------------------------------------------------------------------------------

    @property
    def poll_hz(self) -> float:
        """The rate to poll position at while moving: the best tested rate, at most 20 Hz."""
        return min(self.max_rate_hz or 5.0, 20.0)

    def save(self) -> None:
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "report.json").write_text(json.dumps(self.report, indent=2), encoding="utf-8")

    def safe_stop(self) -> None:
        for action in (self.ptz.stop, self.ptz.zoom_stop):
            try:
                action()
            except Exception:  # noqa: BLE001 - stopping must not raise
                pass

    def section(self, name: str, fn: Callable[[], dict[str, Any]]) -> None:
        if name in self.o.skip:
            self.report[name] = {"skipped": True}
            return
        self.ui.info(f"\n== {name} ==")
        try:
            self.report[name] = fn()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - one broken section must not lose the others
            self.safe_stop()
            self.report[name] = {"error": repr(exc), "traceback": traceback.format_exc()}
            self.ui.info(f"  section {name} failed: {exc!r} (recorded; continuing)")
        self.save()

    def sleep(self, seconds: float) -> None:
        self.clock.sleep(seconds)

    def poll(self, rate_hz: float, duration: float, inquire: Callable[[], Any],
             actions: Sequence[tuple[float, Callable[[], Any]]] = ()) -> tuple[list[Poll], list[tuple[float, Any]]]:
        """Inquire at `rate_hz` for `duration` s. `actions` are (offset s, fn) run once their time comes; returns the
        polls and, per action, (time it ran, its return value)."""
        start = self.clock.monotonic()
        period = 1.0 / rate_hz
        pending = sorted(actions, key=lambda a: a[0])
        done_actions: list[tuple[float, Any]] = []
        polls: list[Poll] = []
        k = 0
        while True:
            now = self.clock.monotonic()
            while pending and now - start >= pending[0][0] - EPSILON_S:
                _, fn = pending.pop(0)
                t = self.clock.monotonic()
                result = fn()
                sent = getattr(result, "sent_at", None)
                done_actions.append((sent if sent is not None else t, result))
                now = self.clock.monotonic()
            if now - start >= duration - EPSILON_S:
                break
            slot = start + k * period
            next_action = start + pending[0][0] if pending else float("inf")
            wake = min(slot, next_action)
            if wake > now + EPSILON_S:
                self.sleep(wake - now)
                if wake == next_action and next_action < slot:
                    continue
            sent = self.clock.monotonic()
            value = inquire()
            polls.append(Poll(sent, self.clock.monotonic(), value))
            k = max(k + 1, int((self.clock.monotonic() - start) / period) + 1)  # skip slots we fell behind on
        return polls, done_actions

    def wait_at_rest(self, start_value: Any = None, expect_change: bool = False,
                     timeout: float = 15.0) -> tuple[float | None, Any]:
        """Wait until the reported position stops changing. Returns (seconds until it settled, final position).

        With `expect_change`, a value equal to `start_value` doesn't count as settled (the move may not have begun,
        or the camera may freeze its report while moving). Without position inquiry: a fixed wait, (None, None).
        """
        t0 = self.clock.monotonic()
        if not self.pt_inquiry:
            self.sleep(FIXED_WAIT_S)
            return None, None
        last: Any = None
        stable_since: float | None = None
        while (now := self.clock.monotonic()) - t0 < timeout:
            self.sleep(0.1)
            value = self.ptz.inquire_pan_tilt()
            now = self.clock.monotonic()
            if value is None or value != last or (expect_change and value == start_value):
                stable_since = now if value is not None else None
                last = value
                continue
            if stable_since is not None and now - stable_since >= self.settle_stable_s:
                return stable_since - t0, value
        return None, last

    def go_home(self) -> tuple[float | None, Any]:
        start = self.ptz.inquire_pan_tilt()
        if self.o.home_preset is not None:
            command = self.ptz.preset_recall(self.o.home_preset)
        else:
            command = self.ptz.home()
        if "home_command" not in self.report:
            self.report["home_command"] = {"preset": self.o.home_preset, "status": command.status}
            if not command.ok:
                self.ui.info(f"  !! the camera did not accept the home command ({command.status}); measuring from "
                             "where it is. Consider --home-preset.")
        self.sleep(0.2)
        return self.wait_at_rest(start, expect_change=False)

    # --- the run ---------------------------------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        try:
            self.section("connection", self.connection)
            if not self.ui.confirm("The camera is about to move by itself (pan, tilt, zoom). Is the area clear?"):
                self.report["aborted"] = "operator did not confirm"
                self.save()
                return self.report
            self.ui.info("Going to the home position...")
            self.go_home()
            self.section("feedback", self.feedback)
            self.section("speed_table", self.speed_table)
            self.section("latency", self.latency)
            self.section("absolute", self.absolute)
            self.section("zoom", self.zoom)
            self.section("repeatability", self.repeatability)
            self.section("units", self.units)
        except KeyboardInterrupt:
            self.report["aborted"] = "interrupted (Ctrl+C)"
            self.ui.info("\nInterrupted: stopping the camera and writing what was measured.")
        finally:
            self.safe_stop()
        self.report["model"] = self.build_model().to_dict()
        branch, reason = ba.verdict(self.while_moving, self.max_rate_hz, self.updates_at_max_rate_hz,
                                    refresh_hz=self.update_rate_hz)
        self.report["verdict"] = {"branch": branch, "reason": reason}
        self.save()
        return self.report

    # --- 0. connection -------------------------------------------------------------------------------------

    def connection(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        info["sequence_reset"] = getattr(self.ptz, "sequence_reset_status", None)
        max_speed = self.ptz.inquire_max_speed() if hasattr(self.ptz, "inquire_max_speed") else None
        info["max_speed_inquiry"] = list(max_speed) if max_speed else "unsupported"
        version = self.ptz.inquire_version() if hasattr(self.ptz, "inquire_version") else None
        info["version_raw"] = version.hex(" ") if version else "unsupported"
        pan_max, tilt_max = 24, 23
        if max_speed and 1 <= max_speed[0] <= 0x18 and 1 <= max_speed[1] <= 0x18:
            pan_max, tilt_max = max_speed
        self.pan_max = self.o.pan_max_step or pan_max
        self.tilt_max = self.o.tilt_max_step or tilt_max
        info["pan_max_step_used"] = self.pan_max
        info["tilt_max_step_used"] = self.tilt_max
        rest = self.ptz.inquire_pan_tilt()
        self.pt_inquiry = rest is not None
        info["pan_tilt_at_start"] = list(rest) if rest else None
        self.ui.info(f"  sequence reset: {info['sequence_reset']}; max speed inquiry: {info['max_speed_inquiry']}; "
                     f"version: {info['version_raw']}; position: {info['pan_tilt_at_start']}")
        return info

    # --- 1. feedback -----------------------------------------------------------------------------------------

    def feedback(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        at_rest = [self._timed(self.ptz.inquire_pan_tilt) for _ in range(5)]
        answered = [p for p in at_rest if p.value is not None]
        self.pt_inquiry = bool(answered)
        result["pan_tilt_at_rest"] = {
            "supported": self.pt_inquiry,
            "answered": f"{len(answered)}/{len(at_rest)}",
            "round_trip_ms_p50": _r(ba.percentile([(p.done - p.sent) * 1000 for p in answered], 50), 1),
        }
        zoom_at_rest = [self._timed(self.ptz.inquire_zoom) for _ in range(3)]
        result["zoom_at_rest"] = {"supported": any(p.value is not None for p in zoom_at_rest)}
        if not self.pt_inquiry:
            result["while_moving"] = "unsupported"
            self.while_moving = None
            self.ui.info("  pan/tilt inquiry: no answer at rest -> unsupported")
            return result

        step = max(1, round(self.pan_max * 0.5))
        per_rate: dict[str, Any] = {}
        all_answered = all_sent = 0
        any_change = False
        best: float | None = None
        best_updates_hz: float | None = None
        rtts: list[float] = []
        for i, rate in enumerate(FEEDBACK_RATES_HZ):
            direction = 1 if i % 2 == 0 else -1
            self.ptz.pan_tilt_velocity(direction * step, 0, ttl=3.0)
            self.sleep(0.4)  # past latency and ramp: from here on the speed is constant
            polls, _ = self.poll(rate, 1.5, self.ptz.inquire_pan_tilt)
            self.safe_stop()
            self.sleep(0.6)
            samples = _axis_samples(polls, 0)
            # The rules (answers, updates, smoothness, jumps) live in bench_analysis.judge_feedback_rate.
            judged = ba.judge_feedback_rate(rate, len(polls), samples)
            if judged["ok"]:
                best = float(rate)
                best_updates_hz = judged["updates_hz"]
            all_answered += len(samples)
            all_sent += len(polls)
            any_change = any_change or ba.value_changes(samples) > 0
            rtts += [(p.done - p.sent) * 1000 for p in polls if p.value is not None]
            per_rate[str(rate)] = {"sent": len(polls), "answered": len(samples), **judged}
            self.ui.info(f"  {rate:>2} Hz while panning: {judged['answers_hz']} answers/s, {judged['updates_hz']} "
                         f"updates/s, cv {judged['smoothness_cv']}, jumps {judged['jumps']}: "
                         f"{'ok' if judged['ok'] else 'NOT ok'} ({judged['why']})")
        self.go_home()
        self.while_moving = ba.classify_feedback(all_answered / all_sent if all_sent else 0.0, any_change)
        self.max_rate_hz = best if self.while_moving == "yes" else None
        # How often the position really changes: while polling at the best passing rate (the verdict needs it),
        # and the most seen at any rate (for a stale-cache camera, that is its refresh rate).
        seen_updates = [row["updates_hz"] for row in per_rate.values()]
        self.updates_at_max_rate_hz = best_updates_hz if self.max_rate_hz is not None else None
        self.update_rate_hz = max(seen_updates) if seen_updates and self.while_moving == "yes" else None
        # A stale cache: the camera answers far more often than its value changes, at the fastest rate it answers.
        fastest = per_rate.get(str(FEEDBACK_RATES_HZ[-1]), {})
        self.refresh_limited = bool(
            self.while_moving == "yes" and fastest and fastest["answers_hz"] > 0
            and fastest["change_fraction"] < ba.MIN_CHANGE_FRACTION * (fastest["expected_change_fraction"] or 1.0))
        if self.update_rate_hz:
            # Between refreshes a moving camera's report holds still; don't mistake that for "at rest".
            self.settle_stable_s = max(SETTLE_STABLE_S, 2.5 / self.update_rate_hz)
        result.update({
            "while_moving": self.while_moving,
            "test_step": step,
            "rates": per_rate,
            "max_rate_hz": self.max_rate_hz,
            "update_rate_hz": self.update_rate_hz,
            "updates_at_max_rate_hz": self.updates_at_max_rate_hz,
            "refresh_limited": self.refresh_limited,
            "round_trip_ms_p50": _r(ba.percentile(rtts, 50), 1),
            "round_trip_ms_p95": _r(ba.percentile(rtts, 95), 1),
            "rules": "a rate is ok when >= 90% of inquiries are answered at >= 80% of the asked rate, the value "
                     "changes on >= 70% of the answers where a live report would, the motion is smooth (cv <= "
                     f"{ba.SMOOTH_CV_LIMIT} or within {ba.QUANTIZATION_UNITS:g} unit of a straight line) and "
                     "there are no jumps",
            "freshness_note": "answers are at most one round trip old as seen from here; "
                              "the camera's own encoder-read delay is not measurable from inquiries alone",
        })
        return result

    def _timed(self, inquire: Callable[[], Any]) -> Poll:
        sent = self.clock.monotonic()
        value = inquire()
        return Poll(sent, self.clock.monotonic(), value)

    # --- 2. speed table --------------------------------------------------------------------------------------

    def speed_table(self) -> dict[str, Any]:
        if self.while_moving == "yes" and self.max_rate_hz is not None:
            method = "moving"
        elif self.pt_inquiry:
            method = "timed"
        else:
            return {"method": "unsupported", "note": "no position inquiry: the speed table needs video analysis"}
        result: dict[str, Any] = {"method": method}
        for axis_name, axis, top, travel, quick in (("pan", 0, self.pan_max, self.o.pan_travel, QUICK_STEPS_PAN),
                                                    ("tilt", 1, self.tilt_max, self.o.tilt_travel, QUICK_STEPS_TILT)):
            steps = [s for s in quick if s <= top] if self.o.quick else list(range(1, top + 1))
            if top not in steps:
                steps.append(top)
            rows: dict[str, Any] = {}
            estimate: float | None = None
            for step in steps:
                duration = 3.0 if estimate is None else min(3.0, max(0.8, travel / estimate))
                row = (self._rate_moving(axis, step, duration) if method == "moving"
                       else self._rate_timed(axis, step, duration))
                rows[str(step)] = row
                for sign in ("+", "-"):
                    rate = row.get(sign, {}).get("rate")
                    if rate is not None:
                        self.rates.setdefault(f"{axis_name}{sign}", {})[step] = rate
                rates_here = [abs(row[s]["rate"]) for s in ("+", "-") if row.get(s, {}).get("rate")]
                if rates_here:
                    estimate = max(rates_here) * 1.3  # the next step is faster
                self.ui.info(f"  {axis_name} step {step:>2}: + {(row.get('+') or {}).get('rate')}, "
                             f"- {(row.get('-') or {}).get('rate')} units/s")
            result[axis_name] = rows
            result[f"{axis_name}_min_smooth_step"] = self._min_smooth_step(rows, method)
            result[f"{axis_name}_sign"] = self._sign(axis_name)
        self.go_home()
        return result

    def _velocity(self, axis: int, step: int, ttl: float) -> Any:
        return self.ptz.pan_tilt_velocity(step if axis == 0 else 0, step if axis == 1 else 0, ttl=ttl)

    def _rate_moving(self, axis: int, step: int, duration: float) -> dict[str, Any]:
        row: dict[str, Any] = {"duration_s": round(duration, 2)}
        for sign, direction in (("+", 1), ("-", -1)):
            polls, actions = self.poll(self.poll_hz, duration + 0.6, self.ptz.inquire_pan_tilt, [
                (0.0, lambda d=direction: self._velocity(axis, d * step, duration + 1.0)),
                (duration, self.ptz.stop),
            ])
            command = actions[0][1] if actions else None
            if command is not None and not command.ok:
                row[sign] = {"rejected": command.status}
                continue
            t_cmd = actions[0][0]
            samples = _axis_samples(polls, axis)
            window = [s for s in samples if t_cmd + max(0.5, duration * 0.4) <= s[0] <= t_cmd + duration]
            rate = ba.fit_rate(window)
            tail = window[len(window) * 3 // 4:]
            tail_rate = ba.fit_rate(tail)
            row[sign] = {
                "rate": _r(rate, 2),
                "smoothness_cv": _r(ba.smoothness_cv(window)),
                "travel": (samples[-1][1] - samples[0][1]) if samples else None,
                "hit_limit": bool(rate and tail_rate is not None and abs(tail_rate) < 0.5 * abs(rate)),
                "samples": len(window),
            }
            self.sleep(0.3)
        return row

    def _rate_timed(self, axis: int, step: int, duration: float) -> dict[str, Any]:
        """Position readable only at rest: two timed moves (T and 2T) per direction. The difference of the two
        travels divided by T is the steady rate, free of latency and ramp; what's left over is the effective lag."""
        short, long_ = duration / 2, duration
        travel: dict[tuple[str, float], float | None] = {}
        rejected: dict[str, str] = {}
        for seconds in (short, long_):
            for sign, direction in (("+", 1), ("-", -1)):
                _, before = self.wait_at_rest()
                result = self._velocity(axis, direction * step, seconds + 1.0)
                if not result.ok:
                    rejected[sign] = result.status
                    continue
                self.sleep(seconds)
                self.ptz.stop()
                _, after = self.wait_at_rest(before, expect_change=True)
                travel[(sign, seconds)] = (after[axis] - before[axis]) if (before and after) else None
        row: dict[str, Any] = {"duration_s": round(duration, 2)}
        for sign in ("+", "-"):
            if sign in rejected:
                row[sign] = {"rejected": rejected[sign]}
                continue
            d1, d2 = travel.get((sign, short)), travel.get((sign, long_))
            if d1 is None or d2 is None:
                row[sign] = {"rate": None}
                continue
            rate = (d2 - d1) / (long_ - short)
            lag = short - d1 / rate if rate else None
            row[sign] = {"rate": _r(rate, 2), "effective_lag_s": _r(lag), "travel_short": d1, "travel_long": d2}
        return row

    def _min_smooth_step(self, rows: dict[str, Any], method: str) -> int | None:
        for step in sorted(rows, key=int):
            row = rows[step]
            ok = True
            for sign in ("+", "-"):
                cell = row.get(sign) or {}
                rate = cell.get("rate")
                if not rate:
                    ok = False
                elif method == "moving" and (cell.get("smoothness_cv") is None or cell["smoothness_cv"] > SMOOTH_CV_LIMIT):
                    ok = False
            if ok:
                return int(step)
        return None

    def _sign(self, axis_name: str) -> int | None:
        positive = [r for r in self.rates.get(f"{axis_name}+", {}).values() if r]
        if not positive:
            return None
        return 1 if statistics.median(positive) > 0 else -1

    # --- 3. latency and ramp ---------------------------------------------------------------------------------

    def latency(self) -> dict[str, Any]:
        if self.while_moving == "no":
            return self._latency_from_silence()
        if not (self.while_moving == "yes" and (self.max_rate_hz or 0) >= 10):
            return {"method": "timed_moves", "effective_lag_s_p50": self._effective_lag(),
                    "note": "position can't be followed during a move, so latency and ramp can't be separated; "
                            "effective lag = start latency + ramp/2 - stop latency - stop ramp/2, from the speed table"}
        rate = self.max_rate_hz or 10.0
        trials = []
        steps = sorted({max(1, self.pan_max // 6), max(1, self.pan_max // 2), self.pan_max})
        plan = [("pan", 0, s) for s in steps] + [("tilt", 1, max(1, self.tilt_max // 2))]
        for i, (axis_name, axis, step) in enumerate(plan):
            direction = 1 if i % 2 == 0 else -1
            hold = 1.5 if axis == 0 else 0.8
            polls, actions = self.poll(rate, 0.3 + hold + 1.2, self.ptz.inquire_pan_tilt, [
                (0.3, lambda a=axis, s=direction * step, h=hold: self._velocity(a, s, h + 1.0)),
                (0.3 + hold, self.ptz.stop),
            ])
            samples = _axis_samples(polls, axis)
            if len(actions) < 2 or len(samples) < 5:
                continue
            t_cmd, t_stop = actions[0][0], actions[1][0]
            pre = [v for t, v in samples if t < t_cmd]
            baseline = statistics.median(pre) if pre else samples[0][1]
            t_first = ba.first_change(samples, baseline, t_cmd)
            steady = ba.fit_rate([s for s in samples if t_cmd + 0.6 <= s[0] <= t_stop])
            ramp_end = ba.time_to_reach_rate(samples, t_first, steady) if (t_first and steady) else None
            drop = ba.time_rate_drops(samples, t_stop, steady) if steady else None
            settled = ba.last_change(samples, t_stop)
            trial = {
                "axis": axis_name, "step": direction * step,
                "latency_s": _r(t_first - t_cmd if t_first else None),
                "ramp_s": _r(ramp_end - t_first if (ramp_end and t_first) else None),
                "stop_latency_s": _r(drop - t_stop if drop else None),
                "stop_total_s": _r(settled - t_stop if settled else None),
                "steady_rate": _r(steady, 2),
            }
            trial["stop_ramp_s"] = (_r(trial["stop_total_s"] - trial["stop_latency_s"])
                                    if trial["stop_total_s"] is not None and trial["stop_latency_s"] is not None else None)
            trials.append(trial)
            self.ui.info(f"  {axis_name} step {direction * step:+d}: latency {trial['latency_s']} s, ramp {trial['ramp_s']} s, "
                         f"stop latency {trial['stop_latency_s']} s, stopped after {trial['stop_total_s']} s")
            self.sleep(0.3)
        self.go_home()

        def med(key: str) -> float | None:
            return _r(ba.percentile([t[key] for t in trials if t[key] is not None], 50))
        return {"method": "polling", "poll_hz": rate, "resolution_s": _r(1 / rate), "trials": trials,
                "latency_s_p50": med("latency_s"), "ramp_s_p50": med("ramp_s"),
                "stop_latency_s_p50": med("stop_latency_s"), "stop_ramp_s_p50": med("stop_ramp_s")}

    def _effective_lag(self) -> float | None:
        lags = [cell["effective_lag_s"]
                for axis in ("pan", "tilt")
                for row in (self.report.get("speed_table", {}).get(axis) or {}).values()
                for cell in (row.get("+"), row.get("-"))
                if isinstance(cell, dict) and cell.get("effective_lag_s") is not None]
        return _r(ba.percentile(lags, 50))

    def _latency_from_silence(self) -> dict[str, Any]:
        """For a camera that answers position only at rest: it falls silent when motion starts and answers again
        when it has stopped. Polling fast around a command times both edges (to about one poll period, plus the
        driver timeout while it is silent)."""
        trials = []
        plan = [("pan", 0, max(1, self.pan_max // 2)), ("pan", 0, self.pan_max), ("tilt", 1, max(1, self.tilt_max // 2))]
        for i, (axis_name, axis, step) in enumerate(plan):
            direction = 1 if i % 2 == 0 else -1
            hold = 1.0 if axis == 0 else 0.6
            polls, actions = self.poll(20.0, 0.3 + hold + 2.0, self.ptz.inquire_pan_tilt, [
                (0.3, lambda a=axis, s=direction * step, h=hold: self._velocity(a, s, h + 1.0)),
                (0.3 + hold, self.ptz.stop),
            ])
            if len(actions) < 2:
                continue
            t_cmd, t_stop = actions[0][0], actions[1][0]
            went_silent = ba.edge_time(polls, t_cmd, answered_before=True)
            answered_again = ba.edge_time(polls, t_stop, answered_before=False)
            trial = {"axis": axis_name, "step": direction * step,
                     "latency_s": _r(went_silent - t_cmd if went_silent is not None else None),
                     "stop_total_s": _r(answered_again - t_stop if answered_again is not None else None)}
            trials.append(trial)
            self.ui.info(f"  {axis_name} step {direction * step:+d}: motion began {trial['latency_s']} s after the "
                         f"command; at rest {trial['stop_total_s']} s after stop")
            self.sleep(0.3)
        self.go_home()
        latency = _r(ba.percentile([t["latency_s"] for t in trials if t["latency_s"] is not None], 50))
        stop_total = _r(ba.percentile([t["stop_total_s"] for t in trials if t["stop_total_s"] is not None], 50))
        timeout = getattr(self.ptz, "timeout", None)
        return {"method": "inquiry_silence", "poll_hz": 20.0, "trials": trials, "latency_s_p50": latency,
                "resolution_s": {"latency": 0.05, "stop_total": _r(max(0.05, timeout or 0.0))},
                "stop_total_s_p50": stop_total, "effective_lag_s_p50": self._effective_lag(),
                "note": "latency = command until the camera stops answering (it's moving); stop_total = stop until "
                        "it answers again. The ramps are inferred from the effective lag of the timed moves."}

    # --- 4. absolute positioning -----------------------------------------------------------------------------

    def absolute(self) -> dict[str, Any]:
        _, start = self.wait_at_rest()
        result: dict[str, Any] = {}
        if start is None and self.pt_inquiry:
            start = self.ptz.inquire_pan_tilt()
        base = start or (0, 0)
        dpan, dtilt = self.o.absolute_offset
        moves = []
        for speed_frac in (0.5, 1.0):
            pan_speed = max(1, round(self.pan_max * speed_frac))
            tilt_speed = max(1, round(self.tilt_max * speed_frac))
            for target in ((base[0] + dpan, base[1] + dtilt), (base[0], base[1])):
                moves.append(self._absolute_move(target, pan_speed, tilt_speed))
                if moves[-1].get("status") == "error":
                    break
            if moves and moves[-1].get("status") == "error":
                break
        result["supported"] = any(m.get("status") in ("acked", "completed") for m in moves)
        result["moves"] = moves
        rel = self.ptz.relative_position(dpan, dtilt, max(1, self.pan_max // 2), max(1, self.tilt_max // 2))
        settle, after = self.wait_at_rest(base, expect_change=True)
        result["relative"] = {"status": rel.status, "supported": rel.ok, "arrived": list(after) if after else None,
                              "expected": [base[0] + dpan, base[1] + dtilt]}
        self.go_home()
        return result

    def _absolute_move(self, target: tuple[float, float], pan_speed: int, tilt_speed: int) -> dict[str, Any]:
        target_i = (int(round(target[0])), int(round(target[1])))
        before = self.ptz.inquire_pan_tilt()
        command = self.ptz.absolute_position(target_i[0], target_i[1], pan_speed, tilt_speed)
        move: dict[str, Any] = {"target": list(target_i), "pan_speed": pan_speed, "tilt_speed": tilt_speed,
                                "status": command.status}
        if not command.ok:
            return move
        t_cmd = command.sent_at if command.sent_at is not None else self.clock.monotonic()
        if not self.pt_inquiry:
            self.sleep(FIXED_WAIT_S)
            return move
        polls: list[Poll] = []
        arrived_at = None
        period = 1.0 / (self.poll_hz if self.while_moving == "yes" else 10.0)
        while self.clock.monotonic() - t_cmd < 15.0:
            p = self._timed(self.ptz.inquire_pan_tilt)
            polls.append(p)
            if p.value is not None and abs(p.value[0] - target_i[0]) <= 2 and abs(p.value[1] - target_i[1]) <= 2:
                arrived_at = p.t
                break
            self.sleep(period)
        settle, final = self.wait_at_rest(before, expect_change=before != target_i)
        samples = _axis_samples(polls, 0)
        moving = [s for s in samples if t_cmd < s[0] < (arrived_at or float("inf"))]
        completion = None
        if hasattr(self.ptz, "completion_time") and command.sequence is not None:
            done = self.ptz.completion_time(command.sequence)
            completion = _r(done - t_cmd) if done else None
        move.update({
            "arrival_s": _r(arrived_at - t_cmd if arrived_at else None),
            "completion_reply_s": completion,
            "final": list(final) if final else None,
            "error": [final[0] - target_i[0], final[1] - target_i[1]] if final else None,
            "smoothness_cv": _r(ba.smoothness_cv(moving)) if self.while_moving == "yes" else None,
            "jumps": ba.count_jumps(moving) if self.while_moving == "yes" else None,
        })
        self.ui.info(f"  absolute to {target_i} at speed {pan_speed}/{tilt_speed}: arrived after {move['arrival_s']} s, "
                     f"error {move['error']}")
        return move

    # --- 5. zoom ---------------------------------------------------------------------------------------------

    def zoom(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        at_rest = self.ptz.inquire_zoom()
        result["inquiry_at_rest"] = at_rest is not None
        if at_rest is None:
            result["note"] = "zoom inquiry unsupported: range, zoom_direct accuracy and FOV positions need a human"
        # While zooming.
        self.ptz.zoom_velocity(-7, ttl=6.0)
        self.sleep(4.0)
        self.ptz.zoom_stop()
        self.sleep(0.8)
        polls, _ = self.poll(10.0, 1.2, self.ptz.inquire_zoom, [(0.0, lambda: self.ptz.zoom_velocity(4, ttl=3.0))])
        self.ptz.zoom_stop()
        samples = _axis_samples(polls, None)
        fraction = len(samples) / len(polls) if polls else 0.0
        result["inquiry_while_zooming"] = (ba.classify_feedback(fraction, len({v for _, v in samples}) > 1)
                                           if at_rest is not None else "unsupported")
        if at_rest is None:
            self.ptz.zoom_velocity(-7, ttl=6.0)
            self.sleep(5.0)
            self.ptz.zoom_stop()
            return result
        # Range: all the way out, then all the way in.
        wide = self._zoom_to_end(-7)
        tele = self._zoom_to_end(+7)
        result["range"] = [wide, tele]
        # Speed per step, from the wide end: zoom in for 1 s, read at rest.
        speeds: dict[str, Any] = {}
        for step in range(1, 8):
            if wide is not None:
                self.ptz.zoom_direct(wide)
            else:
                self.ptz.zoom_velocity(-7, ttl=6.0)
            self._zoom_wait_stable()
            start = self.ptz.inquire_zoom()
            self.ptz.zoom_velocity(step, ttl=2.0)
            self.sleep(1.0)
            self.ptz.zoom_stop()
            end = self._zoom_wait_stable()
            speeds[str(step)] = (end - start) if (start is not None and end is not None) else None
        result["speed_per_step_units_per_s"] = speeds
        result["speed_note"] = "1-second timed moves read at rest: includes latency and ramp, so slightly low"
        # zoom_direct accuracy and time.
        direct = []
        if wide is not None and tele is not None and tele > wide:
            for fraction in (0.25, 0.5, 0.75, 0.0):
                target = int(wide + (tele - wide) * fraction)
                command = self.ptz.zoom_direct(target)
                t0 = command.sent_at if command.sent_at is not None else self.clock.monotonic()
                if not command.ok:
                    direct.append({"target": target, "status": command.status})
                    break
                arrived = None
                while self.clock.monotonic() - t0 < 10.0:
                    z = self.ptz.inquire_zoom()
                    if z is not None and abs(z - target) <= max(2, (tele - wide) // 200):
                        arrived = self.clock.monotonic()
                        break
                    self.sleep(0.1)
                final = self._zoom_wait_stable()
                direct.append({"target": target, "status": command.status,
                               "arrival_s": _r(arrived - t0 if arrived else None),
                               "final": final, "error": (final - target) if final is not None else None})
        result["zoom_direct"] = direct
        result["zoom_direct_supported"] = any(d.get("status") in ("acked", "completed") for d in direct)
        # FOV stops: João takes a screenshot at each and measures the FOV by hand.
        positions = []
        if wide is not None and tele is not None and tele > wide:
            positions = [int(wide + (tele - wide) * k / 8) for k in range(9)]
            for pos in positions:
                if result["zoom_direct_supported"]:
                    self.ptz.zoom_direct(pos)
                    self._zoom_wait_stable()
                actual = self.ptz.inquire_zoom()
                if self.o.fov_pauses:
                    self.ui.pause(f"  zoom at {actual} (target {pos}): take a screenshot for the FOV measurement")
                else:
                    self.ui.info(f"  FOV stop: zoom {actual} (target {pos})")
        result["fov_positions"] = positions
        result["fov_note"] = "FOV per position is measured by hand from the screenshots; fill it in the report"
        if wide is not None and result["zoom_direct_supported"]:
            self.ptz.zoom_direct(wide)
            self._zoom_wait_stable()
        return result

    def _zoom_to_end(self, step: int) -> int | None:
        """Zoom at full speed until the reported zoom stops changing (max 12 s); the end position."""
        self.ptz.zoom_velocity(step, ttl=13.0)
        t0 = self.clock.monotonic()
        last, unchanged_since = None, None
        while self.clock.monotonic() - t0 < 12.0:
            self.sleep(0.2)
            z = self.ptz.inquire_zoom()
            now = self.clock.monotonic()
            if z is not None and z == last:
                unchanged_since = unchanged_since or now
                if now - unchanged_since >= 0.8:
                    break
            else:
                unchanged_since = None
            last = z
        self.ptz.zoom_stop()
        return self._zoom_wait_stable()

    def _zoom_wait_stable(self, timeout: float = 10.0) -> int | None:
        t0 = self.clock.monotonic()
        last, since = None, None
        while self.clock.monotonic() - t0 < timeout:
            self.sleep(0.15)
            z = self.ptz.inquire_zoom()
            now = self.clock.monotonic()
            if z is not None and z == last:
                since = since or now
                if now - since >= SETTLE_STABLE_S:
                    return z
            else:
                since = None
            last = z
        return last

    # --- 6. repeatability ------------------------------------------------------------------------------------

    def repeatability(self) -> dict[str, Any]:
        preset = self.o.repeat_preset if self.o.repeat_preset is not None else self.o.home_preset
        target = f"preset {preset}" if preset is not None else "VISCA home"
        pan_step = max(1, self.pan_max // 2)
        tilt_step = max(1, self.tilt_max // 3)
        starts = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 0.6), (0, -1, 0.6), (1, 1, 0.8),
                  (-1, -1, 0.8), (1, -1, 0.8), (-1, 1, 0.8), (1, 0, 2.0), (-1, 0, 2.0)]
        arrivals, settles = [], []
        for pan_dir, tilt_dir, seconds in starts:
            self.ptz.pan_tilt_velocity(pan_dir * pan_step, tilt_dir * tilt_step, ttl=seconds + 1.0)
            self.sleep(seconds)
            self.ptz.stop()
            _, offset_pose = self.wait_at_rest()
            if preset is not None:
                self.ptz.preset_recall(preset)
            else:
                self.ptz.home()
            settle, pose = self.wait_at_rest(offset_pose, expect_change=True, timeout=20.0)
            if pose is not None:
                arrivals.append(pose)
            if settle is not None:
                settles.append(settle)
        result: dict[str, Any] = {"target": target, "recalls": len(starts), "arrivals": [list(a) for a in arrivals],
                                  "settle_s": [round(s, 2) for s in settles],
                                  "settle_s_p90": _r(ba.percentile(settles, 90))}
        if arrivals:
            pans = [a[0] for a in arrivals]
            tilts = [a[1] for a in arrivals]
            result["pan_spread"] = max(pans) - min(pans)
            result["tilt_spread"] = max(tilts) - min(tilts)
            result["pan_std"] = _r(statistics.pstdev(pans), 2)
            result["tilt_std"] = _r(statistics.pstdev(tilts), 2)
            self.ui.info(f"  {len(arrivals)} recalls of {target}: spread pan {result['pan_spread']}, "
                         f"tilt {result['tilt_spread']} raw units; settle p90 {result['settle_s_p90']} s")
        else:
            result["note"] = "position inquiry unavailable: spread not measurable"
        return result

    # --- 7. units --------------------------------------------------------------------------------------------

    def units(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for axis_name, axis, given in (("pan", 0, self.o.pan_units_per_degree), ("tilt", 1, self.o.tilt_units_per_degree)):
            if given:
                result[f"{axis_name}_units_per_degree"] = given
                result[f"{axis_name}_source"] = "given on the command line (camera documentation)"
                continue
            if not self.pt_inquiry:
                result[f"{axis_name}_units_per_degree"] = None
                result[f"{axis_name}_source"] = "unmeasurable: no position inquiry"
                continue
            angle = self.ui.ask_float(
                f"  {axis_name.upper()} units: aim the camera at mark A with the joystick, web UI or remote, then type "
                f"the angle in degrees between mark A and mark B ({axis_name})")
            if angle is None or angle == 0:
                result[f"{axis_name}_units_per_degree"] = None
                result[f"{axis_name}_source"] = "skipped"
                continue
            a = self.ptz.inquire_pan_tilt()
            self.ui.pause(f"  now aim at mark B ({axis_name})")
            b = self.ptz.inquire_pan_tilt()
            if a is None or b is None:
                result[f"{axis_name}_units_per_degree"] = None
                result[f"{axis_name}_source"] = "inquiry failed"
                continue
            if b[axis] == a[axis]:
                result[f"{axis_name}_units_per_degree"] = None
                result[f"{axis_name}_source"] = "both marks read the same position (camera not moved?)"
                continue
            result[f"{axis_name}_units_per_degree"] = _r(abs(b[axis] - a[axis]) / abs(angle), 4)
            result[f"{axis_name}_source"] = f"two marks {angle:g} deg apart: raw {a[axis]} -> {b[axis]}"
        return result

    # --- the model for the simulator ---------------------------------------------------------------------------

    def build_model(self) -> PtzModel:
        """Defaults from `PtzModel`, overwritten by whatever was measured."""
        m = PtzModel()
        r = self.report
        speed = r.get("speed_table", {})
        for axis in ("pan", "tilt"):
            positive = {k: abs(v) for k, v in self.rates.get(f"{axis}+", {}).items() if v}
            negative = {k: abs(v) for k, v in self.rates.get(f"{axis}-", {}).items() if v}
            if positive:
                setattr(m, f"{axis}_speed_table", positive)
            if negative and positive:
                common = set(positive) & set(negative)
                asymmetric = any(abs(positive[k] - negative[k]) > 0.1 * positive[k] for k in common)
                setattr(m, f"{axis}_speed_table_negative", negative if asymmetric else None)
            sign = speed.get(f"{axis}_sign")
            if sign:
                setattr(m, f"{axis}_sign", sign)
            smooth = speed.get(f"{axis}_min_smooth_step")
            if smooth:
                setattr(m, f"{axis}_min_smooth_step", smooth)
        lat = r.get("latency", {})
        if lat.get("method") == "polling":
            for key, field_name in (("latency_s_p50", "command_latency_s"), ("ramp_s_p50", "ramp_s"),
                                    ("stop_latency_s_p50", "stop_latency_s"), ("stop_ramp_s_p50", "stop_ramp_s")):
                if lat.get(key) is not None:
                    setattr(m, field_name, max(0.0, lat[key]))
        elif lat.get("method") == "inquiry_silence" and lat.get("latency_s_p50") is not None:
            # Known: start latency, total stop time, and the effective lag
            #   lag = latency + ramp/2 - stop_latency - stop_ramp/2.
            # Split the stop time half and half, and solve for the ramp.
            m.command_latency_s = max(0.0, lat["latency_s_p50"])
            stop_total = lat.get("stop_total_s_p50") or (m.stop_latency_s + m.stop_ramp_s)
            m.stop_latency_s = m.stop_ramp_s = max(0.0, stop_total / 2)
            lag = lat.get("effective_lag_s_p50")
            if lag is not None:
                ramp = 2 * (lag - m.command_latency_s + m.stop_latency_s + m.stop_ramp_s / 2)
                m.ramp_s = min(1.0, max(0.0, ramp))
        elif lat.get("effective_lag_s_p50") is not None:
            m.command_latency_s = lat["effective_lag_s_p50"]
            m.stop_latency_s = lat["effective_lag_s_p50"]
            m.ramp_s = m.stop_ramp_s = 0.0
        fb = r.get("feedback", {})
        if fb.get("pan_tilt_at_rest", {}).get("supported") is False:
            m.inquiry_mode = "never"
        elif self.while_moving == "no":
            m.inquiry_mode = "at_rest"
        elif self.while_moving == "frozen":
            m.inquiry_mode = "frozen_while_moving"
        m.inquiry_poll_hz = self.max_rate_hz
        if self.refresh_limited and self.update_rate_hz:
            m.inquiry_refresh_hz = self.update_rate_hz
        zoom = r.get("zoom", {})
        if zoom.get("inquiry_at_rest") is False:
            m.zoom_inquiry_mode = "never"
        elif zoom.get("inquiry_while_zooming") in ("no", "frozen"):
            m.zoom_inquiry_mode = "at_rest" if zoom["inquiry_while_zooming"] == "no" else "frozen_while_moving"
        rng = zoom.get("range")
        if rng and rng[0] is not None and rng[1] is not None and rng[1] > rng[0]:
            m.zoom_limits = (float(rng[0]), float(rng[1]))
        zspeeds = {int(k): float(v) for k, v in (zoom.get("speed_per_step_units_per_s") or {}).items() if v and v > 0}
        if zspeeds:
            m.zoom_speed_table = zspeeds
        if "zoom_direct_supported" in zoom:
            m.supports_zoom_direct = bool(zoom["zoom_direct_supported"])
        absolute = r.get("absolute", {})
        if "supported" in absolute:
            m.supports_absolute = bool(absolute["supported"])
            m.supports_relative = bool(absolute.get("relative", {}).get("supported", True))
        rep = r.get("repeatability", {})
        if rep.get("settle_s_p90") is not None:
            m.preset_settle_s = rep["settle_s_p90"]
        units = r.get("units", {})
        for axis in ("pan", "tilt"):
            if units.get(f"{axis}_units_per_degree"):
                setattr(m, f"{axis}_units_per_degree", units[f"{axis}_units_per_degree"])
        return m


def _options_dict(o: BenchOptions) -> dict[str, Any]:
    d = dict(o.__dict__)
    d["absolute_offset"] = list(o.absolute_offset)
    d["skip"] = list(o.skip)
    return d


# --- the human-readable summary -----------------------------------------------------------------------------------


def summarize(report: dict[str, Any]) -> str:
    """A short text answering the brief's questions, ending with the verdict line."""
    lines = []
    cam = report.get("camera", {})
    lines.append(f"PTZ characterization: {cam.get('ip', '?')}:{cam.get('port', '?')}")
    if report.get("aborted"):
        lines.append(f"!! run incomplete: {report['aborted']}")
    home = report.get("home_command", {})
    if home and home.get("status") not in ("acked", "completed"):
        lines.append(f"!! home command not accepted ({home.get('status')}): measured from wherever the camera was")
    for name in ("feedback", "speed_table", "latency", "absolute", "zoom", "repeatability", "units"):
        if isinstance(report.get(name), dict) and "error" in report[name]:
            lines.append(f"!! section {name} failed: {report[name]['error']}")

    fb = report.get("feedback", {})
    if fb:
        at_rest = fb.get("pan_tilt_at_rest", {})
        lines.append("")
        lines.append("1. Feedback")
        lines.append(f"   pan/tilt inquiry at rest: {'yes' if at_rest.get('supported') else 'NO'} "
                     f"(answered {at_rest.get('answered')}, round trip p50 {at_rest.get('round_trip_ms_p50')} ms)")
        lines.append(f"   while moving: {fb.get('while_moving')}; live feedback up to: {fb.get('max_rate_hz')} Hz; "
                     f"position updates at {fb.get('update_rate_hz')} Hz"
                     + (" (STALE CACHE: answers faster than the position refreshes)" if fb.get("refresh_limited") else ""))
        for rate, row in (fb.get("rates") or {}).items():
            lines.append(f"     {rate:>2} Hz: {row.get('answers_hz')} answers/s ({row['answered']}/{row['sent']}), "
                         f"{row.get('updates_hz')} updates/s, cv {row.get('smoothness_cv')}, jumps {row.get('jumps')}: "
                         f"{'ok' if row.get('ok') else 'NOT ok'} ({row.get('why')})")
        lines.append(f"   zoom inquiry at rest: {'yes' if fb.get('zoom_at_rest', {}).get('supported') else 'no'}")

    st = report.get("speed_table", {})
    if st and "pan" in st:
        lines.append("")
        lines.append(f"2. Speed table (raw units/s, method: {st.get('method')})")
        for axis in ("pan", "tilt"):
            rows = st.get(axis, {})
            cells = []
            for step, row in rows.items():
                plus = (row.get("+") or {}).get("rate")
                minus = (row.get("-") or {}).get("rate")
                cells.append(f"{step}:{plus}/{minus}")
            lines.append(f"   {axis} (+/-): " + "  ".join(cells))
            lines.append(f"   {axis}: minimum smooth step {st.get(f'{axis}_min_smooth_step')}, "
                         f"raw value grows with + step: {st.get(f'{axis}_sign') == 1}")
    elif st:
        lines.append("")
        lines.append(f"2. Speed table: {st.get('method', 'skipped')} {st.get('note', '')}")

    lat = report.get("latency", {})
    if lat:
        lines.append("")
        if lat.get("method") == "polling":
            lines.append(f"3. Latency (polled at {lat.get('poll_hz')} Hz, resolution {lat.get('resolution_s')} s): "
                         f"start {lat.get('latency_s_p50')} s, ramp {lat.get('ramp_s_p50')} s, "
                         f"stop {lat.get('stop_latency_s_p50')} s, stop ramp {lat.get('stop_ramp_s_p50')} s")
        elif lat.get("method") == "inquiry_silence":
            lines.append(f"3. Latency (from when inquiry goes silent / answers again): start {lat.get('latency_s_p50')} s, "
                         f"stop to standstill {lat.get('stop_total_s_p50')} s; effective lag {lat.get('effective_lag_s_p50')} s")
        else:
            lines.append(f"3. Latency: effective lag {lat.get('effective_lag_s_p50')} s (from timed moves)")

    ab = report.get("absolute", {})
    if ab and "moves" in ab:
        lines.append("")
        lines.append(f"4. Absolute positioning: {'works' if ab.get('supported') else 'NOT supported'}; "
                     f"relative: {'works' if ab.get('relative', {}).get('supported') else 'no'}")
        for mv in ab.get("moves", []):
            lines.append(f"   to {mv.get('target')} at {mv.get('pan_speed')}/{mv.get('tilt_speed')}: {mv.get('status')}, "
                         f"arrived {mv.get('arrival_s')} s, error {mv.get('error')}, cv {mv.get('smoothness_cv')}, "
                         f"jumps {mv.get('jumps')}")

    z = report.get("zoom", {})
    if z and "inquiry_at_rest" in z:
        lines.append("")
        lines.append(f"5. Zoom: inquiry at rest {'yes' if z.get('inquiry_at_rest') else 'NO'}, "
                     f"while zooming {z.get('inquiry_while_zooming')}, range {z.get('range')}")
        for d in z.get("zoom_direct", []):
            lines.append(f"   zoom_direct {d.get('target')}: {d.get('status')}, arrived {d.get('arrival_s')} s, "
                         f"error {d.get('error')}")
        if z.get("fov_positions"):
            lines.append(f"   FOV screenshot positions: {z['fov_positions']}")

    rep = report.get("repeatability", {})
    if rep and "recalls" in rep:
        lines.append("")
        lines.append(f"6. Repeatability ({rep.get('target')}, {rep.get('recalls')} recalls): "
                     f"spread pan {rep.get('pan_spread')}, tilt {rep.get('tilt_spread')} raw units; "
                     f"settle p90 {rep.get('settle_s_p90')} s")

    u = report.get("units", {})
    if u and "pan_source" in u:
        lines.append("")
        lines.append(f"7. Units: pan {u.get('pan_units_per_degree')} units/deg ({u.get('pan_source')}), "
                     f"tilt {u.get('tilt_units_per_degree')} units/deg ({u.get('tilt_source')})")

    v = report.get("verdict", {})
    lines.append("")
    lines.append(f"VERDICT: Branch {v.get('branch', '?')}: {v.get('reason', '')}")
    return "\n".join(lines) + "\n"


# --- command line -------------------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m zone.tools.characterize", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", required=True, help="camera IP address")
    ap.add_argument("--port", type=int, default=52381, help="VISCA-over-IP UDP port (default 52381)")
    ap.add_argument("--out", required=True, help="output directory (report.json, summary.txt, visca_log.jsonl)")
    ap.add_argument("--timeout", type=float, default=0.2, help="reply timeout per command, s (default 0.2)")
    ap.add_argument("--home-preset", type=int, help="preset (0-based) to use as home; default: VISCA home command")
    ap.add_argument("--repeat-preset", type=int, help="preset (0-based) for the repeatability test; default: home")
    ap.add_argument("--pan-max-step", type=int, help="top pan speed step (default: ask the camera, else 24)")
    ap.add_argument("--tilt-max-step", type=int, help="top tilt speed step (default: ask the camera, else 23)")
    ap.add_argument("--quick", action="store_true", help="measure ~10 speed steps per axis instead of all")
    ap.add_argument("--pan-travel", type=float, default=1200.0, help="raw units per speed measurement (pan)")
    ap.add_argument("--tilt-travel", type=float, default=400.0, help="raw units per speed measurement (tilt)")
    ap.add_argument("--no-fov-pauses", action="store_true", help="don't stop for FOV screenshots")
    ap.add_argument("--pan-units-per-degree", type=float, help="documented raw units per degree (pan)")
    ap.add_argument("--tilt-units-per-degree", type=float, help="documented raw units per degree (tilt)")
    ap.add_argument("--skip", default="", help="comma-separated sections to skip, e.g. zoom,units")
    ap.add_argument("--yes", action="store_true", help="don't ask anything: confirm the move, skip pauses and units")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from zone.ptz.visca import ViscaPtz  # here, so the tests can import this module without sockets

    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clock = SystemClock()
    options = BenchOptions(
        home_preset=args.home_preset, repeat_preset=args.repeat_preset, pan_max_step=args.pan_max_step,
        tilt_max_step=args.tilt_max_step, quick=args.quick, pan_travel=args.pan_travel, tilt_travel=args.tilt_travel,
        fov_pauses=not args.no_fov_pauses and not args.yes, pan_units_per_degree=args.pan_units_per_degree,
        tilt_units_per_degree=args.tilt_units_per_degree,
        skip=tuple(s.strip() for s in args.skip.split(",") if s.strip()),
    )
    ui: Any = ScriptedUI() if args.yes else ConsoleUI()
    if args.yes:
        ui.info = print  # still show progress
    ptz = ViscaPtz(args.ip, args.port, timeout=args.timeout, clock=clock, log_path=str(out / "visca_log.jsonl"))
    try:
        bench = Characterizer(ptz, clock, ui, options, out, camera_info={"ip": args.ip, "port": args.port})
        report = bench.run()
        report["driver_stats"] = dict(ptz.stats)
        bench.save()
    finally:
        ptz.close()
    text = summarize(report)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"report: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
