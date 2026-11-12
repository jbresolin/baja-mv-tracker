"""A simulated PTZ: the `Ptz` interface on top of a small physics model, driven by an injected clock.

Why it exists: the controller (WP-Z2) must be testable without a camera, without a car, and without waiting in real
time. `SimulatedPtz` behaves like the real camera as far as the bench measured it. The numbers come from the "model"
section of a bench report (`PtzModel.from_report`), or from defaults that look like a typical VISCA PTZ.

What it models, per axis (pan, tilt, zoom):
- **Speed table:** speed step → device units per second (separately per direction if the camera is asymmetric).
- **Command latency:** a command takes effect `command_latency_s` after it is sent (`stop_latency_s` for stops).
- **Ramp:** a change of velocity is linear over `ramp_s` (`stop_ramp_s` when slowing to zero).
- **Travel limits:** the axis stops at its limit.
- **Inquiry behaviour:** "always" answers, "at_rest" answers only when stopped, "never", or "frozen_while_moving"
  (answers, but with the last at-rest value while moving). An optional `inquiry_latency_s` makes answers that old.
- **Position moves** (absolute, relative, presets, zoom direct, home): constant speed to the target, no ramp.
- **TTL dead-man:** the same semantics as the real driver.

How time works: the sim never reads real time. Every call first moves the model forward to `clock.monotonic()`,
integrating in small fixed steps (`STEP_S`), so results are exact and repeatable. Call `tick()` to bring it up to date
without sending a command.
"""
from __future__ import annotations

import bisect
import json
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

from zone.clock import Clock
from zone.ptz.interface import ANSWERED, COMPLETED, ERROR, TIMEOUT, CommandResult

STEP_S = 0.005  # integration step (seconds of simulated time)
HISTORY_S = 2.0  # how much pose history is kept for `inquiry_latency_s`
INQUIRY_MODES = ("always", "at_rest", "never", "frozen_while_moving")
DEFAULT_UNITS_PER_DEGREE = 1 / 0.075  # the usual VISCA value: 0.075 degrees per raw unit


def _default_pan_table() -> dict[int, float]:
    # Roughly geometric, as speed tables usually are: ~1 deg/s at step 1 up to ~80 deg/s at step 24.
    return {s: round(1.0 * 1.21 ** (s - 1) * DEFAULT_UNITS_PER_DEGREE, 1) for s in range(1, 25)}


def _default_tilt_table() -> dict[int, float]:
    return {s: round(0.8 * 1.2 ** (s - 1) * DEFAULT_UNITS_PER_DEGREE, 1) for s in range(1, 24)}


def _default_zoom_table() -> dict[int, float]:
    # Full range (0..0x4000) in ~3 s at step 7, ~20 s at step 1.
    return {s: round(16384 / (20.0 / 1.37 ** (s - 1)), 1) for s in range(1, 8)}


@dataclass
class PtzModel:
    """Everything the simulator needs to know about one camera. A bench report's "model" section is exactly
    `PtzModel.to_dict()`; fields the report doesn't have keep these defaults."""

    pan_speed_table: dict[int, float] = field(default_factory=_default_pan_table)  # |step| -> units/s
    tilt_speed_table: dict[int, float] = field(default_factory=_default_tilt_table)
    pan_speed_table_negative: dict[int, float] | None = None  # set when left differs from right
    tilt_speed_table_negative: dict[int, float] | None = None  # set when down differs from up
    zoom_speed_table: dict[int, float] = field(default_factory=_default_zoom_table)
    command_latency_s: float = 0.10
    stop_latency_s: float = 0.10
    ramp_s: float = 0.20
    stop_ramp_s: float = 0.15
    pan_limits: tuple[float, float] = (-2267.0, 2267.0)  # +/-170 degrees at 0.075 deg/unit
    tilt_limits: tuple[float, float] = (-400.0, 1200.0)  # -30 .. +90 degrees
    zoom_limits: tuple[float, float] = (0.0, 16384.0)
    inquiry_mode: str = "always"  # pan/tilt inquiry: one of INQUIRY_MODES
    zoom_inquiry_mode: str = "always"
    inquiry_latency_s: float = 0.0
    inquiry_poll_hz: float | None = 20.0  # highest rate the bench found reliable (None: inquiry unusable)
    preset_move_speed: float = 1000.0  # units/s for preset recalls and home
    preset_settle_s: float = 2.0  # recall -> at rest, as measured by the bench
    pan_min_smooth_step: int = 1
    tilt_min_smooth_step: int = 1
    pan_units_per_degree: float | None = DEFAULT_UNITS_PER_DEGREE
    tilt_units_per_degree: float | None = DEFAULT_UNITS_PER_DEGREE
    pan_sign: int = 1  # +1: the reported raw pan grows when turning right (positive step)
    tilt_sign: int = 1  # +1: the reported raw tilt grows when tilting up
    supports_absolute: bool = True
    supports_relative: bool = True
    supports_zoom_direct: bool = True

    def __post_init__(self) -> None:
        for name in ("inquiry_mode", "zoom_inquiry_mode"):
            if getattr(self, name) not in INQUIRY_MODES:
                raise ValueError(f"{name} must be one of {INQUIRY_MODES}, not {getattr(self, name)!r}")

    @property
    def pan_step_max(self) -> int:
        return max(self.pan_speed_table)

    @property
    def tilt_step_max(self) -> int:
        return max(self.tilt_speed_table)

    @property
    def zoom_step_max(self) -> int:
        return max(self.zoom_speed_table)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: speed-table keys become strings, tuples become lists."""
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = {str(k): v for k, v in value.items()}
            elif isinstance(value, tuple):
                d[key] = list(value)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PtzModel:
        """Build from `to_dict()` output. Unknown keys are ignored and missing keys keep their defaults."""
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                continue
            if key.endswith("_table") or key.endswith("_table_negative"):
                value = None if value is None else {int(k): float(v) for k, v in value.items()}
            elif key.endswith("_limits"):
                value = (float(value[0]), float(value[1]))
            kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_report(cls, report: str | Path | dict[str, Any]) -> PtzModel:
        """Read the "model" section of a bench report (a path to report.json, or the parsed dict)."""
        if not isinstance(report, dict):
            report = json.loads(Path(report).read_text(encoding="utf-8"))
        return cls.from_dict(report.get("model", report))


def _rate(table: dict[int, float], step: int) -> float:
    """units/s for |step|. A step missing from the table (a quick bench run measures only some) is interpolated
    linearly between its neighbours; steps beyond either end act like that end."""
    magnitude = abs(step)
    if magnitude == 0:
        return 0.0
    if magnitude in table:
        return table[magnitude]
    keys = sorted(table)
    if magnitude < keys[0]:
        return table[keys[0]]
    if magnitude > keys[-1]:
        return table[keys[-1]]
    upper = bisect.bisect_left(keys, magnitude)
    low, high = keys[upper - 1], keys[upper]
    return table[low] + (table[high] - table[low]) * (magnitude - low) / (high - low)


class _Axis:
    """One axis: its position and how it is being driven (a velocity with a ramp, or a move to a target)."""

    def __init__(self, position: float, limits: tuple[float, float]) -> None:
        self.position = position
        self.low, self.high = limits
        self.target: float | None = None  # set during a position move
        self.target_speed = 0.0
        self.ramp_from = 0.0  # velocity ramp: from this velocity...
        self.ramp_to = 0.0  # ...to this one...
        self.ramp_start = 0.0  # ...starting at this time...
        self.ramp_length = 0.0  # ...taking this long.

    def velocity(self, t: float) -> float:
        if self.target is not None:
            return 0.0  # position moves are handled in `advance`
        if self.ramp_length <= 0 or t >= self.ramp_start + self.ramp_length:
            return self.ramp_to
        fraction = max(0.0, (t - self.ramp_start) / self.ramp_length)
        return self.ramp_from + (self.ramp_to - self.ramp_from) * fraction

    def set_velocity(self, t: float, velocity: float, ramp: float) -> None:
        current = self.velocity(t)
        self.target = None
        self.ramp_from, self.ramp_to, self.ramp_start, self.ramp_length = current, velocity, t, ramp

    def move_to(self, target: float, speed: float) -> None:
        self.target = min(self.high, max(self.low, target))
        self.target_speed = abs(speed)
        self.ramp_from = self.ramp_to = 0.0

    def is_moving(self, t: float) -> bool:
        if self.target is not None:
            return abs(self.target - self.position) > 1e-9
        return abs(self.velocity(t)) > 1e-9

    def advance(self, t: float, dt: float) -> None:
        """Integrate from t to t + dt."""
        if self.target is not None:
            remaining = self.target - self.position
            travel = self.target_speed * dt
            if abs(remaining) <= travel:
                self.position = self.target
                self.target = None
                self.ramp_from = self.ramp_to = 0.0
            else:
                self.position += travel if remaining > 0 else -travel
            return
        # Midpoint velocity: exact for a linear ramp.
        self.position += self.velocity(t + dt / 2) * dt
        if self.position <= self.low or self.position >= self.high:
            self.position = min(self.high, max(self.low, self.position))
            self.ramp_from = self.ramp_to = 0.0  # stalled at the end stop


class SimulatedPtz:
    """A simulated camera implementing `zone.ptz.interface.Ptz`. See the module docstring."""

    def __init__(self, model: PtzModel | None = None, clock: Clock | None = None,
                 pan: float = 0.0, tilt: float = 0.0, zoom: float = 0.0) -> None:
        if clock is None:
            raise ValueError("SimulatedPtz needs a clock (a FakeClock in tests, SystemClock to run live)")
        self.model = model or PtzModel()
        self.clock = clock
        m = self.model
        self.pan = _Axis(pan, m.pan_limits)
        self.tilt = _Axis(tilt, m.tilt_limits)
        self.zoom = _Axis(zoom, m.zoom_limits)
        self._now = clock.monotonic()
        self._events: list[tuple[float, int, Callable[[float], None]]] = []  # (time, order, action), sorted
        self._event_order = 0
        self._pan_tilt_deadline: float | None = None
        self._zoom_deadline: float | None = None
        self._presets: dict[int, tuple[float, float, float]] = {}
        self._history: deque[tuple[float, float, float, float]] = deque()  # (t, pan, tilt, zoom)
        self._frozen_pan_tilt: tuple[int, int] = (round(pan), round(tilt))
        self._frozen_zoom: int = round(zoom)
        self.last_inquiry: CommandResult | None = None
        self.commands: list[tuple[float, str, tuple[Any, ...]]] = []  # what was sent, for tests

    # --- time --------------------------------------------------------------------------------------------------

    def tick(self) -> None:
        """Bring the model up to `clock.monotonic()`."""
        target = self.clock.monotonic()
        while True:
            next_event = self._events[0][0] if self._events else None
            deadlines = [d for d in (self._pan_tilt_deadline, self._zoom_deadline) if d is not None]
            next_deadline = min(deadlines) if deadlines else None
            candidates = [t for t in (next_event, next_deadline) if t is not None and t <= target]
            if not candidates:
                break
            when = min(candidates)
            self._integrate_to(when)
            if next_deadline is not None and when == next_deadline:
                self._fire_deadlines(when)
            else:
                _, _, action = self._events.pop(0)
                action(when)
        self._integrate_to(target)

    def _integrate_to(self, t: float) -> None:
        while self._now < t - 1e-12:
            dt = min(STEP_S, t - self._now)
            for axis in (self.pan, self.tilt, self.zoom):
                axis.advance(self._now, dt)
            self._now += dt
            self._record()
        self._now = max(self._now, t)

    def _record(self) -> None:
        self._history.append((self._now, self.pan.position, self.tilt.position, self.zoom.position))
        while self._history and self._history[0][0] < self._now - HISTORY_S:
            self._history.popleft()
        if not self._pan_tilt_moving():
            self._frozen_pan_tilt = self._reported_pan_tilt(self.pan.position, self.tilt.position)
        if not self.zoom.is_moving(self._now):
            self._frozen_zoom = round(self.zoom.position)

    def _schedule(self, delay: float, action: Callable[[float], None]) -> None:
        when = self._now + delay
        self._event_order += 1
        bisect.insort(self._events, (when, self._event_order, action), key=lambda e: (e[0], e[1]))

    def _fire_deadlines(self, t: float) -> None:
        if self._pan_tilt_deadline is not None and t >= self._pan_tilt_deadline:
            self._pan_tilt_deadline = None
            self._schedule(self.model.stop_latency_s, self._apply_pan_tilt_stop)
        if self._zoom_deadline is not None and t >= self._zoom_deadline:
            self._zoom_deadline = None
            self._schedule(self.model.stop_latency_s, self._apply_zoom_stop)

    def _done(self, name: str, *args: Any) -> CommandResult:
        self.commands.append((self._now, name, args))
        return CommandResult(COMPLETED, sent_at=self._now, replied_at=self._now)

    # --- what each command does when it takes effect ---------------------------------------------------------

    def _apply_pan_tilt_velocity(self, pan_step: int, tilt_step: int) -> Callable[[float], None]:
        m = self.model

        def apply(t: float) -> None:
            pan_table = m.pan_speed_table_negative if (pan_step < 0 and m.pan_speed_table_negative) else m.pan_speed_table
            tilt_table = (m.tilt_speed_table_negative if (tilt_step < 0 and m.tilt_speed_table_negative)
                          else m.tilt_speed_table)
            for axis, step, table in ((self.pan, pan_step, pan_table), (self.tilt, tilt_step, tilt_table)):
                velocity = _rate(table, step) * (1 if step > 0 else -1)
                axis.set_velocity(t, velocity, m.stop_ramp_s if step == 0 else m.ramp_s)
        return apply

    def _apply_pan_tilt_stop(self, t: float) -> None:
        self.pan.set_velocity(t, 0.0, self.model.stop_ramp_s)
        self.tilt.set_velocity(t, 0.0, self.model.stop_ramp_s)

    def _apply_zoom_stop(self, t: float) -> None:
        self.zoom.set_velocity(t, 0.0, self.model.stop_ramp_s)

    # --- the Ptz interface -----------------------------------------------------------------------------------

    def pan_tilt_velocity(self, pan_step: int, tilt_step: int, ttl: float | None = None) -> CommandResult:
        self.tick()
        moving = pan_step != 0 or tilt_step != 0
        if moving:
            self._schedule(self.model.command_latency_s, self._apply_pan_tilt_velocity(pan_step, tilt_step))
        else:
            self._schedule(self.model.stop_latency_s, self._apply_pan_tilt_stop)
        self._pan_tilt_deadline = self._now + ttl if (moving and ttl is not None) else None
        return self._done("pan_tilt_velocity", pan_step, tilt_step, ttl)

    def stop(self) -> CommandResult:
        self.tick()
        self._pan_tilt_deadline = None
        self._schedule(self.model.stop_latency_s, self._apply_pan_tilt_stop)
        return self._done("stop")

    def zoom_velocity(self, step: int, ttl: float | None = None) -> CommandResult:
        self.tick()
        if step == 0:
            return self.zoom_stop()
        velocity = _rate(self.model.zoom_speed_table, step) * (1 if step > 0 else -1)
        self._schedule(self.model.command_latency_s, lambda t: self.zoom.set_velocity(t, velocity, self.model.ramp_s))
        self._zoom_deadline = self._now + ttl if ttl is not None else None
        return self._done("zoom_velocity", step, ttl)

    def zoom_stop(self) -> CommandResult:
        self.tick()
        self._zoom_deadline = None
        self._schedule(self.model.stop_latency_s, self._apply_zoom_stop)
        return self._done("zoom_stop")

    def zoom_direct(self, position: int) -> CommandResult:
        self.tick()
        if not self.model.supports_zoom_direct:
            return CommandResult(ERROR, sent_at=self._now, replied_at=self._now, error_code=0x02)
        self._zoom_deadline = None
        speed = self.model.zoom_speed_table[self.model.zoom_step_max]
        self._schedule(self.model.command_latency_s, lambda t: self.zoom.move_to(position, speed))
        return self._done("zoom_direct", position)

    def absolute_position(self, pan: int, tilt: int, pan_speed: int, tilt_speed: int) -> CommandResult:
        self.tick()
        if not self.model.supports_absolute:
            return CommandResult(ERROR, sent_at=self._now, replied_at=self._now, error_code=0x02)
        self._move_to(pan, tilt, pan_speed, tilt_speed)
        return self._done("absolute_position", pan, tilt, pan_speed, tilt_speed)

    def relative_position(self, pan: int, tilt: int, pan_speed: int, tilt_speed: int) -> CommandResult:
        self.tick()
        if not self.model.supports_relative:
            return CommandResult(ERROR, sent_at=self._now, replied_at=self._now, error_code=0x02)
        reported = self._reported_pan_tilt(self.pan.position, self.tilt.position)
        self._move_to(reported[0] + pan, reported[1] + tilt, pan_speed, tilt_speed)
        return self._done("relative_position", pan, tilt, pan_speed, tilt_speed)

    def _move_to(self, reported_pan: float, reported_tilt: float, pan_speed: int, tilt_speed: int) -> None:
        """Start a position move to a *reported* (raw) pose, converting to the internal frame."""
        m = self.model
        self._pan_tilt_deadline = None
        pan_target = reported_pan * m.pan_sign
        tilt_target = reported_tilt * m.tilt_sign
        pan_rate = _rate(m.pan_speed_table, pan_speed)
        tilt_rate = _rate(m.tilt_speed_table, tilt_speed)

        def apply(t: float) -> None:
            self.pan.move_to(pan_target, pan_rate)
            self.tilt.move_to(tilt_target, tilt_rate)
        self._schedule(m.command_latency_s, apply)

    def inquire_pan_tilt(self) -> tuple[int, int] | None:
        self.tick()
        mode = self.model.inquiry_mode
        value: tuple[int, int] | None
        if mode == "never" or (mode == "at_rest" and self._pan_tilt_moving()):
            value = None
        elif mode == "frozen_while_moving":
            value = self._frozen_pan_tilt
        else:
            _, pan, tilt, _ = self._pose_at(self._now - self.model.inquiry_latency_s)
            value = self._reported_pan_tilt(pan, tilt)
        self.last_inquiry = self._inquiry_result(value)
        return value

    def inquire_zoom(self) -> int | None:
        self.tick()
        mode = self.model.zoom_inquiry_mode
        value: int | None
        if mode == "never" or (mode == "at_rest" and self.zoom.is_moving(self._now)):
            value = None
        elif mode == "frozen_while_moving":
            value = self._frozen_zoom
        else:
            value = round(self._pose_at(self._now - self.model.inquiry_latency_s)[3])
        self.last_inquiry = self._inquiry_result(value)
        return value

    def preset_recall(self, n: int) -> CommandResult:
        self.tick()
        # A preset never stored behaves like the factory default: home, fully wide.
        pan, tilt, zoom = self._presets.get(n, (0.0, 0.0, self.model.zoom_limits[0]))
        self._pan_tilt_deadline = None
        self._zoom_deadline = None
        speed = self.model.preset_move_speed
        zoom_speed = self.model.zoom_speed_table[self.model.zoom_step_max]

        def apply(t: float) -> None:
            self.pan.move_to(pan, speed)
            self.tilt.move_to(tilt, speed)
            self.zoom.move_to(zoom, zoom_speed)
        self._schedule(self.model.command_latency_s, apply)
        return self._done("preset_recall", n)

    def preset_store(self, n: int) -> CommandResult:
        self.tick()
        self._presets[n] = (self.pan.position, self.tilt.position, self.zoom.position)
        return self._done("preset_store", n)

    def define_preset(self, n: int, pan: float, tilt: float, zoom: float) -> None:
        """Store preset `n` at a raw (reported-frame) pose without moving the camera.

        Not part of the `Ptz` interface: a helper for tests and for seeding the simulator from a saved zone.json.
        """
        self._presets[n] = (pan * self.model.pan_sign, tilt * self.model.tilt_sign, zoom)

    def home(self) -> CommandResult:
        self.tick()
        self._pan_tilt_deadline = None
        speed = self.model.preset_move_speed

        def apply(t: float) -> None:
            self.pan.move_to(0.0, speed)
            self.tilt.move_to(0.0, speed)
        self._schedule(self.model.command_latency_s, apply)
        return self._done("home")

    def close(self) -> None:
        pass

    # --- extras (the bench script uses them when present) ----------------------------------------------------

    def inquire_max_speed(self) -> tuple[int, int] | None:
        return self.model.pan_step_max, self.model.tilt_step_max

    # --- for tests: the truth, whatever the inquiry mode says ------------------------------------------------

    def true_pan_tilt(self) -> tuple[float, float]:
        """The real (raw, reported-frame) pan and tilt right now, ignoring the inquiry mode."""
        self.tick()
        return self.pan.position * self.model.pan_sign, self.tilt.position * self.model.tilt_sign

    def true_zoom(self) -> float:
        self.tick()
        return self.zoom.position

    def is_moving(self) -> bool:
        self.tick()
        return self._pan_tilt_moving() or self.zoom.is_moving(self._now)

    # --- internals -------------------------------------------------------------------------------------------

    def _pan_tilt_moving(self) -> bool:
        return self.pan.is_moving(self._now) or self.tilt.is_moving(self._now)

    def _reported_pan_tilt(self, pan: float, tilt: float) -> tuple[int, int]:
        return round(pan * self.model.pan_sign), round(tilt * self.model.tilt_sign)

    def _pose_at(self, t: float) -> tuple[float, float, float, float]:
        if t >= self._now or not self._history:
            return self._now, self.pan.position, self.tilt.position, self.zoom.position
        for entry in reversed(self._history):
            if entry[0] <= t:
                return entry
        return self._history[0]

    def _inquiry_result(self, value: object) -> CommandResult:
        if value is None:
            return CommandResult(TIMEOUT, sent_at=self._now)
        return CommandResult(ANSWERED, sent_at=self._now, replied_at=self._now, value=value)
