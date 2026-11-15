"""`ZoneCamera`: the zone's PTZ as the studio sees it through the agent API (contract §3, §5, §6).

It shares the zone's one PTZ driver (`ViscaPtz`, or the simulator) with the control loop: no second VISCA client.

* **Velocities.** The contract's -1..1 becomes a signed speed step, 1 = the top step: from the WP-Z1 report's speed
  table, or the driver's own limit without a report. Same rounding as the studio's own VISCA driver
  (baja-studio visca_packets.speed_step), so a stick feels the same through the agent and through the studio's
  direct fallback (if the studio's camera entry has the same top steps). Every velocity goes out with the driver's
  TTL = the request's ttl_ms: the driver's dead-man stops the camera if the studio link is lost.
* **Position.** Degrees from home and zoom 0..1, only when the report measured units per degree and the camera
  answers position inquiries while moving (Branch A). Otherwise `null` and `has_position_feedback: false`.
  It's polled at `agent.position_hz` (2 Hz) when the zone isn't driving the camera; while AUTO drives it, the
  controller's own inquiry answers are reused, so the agent adds nothing on the wire then.
* **Device status.** Offline when the PTZ stopped answering: 3 failed exchanges in a row, counting the app's health
  probe, the agent's own inquiries and the studio's commands; one answer brings it back. Commands then answer
  503 device_offline (a stop is still sent, just in case).
* **Image controls and crop** are not in this package: `image_controls: []`, `/crop` is `400 not_supported`.
"""
import math
import threading
from typing import Callable, Optional

from zone import director as D
from zone.agent.contract import (
    AUTO, AUTO_PAUSED, DEVICE_OFFLINE, DEVICE_OFFLINE_ERROR, DEVICE_ONLINE, IDLE, LOST, NOT_SUPPORTED, OFFLINE, OK,
    PRESETS_CHANGED, SEARCHING, TRACKER_ACQUIRED, TRACKER_LOST, TRACKING, AgentError,
)
from zone.agent.presets import PresetBook
from zone.controller import answered
from zone.ptz.sim import DEFAULT_UNITS_PER_DEGREE

ZOOM_RANGE_DEFAULT = (0.0, 16384.0)     # VISCA's usual 0..0x4000, when no report says otherwise
LOCAL_SOURCE = "local:zone-page"        # last_command_source for the laptop page's own buttons
OFFLINE_AFTER_FAILURES = 3              # failed exchanges in a row before the PTZ counts as offline


def speed_step(velocity: float, top_step: int) -> int:
    """A normalised velocity -1..1 -> a signed speed step -top..top (0 = stop). |v| = 1 is the top step, the smallest
    non-zero |v| is step 1. Rounded half up, like the studio's own VISCA driver, so equal stick steps are equal speed
    steps."""
    if velocity == 0:
        return 0
    step = 1 + math.floor(abs(velocity) * (top_step - 1) + 0.5)
    step = max(1, min(int(top_step), step))
    return step if velocity > 0 else -step


class ZoneCamera:
    """The one camera of this agent. The routes (server.py) call these methods; every refusal is an AgentError."""

    def __init__(self, zapp, camera_id: str, label: str, presets: PresetBook,
                 emit: Callable[[str, dict], None], position_hz: float = 2.0) -> None:
        self.zapp = zapp
        self.ptz, self.clock, self.runtime = zapp.ptz, zapp.clock, zapp.runtime
        self.id, self.label = camera_id, label
        self.presets = presets
        self.emit = emit                      # emit(event, detail): a contract event about this camera
        self.arbiter = None                   # the ModeArbiter, set by the service once it exists
        self.position_period_s = 1.0 / max(0.1, float(position_hz))
        self.lock = threading.RLock()
        report = zapp.settings.report
        self.model = report.model
        model_raw = (report.raw or {}).get("model", {}) if report.path else {}
        # position feedback: the report measured units per degree, and inquiry works while moving (Branch A)
        self.has_position_feedback = bool(
            report.path and zapp.settings.controller.branch == "A"
            and model_raw.get("pan_units_per_degree") and model_raw.get("tilt_units_per_degree"))
        self.pan_upd = float(self.model.pan_units_per_degree or DEFAULT_UNITS_PER_DEGREE)
        self.tilt_upd = float(self.model.tilt_units_per_degree or DEFAULT_UNITS_PER_DEGREE)
        self.zoom_range = tuple(self.model.zoom_limits) if report.path else ZOOM_RANGE_DEFAULT
        # top speed steps: the report's speed tables, or the driver's limits without a report
        if report.path:
            self.pan_top, self.tilt_top = self.model.pan_step_max, self.model.tilt_step_max
        else:
            self.pan_top = int(getattr(self.ptz, "pan_step_max", None) or self.model.pan_step_max)
            self.tilt_top = int(getattr(self.ptz, "tilt_step_max", None) or self.model.tilt_step_max)
        self.zoom_top = self.model.zoom_step_max
        self.driver = "simulated" if zapp.cfg["ptz"]["driver"] == "sim" else "visca-ip"
        # what the studio is doing with the camera
        self.velocity = (0, 0, 0)             # the studio's speed steps now (pan, tilt, zoom)
        self.velocity_until = -math.inf       # ... until this monotonic time (its ttl)
        self.recall_until = -math.inf         # a studio preset recall is moving the camera until about then
        self.last_command_source: Optional[str] = None
        self.last_command_ts: Optional[str] = None
        # device status: None = unknown (nothing answered or failed yet), True = answering, False = offline
        self.reachable: Optional[bool] = None
        self._failures = 0                    # failed exchanges in a row
        self._seen_probe = None               # the health probe we already took into account
        self._position_raw = None             # (pan, tilt, zoom) raw units of the last answer
        self._position_t = -math.inf          # when we last asked
        self._presets_signature = presets.signature()

    # =============================================================================================== description
    def info(self) -> dict:
        """This camera's entry in GET /agent/info. No PTZ call: /agent/info must answer within 300 ms."""
        m = self.model

        def top_rate(table, step):
            return float(table.get(step) or max(table.values()))
        return {
            "id": self.id, "label": self.label, "kind": "ptz", "driver": self.driver,
            "has_position_feedback": self.has_position_feedback, "has_zoom": True, "supports_auto": True,
            "image_controls": [], "presets": self.presets.listing(),
            "limits": {
                "pan_speed_max": round(top_rate(m.pan_speed_table, self.pan_top) / self.pan_upd, 2),
                "tilt_speed_max": round(top_rate(m.tilt_speed_table, self.tilt_top) / self.tilt_upd, 2),
                "zoom_speed_max": round(top_rate(m.zoom_speed_table, self.zoom_top)
                                        / max(1.0, self.zoom_range[1] - self.zoom_range[0]), 3),
                "preset_max": self.presets.preset_max,
            },
        }

    # =============================================================================================== device status
    def online(self) -> bool:
        return self.reachable is not False

    def note_reachable(self, value: Optional[bool]) -> None:
        """The latest word on whether the PTZ answers. One answer is enough to be online again; it takes
        `OFFLINE_AFTER_FAILURES` failures in a row to be offline (UDP loses a packet now and then: one lost reply
        must not lock the studio out). Sends device_offline / device_online on a change."""
        if value is None:
            return
        with self.lock:
            was = self.online()
            if value:
                self._failures = 0
                self.reachable = True
            else:
                self._failures += 1
                if self._failures >= OFFLINE_AFTER_FAILURES:
                    self.reachable = False
            now = self.online()
        if was and not now:
            self.emit(DEVICE_OFFLINE, {})
        elif now and not was:
            self.emit(DEVICE_ONLINE, {})

    def _note_result(self, result) -> bool:
        """Note what a PTZ call's result says about the device. -> did the camera answer?"""
        ok = answered(result)
        self.note_reachable(ok)
        return ok

    def require_online(self) -> None:
        if not self.online():
            raise AgentError(DEVICE_OFFLINE_ERROR, "the zone's PTZ is not answering (network or power); the zone "
                             "laptop retries every few seconds")

    # =============================================================================================== motion
    def move(self, pan: float, tilt: float, zoom: float, ttl_ms: int, source: str) -> dict:
        """POST /move. All zeros is a stop. -> {"accepted": true, "mode": <mode after>}."""
        steps = (speed_step(pan, self.pan_top), speed_step(tilt, self.tilt_top), speed_step(zoom, self.zoom_top))
        if steps == (0, 0, 0):
            return self.stop(source)
        self.require_online()
        now = self.clock.monotonic()
        ttl = ttl_ms / 1000.0
        self.arbiter.studio_motion(source, motion_ends_at=now + ttl)   # pauses AUTO first: the zone goes quiet
        with self.lock:
            previous = self.velocity if now < self.velocity_until else (0, 0, 0)
            self.velocity, self.velocity_until = steps, now + ttl
            self._commanded_by(source)
        results = []
        if steps[0] or steps[1] or previous[0] or previous[1]:
            results.append(self.ptz.pan_tilt_velocity(steps[0], steps[1], ttl=ttl))
        if steps[2] or previous[2]:
            results.append(self.ptz.zoom_velocity(steps[2], ttl=ttl) if steps[2] else self.ptz.zoom_stop())
        if not all(self._note_result(r) for r in results):
            raise AgentError(DEVICE_OFFLINE_ERROR, "the zone's PTZ did not answer the move")
        return self._reply()

    def stop(self, source: str) -> dict:
        """POST /stop: every axis, now. Never refused because of the mode (a pause in AUTO, like any studio motion)."""
        now = self.clock.monotonic()
        self.arbiter.studio_motion(source, motion_ends_at=now)
        with self.lock:
            self.velocity, self.velocity_until, self.recall_until = (0, 0, 0), -math.inf, -math.inf
            self._commanded_by(source)
        results = []
        for fn in (self.ptz.stop, self.ptz.zoom_stop):
            try:
                results.append(fn())
            except Exception:  # noqa: BLE001 (a dead PTZ must not keep us from trying the other axis)
                results.append(None)
        answers = [self._note_result(r) for r in results if r is not None]
        if not answers or not all(answers):
            raise AgentError(DEVICE_OFFLINE_ERROR, "the zone's PTZ did not answer the stop")
        return self._reply()

    def local_stop(self) -> None:
        """The laptop page's PARAR has stopped the camera: forget any studio velocity."""
        with self.lock:
            self.velocity, self.velocity_until, self.recall_until = (0, 0, 0), -math.inf, -math.inf
            self._commanded_by(LOCAL_SOURCE)

    # =============================================================================================== presets
    def recall_preset(self, n: int, source: str) -> dict:
        self.presets.check_recall(n)
        self.require_online()
        now = self.clock.monotonic()
        self.arbiter.studio_motion(source, motion_ends_at=now)
        with self.lock:
            self.velocity, self.velocity_until = (0, 0, 0), -math.inf
            self.recall_until = now + float(self.zapp.settings.director.settle_s)
            self._commanded_by(source)
        if not self._note_result(self.ptz.preset_recall(int(n))):
            raise AgentError(DEVICE_OFFLINE_ERROR, "the zone's PTZ did not answer the preset recall")
        return self._reply()

    def store_preset(self, n: int, source: str) -> dict:
        self.presets.check_store(n)
        self.require_online()
        result = self.ptz.preset_store(int(n))
        if not self._note_result(result):
            raise AgentError(DEVICE_OFFLINE_ERROR, "the zone's PTZ did not answer the preset store")
        if not getattr(result, "ok", True):
            raise AgentError(DEVICE_OFFLINE_ERROR, "the zone's PTZ refused to store preset %d" % n)
        preset = self.presets.stored(n)
        self._presets_signature = self.presets.signature()
        self.emit(PRESETS_CHANGED, {"n": n})
        return preset

    def label_preset(self, n: int, label: str) -> dict:
        preset = self.presets.rename(n, label)
        self._presets_signature = self.presets.signature()
        self.emit(PRESETS_CHANGED, {"n": n})
        return preset

    def check_presets_changed(self) -> None:
        """The zone may have been re-taught on the laptop page (keyframes added, moved, deleted): tell the studio."""
        signature = self.presets.signature()
        if signature != self._presets_signature:
            old = dict(self._presets_signature)
            self._presets_signature = signature
            changed = sorted(set(old.items()) ^ set(signature))
            self.emit(PRESETS_CHANGED, {"n": changed[0][0] if changed else None})

    # =============================================================================================== position
    def position(self) -> Optional[dict]:
        """GET /position: degrees from home + zoom 0..1, or None without position feedback."""
        if not self.has_position_feedback:
            return None
        self.require_online()
        self.refresh_position()
        return self._position_contract()

    def refresh_position(self, force: bool = False) -> None:
        """Ask the PTZ where it is, at most position_hz times a second (or now, with `force`).

        While AUTO drives the camera (tracking a car, going back to the entry), the controller's own inquiry answers
        are used instead: an inquiry from here would compete with its velocity commands for the driver."""
        if not self.has_position_feedback:
            return
        now = self.clock.monotonic()
        if not force and now - self._position_t < self.position_period_s:
            return
        self._position_t = now
        director = self.runtime.director
        if director.mode == D.AUTO and director.sub in (D.TRACKING, D.RETURNING):
            pose = self.runtime.controller.pose
            if pose.inq is not None and now - pose.inq[2] <= 1.0:
                zoom = pose.zoom if pose.zoom is not None else (self._position_raw or (0, 0, 0))[2]
                self._position_raw = (pose.inq[0], pose.inq[1], zoom)
            return
        try:
            pan_tilt = self.ptz.inquire_pan_tilt()
            zoom = self.ptz.inquire_zoom() if pan_tilt is not None else None
        except Exception:  # noqa: BLE001 (no answer: the device status says so)
            pan_tilt = zoom = None
        self.note_reachable(pan_tilt is not None)   # Branch A: this camera always answers inquiries
        if pan_tilt is not None:
            if zoom is None:
                zoom = (self._position_raw or (0, 0, self.zoom_range[0]))[2]
            self._position_raw = (pan_tilt[0], pan_tilt[1], zoom)

    def _position_contract(self) -> Optional[dict]:
        if not self.has_position_feedback or self._position_raw is None:
            return None
        pan, tilt, zoom = self._position_raw
        lo, hi = self.zoom_range
        return {"pan": round(pan * self.model.pan_sign / self.pan_upd, 2),
                "tilt": round(tilt * self.model.tilt_sign / self.tilt_upd, 2),
                "zoom": round(min(1.0, max(0.0, (zoom - lo) / max(1.0, hi - lo))), 3)}

    # =============================================================================================== state
    def moving(self) -> bool:
        """Is any axis being driven: by the studio (a velocity within its ttl, a preset recall settling) or by the
        zone (AUTO following a car or going back to the entry)?"""
        now = self.clock.monotonic()
        with self.lock:
            studio = (self.velocity != (0, 0, 0) and now < self.velocity_until) or now < self.recall_until
        director = self.runtime.director
        return studio or (director.mode == D.AUTO and director.sub in (D.TRACKING, D.RETURNING))

    def tracker(self) -> dict:
        """The contract's tracker state, from the director: idle outside the auto modes; searching while waiting at
        the entry (or paused, or going back after a car left the zone); tracking a car; lost when the car was
        dropped (lost, stopped, off the line, diverged) and the camera goes back to the entry."""
        mode = self.arbiter.mode
        if mode not in (AUTO, AUTO_PAUSED):
            return {"state": IDLE, "target_id": None, "confidence": None}
        director = self.runtime.director
        if mode == AUTO and director.mode == D.AUTO and director.sub == D.TRACKING and director.locked_id is not None:
            return {"state": TRACKING, "target_id": target_id(director.locked_id), "confidence": None}
        if mode == AUTO and director.sub == D.RETURNING and director.last_drop_reason not in ("", "exit"):
            return {"state": LOST, "target_id": None, "confidence": None}
        return {"state": SEARCHING, "target_id": None, "confidence": None}

    def state(self) -> dict:
        """This camera's CameraState for the WebSocket push. Reads cached values only (no PTZ call), so a push is
        never late because the camera is slow to answer."""
        return {"status": OK if self.online() else OFFLINE, "mode": self.arbiter.mode, "tracker": self.tracker(),
                "position": self._position_contract(), "moving": self.moving(),
                "last_command_source": self.last_command_source, "last_command_ts": self.last_command_ts}

    def on_zone_event(self, ev: dict) -> None:
        """Director events (via the app's event hook): a car locked = tracker_acquired, let go = tracker_lost."""
        if ev.get("type") != "director" or self.arbiter is None or self.arbiter.mode != AUTO:
            return
        if ev.get("event") == "lock":
            self.emit(TRACKER_ACQUIRED, {"target_id": target_id(ev.get("track_id"))})
        elif ev.get("event") == "release":
            self.emit(TRACKER_LOST, {})

    def tick(self) -> None:
        """Several times a second (the agent's background thread): device status from the app's health probe (every
        5 s; while AUTO drives the camera it reports the controller's last command), the position poll, preset changes
        from a re-teach."""
        health = self.zapp.health
        probe = health.get("ptz_probe_wall")
        if probe is not None and probe != self._seen_probe:
            self._seen_probe = probe
            self.note_reachable(health.get("ptz_reachable"))
        self.refresh_position()
        self.check_presets_changed()

    def resync(self, source: str) -> dict:
        """POST /resync: the studio drove the camera directly while we were unreachable. Re-read where it is (and
        whether it answers). The zone's own pose estimate is reset by the preset recall that starts every AUTO."""
        try:
            answer = self.ptz.inquire_zoom()
            if answer is None:
                answer = self.ptz.inquire_pan_tilt()
        except Exception:  # noqa: BLE001
            answer = None
        if answer is not None:
            self.note_reachable(True)
        self.refresh_position(force=True)
        self.require_online()
        return self._reply()

    # =============================================================================================== not supported
    def not_supported(self, what: str) -> None:
        raise AgentError(NOT_SUPPORTED, "%s is a PTZ camera: %s is not supported" % (self.id, what))

    # =============================================================================================== internals
    def _commanded_by(self, source: str) -> None:
        self.last_command_source = source
        self.last_command_ts = iso_now(self.clock)

    def _reply(self) -> dict:
        return {"accepted": True, "mode": self.arbiter.mode}


def target_id(track_id) -> str:
    """The tracker's id of a locked car on the wire: "track-12" (a ByteTrack id, not the car's number)."""
    return "track-%s" % track_id


def iso_now(clock) -> str:
    """The clock's wall time as an ISO 8601 UTC string (contract §2.3)."""
    import datetime as dt
    return dt.datetime.fromtimestamp(clock.wall(), dt.timezone.utc).isoformat()
