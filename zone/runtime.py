"""The zone pipeline, one control cycle per processed spotter frame:

    frame -> spotter (detector on the corridor ROI + ByteTrack) -> progress (s, v per track)
          -> director (which car, which action) -> controller (PTZ commands) -> control log (JSONL)

The same `ZoneRuntime` runs live in the app (zone/app.py, real or simulated PTZ, system clock) and offline in the
replay harness (zone/tools/replay.py, SimulatedPtz on a fake clock), so what's measured offline is what runs live.

Thread safety: two locks.
* `vision_lock`: the detector + tracker + progress part of a cycle (the slow part: ~50-150 ms of detection). Taken by
  the frame loop and by zone edits (`set_zone`), which rebuild the ROI and the progress tracker.
* `lock`: the director + controller part (decide, send the PTZ command). Taken by the frame loop, the mode buttons
  and the big STOP. It is held only for the short control part, so a STOP never waits for a detection to finish.
Lock order, always: vision_lock first, then lock (never the other way round), so they can't deadlock.

The big STOP sends the driver's stop **first**, before taking any lock, then takes `lock` to switch the director to
STOPPED (and sends the stop again through the controller, in case a velocity command was already on its way). It
never waits for the lock forever: if the frame loop is stuck while holding it (a hung PTZ call), `big_stop()` gives
up after `health.stop_lock_timeout_s` and leaves a flag that the next cycle obeys.
"""
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from zone import direction
from zone import director as D
from zone.config import Settings
from zone.controller import CycleCommand, Controller
from zone.director import Decision, Director
from zone.path import PathModel, Pose, PoseCurve
from zone.progress import ProgressTracker, SpeedProfile, TrackView
from zone.zonefile import Zone


class ControlLog:
    """JSONL, one object per line. Every line has `mono` (monotonic s) and `wall` (epoch s, rounded to ms), the
    same convention as rt's events (`wall`)."""

    def __init__(self, path, clock, flush_every=10):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")
        self.closed = False
        self.clock = clock
        self.flush_every = max(1, int(flush_every))
        self.n = 0
        self.lock = threading.Lock()

    def write(self, rec):
        rec = dict(rec)
        rec.setdefault("mono", round(self.clock.monotonic(), 4))
        rec.setdefault("wall", round(self.clock.wall(), 3))
        line = json.dumps(rec, separators=(",", ":"), default=str)
        with self.lock:
            if self.closed:          # a late line during shutdown (e.g. from the watchdog): dropped, never a crash
                return
            self.f.write(line + "\n")
            self.n += 1
            if self.n % self.flush_every == 0:
                self.f.flush()

    def close(self):
        with self.lock:
            self.closed = True
            self.f.close()


@dataclass
class CycleResult:
    t: float                                  # monotonic time of the decision
    t_grab: float
    views: Dict[int, TrackView]
    decision: Decision
    command: Optional[CycleCommand]
    tracker_out: list
    n_dets: int = 0
    det_ms: float = 0.0


class ZoneRuntime:
    def __init__(self, settings: Settings, zone: Zone, ptz, clock, spotter=None, profile: SpeedProfile = None,
                 profile_path=None, log: ControlLog = None, on_event=None):
        self.settings = settings
        self.ptz, self.clock = ptz, clock
        self.spotter = spotter
        self.log = log
        self.on_event_cb = on_event
        self.lock = threading.RLock()            # director + controller (see the module docstring)
        self.vision_lock = threading.RLock()     # spotter + progress
        # how long the big STOP waits for the lock before stopping the camera directly (see big_stop)
        self.stop_lock_timeout_s = float(settings.cfg.get("health", {}).get("stop_lock_timeout_s", 0.5))
        self.stop_requested = False      # set by big_stop when it couldn't get the lock: the next cycle obeys it
        # set by hand_over when it couldn't get the lock: (mode, why, stop_camera), the next cycle obeys it
        self.handover_requested = None
        self.handover_lock_timeout_s = 1.0   # how long hand_over waits for the control loop before queueing instead
        self.profile_path = Path(profile_path) if profile_path else None
        self.profile = profile
        if self.profile is None and self.profile_path and self.profile_path.exists():
            try:
                self.profile = SpeedProfile.load(self.profile_path)
            except Exception as e:  # noqa: BLE001 (a corrupt profile must not stop the race; start a new one)
                self._event(dict(type="warning", msg="speed profile not loaded: %s" % e))
        self.director = Director(settings.director, on_event=self._event)
        self.controller = Controller(ptz, clock, None, settings.pan_map, settings.tilt_map, settings.controller)
        self.zone = None
        self.path: Optional[PathModel] = None
        self.curve: Optional[PoseCurve] = None
        self.progress: Optional[ProgressTracker] = None
        self.last: Optional[CycleResult] = None
        self.sign_warnings = []                  # e.g. "pan sign not measured" (zone/direction.py), for the page
        self.frame_size = None
        self.cycles = 0
        self.fps = 0.0
        self._t_prev = None
        self.set_zone(zone)

    # ------------------------------------------------------------------------------------------- zone
    def set_zone(self, zone: Zone):
        """Install a (new or edited) zone: rebuild the path, the pose curve and the ROI. Tracks restart; the learned
        speed profile is kept (after a big edit of the path, delete zone.speed.json and re-learn it in PASSIVE)."""
        with self.vision_lock, self.lock:
            self.zone = zone
            self._apply_signs(zone)
            path = PathModel.build(zone)
            self.path = path
            if path is None:
                self.curve = None
                if self.director.mode == D.AUTO:
                    self.set_mode(D.STOPPED, why="zone incomplete")
                return
            kind = self.settings.cfg["zone"].get("interpolation", "pchip")
            self.curve = PoseCurve.from_zone(zone, path, kind, float(self.settings.cfg["zone"].get("ease_in", 0.0)))
            self.controller.set_curve(self.curve)
            sp = self.settings.cfg.get("speed_profile", {})
            if self.profile is None:
                self.profile = SpeedProfile(s_lo=min(-0.5, path.s_min), s_hi=max(1.2, path.s_max),
                                            bin_width=float(sp.get("bin_width", 0.05)),
                                            max_passes=int(sp.get("max_passes", 50)),
                                            default_v=float(sp.get("default_v", 0.25)))
            if self.progress is None:
                self.progress = ProgressTracker(path, self.settings.progress, self.profile)
                self.progress.on_finished = self._pass_finished
            else:
                self.progress.set_path(path)
            if self.spotter is not None:
                size = zone.image_size or self.frame_size
                margin = float(self.settings.cfg["spotter"].get("roi_margin_px", 40))
                self.spotter.set_roi(path.corridor_bbox(margin, size))

    def _apply_signs(self, zone: Zone):
        """Use the pan/tilt sign measured while teaching this zone (zone/direction.py), and say so when the sign the
        keyframes depend on was never measured. Warnings are repeated only when they change."""
        if getattr(self.settings, "sign_how", None) is None:
            return
        warnings = direction.apply_zone(self.settings, zone)
        info = direction.summary(self.settings)
        if (info, warnings) != getattr(self, "_signs_said", None):
            self._signs_said = (info, warnings)
            self._event(dict(type="ptz_signs", pan=info["pan"]["sign"], pan_how=info["pan"]["how"],
                             tilt=info["tilt"]["sign"], tilt_how=info["tilt"]["how"],
                             msg="pan %+d (%s), tilt %+d (%s)" % (info["pan"]["sign"], info["pan"]["how"],
                                                                  info["tilt"]["sign"], info["tilt"]["how"])))
            for w in warnings:
                self._event(dict(type="warning", msg=w))
        self.sign_warnings = warnings

    def entry_pose(self):
        k = self.zone.keyframes()[0]
        return Pose(k.pan, k.tilt, k.zoom)

    def _pass_finished(self, tp):
        if self.profile_path:
            try:
                self.profile.save(self.profile_path)
            except Exception as e:  # noqa: BLE001
                self._event(dict(type="warning", msg="speed profile not saved: %s" % e))
        self._event(dict(type="pass_finished", track_id=tp.track_id, n=tp.n_updates,
                         s0=round(tp.samples[0][2], 3), s1=round(tp.samples[-1][2], 3),
                         profile_passes=self.profile.n_passes))

    # ------------------------------------------------------------------------------------------- modes
    def set_mode(self, mode, why="operator"):
        """-> (ok, message). AUTO needs a complete zone (2+ keyframes)."""
        with self.lock:
            mode = str(mode).upper()
            if mode == D.AUTO and (self.path is None or self.zone.entry_preset is None):
                return False, "zona incompleta: ensine pelo menos 2 keyframes antes do AUTO"
            self.handover_requested = None  # this newer choice replaces a hand-over the loop hasn't obeyed yet
            self.director.set_mode(mode, self.clock.monotonic(), why)
            if mode == D.STOPPED:
                self.controller.stop()      # immediately, not at the next frame
            return True, mode

    def hand_over(self, mode, why, stop_camera, lock_timeout_s=None):
        """For the studio (zone/agent): switch the director to PASSIVE or STOPPED *without* its usual one-shot stop at
        the next cycle. The studio's own command (a move, a preset recall) goes out right after this, and a stop from
        the director ~0.1 s later would cut it. `stop_camera=True` stops the PTZ now instead (leaving AUTO for manual
        or hold: the zone's own motion must end). -> (ok, message); not ok only if the control loop held its lock for
        longer than `lock_timeout_s` (a hung PTZ call). Then the hand-over is not lost: `handover_requested` is set,
        and the next cycle obeys it before it decides anything (like the big STOP's `stop_requested`)."""
        mode = str(mode).upper()
        if mode not in (D.PASSIVE, D.STOPPED):
            raise ValueError("hand_over is for PASSIVE or STOPPED, not %r" % mode)
        self.handover_requested = (mode, why, bool(stop_camera))     # first: a cycle about to start sees it
        timeout = self.handover_lock_timeout_s if lock_timeout_s is None else float(lock_timeout_s)
        if not self.lock.acquire(timeout=timeout):
            return False, ("the zone's control loop is busy (a PTZ call is hanging); the zone goes quiet as soon as "
                           "it answers")
        try:
            self._apply_handover()
            return True, mode
        finally:
            self.lock.release()

    def _apply_handover(self):
        """Caller holds self.lock: obey `handover_requested` (director to PASSIVE / STOPPED, no queued action)."""
        request, self.handover_requested = self.handover_requested, None
        if request is None:
            return
        mode, why, stop_camera = request
        self.director.set_mode(mode, self.clock.monotonic(), why)
        self.director.discard_pending()
        if stop_camera:
            self.controller.stop()

    def big_stop(self, lock_timeout_s=None):
        """The big red button: a stop command right now, then STOPPED.

        1. The driver's stop goes out first, before any lock: it never waits for a detection or a control cycle.
        2. Then the lock (held by the frame loop only for its short control part), to switch the director to STOPPED
           and send the stop once more through the controller (a velocity command the loop had already decided on
           may have gone out just after step 1: this second stop is the last word).
        -> True if it got the lock, False if the frame loop is stuck holding it (a hung PTZ call): then the next
        cycle switches to STOPPED as soon as the loop gets going again (`stop_requested`)."""
        timeout = self.stop_lock_timeout_s if lock_timeout_s is None else float(lock_timeout_s)
        self.stop_requested = True       # first, so a cycle that is about to send a command sees it
        self._direct_stop()
        got = self.lock.acquire(timeout=timeout) if timeout > 0 else self.lock.acquire(blocking=False)
        if not got:
            # the frame loop is stuck inside a cycle (hung PTZ call...): don't wait for it forever. The camera has
            # already been told to stop (step 1), and the driver's own lock has a timeout, so that can't hang either.
            self._event(dict(type="warning", msg="big STOP: control loop busy for %.1f s, stop sent directly to the "
                             "PTZ; STOPPED as soon as the loop wakes up" % timeout))
            return False
        try:
            self._apply_stop("big_stop")
            return True
        finally:
            self.lock.release()

    def _direct_stop(self):
        """Stop pan/tilt and zoom through the driver, without the controller (and without the runtime lock)."""
        for fn in (self.ptz.stop, self.ptz.zoom_stop):
            try:
                fn()
            except Exception:  # noqa: BLE001 (a dead PTZ must not stop us from trying the next command)
                pass

    def _apply_stop(self, why):
        """Caller holds self.lock: director to STOPPED and a stop through the controller."""
        self.director.set_mode(D.STOPPED, self.clock.monotonic(), why)
        self.controller.stop()
        self.director._pending = None   # already stopped
        self.stop_requested = False
        self.handover_requested = None

    # ------------------------------------------------------------------------------------------- one cycle
    def cycle(self, frame, tracker_out=None) -> CycleResult:
        """frame: ZoneFrame (img may be None when tracker_out is given, e.g. a cached replay).

        Two parts: vision (detector, tracker, progress) under `vision_lock`, then control (director, controller,
        PTZ commands) under `lock`. The big STOP and the mode buttons only ever wait for the control part."""
        # ---- vision: the slow part, outside the control lock
        with self.vision_lock:
            if frame.img is not None:
                self.frame_size = (frame.img.shape[1], frame.img.shape[0])
            n_dets, det_ms = 0, 0.0
            frame_had_detection = tracker_out is None    # raw detections of THIS frame exist (not a cached replay)
            if tracker_out is None:
                tracker_out = self.spotter.process(frame.img) if self.spotter else []
                if self.spotter:
                    n_dets, det_ms = len(self.spotter.last_dets[0]), self.spotter.last_ms
            # the followed car (if any): while it is in a dropout, progress re-attaches a car reappearing near its
            # prediction to it, also under a new ByteTrack id or as a lone raw detection ByteTrack holds back
            focus = (self.director.locked_id
                     if self.director.mode == D.AUTO and self.director.sub == D.TRACKING else None)
            raw = self._raw_boxes() if frame_had_detection else None
            views = self.progress.update(frame.t, tracker_out, focus, raw) if self.progress else {}
            reacquired = list(self.progress.reacquired) if self.progress else []
        # ---- control: short, under the lock the big STOP takes
        with self.lock:
            if self.handover_requested is not None:
                self._apply_handover()               # the studio took the camera while this loop was busy
            if self.stop_requested:
                self._apply_stop("big_stop (delayed: the loop was busy)")
            if hasattr(self.ptz, "tick"):
                self.ptz.tick()                      # the simulator integrates its motion up to now
            now = self.clock.monotonic()
            for ev in reacquired:
                self.director.reacquired(now, ev)    # logs it; follows the new id if progress adopted one
            decision = self.director.step(now, views)
            command = None
            if decision.action == D.FOLLOW and decision.track is not None:
                tr = decision.track
                command = self.controller.follow(tr.s, tr.v, frame.t, decision.zoom_extra)
                if command.wrong_direction:
                    # Branch A saw the camera turn against the pan/tilt sign we use (zone/direction.py): every
                    # correction would push it further from the car. Give the car up and say what to fix.
                    w = command.wrong_direction
                    self._event(dict(type="direction_mismatch", track_id=tr.track_id, s=round(tr.s, 3),
                                     axis=w["axis"], sign=w["sign"], observed=w["observed"],
                                     msg="a PTZ gira ao contrário: %s_sign está %+d mas a câmera se comporta como %+d "
                                         "(ponha ptz.%s_sign = %d no site.toml, ou ensine com jog)" % (
                                             w["axis"], w["sign"], w["observed"], w["axis"], w["observed"])))
                    decision = self.director.drop(now, "wrong_direction", tr) or decision
                    if decision.action == D.RECALL_ENTRY and self.zone.entry_preset is not None:
                        self.controller.recall(self.zone.entry_preset, self.entry_pose(),
                                               self.settings.director.settle_s)
                elif command.diverged:
                    # the divergence guard: the camera has been too far from the target for too long: give the car
                    # up and go back to the entry for the next one (logged, and shown on the page as a drop)
                    self._event(dict(type="divergence", track_id=tr.track_id, s=round(tr.s, 3),
                                     gap_pan=round(command.gap_pan, 1), gap_tilt=round(command.gap_tilt, 1),
                                     cycles=self.controller.diverged_cycles))
                    decision = self.director.drop(now, "diverged", tr) or decision
                    if decision.action == D.RECALL_ENTRY and self.zone.entry_preset is not None:
                        self.controller.recall(self.zone.entry_preset, self.entry_pose(),
                                               self.settings.director.settle_s)
            elif decision.action == D.RECALL_ENTRY and self.zone.entry_preset is not None:
                self.controller.recall(self.zone.entry_preset, self.entry_pose(), self.settings.director.settle_s)
            elif decision.action == D.STOP:
                self.controller.stop()
            self._stats(now)
            res = CycleResult(now, frame.t, views, decision, command, tracker_out, n_dets, det_ms)
            self.last = res
            self.cycles += 1
            if self.log:
                self.log.write(self._record(frame, res))
            return res

    def _raw_boxes(self):
        """This frame's raw detections (xyxy) confident enough to start a track in ByteTrack (its high threshold),
        for re-attaching a lone detection to the followed car. None without a spotter."""
        if self.spotter is None:
            return None
        dets = tuple(getattr(self.spotter, "last_dets", ()) or ())
        if len(dets) < 2:                         # a spotter without scores (test fakes): no raw re-attachment
            return None
        boxes, scores = dets[0], dets[1]
        high = float(getattr(getattr(self.spotter, "tracker", None), "high", 0.0) or 0.0)
        return [b for b, s in zip(boxes, scores) if s >= high]

    def _stats(self, now):
        if self._t_prev is not None and now > self._t_prev:
            inst = 1.0 / (now - self._t_prev)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._t_prev = now

    def _record(self, frame, res: CycleResult):
        return dict(type="cycle", frame=frame.idx, t_grab=round(frame.t, 4), t_grab_wall=round(frame.t_wall, 3),
                    n_dets=res.n_dets, det_ms=round(res.det_ms, 1),
                    tracks=[v.to_log() for v in res.views.values()],
                    director=self.director.state(), action=res.decision.action, reason=res.decision.reason,
                    cmd=res.command.to_log() if res.command else None)

    def _event(self, ev):
        if self.log:
            self.log.write(ev)
        if self.on_event_cb:
            self.on_event_cb(ev)

    # ------------------------------------------------------------------------------------------- for the page
    def snapshot(self):
        with self.lock:
            st = self.director.state()
            locked = None
            if self.last and st["locked_id"] is not None:
                tr = self.last.views.get(st["locked_id"])
                if tr:
                    locked = dict(id=tr.track_id, s=round(tr.s, 3), v=round(tr.v, 3), predicted=bool(tr.predicted),
                                  age=round(float(tr.since_update_s), 2), held=bool(tr.held),
                                  zoom_extra=round(float(self.director.zoom_extra), 3))
            cmd = self.last.command.to_log() if (self.last and self.last.command) else None
            return dict(director=st, locked=locked, command=cmd, fps=round(self.fps, 1), cycles=self.cycles,
                        tracks=[v.to_log() for v in self.last.views.values()] if self.last else [],
                        branch=self.settings.controller.branch, zone_ready=self.path is not None,
                        profile_passes=self.profile.n_passes if self.profile else 0,
                        ptz_ok=self.controller.ptz_ok)
