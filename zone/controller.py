"""PTZ controller: point the camera where the followed car will be.

The control law (design notes, *Control law*):

    s_ahead   = s + v * latency                       # look ahead to cover the delays
    target    = pose(s_ahead)                         # interpolated between keyframes
    pan_rate  = (dpan/ds)(s_ahead) * v  +  k * (target.pan - pan_now)

The first term (feed-forward) does almost all the work: "the car moves along the path at v, the camera must turn at
dpan/ds * v". The second (feedback) term only corrects slow drift, gently. Tilt uses the same form. Zoom is not a
velocity: it's sent as an absolute position (`zoom_direct`), at most a couple of times per second.

Branches (picked by config from the characterization report, never by editing code):

* **Branch A**: the camera answers position inquiries while moving. `pan_now` comes from inquiry, polled at the rate
  the report says is safe; feedback gain k is small and its contribution is rate-limited.
* **Branch B**: no usable feedback. `pan_now` is dead-reckoned by integrating the commanded speeds through the speed
  table (logged, but not trusted), and k = 0, except right after a preset recall: then the pose is known, and it
  stays trustworthy for `dead_reckoning_trust_s` seconds of motion.

Latency: `latency_s` (config, seeded from the report's command latency) is the command-to-motion delay; when
`include_processing_delay` is on, the time from the frame grab to now (detection + tracking) is added. Every velocity
command carries the driver's TTL, so if this loop dies, the camera stops by itself.
"""
import math
from dataclasses import dataclass
from typing import Optional

from zone.path import Pose, PoseCurve
from zone.speedmap import AxisSpeedMap


@dataclass
class ControllerSettings:
    branch: str = "B"                    # "A" (inquiry feedback) | "B" (feed-forward only)
    latency_s: float = 0.25              # command-to-motion delay (s)
    include_processing_delay: bool = True  # also look ahead by (now - frame grab time)
    k_pan: float = 1.5                   # feedback gain (1/s): rate += k * error
    k_tilt: float = 1.5
    max_feedback_pan: float = 150.0       # cap on |k * error| (units/s), so feedback never yanks the camera
    max_feedback_tilt: float = 100.0
    inquiry_hz: float = 10.0             # Branch A: position poll rate
    inquiry_stale_s: float = 0.35        # an inquiry older than this is not used for feedback
    dead_reckoning_trust_s: float = 4.0  # Branch B: feedback allowed for this much motion after a known pose
    k_quant: float = 1.5                 # quantization compensation gain (1/s), see Controller.follow
    ttl_s: float = 0.3                   # every velocity command expires after this (driver dead-man switch)
    zoom_enabled: bool = True
    zoom_min_interval_s: float = 0.5     # at most 2 zoom commands per second
    zoom_threshold: float = 150.0        # only send a zoom when it changes by more than this (raw units)
    zoom_loosen: float = 0.85            # zoom = wide + loosen * (taught - wide): < 1 frames looser than taught
    zoom_wide: float = 0.0               # raw zoom position of the widest angle
    zoom_limits: tuple = (0, 16384)
    model_latency_s: float = 0.10        # dead reckoning: command -> motion delay (report command_latency_s)
    model_stop_latency_s: float = 0.10   # ... stop -> deceleration starts (report stop_latency_s)
    model_ramp_s: float = 0.20           # ... time to reach a new speed (report ramp_s)
    pan_sign: int = 1                    # +1: the reported raw pan grows with a positive (rightward) step; from the
    tilt_sign: int = 1                   # report. Poses (keyframes, inquiry) are in the reported frame; steps physical


@dataclass
class PoseNow:
    pan: Optional[float]
    tilt: Optional[float]
    zoom: Optional[float]
    source: str          # "inquiry" | "dead_reckoning" | "known" (right after a recall) | "unknown"
    trusted: bool        # may the feedback term use it?
    age_s: float = 0.0


class PoseEstimator:
    """Where is the camera pointing right now?

    Keeps a dead-reckoned pose by integrating every commanded speed step through the speed table, resets it to a
    known pose after a preset recall, and (Branch A) overrides it with fresh inquiry answers."""

    def __init__(self, ptz, clock, pan_map: AxisSpeedMap, tilt_map: AxisSpeedMap, settings: ControllerSettings):
        self.ptz, self.clock = ptz, clock
        self.pan_map, self.tilt_map = pan_map, tilt_map
        self.cfg = settings
        self.dr_pan = self.dr_tilt = None        # dead-reckoned pose (None = never known)
        self.zoom = None                         # last zoom we know of (commanded or inquired)
        self.pan_step = self.tilt_step = 0       # the speed currently commanded
        self.t_last = clock.monotonic()
        self.moving_since_known = math.inf       # seconds of motion since the pose was last known
        self.known_after = None                  # a recall is settling: pose becomes known at this time
        self.pending_known = None
        self.inq = None                          # (pan, tilt, t) of the last successful inquiry
        self.t_last_poll = -math.inf
        self.inquiry_failures = 0
        # the camera's modelled speed (reported units/s): a command takes effect after the latency, and the speed
        # then ramps linearly to the new value over ramp_s, as the bench measured (and as the simulator does)
        self.v = [0.0, 0.0]                      # current modelled speed, pan / tilt
        self.ramp = [None, None]                 # (t_start, v_from, v_to) of a ramp in progress
        self.pending = []                        # (t_effective, v_pan, v_tilt) commands not yet in effect

    def _speed_at(self, axis, t):
        r = self.ramp[axis]
        if r is None:
            return self.v[axis]
        t0, v0, v1 = r
        ramp = max(self.cfg.model_ramp_s, 1e-6)
        if t >= t0 + ramp:
            return v1
        return v0 + (v1 - v0) * max(0.0, t - t0) / ramp

    def _advance(self, t0, t1):
        """Integrate the modelled speed from t0 to t1 (small steps: ramps are short)."""
        t = t0
        while t < t1 - 1e-12:
            h = min(0.01, t1 - t)
            vp = 0.5 * (self._speed_at(0, t) + self._speed_at(0, t + h))
            vt = 0.5 * (self._speed_at(1, t) + self._speed_at(1, t + h))
            if self.dr_pan is not None and self.known_after is None:
                self.dr_pan += vp * h
                self.dr_tilt += vt * h
            if abs(vp) > 1e-9 or abs(vt) > 1e-9:
                self.moving_since_known += h
            t += h
        for axis in (0, 1):
            self.v[axis] = self._speed_at(axis, t1)
            r = self.ramp[axis]
            if r is not None and t1 >= r[0] + self.cfg.model_ramp_s:
                self.ramp[axis] = None

    def _integrate(self, now):
        if now <= self.t_last:
            return
        if self.known_after is not None and now >= self.known_after:
            self.dr_pan, self.dr_tilt, z = self.pending_known
            if z is not None:
                self.zoom = z
            self.known_after = None
            self.moving_since_known = 0.0
            self.v, self.ramp, self.pending = [0.0, 0.0], [None, None], []
            self.t_last = now
            return
        # apply the commands that took effect since the last call, in order
        while self.pending and self.pending[0][0] <= now:
            te, vp, vt = self.pending.pop(0)
            self._advance(self.t_last, te)
            self.t_last = te
            for axis, vnew in ((0, vp), (1, vt)):
                if vnew != self._speed_at(axis, te) or self.ramp[axis] is not None:
                    self.ramp[axis] = (te, self._speed_at(axis, te), vnew)
        self._advance(self.t_last, now)
        self.t_last = now

    def commanded(self, pan_step, tilt_step):
        """Call right after sending a velocity command (or a stop: 0, 0)."""
        now = self.clock.monotonic()
        self._integrate(now)
        self.pan_step, self.tilt_step = int(pan_step), int(tilt_step)
        vp = self.cfg.pan_sign * self.pan_map.rate_of(self.pan_step)
        vt = self.cfg.tilt_sign * self.tilt_map.rate_of(self.tilt_step)
        last = self.pending[-1][1:] if self.pending else None
        if last == (vp, vt) or (last is None and not self.pending and self.ramp == [None, None]
                                and self.v == [vp, vt]):
            return                                # same speed again (TTL refresh): nothing changes
        lat = self.cfg.model_stop_latency_s if (vp == 0 and vt == 0) else self.cfg.model_latency_s
        self.pending.append((now + lat, vp, vt))

    def recalled(self, pose: Pose, settle_s):
        """A preset recall was sent; the camera will be at `pose` once it settles."""
        now = self.clock.monotonic()
        self._integrate(now)
        self.pan_step = self.tilt_step = 0
        self.pending = []
        self.known_after = now + max(0.0, settle_s)
        self.pending_known = (pose.pan, pose.tilt, pose.zoom)
        self.inq = None

    def set_known(self, pan, tilt, zoom=None):
        self._integrate(self.clock.monotonic())
        self.dr_pan, self.dr_tilt = float(pan), float(tilt)
        if zoom is not None:
            self.zoom = float(zoom)
        self.moving_since_known = 0.0
        self.known_after = None

    def zoom_commanded(self, z):
        self.zoom = float(z)

    def poll(self):
        """Branch A: ask the camera where it is, at most inquiry_hz times per second."""
        now = self.clock.monotonic()
        if self.cfg.inquiry_hz <= 0 or now - self.t_last_poll < 1.0 / self.cfg.inquiry_hz - 1e-9:
            return
        self.t_last_poll = now
        try:
            r = self.ptz.inquire_pan_tilt()
        except Exception:  # noqa: BLE001 (a failed inquiry just means no feedback this cycle)
            r = None
        if r is None:
            self.inquiry_failures += 1
            return
        self.inq = (float(r[0]), float(r[1]), now)
        self.set_known(r[0], r[1])

    def estimate(self) -> PoseNow:
        now = self.clock.monotonic()
        self._integrate(now)
        if self.cfg.branch == "A" and self.inq is not None and now - self.inq[2] <= self.cfg.inquiry_stale_s:
            return PoseNow(self.inq[0], self.inq[1], self.zoom, "inquiry", True, now - self.inq[2])
        if self.known_after is not None:
            return PoseNow(self.dr_pan, self.dr_tilt, self.zoom, "settling", False)
        if self.dr_pan is None:
            return PoseNow(None, None, self.zoom, "unknown", False)
        trusted = self.moving_since_known <= self.cfg.dead_reckoning_trust_s
        src = "known" if self.moving_since_known == 0 else "dead_reckoning"
        return PoseNow(self.dr_pan, self.dr_tilt, self.zoom, src, trusted)


@dataclass
class CycleCommand:
    """What the controller decided this cycle (for the control log)."""
    s: float
    v: float
    s_ahead: float
    lookahead_s: float
    target: Pose
    ff_pan: float
    ff_tilt: float
    fb_pan: float
    fb_tilt: float
    pan_step: int
    tilt_step: int
    zoom_sent: Optional[int]
    pose_now: PoseNow
    result_ok: bool = True

    def to_log(self):
        pn = self.pose_now
        return dict(s=round(self.s, 4), v=round(self.v, 4), s_ahead=round(self.s_ahead, 4),
                    lookahead=round(self.lookahead_s, 3),
                    target=dict(pan=round(self.target.pan, 1), tilt=round(self.target.tilt, 1),
                                zoom=round(self.target.zoom, 1)),
                    ff=dict(pan=round(self.ff_pan, 2), tilt=round(self.ff_tilt, 2)),
                    fb=dict(pan=round(self.fb_pan, 2), tilt=round(self.fb_tilt, 2)),
                    steps=dict(pan=self.pan_step, tilt=self.tilt_step), zoom_sent=self.zoom_sent,
                    pan_now=dict(pan=None if pn.pan is None else round(pn.pan, 1),
                                 tilt=None if pn.tilt is None else round(pn.tilt, 1),
                                 zoom=None if pn.zoom is None else round(pn.zoom, 1),
                                 source=pn.source, trusted=pn.trusted), ok=self.result_ok)


class Controller:
    def __init__(self, ptz, clock, curve: PoseCurve, pan_map: AxisSpeedMap, tilt_map: AxisSpeedMap,
                 settings: ControllerSettings = None):
        self.ptz, self.clock = ptz, clock
        self.curve = curve
        self.pan_map, self.tilt_map = pan_map, tilt_map
        self.cfg = settings or ControllerSettings()
        self.pose = PoseEstimator(ptz, clock, pan_map, tilt_map, self.cfg)
        self.pan_step = self.tilt_step = 0
        self.t_last_zoom = -math.inf
        self.last_zoom_sent = None
        self.ptz_ok = True              # did the last command succeed? (health)
        self.q_pan = self.q_tilt = 0.0  # quantization deficit (reported units): wanted minus commanded, integrated
        self.t_follow = None            # time of the last follow() (None = the next one starts a new segment)
        self.want = (0.0, 0.0)          # wanted rates of the last cycle (reported units/s)

    def set_curve(self, curve: PoseCurve):
        self.curve = curve

    def zoom_for(self, taught_zoom):
        c = self.cfg
        z = c.zoom_wide + c.zoom_loosen * (taught_zoom - c.zoom_wide)
        return int(round(min(max(z, c.zoom_limits[0]), c.zoom_limits[1])))

    # ----------------------------------------------------------------------------------- one cycle of following
    def follow(self, s, v, t_grab) -> CycleCommand:
        c = self.cfg
        now = self.clock.monotonic()
        if c.branch == "A":
            self.pose.poll()
        lookahead = c.latency_s + (max(0.0, now - t_grab) if c.include_processing_delay else 0.0)
        s_ahead = s + v * lookahead
        target = self.curve.pose(s_ahead)
        slope = self.curve.dpose_ds(s_ahead)
        ff_pan, ff_tilt = slope.pan * v, slope.tilt * v
        pn = self.pose.estimate()
        fb_pan = fb_tilt = 0.0
        if pn.trusted and pn.pan is not None:
            # compare like with like: the target is where the car will be after `lookahead`, so compare it with
            # where the camera will be by then at its current speed (and the inquiry answer is `age_s` old). Without
            # this, the feedback always sees a phantom error of v * lookahead and overdrives the camera.
            horizon = lookahead + pn.age_s
            pan_then = pn.pan + c.pan_sign * self.pan_map.rate_of(self.pan_step) * horizon
            tilt_then = pn.tilt + c.tilt_sign * self.tilt_map.rate_of(self.tilt_step) * horizon
            fb_pan = _clip(c.k_pan * (target.pan - pan_then), c.max_feedback_pan)
            fb_tilt = _clip(c.k_tilt * (target.tilt - tilt_then), c.max_feedback_tilt)
        # rates are in the reported frame; the sign turns them into physical directions (+ step = right / up)
        want_pan, want_tilt = ff_pan + fb_pan, ff_tilt + fb_tilt
        # Quantization compensation: the camera only has whole speed steps, and with hysteresis the chosen step can
        # stay ~10-15% off the wanted rate for a whole pass. That error is known exactly (we know what we wanted and
        # what we commanded), so it's integrated and fed back in, gently. This is bookkeeping on our own commands,
        # not pose feedback, so it also runs in Branch B (where k = 0 outside the trust window). While the feedback
        # term is active it already contains this error, so the deficit is then reset instead of double-counted.
        if self.t_follow is not None and now > self.t_follow:
            dt = now - self.t_follow
            got_pan = c.pan_sign * self.pan_map.rate_of(self.pan_step)
            got_tilt = c.tilt_sign * self.tilt_map.rate_of(self.tilt_step)
            self.q_pan += (self.want[0] - got_pan) * dt
            self.q_tilt += (self.want[1] - got_tilt) * dt
        if pn.trusted:
            self.q_pan = self.q_tilt = 0.0
        q_pan = _clip(c.k_quant * self.q_pan, c.max_feedback_pan)
        q_tilt = _clip(c.k_quant * self.q_tilt, c.max_feedback_tilt)
        self.want, self.t_follow = (want_pan, want_tilt), now
        self.pan_step = self.pan_map.next_step(c.pan_sign * (want_pan + q_pan), self.pan_step)
        self.tilt_step = self.tilt_map.next_step(c.tilt_sign * (want_tilt + q_tilt), self.tilt_step)
        ok = self._send_velocity()
        zoom_sent = self._maybe_zoom(target.zoom, now)
        return CycleCommand(s, v, s_ahead, lookahead, target, ff_pan, ff_tilt, fb_pan, fb_tilt, self.pan_step,
                            self.tilt_step, zoom_sent, pn, ok)

    def _send_velocity(self):
        try:
            r = self.ptz.pan_tilt_velocity(self.pan_step, self.tilt_step, ttl=self.cfg.ttl_s)
            ok = bool(getattr(r, "ok", True))
        except Exception:  # noqa: BLE001 (the network can fail; the TTL stops the camera, we keep looping)
            ok = False
        self.ptz_ok = ok
        self.pose.commanded(self.pan_step, self.tilt_step)
        return ok

    def _maybe_zoom(self, taught_zoom, now):
        c = self.cfg
        if not c.zoom_enabled:
            return None
        z = self.zoom_for(taught_zoom)
        if now - self.t_last_zoom < c.zoom_min_interval_s:
            return None
        if self.last_zoom_sent is not None and abs(z - self.last_zoom_sent) <= c.zoom_threshold:
            return None
        try:
            self.ptz.zoom_direct(z)
        except Exception:  # noqa: BLE001
            return None
        self.t_last_zoom, self.last_zoom_sent = now, z
        self.pose.zoom_commanded(z)
        return z

    def _reset_quant(self):
        self.q_pan = self.q_tilt = 0.0
        self.t_follow = None
        self.want = (0.0, 0.0)

    # ----------------------------------------------------------------------------------- other actions
    def stop(self):
        """Stop pan/tilt/zoom now (big STOP, mode change, end of a pass)."""
        self.pan_step = self.tilt_step = 0
        self._reset_quant()
        for fn in (self.ptz.stop, self.ptz.zoom_stop):
            try:
                fn()
            except Exception:  # noqa: BLE001
                self.ptz_ok = False
        self.pose.commanded(0, 0)

    def recall(self, preset, pose: Pose, settle_s):
        """Recall a preset whose pose is known (a keyframe's). After `settle_s` the pose estimate is reset to it,
        which is what makes Branch B's feedback usable again for the next car."""
        self.pan_step = self.tilt_step = 0
        self._reset_quant()
        try:
            r = self.ptz.preset_recall(int(preset))
            self.ptz_ok = bool(getattr(r, "ok", True))
        except Exception:  # noqa: BLE001
            self.ptz_ok = False
        self.pose.recalled(pose, settle_s)
        self.last_zoom_sent = int(round(pose.zoom))  # the preset restores the taught zoom
        self.t_last_zoom = self.clock.monotonic()


def _clip(x, lim):
    return max(-lim, min(lim, x))
