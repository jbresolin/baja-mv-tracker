"""Progress and speed: turn each spotter track into `s` (progress along the path) and `v` (ds/dt), and learn the
typical speed profile v_typ(s).

Pieces
------
* `SVFilter`: a 1D constant-velocity Kalman filter on s(t). Each detection gives a noisy `s`; the filter keeps a
  smoothed `s`, a speed `v`, and how uncertain both are. Time steps come from the **frame grab times**, never from
  when we happened to process the frame.
* `TrackProgress`: one car (one tracker id): its filter, how many times it was updated, when it was last seen, how
  many frames in a row it was outside the corridor.
* `ProgressTracker`: fed once per processed frame with the tracker output; projects each box's ground point
  (bottom centre) onto the path, updates the filters, and hands finished tracks to the speed profile.
* `SpeedProfile`: v binned by s across finished passes; the typical profile is the median per bin. It's saved next to
  zone.json and used to predict ahead when detections drop out for a moment.

Dropouts (no detection for a while: dust, another car in front, a few bad frames), see `ProgressTracker.predict_s_v`:
* the speed is **frozen** at the filter's last measured value (prediction steps never change it), clamped to a sane
  band around the car's mean speed over its last second;
* once the speed profile has learned, the predicted speed blends from that frozen value to the typical speed v_typ(s)
  over `typ_blend_s`, since cars follow the typical profile;
* the prediction never runs more than `max_predict_s_ahead` beyond the last measured s: past that it **holds** (v = 0:
  the camera waits at that framing instead of running off ahead of a car it can't see);
* re-acquisition (`ProgressTracker.update(..., focus_id=...)`): a detection near the prediction re-attaches to the
  followed track, also when ByteTrack gives the reappearing car a NEW id, or doesn't report it at all yet (a lone raw
  detection: ByteTrack only reports a new track from its second detection).

Units: `s` is dimensionless (0 = first keyframe, 1 = last keyframe), so `v` is in "zone per second". A car that
crosses the zone in 4 s has v ~ 0.25 /s.
"""
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from zone.path import PathModel


# ===================================================================================================== the filter
class SVFilter:
    """Constant-velocity Kalman filter. State x = [s, v]; covariance P (2x2).

    accel_sigma: how much the speed is allowed to change, as the standard deviation of a random acceleration
        (in s/s^2). Bigger = follows braking faster but is noisier.
    meas_sigma: the noise of one measured s (in s units; the caller converts from pixels).
    """

    def __init__(self, t, s, meas_sigma, accel_sigma, v0=0.0, v_sigma=0.5):
        self.t = float(t)
        self.x = np.array([float(s), float(v0)])
        self.P = np.diag([meas_sigma ** 2, v_sigma ** 2])
        self.r = meas_sigma ** 2
        self.q = accel_sigma ** 2

    def _propagate(self, dt):
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        return F @ self.x, F @ self.P @ F.T + Q

    def predict(self, t):
        """(s, v, var_s, var_v) at time t, without changing the filter (constant speed from the last update)."""
        dt = max(0.0, float(t) - self.t)
        x, P = self._propagate(dt)
        return float(x[0]), float(x[1]), float(P[0, 0]), float(P[1, 1])

    def reseed(self, t, s, v):
        """Move the filter to time t with the mean (s, v) given by the caller (the dropout prediction, which knows more
        than "constant speed": the frozen, clamped speed and the typical profile). The covariance grows as for a
        normal prediction step, so the next measurement is still weighed as a surprise of the right size."""
        dt = float(t) - self.t
        if dt < 0:
            return False
        _x, self.P = self._propagate(dt)
        self.x = np.array([float(s), float(v)])
        self.t = float(t)
        return True

    def update(self, t, s_meas):
        """Fold in a measurement taken at time t. Measurements older than the filter's time are ignored."""
        dt = float(t) - self.t
        if dt < 0:
            return False
        x, P = self._propagate(dt)
        H = np.array([1.0, 0.0])
        innovation = float(s_meas) - x[0]
        S = P[0, 0] + self.r
        K = P[:, 0] / S
        self.x = x + K * innovation
        self.P = P - np.outer(K, H @ P)
        self.t = float(t)
        return True

    @property
    def s(self):
        return float(self.x[0])

    @property
    def v(self):
        return float(self.x[1])


# ===================================================================================================== speed profile
class SpeedProfile:
    """Typical speed per bin of s, learned from finished passes.

    Each finished pass contributes one value per bin it crossed (the median of its own v samples in that bin), so a
    slow car with many samples counts as much as a fast one. Each bin keeps the last `max_passes` values; the profile
    is their median (robust to the odd bad track). Bins with no data borrow from the nearest bin that has some; with
    no data at all the profile is `default_v`.
    """

    def __init__(self, s_lo=-0.5, s_hi=1.2, bin_width=0.05, max_passes=50, default_v=0.25):
        self.s_lo, self.s_hi, self.bin_width = float(s_lo), float(s_hi), float(bin_width)
        self.n_bins = max(1, int(math.ceil((self.s_hi - self.s_lo) / self.bin_width)))
        self.max_passes = int(max_passes)
        self.default_v = float(default_v)
        self.bins: List[deque] = [deque(maxlen=self.max_passes) for _ in range(self.n_bins)]
        self.n_passes = 0
        self._cache = None

    def bin_of(self, s):
        return int(np.clip(math.floor((s - self.s_lo) / self.bin_width), 0, self.n_bins - 1))

    def bin_centre(self, i):
        return self.s_lo + (i + 0.5) * self.bin_width

    def add_pass(self, samples):
        """samples: iterable of (s, v). Returns the number of bins it contributed to."""
        per_bin: Dict[int, list] = {}
        for s, v in samples:
            if s < self.s_lo or s > self.s_hi or not np.isfinite(v):
                continue
            per_bin.setdefault(self.bin_of(s), []).append(v)
        for i, vs in per_bin.items():
            self.bins[i].append(float(np.median(vs)))
        if per_bin:
            self.n_passes += 1
            self._cache = None
        return len(per_bin)

    def medians(self):
        """Per-bin median, or nan for empty bins."""
        if self._cache is None:
            self._cache = np.array([float(np.median(b)) if len(b) else np.nan for b in self.bins])
        return self._cache

    def v_typ(self, s):
        """Typical speed at s (interpolated between bin centres)."""
        med = self.medians()
        ok = np.isfinite(med)
        if not ok.any():
            return self.default_v
        centres = self.s_lo + (np.arange(self.n_bins) + 0.5) * self.bin_width
        return float(np.interp(s, centres[ok], med[ok]))

    @property
    def learned(self):
        return self.n_passes > 0

    # ---- file
    def to_dict(self):
        return dict(s_lo=self.s_lo, s_hi=self.s_hi, bin_width=self.bin_width, max_passes=self.max_passes,
                    default_v=self.default_v, n_passes=self.n_passes,
                    bins=[[round(v, 5) for v in b] for b in self.bins],
                    median=[None if not np.isfinite(m) else round(float(m), 5) for m in self.medians()])

    @classmethod
    def from_dict(cls, d):
        p = cls(d["s_lo"], d["s_hi"], d["bin_width"], d.get("max_passes", 50), d.get("default_v", 0.25))
        for i, b in enumerate(d.get("bins", [])[:p.n_bins]):
            p.bins[i].extend(float(v) for v in b)
        p.n_passes = int(d.get("n_passes", 0))
        return p

    def save(self, path):
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def profile_path_for(zone_path):
    """The speed profile lives next to zone.json: zone.json -> zone.speed.json."""
    p = Path(zone_path)
    return p.with_name(p.stem + ".speed.json")


# ===================================================================================================== tracks
@dataclass
class ProgressSettings:
    meas_sigma_px: float = 6.0          # noise of one ground point along the path, in pixels
    accel_sigma: float = 0.4            # random acceleration allowed, in s/s^2
    v_sigma0: float = 0.5               # initial speed uncertainty (s/s): the first speed comes from the measurements
    window_back_px: float = 40.0        # projection window behind the previous position (pixels of arc length)
    window_ahead_px: float = 400.0      # ... and ahead of it
    window_sigmas: float = 3.0          # after a dropout the window is at least this many standard deviations of the
                                        # predicted s wide on each side (the prediction gets less sure the longer the
                                        # car is unseen: a car that braked out of sight is still found)
    corridor_factor: float = 1.0        # a detection farther than factor * half width from the path is ignored
    finish_after_s: float = 3.0         # a track not seen for this long is finished (fed to the speed profile)
    profile_min_span: float = 0.3       # a finished track must have covered this much s to teach the profile
    profile_min_updates: int = 5
    max_prediction_s: float = 3.0       # never predict further than this past the last update
    max_predict_s_ahead: float = 0.25   # a prediction never runs more than this (in s, zone fraction) beyond the last
                                        # measured s: past it the prediction holds (v = 0, the camera waits there)
    typ_blend_s: float = 1.0            # once the profile has learned: the predicted speed goes from the frozen last
                                        # speed to v_typ(s) linearly over this long of dropout
    speed_band: tuple = (0.5, 1.5)      # the frozen speed is clamped to this x the car's mean speed over ...
    speed_band_window_s: float = 1.0    # ... its last this-many seconds of detections (a speed that jumped on the
                                        # last update doesn't fly the prediction away)
    reacquire_gate_s: float = 0.10      # re-adopting a NEW track id (or a lone raw detection) for the followed car:
    reacquire_gate_max_s: float = 0.35  # it must be within window_sigmas x s_std of the prediction, but at least
                                        # this min and at most this max (in s), or between the last measured s and
                                        # the prediction (a car slower than predicted)
    ground_window_s: float = 1.0        # image speed of the ground point (px/s) is measured over this window: the
                                        # director's lock rule uses it (an entry seen head-on, where s grows slowly)


@dataclass
class TrackProgress:
    track_id: int
    filt: SVFilter
    first_t: float
    last_update_t: float
    last_seen_t: float
    n_updates: int = 1
    off_corridor_frames: int = 0       # consecutive frames whose detection was outside the corridor
    lateral_px: float = 0.0
    box: tuple = ()
    ground: tuple = ()
    samples: list = field(default_factory=list)   # (t, s_raw, s_filtered, v_filtered) at each update
    ground_hist: deque = field(default_factory=lambda: deque(maxlen=200))   # (t, x, y) of every detection's
                                                                            # ground point, for ground_px_s

    def s_raw_last(self):
        return self.samples[-1][1] if self.samples else self.filt.s


@dataclass
class TrackView:
    """What the director, controller, page and log see of one track at the current cycle."""
    track_id: int
    s: float
    v: float
    s_std: float
    v_std: float
    lateral_px: float
    n_updates: int
    since_update_s: float     # how long since a detection updated it (0 = updated this frame)
    off_corridor_frames: int
    predicted: bool           # True when s/v are a prediction (no fresh detection)
    box: tuple
    ground: tuple
    # How fast the ground point (bottom centre of the box) moves in the IMAGE, pixels per second, over the last
    # `ground_window_s` (straight-line distance first -> last detection in the window). None = not enough recent
    # detections to tell. Unlike v it doesn't shrink where the path runs toward the camera (few pixels of path
    # for many metres), so it tells "really stopped" from "moving head-on".
    ground_px_s: Optional[float] = None
    v_typ: Optional[float] = None      # the learned typical speed at s (None until the speed profile has learned)
    held: bool = False                 # the dropout prediction reached max_predict_s_ahead: s holds there, v = 0

    def to_log(self):
        return dict(id=self.track_id, s=round(self.s, 4), v=round(self.v, 4), s_std=round(self.s_std, 4),
                    v_std=round(self.v_std, 4), lat=round(self.lateral_px, 1), n=self.n_updates,
                    age=round(self.since_update_s, 3), off=self.off_corridor_frames, pred=self.predicted,
                    gpx=None if self.ground_px_s is None else round(self.ground_px_s, 1),
                    vtyp=None if self.v_typ is None else round(self.v_typ, 4), held=self.held)


DROPOUT_MIN_S = 0.25   # a gap between two updates longer than this is a "dropout" (more than two missed frames at
                       # 10 fps): the update after it starts from the dropout prediction, and a re-acquisition is logged


@dataclass
class Prediction:
    """Where a track is at time t when no detection updated it at t (see ProgressTracker.prediction)."""
    s: float          # what the director / controller get: the prediction, held at the cap
    v: float          # its speed (0 while held)
    s_free: float     # the prediction without the cap (where the car probably is: used to find it again)
    v_free: float     # ... and its speed
    var_s: float      # the filter's uncertainty (grows with the dropout)
    var_v: float
    held: bool        # the cap was reached


class ProgressTracker:
    def __init__(self, path: PathModel, settings: ProgressSettings = None, profile: SpeedProfile = None):
        self.path = path
        self.cfg = settings or ProgressSettings()
        self.profile = profile or SpeedProfile(s_lo=min(-0.5, path.s_min), s_hi=max(1.2, path.s_max))
        self.tracks: Dict[int, TrackProgress] = {}
        self.finished_passes = 0
        self.on_finished = None          # optional callback(TrackProgress) (e.g. to save the profile)
        # re-acquisitions of the focus track found by the last update(): dicts with old_id, new_id (they differ when
        # a new ByteTrack id was adopted), dropout_s, how ("same_id" | "new_id" | "raw_detection"), s
        self.reacquired: List[dict] = []

    def set_path(self, path: PathModel):
        """The zone was edited: drop the tracks (their s meant something else) but keep the learned profile."""
        self.path = path
        self.tracks.clear()

    # --------------------------------------------------------------------------------------------- per frame
    def update(self, t, tracker_out, focus_id=None, raw_boxes=None):
        """t = grab time of the frame (seconds, monotonic clock). tracker_out = [(track_id, xyxy, score), ...] as
        returned by rt's ByteTracker. Returns {track_id: TrackView} for every live track at time t.

        focus_id: the track the director follows (None = none). While it is in a dropout, a car that reappears near
        its prediction is re-attached to it (see `_reacquire`): a NEW ByteTrack id is adopted (the focus track then
        carries the new id; `self.reacquired` says so), and `raw_boxes` (this frame's raw detections, (N, 4) xyxy)
        lets a lone detection that ByteTrack doesn't report yet update it too."""
        c = self.cfg
        self.reacquired = []
        focus_prev_update = self.tracks[focus_id].last_update_t if focus_id in self.tracks else None
        meas_sigma = c.meas_sigma_px / self.path.scale
        seen = set()
        for tid, box, _score in tracker_out:
            tid = int(tid)
            seen.add(tid)
            x1, y1, x2, y2 = [float(v) for v in box]
            gx, gy = (x1 + x2) / 2.0, y2            # ground point: bottom centre of the box
            tp = self.tracks.get(tid)
            back, ahead = c.window_back_px, c.window_ahead_px
            s_prev = None
            if tp:
                pred = self.prediction(tp, t)
                s_prev, var_s = pred.s_free, pred.var_s
                # widen the search window by the prediction's own uncertainty (in pixels of arc): right after an
                # update it is a few pixels (the fixed window wins); after a long dropout it grows, so a car that
                # did something else than "keep the same speed" while unseen is still within reach
                spread_px = c.window_sigmas * math.sqrt(max(var_s, 0.0)) * self.path.scale
                back, ahead = max(back, spread_px), max(ahead, spread_px)
            proj = self.path.project(gx, gy, s_prev, back, ahead)
            inside = self.path.in_corridor(proj, c.corridor_factor)
            if tp is None:
                if not inside:
                    continue                        # not on the racing line: not a car of the zone (yet)
                v0 = self.profile.v_typ(proj.s) if self.profile.learned else 0.0
                tp = TrackProgress(tid, SVFilter(t, proj.s, meas_sigma, c.accel_sigma, v0, c.v_sigma0),
                                   first_t=t, last_update_t=t, last_seen_t=t)
                tp.ground_hist.append((t, gx, gy))
                tp.samples.append((t, proj.s, proj.s, v0))
                self.tracks[tid] = tp
            else:
                tp.last_seen_t = t
                tp.ground_hist.append((t, gx, gy))
                if not inside:
                    tp.off_corridor_frames += 1
                    tp.box, tp.ground = tuple(box), (gx, gy)
                    continue
                self._fold(tp, t, proj.s)
            tp.off_corridor_frames = 0
            tp.lateral_px = proj.lateral_px
            tp.box, tp.ground = tuple(float(v) for v in box), (gx, gy)
        if focus_id in self.tracks and focus_prev_update is not None:
            self._reacquire(focus_id, focus_prev_update, t, seen, tracker_out, raw_boxes)
        # finish tracks that have been gone long enough
        for tid in [k for k, tp in self.tracks.items() if t - tp.last_seen_t > c.finish_after_s]:
            self._finish(self.tracks.pop(tid))
        return {tid: self.view(tp, t) for tid, tp in self.tracks.items()}

    def _fold(self, tp: TrackProgress, t, s_meas):
        """One measured s for a track. After a dropout the filter first moves to the dropout prediction (frozen /
        clamped speed, typical profile), so the measurement is compared with what we actually predicted."""
        if t - tp.last_update_t > DROPOUT_MIN_S:
            pred = self.prediction(tp, t)
            tp.filt.reseed(t, pred.s_free, pred.v_free)
        if tp.filt.update(t, s_meas):
            tp.n_updates += 1
            tp.last_update_t = t
            tp.samples.append((t, s_meas, tp.filt.s, tp.filt.v))
            return True
        return False

    # --------------------------------------------------------------------------------------------- re-acquisition
    def reacquire_gate(self, tp: TrackProgress, t):
        """(lo, hi): where a car that reappears may be (in s) to be taken for the followed one. Around the free
        prediction by window_sigmas x its standard deviation (at least reacquire_gate_s, at most
        reacquire_gate_max_s), and anywhere between the last measured s and the prediction (a car that went slower
        than predicted), but never more than window_back_px behind the last measured s (cars don't reverse)."""
        c = self.cfg
        pred = self.prediction(tp, t)
        w = c.window_sigmas * math.sqrt(max(pred.var_s, 0.0))
        w = min(max(w, c.reacquire_gate_s), c.reacquire_gate_max_s)
        s_last = tp.filt.s
        lo = max(s_last - c.window_back_px / self.path.scale, min(s_last, pred.s_free - w))
        return lo, pred.s_free + w

    def _reacquire(self, focus_id, prev_update, t, seen, tracker_out, raw_boxes):
        """The focus track after this frame's tracker output: updated under its own id (a plain re-acquisition), or
        still in a dropout. Then, in order:
          1. a track with a NEW id, born during the dropout, detected this frame inside the gate and not going
             backwards, is adopted: its detections are folded into the focus track, which takes the new id. Only
             while the focus id itself is not reported (an off-corridor box of the focus id is still that car);
             tracks that already existed before the dropout are other cars and are never adopted;
          2. else a raw detection that matches no tracker box (ByteTrack holds a new track back until its second
             detection), inside the gate and the corridor, updates the focus track directly."""
        ftp = self.tracks[focus_id]
        how = None
        if ftp.last_update_t >= t:
            how = "same_id"
        elif focus_id not in seen:
            lo, hi = self.reacquire_gate(ftp, t)
            s_ref = self.prediction(ftp, t).s_free
            best = None
            for tid, tp in self.tracks.items():
                if (tid == focus_id or tp.first_t <= prev_update or tp.last_update_t < t
                        or tp.off_corridor_frames > 0):
                    continue
                s_new = tp.s_raw_last()
                if not lo <= s_new <= hi:
                    continue
                if tp.n_updates >= 3 and tp.filt.v <= 0:
                    continue                        # it has a direction of its own, and it's not forward
                if best is None or abs(s_new - s_ref) < abs(best.s_raw_last() - s_ref):
                    best = tp
            if best is not None:
                self._adopt(ftp, best)
                how = "new_id"
            elif raw_boxes is not None and len(raw_boxes):
                if self._attach_raw(ftp, t, lo, hi, raw_boxes, [b for _i, b, _s in tracker_out]):
                    how = "raw_detection"
        if how and t - prev_update > DROPOUT_MIN_S:
            self.reacquired.append(dict(old_id=focus_id, new_id=ftp.track_id, dropout_s=round(t - prev_update, 3),
                                        how=how, s=round(ftp.filt.s, 4)))

    def _adopt(self, ftp: TrackProgress, new: TrackProgress):
        """Fold the new track's detections (all made after the focus track's last update) into the focus track and
        give it the new id. The new track's own filter is thrown away: it started from nothing, the focus one knows
        the car's speed."""
        for (ts, s_raw, _sf, _vf) in new.samples:
            self._fold(ftp, ts, s_raw)
        old_id = ftp.track_id
        del self.tracks[old_id]
        del self.tracks[new.track_id]
        ftp.track_id = new.track_id
        ftp.last_seen_t = max(ftp.last_seen_t, new.last_seen_t)
        ftp.ground_hist.extend(new.ground_hist)
        ftp.box, ftp.ground, ftp.lateral_px = new.box, new.ground, new.lateral_px
        ftp.off_corridor_frames = 0
        self.tracks[new.track_id] = ftp

    def _attach_raw(self, ftp: TrackProgress, t, lo, hi, raw_boxes, tracked_boxes):
        """A lone raw detection inside the gate (and the corridor) that overlaps no box the tracker reported this
        frame: update the focus track with it. -> True if one was used (the one nearest the prediction)."""
        c = self.cfg
        s_ref = self.prediction(ftp, t).s_free
        best = None
        for box in raw_boxes:
            box = tuple(float(v) for v in box)
            if any(_iou(box, tuple(float(v) for v in tb)) > 0.3 for tb in tracked_boxes):
                continue                            # that one is another (tracked) car
            gx, gy = (box[0] + box[2]) / 2.0, box[3]
            proj = self.path.project(gx, gy, lo, 0.0, max(0.0, (hi - lo) * self.path.scale))
            if not self.path.in_corridor(proj, c.corridor_factor) or not lo - 1e-9 <= proj.s <= hi + 1e-9:
                continue
            if best is None or abs(proj.s - s_ref) < abs(best[0].s - s_ref):
                best = (proj, box, (gx, gy))
        if best is None:
            return False
        proj, box, ground = best
        if not self._fold(ftp, t, proj.s):
            return False
        ftp.last_seen_t = t
        ftp.ground_hist.append((t, ground[0], ground[1]))
        ftp.box, ftp.ground, ftp.lateral_px = box, ground, proj.lateral_px
        ftp.off_corridor_frames = 0
        return True

    def flush(self):
        """End of input: every remaining track is finished."""
        for tid in list(self.tracks):
            self._finish(self.tracks.pop(tid))

    def _finish(self, tp: TrackProgress):
        c = self.cfg
        if tp.n_updates < c.profile_min_updates:
            return
        ss = [smp[2] for smp in tp.samples]
        if max(ss) - min(ss) < c.profile_min_span:
            return
        # skip the first two updates: the filter's speed hasn't converged yet
        if self.profile.add_pass([(smp[2], smp[3]) for smp in tp.samples[2:]]):
            self.finished_passes += 1
            if self.on_finished:
                self.on_finished(tp)

    # --------------------------------------------------------------------------------------------- prediction
    def frozen_speed(self, tp: TrackProgress):
        """The speed a dropout prediction starts from: the filter's speed at its last update (prediction steps never
        change it), clamped to speed_band x the car's mean speed over its last speed_band_window_s of detections
        (the slope of a straight line through them). A speed that jumped on the last noisy update can't fly the
        prediction away. No clamp when there are too few detections to tell, or the mean isn't forward."""
        c = self.cfg
        v0 = tp.filt.v
        w = c.speed_band_window_s
        recent = []
        for smp in reversed(tp.samples):
            if smp[0] < tp.last_update_t - w - 1e-9:
                break
            recent.append(smp)
        if len(recent) < 3 or recent[0][0] - recent[-1][0] < w / 2 - 1e-9:
            return v0
        ts = np.array([smp[0] for smp in recent])
        ss = np.array([smp[1] for smp in recent])
        mean_v = float(np.polyfit(ts - ts.mean(), ss, 1)[0])
        if mean_v <= 1e-3:
            return v0
        return float(np.clip(v0, c.speed_band[0] * mean_v, c.speed_band[1] * mean_v))

    def prediction(self, tp: TrackProgress, t) -> Prediction:
        """Where the track is at time t. Right after an update (t = its time) that's the filter. During a dropout:
        * the speed starts from `frozen_speed` (the last measured one, clamped);
        * before the speed profile has learned: constant at that speed;
        * once learned: it blends linearly from the frozen speed to the typical speed at the predicted s, reaching
          it after typ_blend_s of dropout (small Euler steps, so a car that reaches a braking spot slows down);
        * at most max_prediction_s of it; and never more than max_predict_s_ahead past the last measured s: there
          the prediction holds (v = 0), so the camera waits for the car instead of running off ahead of it."""
        c = self.cfg
        dt = t - tp.last_update_t
        s_k, v_k, vs, vv = tp.filt.predict(t)
        if dt <= 1e-9:
            return Prediction(s_k, v_k, s_k, v_k, vs, vv, False)
        dt = min(dt, c.max_prediction_s)
        s0 = s = tp.filt.s
        v0 = v = self.frozen_speed(tp)
        if self.profile.learned:
            n = max(1, int(math.ceil(dt / 0.05)))
            h = dt / n
            for i in range(n):
                v = self._blend(v0, self.profile.v_typ(s), (i + 0.5) * h)
                s += v * h
            v = self._blend(v0, self.profile.v_typ(s), dt)
        else:
            s = s0 + v0 * dt
        s_free, v_free = s, v
        held = c.max_predict_s_ahead > 0 and s > s0 + c.max_predict_s_ahead
        if held:
            s, v = s0 + c.max_predict_s_ahead, 0.0
        return Prediction(s, v, s_free, v_free, vs, vv, held)

    def _blend(self, v0, v_typ, tau):
        w = 1.0 if self.cfg.typ_blend_s <= 0 else min(1.0, tau / self.cfg.typ_blend_s)
        return (1.0 - w) * v0 + w * v_typ

    def predict_s_v(self, tp: TrackProgress, t):
        """(s, v, var_s, var_v) of a track at time t, as the director and controller see it (held at the cap)."""
        p = self.prediction(tp, t)
        return p.s, p.v, p.var_s, p.var_v

    def ground_speed(self, tp: TrackProgress, t):
        """Image speed of the ground point (px/s) over the last ground_window_s, or None: fewer than two detections
        in the window, spanning less than half of it, or the newest one older than half the window."""
        w = self.cfg.ground_window_s
        recent = [h for h in tp.ground_hist if h[0] >= t - w - 1e-9]
        if len(recent) < 2 or t - recent[-1][0] > w / 2:
            return None
        dt = recent[-1][0] - recent[0][0]
        if dt < w / 2 - 1e-9:
            return None
        return math.hypot(recent[-1][1] - recent[0][1], recent[-1][2] - recent[0][2]) / dt

    def view(self, tp: TrackProgress, t) -> TrackView:
        p = self.prediction(tp, t)
        age = max(0.0, t - tp.last_update_t)
        return TrackView(tp.track_id, p.s, p.v, math.sqrt(max(p.var_s, 0)), math.sqrt(max(p.var_v, 0)),
                         tp.lateral_px, tp.n_updates, age, tp.off_corridor_frames, bool(age > 1e-9), tp.box,
                         tp.ground, self.ground_speed(tp, t), self.profile.v_typ(p.s) if self.profile.learned else None,
                         bool(p.held))


def _iou(a, b):
    """Intersection over union of two xyxy boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
