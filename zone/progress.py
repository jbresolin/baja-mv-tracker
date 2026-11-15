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
        centres = np.array([self.bin_centre(i) for i in range(self.n_bins)])
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
    ratio_clip: tuple = (0.4, 2.5)      # a car's speed relative to the typical profile, when predicting ahead
    max_prediction_s: float = 3.0       # never predict further than this past the last update
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
    ratio: float = 1.0                  # v / v_typ(s) at the last update (car-specific speed factor)
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

    def to_log(self):
        return dict(id=self.track_id, s=round(self.s, 4), v=round(self.v, 4), s_std=round(self.s_std, 4),
                    v_std=round(self.v_std, 4), lat=round(self.lateral_px, 1), n=self.n_updates,
                    age=round(self.since_update_s, 3), off=self.off_corridor_frames, pred=self.predicted,
                    gpx=None if self.ground_px_s is None else round(self.ground_px_s, 1),
                    vtyp=None if self.v_typ is None else round(self.v_typ, 4))


class ProgressTracker:
    def __init__(self, path: PathModel, settings: ProgressSettings = None, profile: SpeedProfile = None):
        self.path = path
        self.cfg = settings or ProgressSettings()
        self.profile = profile or SpeedProfile(s_lo=min(-0.5, path.s_min), s_hi=max(1.2, path.s_max))
        self.tracks: Dict[int, TrackProgress] = {}
        self.finished_passes = 0
        self.on_finished = None          # optional callback(TrackProgress) (e.g. to save the profile)

    def set_path(self, path: PathModel):
        """The zone was edited: drop the tracks (their s meant something else) but keep the learned profile."""
        self.path = path
        self.tracks.clear()

    # --------------------------------------------------------------------------------------------- per frame
    def update(self, t, tracker_out):
        """t = grab time of the frame (seconds, monotonic clock). tracker_out = [(track_id, xyxy, score), ...] as
        returned by rt's ByteTracker. Returns {track_id: TrackView} for every live track at time t."""
        c = self.cfg
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
                s_prev, _, var_s, _ = tp.filt.predict(t)
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
                if tp.filt.update(t, proj.s):
                    tp.n_updates += 1
                    tp.last_update_t = t
                    tp.samples.append((t, proj.s, tp.filt.s, tp.filt.v))
                    vt = self.profile.v_typ(tp.filt.s)
                    if vt > 1e-3 and tp.n_updates >= 3:
                        tp.ratio = float(np.clip(tp.filt.v / vt, *c.ratio_clip))
            tp.off_corridor_frames = 0
            tp.lateral_px = proj.lateral_px
            tp.box, tp.ground = tuple(float(v) for v in box), (gx, gy)
        # finish tracks that have been gone long enough
        for tid in [k for k, tp in self.tracks.items() if t - tp.last_seen_t > c.finish_after_s]:
            self._finish(self.tracks.pop(tid))
        return {tid: self.view(tp, t) for tid, tp in self.tracks.items()}

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
    def predict_s_v(self, tp: TrackProgress, t):
        """s and v of a track at time t. Right after an update this is the filter's constant-speed prediction. After
        a dropout (no fresh detection), it rides the learned speed profile scaled by the car's own speed ratio, so a
        car that brakes where cars usually brake is predicted to brake too."""
        dt = t - tp.last_update_t
        s, v, vs, vv = tp.filt.predict(t)
        if dt <= 0 or not self.profile.learned:
            return s, v, vs, vv
        dt = min(dt, self.cfg.max_prediction_s)
        s = tp.filt.s
        n = max(1, int(math.ceil(dt / 0.05)))
        h = dt / n
        for _ in range(n):  # small Euler steps along the profile
            v = self.profile.v_typ(s) * tp.ratio
            s += v * h
        return s, v, vs, vv

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
        s, v, vs, vv = self.predict_s_v(tp, t)
        age = max(0.0, t - tp.last_update_t)
        return TrackView(tp.track_id, s, v, math.sqrt(max(vs, 0)), math.sqrt(max(vv, 0)), tp.lateral_px,
                         tp.n_updates, age, tp.off_corridor_frames, bool(age > 1e-9), tp.box, tp.ground,
                         self.ground_speed(tp, t), self.profile.v_typ(s) if self.profile.learned else None)
