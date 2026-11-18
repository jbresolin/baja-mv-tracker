"""Path geometry: turn the taught polyline into progress `s`, and keyframe poses into a pose curve pose(s).

Progress `s`
------------
`s` is the arc length along the polyline (in pixels of the spotter image), rescaled so that the **first keyframe is
s = 0 and the last keyframe is s = 1**. Points upstream of the first keyframe get negative `s` (the approach, where
the car's speed is measured before the PTZ starts moving); points after the last keyframe get s > 1.

Pixel arc length is not proportional to metres (perspective), and it doesn't need to be: `s` only has to grow
monotonically along the path. The learned speed profile absorbs the non-uniformity.

Projection
----------
A car's ground point (bottom centre of its box) is projected onto the polyline: the closest point on it gives `s`,
and the distance to it is the *lateral offset*. If the path comes back near itself in the image (a hairpin seen from
the side), the closest point could jump to the wrong leg. So, when we know where the car was a moment ago, we search
only a window of the path around that previous position ("monotonic window": a little behind, more ahead).

Corridor width
--------------
The corridor is the band around the path where detections count as cars of the zone. Its half-width is the zone's
global `corridor_half_width_px`, unless points carry their own `half_width_px`: then it is interpolated along the path
(`Zone.point_half_widths`, `half_width_at_arc`). The detector ROI, the in-corridor test and the page's drawing all use
the varying width.

Overlapping stretches ("legs")
------------------------------
When the path passes near itself, a NEW detection may be inside the corridor of two stretches that are far apart along
the path. `candidates()` lists them; the progress tracker (zone/progress.py) then picks one with the learned box-size
profile, or waits.

Pose curve
----------
Each keyframe has an `s` (the arc length at its point) and a pose (pan, tilt, zoom). `PoseCurve` interpolates them:
"pchip" (monotone cubic, the default: smooth speed, never overshoots between keyframes) or "linear". Before s = 0 the
pose is the entry keyframe's, after s = 1 the exit keyframe's. Its derivative d(pose)/ds is the feed-forward term of
the controller: pan rate = dpan/ds * v.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from zone.zonefile import Zone


@dataclass
class Candidate:
    """One stretch of path a detection could belong to (see PathModel.candidates)."""
    projection: "Projection"
    distance_px: float      # distance from the detection to the path there
    half_width_px: float    # corridor half-width at that place


@dataclass
class Projection:
    s: float            # progress (normalized: 0 = first keyframe, 1 = last keyframe)
    lateral_px: float   # signed distance from the path (pixels); positive = left of the direction of travel
    arc_px: float       # arc length from the first path point (pixels)
    x: float            # the projected point on the path (pixels)
    y: float

    @property
    def offset_px(self):
        return abs(self.lateral_px)


class PathModel:
    """Built from a Zone. Needs at least two keyframes (to define s = 0 and s = 1); `PathModel.build` returns None
    otherwise, so callers can tell 'not taught yet' apart from a real path."""

    def __init__(self, zone: Zone):
        pts = np.array([[p.x, p.y] for p in zone.points], dtype=np.float64)
        kfi = zone.keyframe_indices()
        if len(pts) < 2 or len(kfi) < 2:
            raise ValueError("a path needs at least 2 points and 2 keyframes (has %d points, %d keyframes)"
                             % (len(pts), len(kfi)))
        seg = np.diff(pts, axis=0)
        seglen = np.hypot(seg[:, 0], seg[:, 1])
        self.points = pts
        self.seg = seg
        self.seglen = seglen
        self.cum = np.r_[0.0, np.cumsum(seglen)]          # arc length at each point
        self.arc0 = self.cum[kfi[0]]                       # arc length of s = 0
        self.arc1 = self.cum[kfi[-1]]                      # arc length of s = 1
        if self.arc1 - self.arc0 <= 1e-6:
            raise ValueError("the first and last keyframes are at the same place on the path")
        self.scale = self.arc1 - self.arc0                 # pixels per unit of s
        self.half_width = float(zone.corridor_half_width_px)          # the global value (fallback)
        self.point_widths = np.array(zone.point_half_widths(), dtype=np.float64)   # half-width at each point
        self.max_half_width = float(self.point_widths.max())
        self.keyframe_indices = kfi

    @classmethod
    def build(cls, zone: Zone) -> Optional["PathModel"]:
        try:
            return cls(zone)
        except ValueError:
            return None

    # ----------------------------------------------------------------- s <-> arc length
    def s_of_arc(self, arc):
        return (np.asarray(arc, dtype=np.float64) - self.arc0) / self.scale

    def arc_of_s(self, s):
        return np.asarray(s, dtype=np.float64) * self.scale + self.arc0

    @property
    def s_min(self):
        """s of the first path point (<= 0: the approach)."""
        return float(self.s_of_arc(0.0))

    @property
    def s_max(self):
        """s of the last path point (>= 1)."""
        return float(self.s_of_arc(self.cum[-1]))

    def point_at(self, s):
        """The image point (x, y) at progress s (clamped to the ends of the polyline)."""
        a = float(np.clip(self.arc_of_s(s), 0.0, self.cum[-1]))
        i = int(np.clip(np.searchsorted(self.cum, a, side="right") - 1, 0, len(self.seg) - 1))
        t = (a - self.cum[i]) / self.seglen[i] if self.seglen[i] > 0 else 0.0
        p = self.points[i] + t * self.seg[i]
        return float(p[0]), float(p[1])

    def keyframe_s(self):
        return [float(self.s_of_arc(self.cum[i])) for i in self.keyframe_indices]

    # ----------------------------------------------------------------- projection
    def project(self, x, y, s_prev=None, back_px=40.0, ahead_px=400.0) -> Projection:
        """Closest point on the path to (x, y).

        With `s_prev` (the car's previous progress), only the stretch of path from `back_px` behind to `ahead_px`
        ahead of it (in arc length) is searched, so a path that passes near itself can't make s jump to the other
        leg. Without it, the whole path is searched."""
        p = np.array([x, y], dtype=np.float64)
        lo_arc, hi_arc = 0.0, self.cum[-1]
        if s_prev is not None:
            a = float(self.arc_of_s(s_prev))
            lo_arc, hi_arc = max(0.0, a - back_px), min(self.cum[-1], a + ahead_px)
        best = None
        for i in range(len(self.seg)):
            a0, a1 = self.cum[i], self.cum[i + 1]
            if a1 < lo_arc or a0 > hi_arc or self.seglen[i] <= 0:
                continue
            # allowed part of this segment, as a fraction 0..1 along it
            t_lo = max(0.0, (lo_arc - a0) / self.seglen[i])
            t_hi = min(1.0, (hi_arc - a0) / self.seglen[i])
            d = self.seg[i]
            t = float(np.dot(p - self.points[i], d) / (self.seglen[i] ** 2))
            t = min(max(t, t_lo), t_hi)
            q = self.points[i] + t * d
            dist = float(np.hypot(*(p - q)))
            if best is None or dist < best[0] - 1e-9:
                cross = d[0] * (p[1] - q[1]) - d[1] * (p[0] - q[0])
                # image y points down, so a positive cross product is to the RIGHT of travel; flip for "left = +"
                lateral = -dist if cross > 0 else dist
                best = (dist, a0 + t * self.seglen[i], lateral, q)
        if best is None:  # window fell outside the path (can't happen with a sane s_prev); search everything
            return self.project(x, y)
        _, arc, lateral, q = best
        return Projection(s=float(self.s_of_arc(arc)), lateral_px=float(lateral), arc_px=float(arc),
                          x=float(q[0]), y=float(q[1]))

    # ----------------------------------------------------------------- corridor width
    def half_width_at_arc(self, arc):
        """Corridor half-width (pixels) at arc length `arc`: linear between the points' widths, held at the ends."""
        return float(np.interp(float(arc), self.cum, self.point_widths))

    def half_width_at_s(self, s):
        return self.half_width_at_arc(float(self.arc_of_s(s)))

    def in_corridor(self, proj: Projection, factor=1.0):
        return proj.offset_px <= self.half_width_at_arc(proj.arc_px) * factor

    # ----------------------------------------------------------------- several legs near one detection
    def candidates(self, x, y, factor=1.0, min_separation_px=200.0):
        """Every stretch of path whose corridor contains (x, y), as a list of Candidate sorted by arc length.

        Each segment gives its closest point to (x, y); segments in a row whose closest point is inside the corridor
        form one stretch, represented by its closest point. Stretches less than `min_separation_px` apart along the
        path are the same place (e.g. the two sides of a tight bend): only the closer one is kept. So:
          * []: the detection is outside the corridor;
          * one candidate: unambiguous, and it is the plain closest point (`project(x, y)`) when that is inside;
          * two or more: the path passes near itself here (overlapping legs): the caller must choose."""
        p = np.array([float(x), float(y)])
        runs = []                      # [(distance, arc, segment index, t)] best of each run of inside segments
        current = None
        for i in range(len(self.seg)):
            if self.seglen[i] <= 0:
                continue
            d = self.seg[i]
            t = min(1.0, max(0.0, float(np.dot(p - self.points[i], d) / (self.seglen[i] ** 2))))
            q = self.points[i] + t * d
            dist = float(np.hypot(*(p - q)))
            arc = self.cum[i] + t * self.seglen[i]
            inside = dist <= self.half_width_at_arc(arc) * factor
            if not inside:
                current = None
                continue
            if current is None:
                current = [dist, arc, i, t]
                runs.append(current)
            elif dist < current[0] - 1e-9:
                current[:] = [dist, arc, i, t]
        runs.sort(key=lambda r: r[1])
        kept = []
        for r in runs:
            if kept and r[1] - kept[-1][1] < float(min_separation_px):
                if r[0] < kept[-1][0]:
                    kept[-1] = r          # same place: keep the closer one
                continue
            kept.append(r)
        out = []
        for dist, arc, i, t in kept:
            q = self.points[i] + t * self.seg[i]
            d = self.seg[i]
            cross = d[0] * (p[1] - q[1]) - d[1] * (p[0] - q[0])
            lateral = -dist if cross > 0 else dist
            proj = Projection(s=float(self.s_of_arc(arc)), lateral_px=float(lateral), arc_px=float(arc),
                              x=float(q[0]), y=float(q[1]))
            out.append(Candidate(proj, dist, self.half_width_at_arc(arc)))
        return out

    # ----------------------------------------------------------------- region of interest
    def corridor_bbox(self, margin_px=0.0, image_size=None):
        """(x0, y0, x1, y1) integer bounding box of the corridor (polyline +- its half width, which may vary from
        point to point) plus a margin, clipped to the image when its size is given. Each segment's band lies inside
        the two discs at its ends (radius = the width there) and the hull between them, so the box of those discs
        holds the whole band."""
        r = self.point_widths[:, None] + float(margin_px)
        x0, y0 = (self.points - r).min(0)
        x1, y1 = (self.points + r).max(0)
        if image_size:
            w, h = image_size
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(w), x1), min(float(h), y1)
        return int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))


# ===================================================================================================== pose curve
def _pchip_slopes(x, y):
    """Fritsch-Carlson slopes for monotone cubic Hermite interpolation (what scipy's PchipInterpolator does)."""
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.zeros(n)
    if n == 2:
        m[:] = delta[0]
        return m
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] <= 0:
            m[k] = 0.0          # local extremum (or flat): zero slope, so the curve never overshoots
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    # end slopes: one-sided three-point estimate, limited so the end segments stay monotone too
    for k, (d0, d1, h0, h1) in ((0, (delta[0], delta[1], h[0], h[1])),
                                (n - 1, (delta[-1], delta[-2], h[-1], h[-2]))):
        e = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(e) != np.sign(d0):
            e = 0.0
        elif np.sign(d0) != np.sign(d1) and abs(e) > abs(3 * d0):
            e = 3 * d0
        m[k] = e
    return m


class _Curve1D:
    """y(x) through the knots, linear or pchip, constant outside [x0, xn] (derivative 0 there)."""

    def __init__(self, x, y, kind="pchip"):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.kind = kind
        self.m = _pchip_slopes(self.x, self.y) if kind == "pchip" else None

    def _locate(self, s):
        i = int(np.clip(np.searchsorted(self.x, s, side="right") - 1, 0, len(self.x) - 2))
        h = self.x[i + 1] - self.x[i]
        return i, h, (s - self.x[i]) / h

    def value(self, s):
        if s <= self.x[0]:
            return float(self.y[0])
        if s >= self.x[-1]:
            return float(self.y[-1])
        i, h, t = self._locate(s)
        y0, y1 = self.y[i], self.y[i + 1]
        if self.kind == "linear":
            return float(y0 + t * (y1 - y0))
        m0, m1 = self.m[i], self.m[i + 1]
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        return float(h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1)

    def derivative(self, s):
        if s < self.x[0] or s > self.x[-1]:
            return 0.0
        i, h, t = self._locate(min(s, self.x[-1] - 1e-12))
        y0, y1 = self.y[i], self.y[i + 1]
        if self.kind == "linear":
            return float((y1 - y0) / h)
        m0, m1 = self.m[i], self.m[i + 1]
        d00 = (6 * t ** 2 - 6 * t) / h
        d10 = 3 * t ** 2 - 4 * t + 1
        d01 = (-6 * t ** 2 + 6 * t) / h
        d11 = 3 * t ** 2 - 2 * t
        return float(d00 * y0 + d10 * m0 + d01 * y1 + d11 * m1)


@dataclass
class Pose:
    pan: float
    tilt: float
    zoom: float


class PoseCurve:
    """pose(s) and d(pose)/ds from the keyframes. `kind` = "pchip" (default) or "linear".

    `ease_in` (in s, default 0 = off): the camera can't jump from standing still to the car's pan rate the moment the
    car crosses s = 0, so it starts behind (the "entry lag"). With ease_in = p, the pose follows a parabola from the
    entry pose at s = -p (zero slope) to the taught curve at s = +p (same value, about the same slope): the camera
    starts turning a moment before the car reaches the entry keyframe. Then the director must lock at s_lock <= -p.
    """

    def __init__(self, s_knots, poses, kind="pchip", ease_in=0.0):
        s = np.asarray(s_knots, dtype=np.float64)
        if len(s) < 2:
            raise ValueError("need at least two keyframes")
        if np.any(np.diff(s) <= 0):
            raise ValueError("keyframes must be in increasing s order (two keyframes at the same place?)")
        self.s = s
        self.kind = kind
        self.pan = _Curve1D(s, [p.pan for p in poses], kind)
        self.tilt = _Curve1D(s, [p.tilt for p in poses], kind)
        self.zoom = _Curve1D(s, [p.zoom for p in poses], kind)
        self.ease = max(0.0, min(float(ease_in), (s[1] - s[0]) / 2))   # never past halfway to the 2nd keyframe

    @classmethod
    def from_zone(cls, zone: Zone, path: PathModel, kind="pchip", ease_in=0.0):
        kfs = zone.keyframes()
        return cls(path.keyframe_s(), [Pose(k.pan, k.tilt, k.zoom) for k in kfs], kind, ease_in)

    def _eased(self, curve, s):
        """(value, derivative) of one axis with the ease-in applied."""
        p = self.ease
        s0 = self.s[0]
        if p <= 0 or not (s0 - p < s < s0 + p):
            return curve.value(s), curve.derivative(s)
        y0 = curve.value(s0)
        m = (curve.value(s0 + p) - y0) / p          # secant slope over the first p of the zone
        u = s - s0 + p                              # 0 .. 2p
        return y0 + m * u * u / (4 * p), m * u / (2 * p)

    def pose(self, s) -> Pose:
        return Pose(self._eased(self.pan, s)[0], self._eased(self.tilt, s)[0], self._eased(self.zoom, s)[0])

    def dpose_ds(self, s) -> Pose:
        return Pose(self._eased(self.pan, s)[1], self._eased(self.tilt, s)[1], self._eased(self.zoom, s)[1])
