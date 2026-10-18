"""WP2c: broadcast overlay regions per video -> work/<id>/overlay_mask.png, overlays.csv, overlay_preview.jpg, overlay.done

usage: python teacher/overlay.py [--workers 3] [--force] [--reuse-p] [id ...]
       (--reuse-p: reuse work/<id>/overlay_P.npy instead of recomputing the persistence map, for tuning steps 3-5)

Overlays are screen regions that stay pixel-identical while the scene behind them changes: leaderboard towers,
tickers, corner logos, lower thirds, bracket boxes, rotating sponsor logos (fixed slot, changing content), picture-in-
picture insets (fixed frame, changing content).

Method (CPU; keyframes only: the streams have a 5 s GOP, so decoding I-frames is ~100-200x real time):
 1. Decode every keyframe at 640x360 -> Canny edges E_t, plus a frame signature (32x18 edge-density grid, z-scored).
 2. Persistence map P(x) (two passes; pass 2 leaves the pass-1 regions out of the signatures, so a big overlay
    such as the SPP bracket does not make the same static view look like a different scene): for each keyframe i, up to K partner keyframes j that are >= 2 min away AND show a
    different scene (edge-layout signature correlation < SCENE_CORR and colour histogram distance > HIST_MIN; the
    colour test stops studio shots, whose layout changes as people move, from pairing with other studio shots). Coincident edges C = E_i & E_j count only where they are a
    near-complete local match (box(C) >= 0.7 * max(box(E_i), box(E_j)) in a 9x9 window): overlay text and borders
    coincide as whole structures, random texture coincides sparsely. P(x) = share of frames i with such a match with
    at least one partner ~ share of time an overlay is on screen there.
    The scene-difference condition is what removes static cameras (19SU start-line camera, 24SU SPP camera, studio
    shots): their background coincides with frames of the same view, never with frames of a different scene.
 3. P >= P_MIN, closing, connected components -> boxes (1280x720 coordinates). Thin streaks (macroblock-grid
    artifacts of the low-bitrate encode are persistent too) and tiny blobs are dropped.
 4. Per box: s(t) = share of its persistent edge pixels present in keyframe t (1 px tolerance). on(t) = s(t) above
    an adaptive threshold (half of its 95th percentile, >= 0.25). Runs of 'on' keyframes (gaps <= 2 keyframes
    closed, runs >= 2 keyframes) -> visibility intervals.
 5. Kind guess from position and shape; mask = union of boxes (+4 px), 255 = overlay.

Overrides: teacher/overlay_overrides.csv (id,action,x1,y1,x2,y2,kind,t_ref,note; 1280x720 coordinates) with action
add | drop. drop removes detected boxes whose IoU with the given box is > 0.3. add inserts a box; its visibility is
the share of its thumbnail cells that match the keyframe nearest to t_ref (a time where the graphic is on screen),
or always-on without t_ref. Used for the few graphics the automatic step gets wrong (text-only lower thirds).
"""
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, cli_ids, ffmpeg_frames, fmt_t, frame_at, run_stage, select_videos, wdir  # noqa: E402

W, H = 640, 360
SCALE = 2  # to 1280x720
K = 24            # partners per frame
MIN_GAP_S = 120   # partners at least this far apart in time
TRIES = 3         # pass 1: partners = the K least similar of TRIES*K random far keyframes; pass 2: tries
SAME_MAX = float(os.environ.get("OVL_SAME_MAX", "0.3"))  # pass 2: max share of same-looking cells (other view)
TW, TH = 64, 36   # thumbnail grid for the context test (10x10 px cells at 640x360)
LDIFF = 0.6       # thumbnail cell 'same': normalised lightness difference below this ...
CDIFF = 10.0      # ... and chroma (Lab a, b) difference below this
RING_R0, RING_R1 = 3, 9   # context annulus radius in cells (30-90 px at 640x360, 60-180 px at 1280x720)
CTX_MIN = float(os.environ.get("OVL_CTX_MIN", "0.3"))  # share of differing cells on the annulus
WIN = 9           # local window for the coincidence test (px at 640x360)
MATCH = 0.7       # coincident share of the local edges for a 'structure match'
P_MIN = 0.004     # min persistence for an overlay pixel (rare lower thirds are ~0.005-0.02)
MIN_MASS = 3.0    # min sum of P over a region's persistent (non-line) pixels
BORDER = 5        # ignore the outer pixels (encoder / letterbox borders are persistent lines)
LINE_LEN = 31     # straight runs at least this long (px at 640x360) are treated as lines
LINE_P = 0.05     # a free-standing line is a region only if this persistent (e.g. a ticker bar edge)
TRIM = 0.15       # region box = extent of its pixels with P >= TRIM * (region's P90): drops faint tails
VIS_LIFT = 3.0    # visibility: reference edges present at >= 3x the chance rate
MIN_COVER = 0.01  # regions visible less than 1% of the video are dropped (spurious)
MANUAL_MATCH = 0.5  # manual regions with t_ref: on when >= this share of thumbnail cells match the reference
OVR = Path(__file__).resolve().parent / "overlay_overrides.csv"


def edges(f):
    return cv2.Canny(f, 60, 150) > 0


def load_keyframes(path, cache=None):
    """Keyframe times, Canny edge maps (bool, n x 360 x 640) and 64x36 Lab thumbnails (uint8). Frames are not
    kept. `cache`: optional .npz path for the packed edges + thumbnails (set OVL_CACHE=1 while tuning)."""
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        if "T" in z:
            E = np.unpackbits(z["E"], axis=1)[:, :W * H].reshape(-1, H, W).astype(bool)
            return z["ts"], E, z["T"]
    ts, E, T = [], [], []
    for t, f in ffmpeg_frames(path, size=(W, H), pix="rgb24", threads=2, keyframes_only=True):
        ts.append(t)
        E.append(edges(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)))
        T.append(cv2.cvtColor(cv2.resize(f, (TW, TH), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB))
    E, T = np.stack(E), np.stack(T)
    if cache is not None:
        np.savez(cache, E=np.packbits(E.reshape(len(E), -1), axis=1), ts=np.array(ts), T=T)
    return np.array(ts), E, T


def cell_features(T):
    """Per thumbnail cell: lightness normalised per frame by median / IQR (robust to exposure changes and to
    overlays covering part of the frame) and the two chroma channels. -> float32 (n, TH, TW, 3)."""
    T = T.astype(np.float32)
    L = T[..., 0].reshape(len(T), -1)
    med = np.median(L, 1)[:, None, None]
    q75, q25 = np.percentile(L, [75, 25], axis=1)
    Ln = (T[..., 0] - med) / ((q75 - q25)[:, None, None] + 5.0)
    return np.concatenate([Ln[..., None], T[..., 1:] - 128.0], -1)


def cells_same(fi, fj):
    """Thumbnail cells that look the same in two frames (same scene content at that screen position)."""
    return (np.abs(fi[..., 0] - fj[..., 0]) < LDIFF) & (np.abs(fi[..., 1:] - fj[..., 1:]).max(-1) < CDIFF)


def _annulus(r0, r1):
    y, x = np.mgrid[-r1:r1 + 1, -r1:r1 + 1]
    d = np.hypot(x, y)
    return ((d >= r0) & (d <= r1)).astype(np.float32)


RING = None


def context_differs(fi, fj):
    """Per pixel (640x360 bool): do the SURROUNDINGS of this point differ between frames i and j? Share of
    differing thumbnail cells (10x10 px) on an annulus of RING_R0..RING_R1 cells around each point >= CTX_MIN.
    An overlay keeps its own pixels but sits on a changing background; a fixed camera's scene does not."""
    global RING
    if RING is None:
        RING = _annulus(RING_R0, RING_R1)
    dz = (~cells_same(fi, fj)).astype(np.float32)
    num = cv2.filter2D(dz, -1, RING, borderType=cv2.BORDER_CONSTANT)
    den = cv2.filter2D(np.ones_like(dz), -1, RING, borderType=cv2.BORDER_CONSTANT)
    ok = (num / np.maximum(den, 1e-6)) >= CTX_MIN
    return cv2.resize(ok.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def persistence(ts, E, F, excl=None, rng=None):
    """P(x): share of keyframes i whose edge at x (a) is part of a full local structure match with a partner j and
    (b) has surroundings that differ between i and j.
    Partners: up to K keyframes >= MIN_GAP_S away that show a DIFFERENT VIEW: outside `excl` (the regions of a
    first pass), fewer than SAME_MAX of the thumbnail cells look the same. Pass 1 (excl=None) cannot discount the
    overlays themselves, so it uses the K least similar of TRIES*K random far keyframes instead (robust for static
    cameras, but it misses graphics that only appear over one kind of shot, e.g. DIN lower thirds)."""
    rng = rng or np.random.default_rng(0)
    n = len(E)
    box = lambda a: cv2.boxFilter(a.astype(np.uint8), cv2.CV_16U, (WIN, WIN), normalize=False,  # noqa: E731
                                  borderType=cv2.BORDER_CONSTANT)
    if excl is None:
        Zf = F[..., 0].reshape(n, -1)
        Zf = (Zf - Zf.mean(1, keepdims=True)) / (Zf.std(1, keepdims=True) + 1e-3)
    else:
        keep = ~(cv2.resize(excl.astype(np.uint8), (TW, TH), interpolation=cv2.INTER_AREA) > 0)
    P = np.zeros((H, W), np.float32)
    used = 0
    for i in range(n):
        far = np.where(np.abs(ts - ts[i]) >= MIN_GAP_S)[0]
        if len(far) == 0:
            continue
        cand = rng.permutation(far)[: TRIES * K]
        if excl is None:
            corr = (Zf[cand] @ Zf[i]) / Zf.shape[1]
            js = cand[np.argsort(corr)[:K]]
        else:
            same = np.array([cells_same(F[i], F[j])[keep].mean() for j in cand])
            js = cand[same < SAME_MAX][:K]
        if len(js) == 0:
            continue
        Di = box(E[i])
        best = np.zeros((H, W), bool)
        for j in js:
            Cc = E[i] & E[j]
            if Cc.sum() < 20:
                continue
            dc = box(Cc)
            m = Cc & (dc >= 0.06 * WIN * WIN) & (dc >= MATCH * np.maximum(Di, box(E[j])))
            if m.any():
                best |= m & context_differs(F[i], F[j])
        used += 1
        P += best
    return P / max(used, 1), used


def boxes_from(P):
    """Regions from the persistence map. Long straight lines (ticker edges, box borders, full-width bars, encoder
    grid streaks) would chain unrelated overlays into one frame-sized component, so components are built from the
    non-line pixels first; a line then extends the region it borders, and a strong line that borders nothing becomes
    a thin region of its own (e.g. a ticker bar edge)."""
    P = P.copy()
    b = BORDER
    P[:b], P[-b:], P[:, :b], P[:, -b:] = 0, 0, 0, 0
    pm = (P >= P_MIN).astype(np.uint8)
    hl = cv2.morphologyEx(pm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_LEN, 1)))
    vl = cv2.morphologyEx(pm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, LINE_LEN)))
    lines = (hl | vl).astype(bool)
    core = pm.astype(bool) & ~lines
    m = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    parts = []
    for k in range(1, n):
        x, y, w, h, _ = stats[k]
        comp = lab == k
        if (comp & core).sum() < 60 or w < 8 or h < 8:
            continue
        parts += guillotine(comp)
    regs = []
    for comp in parts:
        ref = comp & core
        if ref.sum() < 60:
            continue
        mass = float(P[ref].sum())
        if mass < MIN_MASS:
            continue
        strong = ref & (P >= TRIM * np.percentile(P[ref], 90))
        ys, xs = np.nonzero(strong)
        if len(xs) < 20:
            continue
        x, y, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        if x2 - x < 8 or y2 - y < 8:
            continue
        box_ref = np.zeros_like(ref)
        box_ref[y:y2, x:x2] = ref[y:y2, x:x2]
        regs.append([x, y, x2, y2, box_ref, (x, y, x2, y2), "core"])
    # closed rectangles drawn by lines (sponsor slots, picture-in-picture frames, boxed labels): the enclosed area
    # is a region even when its content changes (rotating logos)
    L = cv2.morphologyEx((hl | vl), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    nh, hlab, hst, _ = cv2.connectedComponentsWithStats((1 - L).astype(np.uint8), connectivity=4)
    for k in range(1, nh):
        x, y, w, h, a = hst[k]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:  # touches the frame border: not enclosed
            continue
        if a < 300 or a < 0.7 * w * h or w < 12 or h < 12 or a > 0.15 * W * H:
            continue
        hole = (hlab == k).astype(np.uint8)
        rim = (cv2.dilate(hole, np.ones((7, 7), np.uint8)) > 0) & ~hole.astype(bool) & (hl | vl).astype(bool)
        if rim.sum() < 20 or P[rim].mean() < 0.01:  # enclosed only by faint lines (encoder grid): not a box
            continue
        x1r, y1r, x2r, y2r = max(x - 3, 0), max(y - 3, 0), min(x + w + 3, W), min(y + h + 3, H)
        ref = np.zeros((H, W), bool)
        ref[y1r:y2r, x1r:x2r] = pm[y1r:y2r, x1r:x2r].astype(bool)
        add_region(regs, x1r, y1r, x2r, y2r, ref, "rect")
    # boxes whose frame is only partly persistent: two parallel lines with the same extent (left + right side of a
    # sponsor slot) span a region
    for lm, axis in ((vl, 1), (hl, 0)):
        nl, llab, lst, _ = cv2.connectedComponentsWithStats(lm, connectivity=8)
        cand = [(lst[k], float(P[llab == k].mean())) for k in range(1, nl)]
        cand = [(st, mp) for st, mp in cand if mp >= 0.01 and max(st[2], st[3]) <= 150]
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                (xa, ya, wa, ha, _), (xb, yb, wb, hb, _) = cand[i][0], cand[j][0]
                if axis == 1:  # vertical pair: similar length, top or bottom aligned
                    same = (abs(ya - yb) <= 6 or abs((ya + ha) - (yb + hb)) <= 6) and min(ha, hb) >= 0.6 * max(ha, hb)
                    gap = abs(xa - xb)
                    box = (min(xa, xb), min(ya, yb), max(xa + wa, xb + wb), max(ya + ha, yb + hb))
                else:
                    same = (abs(xa - xb) <= 6 or abs((xa + wa) - (xb + wb)) <= 6) and min(wa, wb) >= 0.6 * max(wa, wb)
                    gap = abs(ya - yb)
                    box = (min(xa, xb), min(ya, yb), max(xa + wa, xb + wb), max(ya + ha, yb + hb))
                length = max(ha, hb) if axis == 1 else max(wa, wb)
                if same and 12 <= gap <= min(200, 2 * length):  # a box, not two unrelated parallel edges
                    ref = np.zeros((H, W), bool)
                    ref[box[1]:box[3], box[0]:box[2]] = pm[box[1]:box[3], box[0]:box[2]].astype(bool)
                    add_region(regs, *box, ref, "line_pair")
    # straight line segments (horizontal and vertical separately): attach to the region they border, else keep
    # strong long ones as thin regions of their own (e.g. a ticker bar edge)
    segs = []
    for lm in (hl, vl):
        nl, llab, lst, _ = cv2.connectedComponentsWithStats(lm, connectivity=8)
        segs += [(lst[k], llab == k) for k in range(1, nl)]
    for (x, y, w, h, _), seg in segs:
        meanp = float(P[seg].mean())
        x2, y2 = x + w, y + h
        host = None
        for r in regs:
            if r[5] is None:  # free-standing line regions do not host other lines
                continue
            cx1, cy1, cx2, cy2 = r[5]  # original core box: no chain growth through successive lines
            near = x <= cx2 + 6 and x2 >= cx1 - 6 and y <= cy2 + 6 and y2 >= cy1 - 6
            # a border of the host: lies mostly within the host's extent along its own direction
            if w >= h:
                fits = max(0, min(x2, cx2 + 6) - max(x, cx1 - 6)) >= 0.7 * w
            else:
                fits = max(0, min(y2, cy2 + 6) - max(y, cy1 - 6)) >= 0.7 * h
            if near and fits:
                host = r
                break
        if host is not None:
            host[0], host[1] = min(host[0], int(x)), min(host[1], int(y))
            host[2], host[3] = max(host[2], int(x2)), max(host[3], int(y2))
            host[4] = host[4] | seg
        elif meanp >= LINE_P and w >= 3 * LINE_LEN and h < w:
            # free-standing horizontal edge (ticker / bar); near the top or bottom, the bar reaches the frame edge
            ya, yb = max(int(y) - 2, 0), min(int(y2) + 2, H)
            if y > 0.85 * H:
                yb = H
            elif y2 < 0.15 * H:
                ya = 0
            regs.append([int(x), ya, int(x2), yb, seg, None, "bar_edge"])
    out = []
    for x1, y1, x2, y2, ref, _, src in regs:
        out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "ref": ref, "p": float(P[ref].mean()) if ref.any() else 0.0,
                    "mass": float(P[ref].sum()), "npx": int(ref.sum()), "src": src})
    return out


def guillotine(comp, min_gain=0.3, depth=0):
    """Split a component that a single box describes badly (L shapes such as a tower meeting a ticker, or two
    overlays joined by a thin bridge) with straight cuts: pick the horizontal or vertical cut that minimises the
    summed bounding-box area of the two sides; recurse while that saves >= min_gain of the box area."""
    ys, xs = np.nonzero(comp)
    if len(xs) == 0:
        return []
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    area = (x2 - x1) * (y2 - y1)
    if depth >= 6 or area < 400 or len(xs) / area > 0.6:
        return [comp]
    best = (area, None)
    for axis in (0, 1):  # 0: horizontal cut (by rows), 1: vertical cut (by columns)
        a = comp[y1:y2, x1:x2] if axis == 0 else comp[y1:y2, x1:x2].T
        has = a.any(1)
        idx = np.arange(a.shape[1])
        lo = np.where(has, np.where(a, idx, 1 << 20).min(1), 1 << 20)
        hi = np.where(has, np.where(a, idx, -1).max(1), -1)
        r = np.arange(len(has))
        # prefix (rows < c) and suffix (rows >= c) extents
        pl, ph = np.minimum.accumulate(lo), np.maximum.accumulate(hi)
        sl, sh = np.minimum.accumulate(lo[::-1])[::-1], np.maximum.accumulate(hi[::-1])[::-1]
        first = np.where(has, r, 1 << 20)
        last = np.where(has, r, -1)
        pf, pls = np.minimum.accumulate(first), np.maximum.accumulate(last)
        sf, sls = np.minimum.accumulate(first[::-1])[::-1], np.maximum.accumulate(last[::-1])[::-1]
        for c in range(2, len(has) - 2):
            if not (has[:c].any() and has[c:].any()):
                continue
            a1 = (ph[c - 1] - pl[c - 1] + 1) * (pls[c - 1] - pf[c - 1] + 1)
            a2 = (sh[c] - sl[c] + 1) * (sls[c] - sf[c] + 1)
            if a1 + a2 < best[0]:
                best = (a1 + a2, (axis, c))
    if best[1] is None or best[0] > (1 - min_gain) * area:
        return [comp]
    axis, c = best[1]
    p1, p2 = np.zeros_like(comp), np.zeros_like(comp)
    if axis == 0:
        p1[:y1 + c] = comp[:y1 + c]
        p2[y1 + c:] = comp[y1 + c:]
    else:
        p1[:, :x1 + c] = comp[:, :x1 + c]
        p2[:, x1 + c:] = comp[:, x1 + c:]
    return guillotine(p1, min_gain, depth + 1) + guillotine(p2, min_gain, depth + 1)


def add_region(regs, x1, y1, x2, y2, ref, src):
    """Add a box region, or merge it into an existing region it mostly overlaps (or contains)."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    a = (x2 - x1) * (y2 - y1)
    for r in regs:
        ix = max(0, min(r[2], x2) - max(r[0], x1))
        iy = max(0, min(r[3], y2) - max(r[1], y1))
        ra = (r[2] - r[0]) * (r[3] - r[1])
        # merge when the new box lies mostly inside r, or r lies mostly inside a new box of comparable size
        if (a and ix * iy / a > 0.6) or (ra and ix * iy / ra > 0.6 and a <= 2 * ra):
            r[0], r[1], r[2], r[3] = min(r[0], x1), min(r[1], y1), max(r[2], x2), max(r[3], y2)
            r[4] = r[4] | ref
            if r[5] is not None:
                r[5] = (r[0], r[1], r[2], r[3])
            if src not in r[6]:
                r[6] += "+" + src
            return
    regs.append([x1, y1, x2, y2, ref, (x1, y1, x2, y2), src])


def visibility(box, E, T=None):
    """Two tests per keyframe, on = either:
    edges: s(t) = share of the box's persistent edge pixels that are edges at t (exact), c(t) = edge density of the
      rest of the box; lift = s / c is ~1 when the overlay is off and >> 1 when it is on (lift >= VIS_LIFT, s >= 0.12).
    colour (needs the Lab thumbnails T): the box's thumbnail cells match the median look of the clearly-on keyframes
      (L within 18, a/b within 10) in >= 60% of the cells. Catches scrolling tickers and graphics whose text changes,
      where only a few edges are persistent.
    Returns (lift, on)."""
    y1, y2, x1, x2 = box["y1"], box["y2"], box["x1"], box["x2"]
    r = box["ref"][y1:y2, x1:x2]
    n = len(E)
    if r.sum() == 0:
        return np.zeros(n), np.zeros(n, bool)
    nr = ~r
    e = E[:, y1:y2, x1:x2]
    s = (e & r).sum((1, 2)) / r.sum()
    c = (e & nr).sum((1, 2)) / max(int(nr.sum()), 1)
    lift = s / np.maximum(c, 0.01)
    on = (lift >= VIS_LIFT) & (s >= 0.12)
    if T is not None:
        cs = W // TW
        cx1, cy1, cx2, cy2 = -(-x1 // cs), -(-y1 // cs), x2 // cs, y2 // cs  # cells fully inside the box
        strong = on & (lift >= 1.5 * VIS_LIFT)
        if cx2 > cx1 and cy2 > cy1 and strong.sum() >= 10:
            cells = T[:, cy1:cy2, cx1:cx2].astype(np.int16)
            tmpl = np.median(cells[strong], 0)
            d = np.abs(cells - tmpl)
            match = ((d[..., 0] < 15) & (d[..., 1:].max(-1) < 8)).mean((1, 2)) >= 0.7
            bar = (x2 - x1) >= 0.6 * W  # full-width tickers scroll: no persistent edges, colour alone decides
            on = on | (match if bar else match & (lift >= 1.5))
    return lift, on


def intervals(ts, on, gap=2, min_len=2):
    # close gaps of <= gap keyframes, drop runs shorter than min_len keyframes
    n = len(on)
    i = 0
    runs = []
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and on[j + 1]:
            j += 1
        runs.append([i, j])
        i = j + 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    merged = [r for r in merged if r[1] - r[0] + 1 >= min_len]
    dt = np.median(np.diff(ts)) if len(ts) > 1 else 5.0
    return [(float(ts[a]), float(ts[b] + dt)) for a, b in merged]


def kind_guess(b):
    x1, y1, x2, y2 = (b[k] * SCALE for k in ("x1", "y1", "x2", "y2"))
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if h > 300 and w < 400 and (cx > 900 or cx < 380):
        return "leaderboard_tower"
    if w > 700 and h < 90 and (cy > 620 or cy < 80):
        return "ticker_bar"
    if w > 500 and h > 250:
        return "bracket_or_fullscreen_graphic"
    if cy > 520 and w > 200:
        return "lower_third"
    if cy < 120 and (cx < 300 or cx > 980) and w < 360:
        return "corner_logo_or_label"
    if cy < 120:
        return "top_label"
    if w > 150 and h > 100:
        return "inset_or_logo_box"
    return "graphic"


def merge_contained(boxes, ts, dur, frac=0.8, corr_min=0.5):
    """A region lying >= frac inside another one and switching on/off with it (phi >= corr_min) is part of it
    (tower header / name column / number column). A label that only sits inside a bigger graphic's box but has its
    own timing (DIN test-name label vs bracket) stays separate."""
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(boxes):
            for j, b in enumerate(boxes):
                if i == j:
                    continue
                ix = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
                iy = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
                aa = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
                oa, ob = a["on"].astype(float), b["on"].astype(float)
                same_timing = oa.std() > 0 and ob.std() > 0 and np.corrcoef(oa, ob)[0, 1] >= corr_min
                if aa and ix * iy >= frac * aa and same_timing:  # a inside b, same timing
                    b["ref"] = a["ref"] | b["ref"]
                    b["on"] = a["on"] | b["on"]
                    b["vis"] = np.maximum(a["vis"], b["vis"])
                    b["iv"] = intervals(ts, b["on"])
                    b["cover"] = sum(t1 - t0 for t0, t1 in b["iv"]) / dur
                    boxes = [x for k, x in enumerate(boxes) if k != i]
                    changed = True
                    break
            if changed:
                break
    return boxes


def merge_by_visibility(boxes, ts, dur, corr_min=0.8):
    """Parts of one graphic (bracket boxes, the pieces of a lower third) switch on and off together: merge
    overlapping or nearby regions whose on/off series agree (phi correlation >= corr_min)."""
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                gapx = max(a["x1"], b["x1"]) - min(a["x2"], b["x2"])
                gapy = max(a["y1"], b["y1"]) - min(a["y2"], b["y2"])
                if gapx > 12 or gapy > 12:
                    continue
                oa, ob = a["on"].astype(float), b["on"].astype(float)
                if oa.std() == 0 or ob.std() == 0:
                    continue
                if np.corrcoef(oa, ob)[0, 1] < corr_min:
                    continue
                m = dict(a)
                for k, f in (("x1", min), ("y1", min), ("x2", max), ("y2", max)):
                    m[k] = f(a[k], b[k])
                m["ref"] = a["ref"] | b["ref"]
                m["mass"] = a["mass"] + b["mass"]
                m["npx"] = a["npx"] + b["npx"]
                m["p"] = (a["p"] * a["npx"] + b["p"] * b["npx"]) / max(m["npx"], 1)
                m["on"] = a["on"] | b["on"]
                m["vis"] = np.maximum(a["vis"], b["vis"])
                m["iv"] = intervals(ts, m["on"])
                m["cover"] = sum(t1 - t0 for t0, t1 in m["iv"]) / dur
                m["kind"] = kind_guess(m)
                m["src"] = "merged"
                boxes = [x for k, x in enumerate(boxes) if k not in (i, j)] + [m]
                changed = True
                break
            if changed:
                break
    return boxes


def load_overrides(vid):
    if not OVR.exists():
        return []
    with open(OVR, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["id"] == vid]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def process(row, reuse_p=False):
    vid = row["id"]
    d = wdir(vid)
    cache = d / "overlay_edges.npz" if os.environ.get("OVL_CACHE") else None
    ts, E, T = load_keyframes(ROOT / row["path"], cache)
    F = cell_features(T)
    pfile = d / "overlay_P.npy"
    if reuse_p and pfile.exists():
        P = np.load(pfile)
    else:
        P1, _ = persistence(ts, E, F)
        excl = np.zeros((H, W), bool)
        for b in boxes_from(P1):
            excl[max(b["y1"] - 4, 0):b["y2"] + 4, max(b["x1"] - 4, 0):b["x2"] + 4] = True
        P2, _ = persistence(ts, E, F, excl)
        # pass-1 regions are kept (pass 2 may see fewer partners for them)
        P = np.maximum(P2, np.where(excl, P1, 0))
        np.save(pfile, P.astype(np.float32))
    boxes = boxes_from(P)
    dur = float(row["duration_s"]) or float(ts[-1])
    rows = []
    for b in boxes:
        vis, on = visibility(b, E, T)
        iv = intervals(ts, on)
        cover = sum(t1 - t0 for t0, t1 in iv) / dur
        b.update(vis=vis, on=on, iv=iv, cover=cover, kind=kind_guess(b))
    boxes = merge_by_visibility(boxes, ts, dur)
    boxes = merge_contained(boxes, ts, dur)
    # overrides
    ov = load_overrides(vid)
    for o in ov:
        bb = [int(o[k]) // SCALE for k in ("x1", "y1", "x2", "y2")]
        if o["action"] == "drop":
            boxes = [b for b in boxes if iou((b["x1"], b["y1"], b["x2"], b["y2"]), bb) <= 0.3]
        elif o["action"] == "add":
            b = {"x1": bb[0], "y1": bb[1], "x2": bb[2], "y2": bb[3], "p": 0.0, "mass": 0.0, "npx": 0,
                 "kind": o["kind"] or "manual", "src": "manual"}
            b["ref"] = np.zeros((H, W), bool)
            if o.get("t_ref"):
                # visibility from a reference time where the overlay is on screen: share of the box's thumbnail
                # cells that look like the reference (graphic backgrounds keep their colour while the text changes)
                k = int(np.argmin(np.abs(ts - float(o["t_ref"]))))
                cs = W // TW
                cx1, cy1 = -(-bb[0] // cs), -(-bb[1] // cs)
                cx2, cy2 = max(bb[2] // cs, cx1 + 1), max(bb[3] // cs, cy1 + 1)
                cells = T[:, cy1:cy2, cx1:cx2].astype(np.int16)
                dd = np.abs(cells - cells[k])
                share = ((dd[..., 0] < 15) & (dd[..., 1:].max(-1) < 8)).mean((1, 2))
                b["vis"], b["on"] = share, share >= MANUAL_MATCH
                b["ref"][bb[1]:bb[3], bb[0]:bb[2]] = E[k][bb[1]:bb[3], bb[0]:bb[2]]
            else:
                b["vis"], b["on"] = np.ones(len(ts)), np.ones(len(ts), bool)  # no reference: always on
            b["iv"] = intervals(ts, b["on"]) or [(0.0, dur)]
            b["cover"] = sum(t1 - t0 for t0, t1 in b["iv"]) / dur
            boxes.append(b)
    boxes = [b for b in boxes if b["iv"] and (b["cover"] >= MIN_COVER or b.get("src") == "manual")]
    boxes.sort(key=lambda b: (b["y1"], b["x1"]))
    np.savez_compressed(d / "overlay_vis.npz", ts=ts, vis=np.array([b["vis"] for b in boxes]).reshape(len(boxes), -1),
                        on=np.array([b["on"] for b in boxes]).reshape(len(boxes), -1),
                        box=np.array([[b[k] for k in ("x1", "y1", "x2", "y2")] for b in boxes]).reshape(-1, 4))

    mask = np.zeros((720, 1280), np.uint8)
    out_rows = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = (b[k] * SCALE for k in ("x1", "y1", "x2", "y2"))
        x1, y1, x2, y2 = max(x1 - 4, 0), max(y1 - 4, 0), min(x2 + 4, 1280), min(y2 + 4, 720)
        mask[y1:y2, x1:x2] = 255
        out_rows.append({"region": i, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "kind": b["kind"],
                         "source": b.get("src", "manual"), "persistence": round(b["p"], 3),
                         "coverage": round(b["cover"], 3),
                         "n_intervals": len(b["iv"]),
                         "intervals": ";".join(f"{t0:.0f}-{t1:.0f}" for t0, t1 in b["iv"])})
    cv2.imwrite(str(d / "overlay_mask.png"), mask)
    with open(d / "overlays.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["region", "x1", "y1", "x2", "y2", "kind", "source", "persistence", "coverage", "n_intervals",
                               "intervals"])
        w.writeheader()
        w.writerows(out_rows)
    preview(d, ROOT / row["path"], ts, boxes, out_rows, P)
    return f"{len(out_rows)} regions: " + ", ".join(f"{r['kind']}({r['coverage']:.0%})" for r in out_rows)


def preview(d, path, ts, boxes, rows, P):
    """5 frames where the most regions are visible (spread in time) + the persistence map, 3x2 sheet."""
    n = len(ts)
    if boxes:
        cnt = np.sum([b["on"] for b in boxes], 0)
    else:
        cnt = np.zeros(n)
    picks = []
    order = np.argsort(-cnt + np.random.default_rng(1).random(n) * 0.5)
    for i in order:
        if all(abs(ts[i] - ts[j]) > 600 for j in picks):
            picks.append(i)
        if len(picks) == 5:
            break
    picks.sort()
    tiles = []
    colors = [(255, 60, 60), (60, 200, 255), (255, 200, 0), (120, 255, 120), (255, 120, 255), (0, 255, 255)]
    for i in picks:
        im = frame_at(path, ts[i] + 0.01, (W, H), "rgb24")
        im = im.copy() if im is not None else np.zeros((H, W, 3), np.uint8)
        ov = im.copy()
        for k, (b, r) in enumerate(zip(boxes, rows)):
            vis = bool(b["on"][i])
            x1, y1, x2, y2 = (r[c] // SCALE for c in ("x1", "y1", "x2", "y2"))
            c = colors[k % len(colors)]
            if vis:
                ov[y1:y2, x1:x2] = (0.5 * ov[y1:y2, x1:x2] + 0.5 * np.array(c)).astype(np.uint8)
            cv2.rectangle(ov, (x1, y1), (x2 - 1, y2 - 1), c, 1 if not vis else 2)
            cv2.putText(ov, f"{k}", (x1 + 2, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
        im = cv2.addWeighted(im, 0.35, ov, 0.65, 0)
        cv2.putText(im, fmt_t(ts[i]), (W - 70, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append(im)
    lp = np.clip((np.log10(P + 1e-4) + 3) / 3, 0, 1)  # log scale: 0.001 -> black, 1 -> white
    pm = np.ascontiguousarray(cv2.applyColorMap((lp * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1])
    for k, r in enumerate(rows):
        x1, y1, x2, y2 = (r[c] // SCALE for c in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(pm, (x1, y1), (x2 - 1, y2 - 1), colors[k % len(colors)], 1)
        cv2.putText(pm, f"{k} {r['kind'][:12]} {r['coverage']:.0%}", (x1 + 2, min(y2 - 3, H - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(pm)
    while len(tiles) < 6:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
    cv2.imwrite(str(d / "overlay_preview.jpg"), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    workers, force, ids, rest = cli_ids(sys.argv[1:])
    run_stage("overlay", process, select_videos(ids), workers=workers, force=force, reuse_p="--reuse-p" in rest)
