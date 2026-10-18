"""WP3 step 2 (CPU): per-shot tracking of the stored detections + per-track stats and flags.

Inputs:  work/<id>/dets_raw.parquet (+ detect.done), work/<id>/shots.csv (+ shots.done),
         optional work/<id>/overlays.csv (+ overlay.done) for `in_overlay`.
Outputs: work/<id>/dets.parquet   = dets_raw + shot_id + track_id (-1: not in a kept track)
         work/<id>/tracks.parquet = one row per kept track (>= MIN_LEN detections)
         work/<id>/track.done

Tracker (4 fps, so cars move a lot between frames in close-ups and onboard): a ByteTrack-style two-stage greedy-free
(Hungarian) matcher, run separately per class (buggy, person) and per shot, never across a shot boundary:
  stage 1: active + lost tracks vs dets with conf >= HIGH. Similarity = max(IoU(predicted box, det),
           0.6 * centre-distance score) where the centre-distance score is 1 - d / (GATE * track size) and only counts
           when the box heights agree within a factor HR. Predicted box = last box + constant velocity (EMA).
  stage 2: still-unmatched tracks vs dets with LOW <= conf < HIGH, IoU >= 0.3 only.
  new tracks from unmatched dets with conf >= HIGH; a track is dropped after MAX_AGE frames without a match.

Per-track stats come from the boxes plus a cheap second decode pass (ffmpeg 2 fps, 192x108 gray) that measures, for
every track, how much the picture changes outside its box (bg_motion, i.e. camera / scene motion) and inside it
(inbox_change), and keeps a small appearance signature (box patch + the rest of the frame) for the recurrence test.

Flags:
  small        median box height < SMALL_PX source pixels (detector data only, no number reading).
  screen_fixed = fixed_persistent | fixed_bg_moving | fixed_recurrent (cars only):
    fixed_persistent  onboard ego car. Seeds: big bottom-anchored box (y2 >= BOTTOM*H) while the scene moves, at the same
                      place in >= PERSIST_MIN of the frames of its shot (or of +-PERSIST_WIN s), shot >= PERSIST_SHOT_S,
                      and the same place (IoU >= PLACE_IOU) + look (patch NCC >= PERSIST_LOOK) in >= PERSIST_SHOTS-1
                      other shots (the mount is rigid; handheld / PTZ close-ups of real cars fail the look test).
                      Then propagated to other bottom-anchored fragments of the same shot (IoU >= FRAG_IOU with a seed)
                      and to big bottom-anchored boxes at an established ego place (EGO_PLACE_IOU, EGO_PLACE_SHOTS).
    fixed_bg_moving   static box (centre MAD <= STATIC_C*H, height MAD <= STATIC_H) while bg_motion >= BG_MIN and
                      >= BG_RATIO * inbox_change (graphic over moving video), at the same place in >= 2 shots or for
                      >= BGFIX_S (a PTZ operator can hold a small far car still for a few seconds).
    fixed_recurrent   the same static place + look in >= RECUR_MIN shots, counting only instances whose surroundings move.
    A parked car under a static camera is NOT flagged (bg_motion of static cameras is 1-8); `long_static` marks it.
  long_static  (info) static box for >= LONG_STATIC_S: parked cars, parked road vehicles, props under a fixed camera.
  in_overlay   >= 50% of the track's detections lie mostly (>= 60% of box area) inside an overlay region that is
               visible at that time (overlays.csv). Null when the video has no overlays yet; `--reflag` fills it later.
               Regions covering > 6% of the frame and the kinds bracket_or_fullscreen_graphic / inset_or_logo_box
               (bounding boxes of sparse layouts, mostly live video inside) are ignored.

usage (repo root): python teacher/track.py [id ...] [--force] [--reflag] [--no-people]
"""
import datetime as dt
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, log_error, select_videos, tool, wdir  # noqa: E402

FPS = 4.0
HIGH, LOW = 0.25, 0.08
MAX_AGE = 12          # frames (3 s at 4 fps)
GATE = 1.2            # centre distance gate, in units of the track's box size (max(w, h))
HR = 1.8              # max height ratio for a centre-distance match
MIN_SIM = 0.15
MIN_LEN = 3
SMALL_PX = 60
LONG_STATIC_S = 20.0
OVL_MAX_AREA = 0.06    # overlay regions larger than this share of the frame are ignored (WP2 boxes are unions of positions)
OVL_SKIP_KINDS = {"bracket_or_fullscreen_graphic", "inset_or_logo_box"}
# screen_fixed
STATIC_C = 0.012      # centre MAD / frame height
STATIC_H = 0.04       # relative MAD of box height
BG_MIN = 10.0         # median |dI| outside the box (gray levels, 192x108, 0.5 s apart); static cameras: 1-8
BGFIX_S = 10.0        # s; or the same place in >= 2 shots
BG_RATIO = 1.6        # bg_motion / inbox_change
RECUR_MIN = 4         # distinct shots
RECUR_IOU = 0.7
PATCH_CORR = 0.6
RING_CORR = 0.5
PERSIST_IOU = 0.7
PERSIST_MIN = 0.2     # share of frames with a car box at the track's place
PERSIST_SHOT_S = 10.0
PERSIST_WIN = 10.0
PERSIST_LOOK = 0.3     # patch NCC for 'looks the same' (ego seeds)
PERSIST_SHOTS = 3      # distinct shots with the same persistent box
PLACE_IOU = 0.85      # median-box IoU for "same place" across shots (ego seeds: the mount is rigid)
FRAG_IOU = 0.6        # IoU with a flagged ego box of the same shot (ego fragments)
EGO_PLACE_IOU, EGO_PLACE_SHOTS = 0.7, 5
BOTTOM = 0.97          # y2 / frame height for 'anchored on the bottom edge'     # s around the track for the local persistence
SCENE_MIN = 8.0       # inbox_change / bg_motion that counts as 'the scene moves'
DEC_FPS, DW, DH = 2.0, 192, 108
STRIDE = int(FPS / DEC_FPS)


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


class _T:
    __slots__ = ("id", "box", "v", "last", "idx")

    def __init__(self, tid, box, f, i):
        self.id, self.box, self.v, self.last, self.idx = tid, box.copy(), np.zeros(2), f, [i]

    def pred(self, f):
        g = f - self.last
        b = self.box.copy()
        s = self.v * g * (0.8 ** max(g - 1, 0))
        b[[0, 2]] += s[0]
        b[[1, 3]] += s[1]
        return b

    def update(self, box, f, i):
        g = max(f - self.last, 1)
        dc = ((box[:2] + box[2:]) - (self.box[:2] + self.box[2:])) / 2 / g
        self.v = 0.5 * self.v + 0.5 * dc if len(self.idx) > 1 else dc
        self.box, self.last = box.copy(), f
        self.idx.append(i)


def _sim(P, D, sizes):
    s = iou(P, D)
    if not s.size:
        return s
    pc = (P[:, :2] + P[:, 2:]) / 2
    dc = (D[:, :2] + D[:, 2:]) / 2
    d = np.linalg.norm(pc[:, None] - dc[None], axis=2)
    cs = np.clip(1 - d / (GATE * sizes[:, None]), 0, None)
    ph, dh = P[:, 3] - P[:, 1], D[:, 3] - D[:, 1]
    r = np.maximum(ph[:, None], dh[None]) / np.maximum(np.minimum(ph[:, None], dh[None]), 1)
    cs[r > HR] = 0
    return np.maximum(s, 0.6 * cs)


def _match(sim, thr):
    if not sim.size:
        return []
    r, c = linear_sum_assignment(-sim)
    return [(i, j) for i, j in zip(r, c) if sim[i, j] >= thr]


def track_shot(fr, boxes, conf, next_id):
    """fr, boxes, conf: arrays of one class in one shot, sorted by frame. Returns per-det track ids (-1 none)."""
    tid = np.full(len(fr), -1, np.int64)
    tracks, done = [], []
    starts = np.flatnonzero(np.r_[True, fr[1:] != fr[:-1]])
    ends = np.r_[starts[1:], len(fr)]
    for s, e in zip(starts, ends):
        f = fr[s]
        alive = [t for t in tracks if f - t.last <= MAX_AGE]
        done += [t for t in tracks if f - t.last > MAX_AGE]
        tracks = alive
        idx = np.arange(s, e)
        hi = idx[conf[idx] >= HIGH]
        lo = idx[(conf[idx] >= LOW) & (conf[idx] < HIGH)]
        un_t = list(range(len(tracks)))
        if tracks and len(hi):
            P = np.array([t.pred(f) for t in tracks])
            sz = np.maximum(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
            m = _match(_sim(P, boxes[hi], sz), MIN_SIM)
            for i, j in m:
                tracks[i].update(boxes[hi[j]], f, hi[j])
            mt = {i for i, _ in m}
            md = {j for _, j in m}
            un_t = [i for i in un_t if i not in mt]
            hi_left = [hi[j] for j in range(len(hi)) if j not in md]
        else:
            hi_left = list(hi)
        if un_t and len(lo):
            P = np.array([tracks[i].pred(f) for i in un_t])
            m = _match(iou(P, boxes[lo]), 0.3)
            for i, j in m:
                tracks[un_t[i]].update(boxes[lo[j]], f, lo[j])
        for j in hi_left:
            tracks.append(_T(next_id, boxes[j], f, j))
            next_id += 1
    for t in tracks + done:
        if len(t.idx) >= MIN_LEN:
            tid[t.idx] = t.id
    return tid, next_id


def assign_shots(pts, shots):
    t0 = shots.t0.to_numpy()
    k = np.searchsorted(t0, pts, side="right") - 1
    return shots.shot_id.to_numpy()[np.clip(k, 0, len(shots) - 1)]


def run_tracking(d, shots, people=True):
    d = d.sort_values(["frame_idx", "conf"], ascending=[True, False], kind="stable").reset_index(drop=True)
    d["shot_id"] = assign_shots(d.pts.to_numpy() + 1e-6, shots).astype(np.int32)
    d["track_id"] = np.int64(-1)
    next_id = 0
    for cls in (["buggy", "person"] if people else ["buggy"]):
        sel = d.index[(d.cls == cls) & (d.conf >= LOW)]
        sub = d.loc[sel]
        for _, g in sub.groupby("shot_id", sort=True):
            tid, next_id = track_shot(g.frame_idx.to_numpy(), g[["x1", "y1", "x2", "y2"]].to_numpy(np.float64),
                                      g.conf.to_numpy(), next_id)
            d.loc[g.index, "track_id"] = tid
    return d


# ---------------------------------------------------------------- per-track stats


def box_stats(d, H):
    t = d[d.track_id >= 0].copy()
    t["h"] = t.y2 - t.y1
    t["w"] = t.x2 - t.x1
    t["cx"] = (t.x1 + t.x2) / 2
    t["cy"] = (t.y1 + t.y2) / 2
    rows = []
    for tid, g in t.groupby("track_id", sort=True):
        g = g.sort_values("frame_idx")
        cx, cy, h = g.cx.to_numpy(), g.cy.to_numpy(), g.h.to_numpy()
        step = np.hypot(np.diff(cx), np.diff(cy)) / np.maximum(np.diff(g.frame_idx.to_numpy()), 1)
        mad_c = np.median(np.hypot(cx - np.median(cx), cy - np.median(cy)))
        rows.append(dict(
            track_id=tid, cls=g.cls.iat[0], shot_id=int(g.shot_id.iat[0]), t0=g.pts.iat[0], t1=g.pts.iat[-1],
            n=len(g), span_frames=int(g.frame_idx.iat[-1] - g.frame_idx.iat[0] + 1),
            conf_mean=float(g.conf.mean()), conf_median=float(g.conf.median()), conf_max=float(g.conf.max()),
            conf_n025=int((g.conf >= 0.25).sum()),
            h_px_median=float(np.median(h)), h_px_min=float(h.min()), h_px_max=float(h.max()),
            h_rel_median=float(np.median(h) / H), w_px_median=float(g.w.median()),
            x1_med=float(g.x1.median()), y1_med=float(g.y1.median()), x2_med=float(g.x2.median()),
            y2_med=float(g.y2.median()),
            speed_rel_median=float(np.median(step) / H) if len(step) else 0.0,
            net_disp_rel=float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0]) / H),
            path_rel=float(step.sum() / H) if len(step) else 0.0,
            center_mad_rel=float(mad_c / H),
            h_mad_rel=float(np.median(np.abs(h - np.median(h))) / max(np.median(h), 1)),
            size_ratio=float(h.max() / max(h.min(), 1)),
            edge_touch=float(np.mean((g.x1 <= 2) | (g.y1 <= 2) | (g.x2 >= 1278 * H / 720) | (g.y2 >= H - 2))),
        ))
    return pd.DataFrame(rows)


def nested(d, tr):
    """Car tracks only: id of a larger coexisting car track whose box contains this one (session-1 double-box rule:
    median intersection / own area > 0.6 over >= half of its detections). -1 if none."""
    out = {}
    t = d[(d.track_id >= 0) & (d.cls == "buggy")]
    for (_, _), g in t.groupby(["shot_id", "cls"]):
        if g.track_id.nunique() < 2:
            continue
        a = g.merge(g, on="frame_idx", suffixes=("", "_o"))
        a = a[a.track_id != a.track_id_o]
        if a.empty:
            continue
        ix = np.clip(np.minimum(a.x2, a.x2_o) - np.maximum(a.x1, a.x1_o), 0, None)
        iy = np.clip(np.minimum(a.y2, a.y2_o) - np.maximum(a.y1, a.y1_o), 0, None)
        ar = (a.x2 - a.x1) * (a.y2 - a.y1)
        ao = (a.x2_o - a.x1_o) * (a.y2_o - a.y1_o)
        a = a.assign(cont=(ix * iy) / np.maximum(ar, 1), smaller=ar < ao)
        a = a[a.smaller]
        s = a.groupby(["track_id", "track_id_o"]).cont.agg(["median", "size"]).reset_index()
        n = tr.set_index("track_id").n
        s = s[(s["median"] > 0.6) & (s["size"] >= 0.5 * s.track_id.map(n))]
        for r in s.sort_values("size").itertuples():
            out[r.track_id] = r.track_id_o
    return tr.track_id.map(out).fillna(-1).astype(np.int64)


def decode_feats(path, d, tr, W, H, shots, start=0.0, dur=None):
    """Second decode pass (2 fps, gray 192x108) for the car tracks: bg_motion / inbox_change = median over the track's
    frames of the mean |dI| (0.5 s apart, same shot only) outside / inside its box, and an appearance signature
    (box patch + rest of the frame) at the track's middle for the recurrence test."""
    t = d[(d.track_id >= 0) & (d.cls == "buggy") & (d.frame_idx % STRIDE == 0)]
    by_k = {k: g for k, g in t.groupby(t.frame_idx // STRIDE)}
    sx, sy = DW / W, DH / H
    mid = {r.track_id: (r.t0 + r.t1) / 2 for r in tr[tr.cls == "buggy"].itertuples()}
    bg, inb, sig, best_mid = {}, {}, {}, {}
    seek = (["-ss", f"{start:.3f}"] if start else []) + (["-t", f"{dur:.3f}"] if dur else [])
    cmd = [tool("ffmpeg"), "-v", "error", "-nostdin", "-threads", "4", *seek, "-i", str(path), "-an", "-sn",
           "-vf", f"fps={DEC_FPS:g},scale={DW}:{DH}:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=DW * DH * 64)
    k = int(round(start * DEC_FPS))
    prev, prev_shot = None, -1
    yy, xx = np.mgrid[0:DH, 0:DW]
    try:
        while True:
            b = p.stdout.read(DW * DH)
            if len(b) < DW * DH:
                break
            g = np.frombuffer(b, np.uint8).reshape(DH, DW).astype(np.float32)
            sh = int(assign_shots(np.array([k / DEC_FPS + 1e-6]), shots)[0])
            rows = by_k.get(k)
            if rows is not None:
                diff = np.abs(g - prev) if (prev is not None and sh == prev_shot) else None
                for r in rows.itertuples():
                    x1, y1, x2, y2 = r.x1 * sx, r.y1 * sy, r.x2 * sx, r.y2 * sy
                    if diff is not None:
                        mw, mh = 0.1 * (x2 - x1) + 1, 0.1 * (y2 - y1) + 1
                        out_m = (xx < x1 - mw) | (xx > x2 + mw) | (yy < y1 - mh) | (yy > y2 + mh)
                        in_m = (xx > x1 + mw) & (xx < x2 - mw) & (yy > y1 + mh) & (yy < y2 - mh)
                        if out_m.sum() > 200:
                            bg.setdefault(r.track_id, []).append(float(diff[out_m].mean()))
                        if in_m.sum() >= 4:
                            inb.setdefault(r.track_id, []).append(float(diff[in_m].mean()))
                    dm = abs(k / DEC_FPS - mid.get(r.track_id, 0))
                    if r.track_id not in best_mid or dm < best_mid[r.track_id]:
                        best_mid[r.track_id] = dm
                        sig[r.track_id] = _signature(g, x1, y1, x2, y2)
            prev, prev_shot = g, sh
            k += 1
    finally:
        p.kill() if p.poll() is None else None
        p.wait()
    f = pd.DataFrame({"track_id": tr.track_id})
    f["bg_motion"] = f.track_id.map(lambda i: float(np.median(bg[i])) if i in bg else np.nan)
    f["inbox_change"] = f.track_id.map(lambda i: float(np.median(inb[i])) if i in inb else np.nan)
    f["n_motion"] = f.track_id.map(lambda i: len(inb.get(i, [])))
    return f, sig, k


def persistence(d, tr, shots):
    """pos_persist: share of 4 fps frames that have a car detection (conf >= LOW, any track) at this track's place
    (IoU >= PERSIST_IOU with its median box), over the whole shot or over the track's span +- PERSIST_WIN s inside the
    shot, whichever is larger (layouts can switch without a detected cut). Survives fragmentation of low-conf boxes."""
    out = pd.Series(0.0, index=tr.index)
    sh = shots.set_index("shot_id")
    car = d[(d.cls == "buggy") & (d.conf >= LOW)]
    groups = {s: g for s, g in car.groupby("shot_id")}
    for s, tt in tr[tr.cls == "buggy"].groupby("shot_id"):
        g = groups.get(s)
        if g is None or s not in sh.index:
            continue
        s0, s1 = sh.at[s, "t0"], sh.at[s, "t1"]
        D = g[["x1", "y1", "x2", "y2"]].to_numpy()
        T = g.pts.to_numpy()
        F = g.frame_idx.to_numpy()
        M = iou(tt[["x1_med", "y1_med", "x2_med", "y2_med"]].to_numpy(), D) >= PERSIST_IOU
        for j, (ix, a, b) in enumerate(zip(tt.index, tt.t0, tt.t1)):
            whole = len(np.unique(F[M[j]])) / max((s1 - s0) * FPS, 1)
            w0, w1 = max(s0, a - PERSIST_WIN), min(s1, b + PERSIST_WIN)
            sel = M[j] & (T >= w0) & (T <= w1)
            local = len(np.unique(F[sel])) / max((w1 - w0) * FPS, 1)
            out[ix] = max(whole, local)
    return out


def _signature(g, x1, y1, x2, y2):
    """Box patch (12x12) + the rest of the frame (48x27 thumbnail, box area masked) for the recurrence test."""
    import cv2
    ix1, iy1 = int(max(np.floor(x1), 0)), int(max(np.floor(y1), 0))
    ix2, iy2 = int(min(np.ceil(x2), DW)), int(min(np.ceil(y2), DH))
    patch = g[iy1:max(iy2, iy1 + 1), ix1:max(ix2, ix1 + 1)]
    patch = cv2.resize(patch, (12, 12), interpolation=cv2.INTER_AREA).ravel()
    ring = cv2.resize(g, (48, 27), interpolation=cv2.INTER_AREA)
    m = np.ones_like(ring, bool)
    m[int(y1 / 4):int(np.ceil(y2 / 4)) + 1, int(x1 / 4):int(np.ceil(x2 / 4)) + 1] = False
    return patch, ring.ravel(), m.ravel()


def _ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-6 else (1.0 if np.allclose(a, b, atol=2) else 0.0)


def recurrence(tr, sig):
    """recur_shots: distinct shots in which the same static box (place + look) appears over a different scene.
    Only instances whose surroundings move (bg_motion >= BG_MIN) count as hits besides the track itself: a fixed
    camera that comes back to the same view shows a parked car / prop at the same place with the same look too, and
    people walking or an overlay elsewhere can make its frame-level 'ring' look different. Graphics over a still
    picture are left to in_overlay (WP2 overlay masks)."""
    st = tr[(tr.cls == "buggy") & (tr.static_box | (tr.pos_persist >= PERSIST_MIN))].copy()
    out = pd.Series(0, index=tr.index, dtype=np.int64)
    if len(st) < 2:
        return out
    B = st[["x1_med", "y1_med", "x2_med", "y2_med"]].to_numpy()
    M = iou(B, B)
    ids = st.track_id.to_numpy()
    shots = st.shot_id.to_numpy()
    bgm = st.bg_motion.to_numpy()
    for a in range(len(st)):
        cand = np.flatnonzero((M[a] >= RECUR_IOU) & (shots != shots[a]))
        if len(cand) < RECUR_MIN - 1 or ids[a] not in sig:
            continue
        pa, ra, ma = sig[ids[a]]
        hit = {shots[a]}
        for c in cand:
            if ids[c] not in sig or shots[c] in hit or not (bgm[c] >= BG_MIN):
                continue
            pc, rc, mc = sig[ids[c]]
            m = ma & mc
            if _ncc(pa, pc) >= PATCH_CORR and (m.sum() < 50 or _ncc(ra[m], rc[m]) < RING_CORR):
                hit.add(shots[c])
        out.loc[st.index[a]] = len(hit)
    return out


def load_overlays(d, W, H):
    p = d / "overlays.csv"
    if not (d / "overlay.done").exists() or not p.exists():
        return None
    o = pd.read_csv(p)
    regs = []
    for r in o.itertuples():
        # the region boxes are bounding boxes; for sparse layouts (brackets, picture-in-picture insets, big towers)
        # most of the box is live video with real cars in it, so only compact graphics count
        if (r.x2 - r.x1) * (r.y2 - r.y1) > OVL_MAX_AREA * W * H or str(r.kind) in OVL_SKIP_KINDS:
            continue
        iv = []
        for s in str(r.intervals).split(";"):
            if "-" in s:
                a, b = s.split("-")
                iv.append((float(a), float(b)))
        regs.append((np.array([r.x1, r.y1, r.x2, r.y2], float), iv))
    return regs


def overlay_flag(d, tr, regs):
    if regs is None:
        return pd.Series([None] * len(tr), index=tr.index, dtype=object), None
    t = d[d.track_id >= 0]
    inside = np.zeros(len(t), bool)
    bx = t[["x1", "y1", "x2", "y2"]].to_numpy()
    area = np.maximum((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]), 1)
    pts = t.pts.to_numpy()
    for box, iv in regs:
        ix = np.clip(np.minimum(bx[:, 2], box[2]) - np.maximum(bx[:, 0], box[0]), 0, None)
        iy = np.clip(np.minimum(bx[:, 3], box[3]) - np.maximum(bx[:, 1], box[1]), 0, None)
        cov = ix * iy / area >= 0.6
        if not cov.any():
            continue
        vis = np.zeros(len(t), bool)
        for a, b in iv:
            vis |= (pts >= a - 2.5) & (pts <= b + 2.5)  # keyframe-based intervals (5 s GOP)
        inside |= cov & vis
    share = pd.Series(inside, index=t.index).groupby(t.track_id).mean()
    s = tr.track_id.map(share)
    return (s >= 0.5).astype(object), s


def same_place_look(tr, cand, sig):
    """Number of other shots with a candidate at the same place (IoU >= PLACE_IOU) that also looks the same (12x12 patch
    NCC >= PERSIST_LOOK). The ego car looks the same shot after shot (helmet, cage: median NCC ~0.45 on 26BR_END);
    handheld / PTZ close-ups of different cars that fill the frame the same way do not (~0.07 on 25BR_END)."""
    out = pd.Series(0, index=tr.index, dtype=np.int64)
    c = tr[cand & tr.track_id.isin(list(sig))]
    if len(c) < 2:
        return out
    M = iou(*(2 * [c[["x1_med", "y1_med", "x2_med", "y2_med"]].to_numpy()])) >= PLACE_IOU
    ids, sh = c.track_id.to_numpy(), c.shot_id.to_numpy()
    for a, ix in enumerate(c.index):
        ok = {sh[b] for b in np.flatnonzero(M[a] & (sh != sh[a])) if _ncc(sig[ids[a]][0], sig[ids[b]][0]) >= PERSIST_LOOK}
        out[ix] = len(ok)
    return out


def same_place_shots(tr, cand):
    """For each candidate track: number of distinct shots with a candidate track at the same place (median IoU)."""
    out = pd.Series(0, index=tr.index, dtype=np.int64)
    c = tr[cand]
    if len(c) == 0:
        return out
    M = iou(*(2 * [c[["x1_med", "y1_med", "x2_med", "y2_med"]].to_numpy()])) >= PLACE_IOU
    sh = c.shot_id.to_numpy()
    for j, ix in enumerate(c.index):
        out[ix] = len(set(sh[M[j]]))
    return out


def ego_fragments(tr, cand):
    out = pd.Series(False, index=tr.index)
    ego = tr[tr.fixed_persistent]
    for sid, g in tr[cand].groupby("shot_id"):
        e = ego[ego.shot_id == sid]
        if e.empty:
            continue
        M = iou(g[["x1_med", "y1_med", "x2_med", "y2_med"]].to_numpy(), e[["x1_med", "y1_med", "x2_med", "y2_med"]].to_numpy())
        out[g.index[M.max(axis=1) >= FRAG_IOU]] = True
    return out


def ego_places(tr, cand):
    out = pd.Series(False, index=tr.index)
    ego = tr[tr.fixed_persistent]
    c = tr[cand]
    if ego.empty or c.empty:
        return out
    M = iou(c[["x1_med", "y1_med", "x2_med", "y2_med"]].to_numpy(),
            ego[["x1_med", "y1_med", "x2_med", "y2_med"]].to_numpy()) >= EGO_PLACE_IOU
    sh = ego.shot_id.to_numpy()
    for j, ix in enumerate(c.index):
        out[ix] = len(set(sh[M[j]])) >= EGO_PLACE_SHOTS
    return out


def flags(d, tr, feats, sig, regs, shots, H=720):
    tr = tr.merge(feats, on="track_id", how="left")
    car = tr.cls == "buggy"
    sdur = tr.shot_id.map(shots.set_index("shot_id").dur)
    tr["static_box"] = (tr.center_mad_rel <= STATIC_C) & (tr.h_mad_rel <= STATIC_H) & (tr.n >= 4)
    tr["pos_persist"] = persistence(d, tr, shots)
    scene_moves = (tr.inbox_change.fillna(0) >= SCENE_MIN) | (tr.bg_motion.fillna(0) >= SCENE_MIN)
    # 1) onboard ego car: a box anchored on the bottom edge that holds its screen position for a good part of a long
    #    shot while the picture changes (low-conf, fragmented boxes: measured on all detections, not the track)
    #    and that recurs at the same place in >= PERSIST_SHOTS shots (a zoomed PTZ close-up of a real car also holds
    #    a big bottom-anchored box for a while, but not at the same place shot after shot)
    cand = car & (tr.pos_persist >= PERSIST_MIN) & (sdur >= PERSIST_SHOT_S) & scene_moves & (tr.y2_med >= BOTTOM * H)
    tr["persist_shots"] = same_place_shots(tr, cand)
    tr["persist_look_shots"] = same_place_look(tr, cand, sig)
    tr["fixed_persistent"] = cand & (tr.persist_look_shots >= PERSIST_SHOTS - 1)
    # short / low-persistence fragments of the same ego box: bottom-anchored, scene moving, and at the place of a
    # flagged ego track of the same shot
    frag = car & scene_moves & (tr.y2_med >= BOTTOM * H) & ~tr.fixed_persistent
    tr["fixed_persistent"] |= ego_fragments(tr, frag)
    # ... and in shots without a flagged ego track (short onboard shots): a big bottom-anchored box at a well-established
    #    ego place (IoU >= EGO_PLACE_IOU with ego boxes of >= EGO_PLACE_SHOTS distinct shots)
    tr["fixed_persistent"] |= ego_places(tr, frag & ~tr.fixed_persistent & (tr.h_rel_median >= 0.4))
    # 2) a box that does not move at all while the scene around it moves and its own content does not (graphic)
    moving_bg = (tr.bg_motion >= BG_MIN) & (tr.bg_motion >= BG_RATIO * tr.inbox_change.fillna(0))
    #    and is not a one-off: same place in >= 2 shots, or >= BGFIX_S long (a PTZ operator following a small far car
    #    can hold it still on screen for a couple of seconds)
    cand2 = car & tr.static_box & moving_bg.fillna(False)
    tr["bgfix_shots"] = same_place_shots(tr, cand2)
    tr["fixed_bg_moving"] = cand2 & ((tr.bgfix_shots >= 2) | ((tr.t1 - tr.t0) >= BGFIX_S))
    # 3) the same place + look recurring over different scenes in many shots
    tr["recur_shots"] = recurrence(tr, sig)
    tr["fixed_recurrent"] = car & (tr.recur_shots >= RECUR_MIN)
    tr["screen_fixed"] = tr.fixed_persistent | tr.fixed_bg_moving | tr.fixed_recurrent
    tr["small"] = tr.h_px_median < SMALL_PX
    # informational: a box that does not move for >= LONG_STATIC_S (parked car, static prop such as a tyre, banner)
    tr["long_static"] = car & tr.static_box & ((tr.t1 - tr.t0) >= LONG_STATIC_S)
    tr["in_overlay"], tr["overlay_share"] = overlay_flag(d, tr, regs)
    return tr


def process(row, people=True, reflag=False):
    vid = row["id"]
    d = wdir(vid)
    W, H = row["width"], row["height"]
    path = ROOT / row["path"]
    t_start = time.time()
    shots = pd.read_csv(d / "shots.csv").sort_values("t0").reset_index(drop=True)
    if reflag and (d / "dets.parquet").exists() and (d / "track_feats.npz").exists():
        dets = pd.read_parquet(d / "dets.parquet")
        tr = box_stats(dets, H)
        tr["nested_in"] = nested(dets, tr).to_numpy()
        z = np.load(d / "track_feats.npz")
        feats = pd.DataFrame({c: z[c] for c in ("track_id", "bg_motion", "inbox_change", "n_motion")})
        sig = {int(i): (p, r, m) for i, p, r, m in zip(z["sig_id"], z["patch"], z["ring"], z["mask"])}
    else:
        raw = pd.read_parquet(d / "dets_raw.parquet")
        dets = run_tracking(raw, shots, people)
        tr = box_stats(dets, H)
        if tr.empty:
            tr = pd.DataFrame(columns=["track_id"])
        tr["nested_in"] = nested(dets, tr).to_numpy() if len(tr) else []
        feats, sig, nk = decode_feats(path, dets, tr, W, H, shots)
        ids = np.array(sorted(sig), np.int64)
        np.savez_compressed(d / "track_feats.npz", **{c: feats[c].to_numpy() for c in feats.columns},
                            sig_id=ids,
                            patch=np.array([sig[i][0] for i in ids], np.float32).reshape(len(ids), -1),
                            ring=np.array([sig[i][1] for i in ids], np.float32).reshape(len(ids), -1),
                            mask=np.array([sig[i][2] for i in ids], bool).reshape(len(ids), -1))
        dets.to_parquet(d / "dets.parquet", index=False)
    regs = load_overlays(d, W, H)
    tr = flags(dets, tr, feats, sig, regs, shots, H)
    tr.to_parquet(d / "tracks.parquet", index=False)
    car = tr[tr.cls == "buggy"]
    hours = row["duration_s"] / 3600
    info = dict(video=vid, hours=round(hours, 2), dets=int(len(dets)),
                buggy_dets_025=int(((dets.cls == "buggy") & (dets.conf >= 0.25)).sum()),
                car_tracks=int(len(car)), car_tracks_per_h=round(len(car) / hours, 1),
                person_tracks=int((tr.cls == "person").sum()),
                screen_fixed=round(float(car.screen_fixed.mean()), 4) if len(car) else 0,
                fixed_persistent=int(car.fixed_persistent.sum()), fixed_bg_moving=int(car.fixed_bg_moving.sum()),
                fixed_recurrent=int(car.fixed_recurrent.sum()),
                small=round(float(car.small.mean()), 4) if len(car) else 0,
                long_static=round(float(car.long_static.mean()), 4) if len(car) else 0,
                in_overlay=(round(float(car.in_overlay.astype(float).mean()), 4) if regs is not None and len(car)
                            else None),
                overlays=regs is not None, n_overlay_regions=len(regs) if regs is not None else 0,
                nested=int((car.nested_in >= 0).sum()), runtime_s=round(time.time() - t_start, 1))
    (d / "track_summary.json").write_text(json.dumps(info, indent=1))
    (d / "track.done").write_text(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {json.dumps(info)}\n")
    print(json.dumps(info), flush=True)
    return info


def main(argv):
    force, reflag, people = "--force" in argv, "--reflag" in argv, "--no-people" not in argv
    ids = [a for a in argv if not a.startswith("--")]
    for r in select_videos(ids or None):
        d = wdir(r["id"])
        need = [s for s in ("detect", "shots") if not (d / f"{s}.done").exists()]
        if need:
            print(f"{r['id']}: needs {', '.join(need)} first, skip")
            continue
        rf = reflag
        if (d / "track.done").exists() and not force and not reflag:
            if '"overlays": false' in (d / "track.done").read_text() and (d / "overlay.done").exists():
                print(f"{r['id']}: overlays now available, re-flagging")
                rf = True
            else:
                print(f"{r['id']}: track already done, skip")
                continue
        try:
            process(r, people, reflag=rf)
        except Exception as e:  # noqa: BLE001
            log_error(r["id"], "track", traceback.format_exc())
            print(f"{r['id']}: track FAILED {e!r}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
