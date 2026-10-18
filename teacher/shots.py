"""WP2a: shot boundaries (hard cuts, dissolves, fades) -> work/<id>/shots.csv + shots.done

usage: python teacher/shots.py [--workers 4] [--force] [id ...]       (default: every active video in data/videos.csv)

Method (CPU only; decodes the whole file once through ffmpeg at 10 fps, 128x72 RGB, ~40x real time per 3 threads):
  per frame t, on a 64x36 HSV thumbnail:
  * d1[t]  = mean |HSV(t) - HSV(t-1)| (PySceneDetect ContentDetector-style score, 0..255)
  * hd[t]  = 1 - intersection of 8x4x4 HSV histograms of t-1 and t
  * hard cut at t when d1 is large in absolute terms AND much larger than the d1 of its neighbours (adaptive ratio, so
    fast PTZ pans are not cuts) AND the colour histogram changed.
  * dissolve / fade centred at t (window k = 3, 6, 10 frames): A = F[t-k], B = F[t+k], M = F[t]. If A and B differ a
    lot, M is close to a linear blend aA + (1-a)B (relative residual small, 0.2 < a < 0.8) and no single step inside the
    window carries most of the change, it is a gradual transition. Pans and zooms give large blend residuals.
    Fades to/from black or white are blends with a flat frame and come out the same way; flat (blank) frames are
    also detected directly.
  Transitions become their own short shots with is_transition=1, so tracking never runs across a blend.
  Prefers over-segmenting: a missed cut merges two shots (bad for tracking), an extra cut only splits one.

shots.csv columns: shot_id, t0, t1, dur, cut_in (start|cut|dissolve|flat), is_transition; t0/t1 in seconds of the file.
Also writes work/<id>/shotsig.npz (per-frame signals, a few MB) for tuning / QA.
"""
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, cli_ids, ffmpeg_frames, run_stage, select_videos, wdir  # noqa: E402

FPS = 10
DEC = (128, 72)
TH = (64, 36)
BL = (32, 18)
KS = (3, 6, 10)

# thresholds (tuned on 25SU_DIN / 26BR_END, see LESSONS.md session 2)
CUT_ABS = 18.0      # d1 at a cut (0..255 scale)
CUT_RATIO = 2.6     # d1[t] / max(neighbour levels)
CUT_HIST = 0.25     # histogram change at a cut
CUT_STRONG = 45.0   # d1 so large that the ratio test alone suffices (cut during a fast pan)
DIS_ABS = 20.0      # |A - B| (mean abs S/V, 0..255) for a gradual transition
DIS_RES = 0.20      # blend residual / |A - B|: a dissolve is (nearly) a linear blend of the frames before and after
DIS_ABS_STRONG = 45.0  # ... or a very large change with a looser blend fit (dissolve during camera motion)
DIS_RES_STRONG = 0.45
FLAT_STD = 6.0      # luma std of a blank (black / white / single-colour) frame
MRES = 0.70         # motion-compensated residual: below = pan / zoom / moving subject, not a blend


def signals(path, threads=3, duration=None):
    """Decode at FPS and return a dict of per-frame float32 signals (streaming; memory ~1.2 KB per frame)."""
    d1, dsv, hd, lum, lstd, X = [], [], [], [], [], []
    prev_hsv = prev_hist = None
    K = max(KS)
    ring, gring = [], []  # last 2K+1 blurred (float, raveled) / raw uint8 gray 128x72 frames
    rhi = {k: [] for k in KS}
    mres = []  # motion-compensated residual of the best blend window (9 = not computed)
    for _, fr in ffmpeg_frames(path, fps=FPS, size=DEC, pix="rgb24", threads=threads, duration=duration):
        hsv_big = cv2.cvtColor(fr, cv2.COLOR_RGB2HSV_FULL)
        hsv = cv2.resize(hsv_big, TH, interpolation=cv2.INTER_AREA).astype(np.int16)
        h = cv2.calcHist([hsv_big], [0, 1, 2], None, [8, 4, 4], [0, 256, 0, 256, 0, 256]).ravel()
        h /= max(h.sum(), 1)
        if prev_hsv is None:
            d1.append(0.0), dsv.append(0.0), hd.append(0.0)
        else:
            dh = np.abs(hsv[..., 0] - prev_hsv[..., 0])
            dh = np.minimum(dh, 256 - dh)
            sv = np.abs(hsv[..., 1:] - prev_hsv[..., 1:]).mean()
            d1.append((dh.mean() + 2 * sv) / 3)
            dsv.append(sv)
            hd.append(1 - np.minimum(h, prev_hist).sum())
        prev_hsv, prev_hist = hsv, h
        g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
        lum.append(g.mean())
        lstd.append(g.std())
        X.append(cv2.resize(hsv_big[..., 1:], BL, interpolation=cv2.INTER_AREA).ravel())
        ring.append(cv2.GaussianBlur(g, (5, 5), 1.0).astype(np.float32).ravel())
        gring.append(g)
        if len(ring) > 2 * K + 1:
            ring.pop(0)
            gring.pop(0)
        # residuals for the frame centred K frames back, computed when its window is complete
        if len(ring) == 2 * K + 1:
            M = ring[K]
            best, bk = 9.0, None
            for k in KS:
                r = _blend_res(ring[K - k], ring[K + k], M)
                rhi[k].append(r)
                if r < best and np.abs(ring[K - k] - ring[K + k]).mean() > 6:
                    best, bk = r, k
            mres.append(_motion_res(gring[K - bk], gring[K + bk]) if best < 0.5 else 9.0)
    X = np.stack(X)  # (n, 32*18*2) uint8: S and V (hue does not blend linearly)
    n = len(X)
    sig = {"d1": np.array(d1, np.float32), "dsv": np.array(dsv, np.float32), "hd": np.array(hd, np.float32),
           "lum": np.array(lum, np.float32), "lstd": np.array(lstd, np.float32)}
    # blend residuals for gradual transitions, chunked
    for k in KS:
        r = np.full(n, 9.0, np.float32)
        d = np.zeros(n, np.float32)
        al = np.zeros(n, np.float32)
        for c0 in range(k, n - k, 20000):
            c1 = min(c0 + 20000, n - k)
            A = X[c0 - k: c1 - k].astype(np.float32)
            B = X[c0 + k: c1 + k].astype(np.float32)
            M = X[c0: c1].astype(np.float32)
            AB = A - B
            nab = (AB ** 2).sum(1)
            a = ((M - B) * AB).sum(1) / np.maximum(nab, 1e-6)
            ac = np.clip(a, 0, 1)
            resid = M - (B + ac[:, None] * AB)
            r[c0:c1] = np.sqrt((resid ** 2).sum(1) / np.maximum(nab, 1e-6))
            d[c0:c1] = np.abs(AB).mean(1)
            al[c0:c1] = a
        sig[f"res{k}"], sig[f"dab{k}"], sig[f"alpha{k}"] = r, d, al
        h = np.full(n, 9.0, np.float32)
        if rhi[k]:
            h[K: K + len(rhi[k])] = rhi[k]
        sig[f"rhi{k}"] = h
    m = np.full(n, 9.0, np.float32)
    m[K: K + len(mres)] = mres
    sig["mres"] = m
    return sig


def _motion_res(a, b):
    """How much of the A->B change is NOT explained by smooth motion (dense flow on 128x72): ~0.1-0.4 for
    pans / zooms / moving subjects, ~0.6-1 for a blend between different shots."""
    a = cv2.GaussianBlur(a, (5, 5), 1.0)
    b = cv2.GaussianBlur(b, (5, 5), 1.0)
    raw = np.abs(a.astype(np.float32) - b).mean()
    if raw < 1e-3:
        return 0.0
    fl = cv2.calcOpticalFlowFarneback(b, a, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = a.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    wa = cv2.remap(a, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = 6  # ignore borders (content entering the frame)
    return float(np.abs(wa.astype(np.float32) - b)[m:-m, m:-m].mean() / max(raw, 1e-3))


def _blend_res(A, B, M):
    AB = A - B
    nab = float(AB @ AB)
    if nab < 1e-6:
        return 9.0
    a = min(max(float((M - B) @ AB) / nab, 0.0), 1.0)
    r = M - B - a * AB
    return float(np.sqrt((r @ r) / nab))


def _neigh_level(d1, w=5):
    """Per t: max of the medians of d1 in the w frames before and after t (t excluded)."""
    n = len(d1)
    pad = np.pad(d1, w, mode="edge")
    before = np.stack([pad[w - j: w - j + n] for j in range(1, w + 1)], 1)
    after = np.stack([pad[w + j: w + j + n] for j in range(1, w + 1)], 1)
    # a neighbouring cut (2 cuts close together) should not mask this one: use the 2nd smallest-ish (median)
    return np.maximum(np.median(before, 1), np.median(after, 1))


def detect(sig):
    """Return list of (t_frame_start, t_frame_end, kind) transitions in frame indices.
    Hard cut at frame t -> boundary between t-1 and t (start=end=t)."""
    d1, hd, lstd = sig["d1"], sig["hd"], sig["lstd"]
    n = len(d1)
    lvl = _neigh_level(d1)
    cut = (d1 > CUT_ABS) & (hd > CUT_HIST) & ((d1 > CUT_RATIO * lvl + 4) | ((d1 > CUT_STRONG) & (d1 > 1.6 * lvl)))
    # local maximum within +-1 frame
    for t in np.where(cut)[0]:
        if (t > 0 and d1[t - 1] > d1[t] and cut[t - 1]) or (t + 1 < n and d1[t + 1] >= d1[t] and cut[t + 1]):
            cut[t] = False
    events = [(int(t), int(t), "cut") for t in np.where(cut)[0]]

    # gradual transitions
    cand = np.zeros(n, bool)
    span = np.zeros(n, np.int32)
    score = np.full(n, 9.0, np.float32)
    for k in KS:
        r, dab, al = sig[f"res{k}"], sig[f"dab{k}"], sig[f"alpha{k}"]
        # largest single S/V step inside (t-k, t+k]: a dissolve spreads the change, a cut concentrates it
        pad = np.pad(sig["dsv"], k, mode="constant")
        mstep = np.max(np.stack([pad[k + j: k + j + n] for j in range(-k + 1, k + 1)], 1), 1)
        mr = sig["mres"]
        blend = ((r < DIS_RES) & (mr > MRES)) | ((dab > DIS_ABS_STRONG) & (r < DIS_RES_STRONG) & (mr > MRES + 0.05))
        ok = (dab > DIS_ABS) & (al > 0.15) & (al < 0.85) & (mstep < 0.5 * dab) & (mr < 5) & blend
        better = ok & (r < score)
        cand |= ok
        score[better] = r[better]
        span[better] = k
    # group consecutive candidate frames into one transition each
    t = 0
    while t < n:
        if not cand[t]:
            t += 1
            continue
        u = t
        while u + 1 < n and cand[u + 1]:
            u += 1
        c = t + int(np.argmin(score[t: u + 1]))
        k = int(span[c])
        events.append((max(t - k, 0), min(u + k, n - 1), "dissolve"))
        t = u + 1

    # flat frames (black / white / single colour cards): runs of >= 2 frames
    flat = lstd < FLAT_STD
    t = 0
    while t < n:
        if not flat[t]:
            t += 1
            continue
        u = t
        while u + 1 < n and flat[u + 1]:
            u += 1
        if u - t + 1 >= 2:
            events.append((t, u, "flat"))
        t = u + 1
    return events


def build_shots(events, n, fps=FPS, min_shot=0.0):
    """Turn transitions into a shot table. A hard cut at frame t: boundary at time (t - 0.5)/fps. A gradual or flat
    transition [a, b]: its own is_transition shot from (a - 0.5)/fps to (b + 0.5)/fps."""
    marks = []  # (time, kind, is_transition_interval_end)
    ivals = []
    for a, b, kind in events:
        if kind == "cut":
            marks.append(((a - 0.5) / fps, "cut"))
        else:
            ivals.append([(a - 0.5) / fps, (b + 0.5) / fps, kind])
    # merge overlapping transition intervals
    ivals.sort()
    merged = []
    for iv in ivals:
        if merged and iv[0] <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], iv[1])
            if iv[2] == "flat":
                merged[-1][2] = "flat"
        else:
            merged.append(list(iv))
    # hard cuts inside a transition interval are absorbed by it
    cuts = [m for m in marks if not any(iv[0] - 0.05 <= m[0] <= iv[1] + 0.05 for iv in merged)]
    dur = n / fps
    bounds = [(0.0, "start", 0)]
    for t, k in cuts:
        bounds.append((t, k, 0))
    for a, b, k in merged:
        bounds.append((max(a, 0.0), k, 1))   # transition shot starts
        bounds.append((min(b, dur), "end_" + k, 0))  # next normal shot starts after it
    bounds.sort(key=lambda x: x[0])
    shots = []
    for i, (t0, kind, istr) in enumerate(bounds):
        t1 = bounds[i + 1][0] if i + 1 < len(bounds) else dur
        if t1 - t0 < 1e-3:
            continue
        if kind.startswith("end_"):
            kind = kind[4:]
        shots.append({"t0": round(max(t0, 0), 2), "t1": round(t1, 2), "cut_in": kind, "is_transition": istr})
    for i, s in enumerate(shots):
        s["shot_id"] = i
        s["dur"] = round(s["t1"] - s["t0"], 2)
    return shots


def write_shots(path, shots):
    # new boundaries invalidate the per-shot tags
    (Path(path).parent / "shot_tags.done").unlink(missing_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["shot_id", "t0", "t1", "dur", "cut_in", "is_transition"])
        w.writeheader()
        for s in shots:
            w.writerow({k: s[k] for k in w.fieldnames})


def process(row, threads=3):
    d = wdir(row["id"])
    sig = signals(ROOT / row["path"], threads=threads)
    np.savez_compressed(d / "shotsig.npz", **sig)
    n = len(sig["d1"])
    shots = build_shots(detect(sig), n)
    write_shots(d / "shots.csv", shots)
    norm = [s["dur"] for s in shots if not s["is_transition"]]
    return f"{len(shots)} shots ({sum(s['is_transition'] for s in shots)} transitions), median {np.median(norm):.1f}s"


def redetect(row):
    """Re-run detection from the stored signals (no decoding)."""
    d = wdir(row["id"])
    sig = dict(np.load(d / "shotsig.npz"))
    shots = build_shots(detect(sig), len(sig["d1"]))
    write_shots(d / "shots.csv", shots)
    return len(shots)


if __name__ == "__main__":
    workers, force, ids, rest = cli_ids(sys.argv[1:])
    rows = select_videos(ids)
    if "--redetect" in rest:
        for r in rows:
            if (wdir(r["id"]) / "shotsig.npz").exists():
                print(r["id"], redetect(r))
    else:
        run_stage("shots", process, rows, workers=workers, force=force)
