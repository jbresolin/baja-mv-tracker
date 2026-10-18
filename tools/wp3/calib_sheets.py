"""WP3 calibration: contact sheets to judge detector settings by eye.

usage (repo root):
  python tools/wp3/calib_sheets.py crops <video_id> <dets.parquet> <out.jpg> [--n=100] [--conf=0.25] [--seed=0]
      random buggy detections (conf >= conf), cropped with context; tile index + conf printed on each tile.
  python tools/wp3/calib_sheets.py diff <video_id> <X.parquet> <Y.parquet> <out.jpg> [--n=60]
      buggy dets (>= 0.25) of X with no buggy of Y (IoU >= 0.5, conf >= 0.15); label = Y's best overlapping class.
  python tools/wp3/calib_sheets.py frames <video_id> <X.parquet> <Y.parquet> <out_prefix> [--n=36] [--per=6]
      random frames (half random, half where X and Y disagree on buggies >= 0.25) with boxes drawn:
      X buggy >= 0.25 green (thick), X buggy 0.10-0.25 yellow, Y buggy >= 0.25 magenta (thin, inset).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "teacher"))
from common import select_videos  # noqa: E402


def arg(name, default):
    for a in sys.argv:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


def grab(video_id, pts_list):
    row = select_videos([video_id], active_only=False)[0]
    cap = cv2.VideoCapture(str(ROOT / row["path"]))
    out = {}
    for t in sorted(set(pts_list)):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, f = cap.read()
        if ok:
            out[t] = f
    return out


def iou_mat(a, b):
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


def tile(frame, r, label, size=180):
    x1, y1, x2, y2 = r.x1, r.y1, r.x2, r.y2
    w, h = x2 - x1, y2 - y1
    pad = 0.25 * max(w, h) + 8
    cx1, cy1 = int(max(x1 - pad, 0)), int(max(y1 - pad, 0))
    cx2, cy2 = int(min(x2 + pad, frame.shape[1])), int(min(y2 + pad, frame.shape[0]))
    c = frame[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(c, (int(x1 - cx1), int(y1 - cy1)), (int(x2 - cx1), int(y2 - cy1)), (0, 255, 0), 1)
    s = size / max(c.shape[:2])
    c = cv2.resize(c, (max(int(c.shape[1] * s), 1), max(int(c.shape[0] * s), 1)),
                   interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    t = np.full((size + 18, size, 3), 40, np.uint8)
    t[:c.shape[0], :c.shape[1]] = c
    cv2.putText(t, label, (2, size + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def grid(tiles, cols=10):
    if not tiles:
        return np.zeros((50, 200, 3), np.uint8)
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    g = np.full((rows * h, cols * w, 3), 20, np.uint8)
    for i, t in enumerate(tiles):
        g[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = t
    return g


def crops(vid, path, out):
    d = pd.read_parquet(path)
    d = d[(d.cls == "buggy") & (d.conf >= arg("conf", 0.25))]
    n = min(arg("n", 100), len(d))
    s = d.sample(n, random_state=arg("seed", 0)).sort_values("pts") if n else d
    fr = grab(vid, s.pts.tolist())
    tiles = []
    for i, r in enumerate(s.itertuples()):
        if r.pts in fr:
            tiles.append(tile(fr[r.pts], r, f"{i} {r.conf:.2f} h{int(r.y2 - r.y1)}"))
    cv2.imwrite(out, grid(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{out}: {len(tiles)} tiles of {len(d)} buggy dets >= conf")


def diff(vid, px, py, out):
    X, Y = pd.read_parquet(px), pd.read_parquet(py)
    xb = X[(X.cls == "buggy") & (X.conf >= 0.25)]
    lost = []
    yg = {k: g for k, g in Y.groupby("frame_idx")}
    for fi, g in xb.groupby("frame_idx"):
        yy = yg.get(fi)
        a = g[["x1", "y1", "x2", "y2"]].to_numpy()
        if yy is None:
            lost += [(r, "none") for r in g.itertuples()]
            continue
        yb = yy[(yy.cls == "buggy") & (yy.conf >= 0.15)]
        m = iou_mat(a, yb[["x1", "y1", "x2", "y2"]].to_numpy())
        mall = iou_mat(a, yy[["x1", "y1", "x2", "y2"]].to_numpy())
        for j, r in enumerate(g.itertuples()):
            if m.shape[1] and m[j].max() >= 0.5:
                continue
            k = int(np.argmax(mall[j])) if mall.shape[1] else -1
            lab = f"{yy.cls.iloc[k]}{yy.conf.iloc[k]:.2f}" if k >= 0 and mall[j, k] >= 0.5 else "none"
            lost.append((r, lab))
    print(f"{out}: {len(lost)} of {len(xb)} X buggy dets >= 0.25 have no Y buggy (IoU 0.5, conf 0.15)")
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(lost), min(arg("n", 60), len(lost)), replace=False)) if lost else []
    sel = [lost[i] for i in pick]
    fr = grab(vid, [r.pts for r, _ in sel])
    tiles = [tile(fr[r.pts], r, f"{i} {r.conf:.2f} ->{lab}") for i, (r, lab) in enumerate(sel) if r.pts in fr]
    cv2.imwrite(out, grid(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    from collections import Counter
    print("  Y label of lost ones:", Counter(lab.rstrip("0123456789.") for _, lab in lost).most_common())


def frames(vid, px, py, prefix):
    X, Y = pd.read_parquet(px), pd.read_parquet(py)
    meta = pd.read_json(Path(px).with_suffix(".json"), typ="series")
    f0 = int(round(meta["start"] * meta["fps"]))
    allf = np.arange(f0, f0 + int(meta["n_frames"]))
    cx = X[(X.cls == "buggy") & (X.conf >= 0.25)].groupby("frame_idx").size()
    cy = Y[(Y.cls == "buggy") & (Y.conf >= 0.25)].groupby("frame_idx").size()
    c = pd.DataFrame({"x": cx, "y": cy}).reindex(allf).fillna(0)
    dis = c.index[c.x != c.y].to_numpy()
    n = arg("n", 36)
    rng = np.random.default_rng(1)
    nd = min(n // 2, len(dis))
    pick = list(rng.choice(dis, nd, replace=False)) if nd else []
    rest = np.setdiff1d(allf, pick)
    pick += list(rng.choice(rest, n - nd, replace=False))
    pick = sorted(int(p) for p in pick)
    fps = meta["fps"]
    fr = grab(vid, [p / fps for p in pick])
    per = arg("per", 6)
    tiles = []
    for p in pick:
        f = fr.get(p / fps)
        if f is None:
            continue
        f = f.copy()
        for r in X[(X.frame_idx == p) & (X.cls == "buggy") & (X.conf >= 0.10)].itertuples():
            col, th = ((0, 255, 0), 3) if r.conf >= 0.25 else ((0, 255, 255), 1)
            cv2.rectangle(f, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, th)
        for r in Y[(Y.frame_idx == p) & (Y.cls == "buggy") & (Y.conf >= 0.25)].itertuples():
            cv2.rectangle(f, (int(r.x1) + 3, int(r.y1) + 3), (int(r.x2) - 3, int(r.y2) - 3), (255, 0, 255), 1)
        tag = "D" if p in set(int(x) for x in dis) else "R"
        cv2.putText(f, f"{tag} f{p} t={p / fps:.0f}s X{int(c.x.get(p, 0))} Y{int(c.y.get(p, 0))}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(f, (800, 450), interpolation=cv2.INTER_AREA))
    for k in range(0, len(tiles), per):
        out = f"{prefix}_{k // per}.jpg"
        cv2.imwrite(out, grid(tiles[k:k + per], cols=2), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{prefix}_*.jpg: {len(tiles)} frames ({nd} disagreement) in {(len(tiles) + per - 1) // per} sheets; "
          f"disagreement frames {len(dis)} of {len(allf)}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    {"crops": lambda: crops(a[1], a[2], a[3]), "diff": lambda: diff(a[1], a[2], a[3], a[4]),
     "frames": lambda: frames(a[1], a[2], a[3], a[4])}[a[0]]()
