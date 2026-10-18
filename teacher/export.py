"""WP8 (CPU + decode): export the teacher labels as training datasets under datasets/ (gitignored).

Two steps, so the expensive part (decoding frames) is resumable per video and the cheap part (splits, balancing, YOLO
files) can be redone in seconds with other settings:

  pool   per video -> datasets/_pool/<id>/{frames/*.jpg, frames.parquet, status.parquet, pool.json}
         + datasets/_state/<id>.pool.done (holds the mtimes of the inputs; a video is redone automatically when
         labels.parquet / tracks.parquet changed since). Failures -> datasets/_state/errors.log, the video is skipped.
  build  -> datasets/{cars, numbers_crop, numbers_full, digits, ids}/ + datasets/README.md + datasets/stats.json
            (images are hardlinks into the pool / work dirs, so a rebuild costs no disk).
  qa     -> datasets/_qa/<dataset>_sheet*.jpg: ~50 random training images per dataset with the labels drawn.

usage (repo root, CPU only; CUDA_VISIBLE_DEVICES is forced empty):
  python teacher/export.py pool [id ...] [--workers 3] [--force] [--allow-unlabelled]
  python teacher/export.py build [--val 25SU] [--neg-frac 0.12] [--max-share 0.3] [--tag-share 0.5] [--max-train N]
  python teacher/export.py qa [--n 50]
  python teacher/export.py all [--workers 3] [...]           (pool + build + qa)

Track status (car tracks of tracks.parquet; labels.parquet from WP5 when present):
  bg    screen_fixed or ego residue (onboard ego car, graphics), or car_verified == 'no': never labelled, does not block
        the frame (the box stays in the picture as background).
  car   car_verified == 'yes' and not in_overlay (in_overlay & yes -> unres: the corner logo has a buggy drawing).
  unres car_verified == 'unsure', or not checked, or anything else -> a frame where such a box (conf >= 0.25) is
        visible is not exported.
  Fallback when the video has no verification yet (labels.parquet without car_verified, or verify.done missing):
        car  = identity label from WP5 (VLM consensus / overlay) or conf_median >= 0.40 (>= 0.50 if long_static),
               not in_overlay. On the 125 eye-labelled crops of work/verify_val this rule is ~95% precise at ~70%
               recall (the VLM check: 97.5% / 76%).
  Verified video, track not checked (median box < 30 px): car if conf_median >= 0.50 and not long_static, else unres.
  Tracks whose conf_max < 0.25 never block a frame (the guard only looks at boxes >= 0.25) and are labelled only
  when 'car'.

Dataset A (cars), per 4 fps frame:
  label boxes   every detection of a 'car' track (any conf), plus linear interpolation over gaps of <= INTERP_GAP
                missing frames (0.5 s); a longer gap inside a car track makes those frames unusable (car probably
                still visible, unlabelled). Two label boxes that nest (intersection >= NEST of the smaller): a big box
                containing >= 2 others is a group box (dropped), otherwise the inner one is a partial box (dropped).
  poison guard  every buggy detection with conf >= 0.25 must be a 'car' or 'bg' track, or be covered by a label box
                (IoU >= 0.5 or >= 80% of it inside one: duplicate / partial box of a labelled car). Otherwise the frame
                is skipped. Frames of transition shots and within EDGE_S of a shot boundary are skipped.
  kinds         pos (>= 1 label box), hardneg (no label box, >= 1 'bg' box >= 0.25: road cars, tyre, ego car, logo),
                neg (work/<id>/negatives.parquet: no car detection >= 0.10 within +-1 s).
  sampling      <= 1 frame per second per shot for pos (the frame with the most confident labels in each 1 s bin),
                <= 1 per HARDNEG_S for hardneg, then perceptual-hash dedupe per video (64-bit DCT pHash; a frame is a
                duplicate when the hash is within PHASH_T bits of an already kept frame AND the label boxes match,
                IoU >= 0.7 each: a static camera with a parked car, not a PTZ shot following a moving car).
  build         splits by whole event (data/events.csv test; --val events; rest train). Train is balanced: no video
                above --max-share of the split, no shot tag above --tag-share of a video (water-filling, random within
                a cell). Negatives fill --neg-frac of every split (half hardneg, half neg when both are available).
"""
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, select_videos, wdir  # noqa: E402
from detect import frame_pipe  # noqa: E402  (the exact decode detect.py used; frame k of the pipe = frame_idx k)

POOL_VERSION = 4        # bump when the pool rules change: every video is redone on the next `pool`
DS = ROOT / "datasets"
POOL = DS / "_pool"
STATE = DS / "_state"
QA = DS / "_qa"

FPS = 4.0
GUARD_CONF = 0.25
INTERP_GAP = 2          # missing 4 fps frames that are interpolated (<= 0.5 s)
INTERP_IOU = 0.3        # ... only when the boxes at both ends overlap this much (no interpolation across fast pans)
LABEL_NMS = 0.5         # two label boxes overlapping this much in one frame: keep the real / more confident one
GUARD_TRACK_CONF = 0.10 # detections of unresolved tracks block the frame from this conf on
GUARD_UNTRACKED_CONF = 0.15  # ... and detections outside any kept track from this conf on
EDGE_S = 0.5            # s from a shot boundary
COVER_IOU, COVER_IN = 0.5, 0.8
NEST = 0.8
GROUP_COVER = 0.6       # a group box must be mostly covered by the boxes it contains
PART_RATIO = 0.35       # a single nested box this big (area share) is the same car
SAMPLE_S = 1.0
HARDNEG_S = 5.0
PHASH_T = 6
DUP_IOU = 0.7
SPAN_GAP = 24           # frames: decode spans merged when closer (as crops.py)
JPEG_Q = 90
FB_CONF, FB_CONF_STATIC, TINY_CONF = 0.40, 0.50, 0.50
EGO_H = 0.8
NUM_MIN_CAR_H = 60      # numbers_full: other labelled cars at least this tall must have number boxes too
DIGIT_PAD = 0.10
ID_SOURCES = ("both-agree", "vlm", "overlay", "heat-vlm-weak")
DEFAULT_VAL = ("25SU",)
HBINS = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.01]


def log(msg):
    STATE.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATE / "export.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- track status

def ego_residue(d, tr, H):
    """Same rule as teacher/verify.py ego_residue (copied: verify.py belongs to WP5): in a shot with a screen_fixed
    track, car tracks whose boxes are mostly > EGO_H x H tall and bottom-anchored."""
    car = tr[tr.cls == "buggy"]
    ego_shots = set(car.shot_id[car.screen_fixed.astype(bool)])
    if not ego_shots:
        return set()
    D = pd.read_parquet(d / "dets.parquet", columns=["shot_id", "track_id", "cls", "y1", "y2"])
    D = D[(D.cls == "buggy") & (D.track_id >= 0) & D.shot_id.isin(ego_shots)]
    D["ego"] = ((D.y2 - D.y1) > EGO_H * H) & (D.y2 >= 0.97 * H)
    share = D.groupby("track_id").ego.mean()
    return set(share[share >= 0.5].index) - set(car.track_id[car.screen_fixed.astype(bool)])


def track_status(d, row):
    """-> DataFrame track_id, status (car / bg / unres), why, label, source, rule ('verify' / 'fallback' / ...)."""
    tr = pd.read_parquet(d / "tracks.parquet")
    car = tr[tr.cls == "buggy"].copy()
    L = pd.read_parquet(d / "labels.parquet") if (d / "labels.parquet").exists() else None
    if L is not None:
        L = L.set_index("track_id")
    if L is not None and "ego_residue" in L.columns:
        ego = set(L.index[L.ego_residue.astype(bool)])
    else:
        ego = ego_residue(d, tr, row["height"])
    cv = L.car_verified.reindex(car.track_id) if L is not None and "car_verified" in L.columns else None
    verified_video = (d / "verify.done").exists() and cv is not None and cv.isin(["yes", "unsure"]).any()
    lab = L.label.reindex(car.track_id) if L is not None else pd.Series(pd.NA, index=car.track_id)
    src = L.source.reindex(car.track_id) if L is not None else pd.Series(None, index=car.track_id, dtype=object)
    io = car.in_overlay.map(lambda v: bool(v) if v is not None and v == v else False).to_numpy(bool)
    out = []
    for i, r in enumerate(car.itertuples()):
        tid = int(r.track_id)
        v = cv.iloc[i] if cv is not None else None
        v = None if v is None or (isinstance(v, float) and v != v) or v is pd.NA else v
        has_lab = not pd.isna(lab.iloc[i])
        if bool(r.screen_fixed) or tid in ego:
            st, why = "bg", "screen_fixed" if bool(r.screen_fixed) else "ego_residue"
        elif v == "no":
            st, why = "bg", "verified_no"
        elif v == "yes":
            st, why = ("unres", "overlay_yes") if io[i] else ("car", "verified_yes")
        elif v == "unsure":
            st, why = "unres", "unsure"
        elif r.conf_max < GUARD_CONF:
            st, why = "low", "conf_max<0.25"   # never blocks a frame, never labelled
        elif verified_video:
            ok = r.conf_median >= TINY_CONF and not bool(r.long_static) and not io[i]
            st, why = ("car", "tiny_conf") if ok else ("unres", "tiny_unchecked")
        else:
            thr = FB_CONF_STATIC if bool(r.long_static) else FB_CONF
            if io[i]:
                st, why = "unres", "fb_overlay"
            elif has_lab:
                st, why = "car", "fb_identity"
            elif r.conf_median >= thr:
                st, why = "car", "fb_conf"
            else:
                st, why = "unres", "fb_lowconf"
        out.append((tid, int(r.shot_id), st, why, None if not has_lab else int(lab.iloc[i]),
                    None if src.iloc[i] is None or (isinstance(src.iloc[i], float)) else str(src.iloc[i]),
                    float(r.h_px_median), float(r.conf_median), bool(r.long_static), bool(io[i])))
    S = pd.DataFrame(out, columns=["track_id", "shot_id", "status", "why", "label", "source", "h_px_median",
                                   "conf_median", "long_static", "in_overlay"])
    S["label"] = S.label.astype("Int64")
    rule = "verify" if verified_video else "fallback"
    return S, rule


# ---------------------------------------------------------------- per-frame labels + guard

def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-9), inter / max(aa, 1e-9), inter / max(ab, 1e-9)


def label_boxes(D, S):
    """-> {frame: [(x1, y1, x2, y2, track_id, conf or -1 (interp))]}, set of hole frames (car visible, unlabelled)."""
    cars = set(S.track_id[S.status == "car"])
    C = D[D.track_id.isin(cars)].sort_values(["track_id", "frame_idx"])
    lab, holes = {}, set()
    for tid, g in C.groupby("track_id", sort=False):
        f = g.frame_idx.to_numpy()
        b = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        c = g.conf.to_numpy(float)
        for k in range(len(f)):
            lab.setdefault(int(f[k]), []).append((*b[k], int(tid), float(c[k])))
            if k + 1 < len(f):
                gap = int(f[k + 1] - f[k] - 1)
                if 0 < gap <= INTERP_GAP and box_iou(b[k], b[k + 1])[0] >= INTERP_IOU:
                    for j in range(1, gap + 1):
                        a = j / (gap + 1)
                        lab.setdefault(int(f[k] + j), []).append((*(b[k] * (1 - a) + b[k + 1] * a), int(tid), -1.0))
                elif gap > 0:
                    holes.update(range(int(f[k]) + 1, int(f[k + 1])))
    return lab, holes


def label_nms(boxes):
    """Duplicate label boxes on one car (two tracks, or an interpolated box next to a real one): keep real detections
    first, then the more confident one."""
    if len(boxes) < 2:
        return boxes, []
    order = sorted(range(len(boxes)), key=lambda i: -(boxes[i][5] if boxes[i][5] >= 0 else -1.0))
    keep = []
    for i in order:
        if all(box_iou(boxes[i], boxes[j])[0] < LABEL_NMS for j in keep):
            keep.append(i)
    return [boxes[i] for i in sorted(keep)], [b for i, b in enumerate(boxes) if i not in keep]


def resolve_nesting(boxes):
    """Nested label boxes (intersection >= NEST of the smaller). Per outer box, its direct children (nested boxes not
    themselves inside another child):
      >= 2 children covering >= GROUP_COVER of it -> group box (several cars): the outer box is dropped;
      1 child with >= PART_RATIO of its area      -> the same car twice (partial / loose box): the child is dropped;
      anything else (outer box much bigger than what it contains) -> ambiguous: it may hold an unlabelled car, the frame
      is skipped. -> (boxes, dropped, ambiguous)"""
    n = len(boxes)
    if n < 2:
        return boxes, [], False
    area = [max((b[2] - b[0]) * (b[3] - b[1]), 1e-6) for b in boxes]
    inside = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and area[j] <= area[i] and box_iou(boxes[i], boxes[j])[2] >= NEST:
                inside[i].add(j)
    drop, amb = set(), False
    for i in range(n):
        if not inside[i]:
            continue
        direct = [j for j in inside[i] if not any(j in inside[k] for k in inside[i] if k != j)]
        if len(direct) >= 2:
            m = np.zeros((64, 64), bool)
            x1, y1, x2, y2 = boxes[i][:4]
            for j in direct:
                a, b, c, e = boxes[j][:4]
                m[int(max(0, (b - y1) / (y2 - y1) * 64)):int(np.ceil(min(1, (e - y1) / (y2 - y1)) * 64)),
                  int(max(0, (a - x1) / (x2 - x1) * 64)):int(np.ceil(min(1, (c - x1) / (x2 - x1)) * 64))] = True
            if m.mean() >= GROUP_COVER:
                drop.add(i)
            else:
                amb = True
        else:
            j = direct[0]
            if area[j] / area[i] >= PART_RATIO:
                drop.add(j)
            else:
                amb = True
    return [b for k, b in enumerate(boxes) if k not in drop], [b for k, b in enumerate(boxes) if k in drop], amb


def frame_table(d, row, S):
    """Evaluate every 4 fps frame with any car-ish content. -> DataFrame frame_idx, shot_id, ok, kind, why, score,
    boxes (list), and a few counters."""
    D = pd.read_parquet(d / "dets.parquet", columns=["frame_idx", "cls", "conf", "x1", "y1", "x2", "y2", "shot_id",
                                                     "track_id"])
    D = D[D.cls == "buggy"].drop(columns="cls")
    shots = pd.read_csv(d / "shots.csv")
    st = S.set_index("track_id").status.to_dict()
    lab, holes = label_boxes(D, S)
    gst = D.track_id.map(lambda t: st.get(int(t), "none") if t >= 0 else "none").to_numpy()
    G = D.assign(st=gst)
    G = G[(G.conf >= GUARD_CONF) | ((G.st == "unres") & (G.conf >= GUARD_TRACK_CONF)) |
          ((G.st == "none") & (G.conf >= GUARD_UNTRACKED_CONF))]
    Gby = {f: g for f, g in G.groupby("frame_idx")}
    # shot of every frame + usable (not transition, not near a boundary)
    nf = int(D.frame_idx.max()) + 2 if len(D) else 0
    shot_of = np.full(nf, -1, np.int32)
    usable = np.zeros(nf, bool)
    for s in shots.itertuples():
        a, b = int(np.ceil(s.t0 * FPS)), int(np.floor(s.t1 * FPS - 1e-6))
        a, b = max(a, 0), min(b, nf - 1)
        if b < a:
            continue
        shot_of[a:b + 1] = s.shot_id
        if not s.is_transition:
            ua, ub = int(np.ceil((s.t0 + EDGE_S) * FPS)), int(np.floor((s.t1 - EDGE_S) * FPS))
            ua, ub = max(ua, 0), min(ub, nf - 1)
            if ub >= ua:
                usable[ua:ub + 1] = True
    frames = sorted(set(lab) | set(Gby))
    rows = []
    for f in frames:
        if f >= nf:
            continue
        boxes, d1 = label_nms(lab.get(f, []))
        boxes, d2, amb = resolve_nesting(boxes)
        dropped = d1 + d2
        why = ""
        if not usable[f]:
            why = "transition_or_edge"
        elif f in holes:
            why = "track_gap"
        elif amb:
            why = "nest_ambiguous"
        g = Gby.get(f)
        n_bg = n_bad = 0
        if g is not None:
            for r in g.itertuples():
                if r.st == "car":
                    continue
                if r.st in ("bg", "low"):
                    n_bg += 1
                    continue
                db = (r.x1, r.y1, r.x2, r.y2)
                covered = False
                for b in boxes:
                    iou, _, in_d = box_iou(b, db)
                    if iou >= COVER_IOU or in_d >= COVER_IN:
                        covered = True
                        break
                if not covered:
                    n_bad += 1
        if not why and n_bad:
            why = "unresolved_box"
        kind = "pos" if boxes else ("hardneg" if n_bg else "")
        if not kind:
            continue
        sc = float(np.mean([b[5] if b[5] >= 0 else 0.2 for b in boxes])) if boxes else 0.0
        rows.append((f, int(shot_of[f]), why == "", kind, why, sc,
                     [[round(float(v), 1) for v in b[:4]] + [int(b[4]), round(float(b[5]), 3)] for b in boxes],
                     n_bg, n_bad, len(dropped), sum(b[5] < 0 for b in boxes),
                     [[round(float(v), 1) for v in b[:4]] + [int(b[4]), round(float(b[5]), 3)] for b in dropped]))
    F = pd.DataFrame(rows, columns=["frame_idx", "shot_id", "ok", "kind", "why", "score", "boxes", "n_bg", "n_bad",
                                    "n_nest_drop", "n_interp", "dropped"])
    tag = shots.set_index("shot_id").tag.to_dict()
    F["tag"] = F.shot_id.map(tag)
    return F, shots, lab


def sample_frames(F):
    """Per shot, greedy by score: a pos frame is taken only if no taken pos frame is closer than SAMPLE_S (hardneg:
    HARDNEG_S), so at most ~1 frame per second per shot and never two neighbouring frames."""
    out = []
    for kind, step in (("pos", SAMPLE_S), ("hardneg", HARDNEG_S)):
        X = F[F.ok & (F.kind == kind)]
        if not len(X):
            continue
        gap = int(round(step * FPS))
        for _, g in X.groupby("shot_id"):
            g = g.sort_values(["score", "frame_idx"], ascending=[False, True])
            taken = []
            fr = g.frame_idx.to_numpy()
            for i, f in enumerate(fr):
                if all(abs(f - t) >= gap for t in taken):
                    taken.append(f)
                    out.append(g.iloc[[i]])
    return pd.concat(out).sort_values("frame_idx") if out else F.iloc[:0]


# ---------------------------------------------------------------- decode + dedupe

def phash(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    c = cv2.dct(g)[:8, :8].flatten()
    bits = c > np.median(c[1:])
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def popcount64(x):
    x = np.ascontiguousarray(x, dtype=np.uint64)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def boxes_match(a, b):
    if len(a) != len(b):
        return False
    used = set()
    for x in a:
        best, bj = 0.0, -1
        for j, y in enumerate(b):
            if j in used:
                continue
            iou = box_iou(x, y)[0]
            if iou > best:
                best, bj = iou, j
        if best < DUP_IOU:
            return False
        used.add(bj)
    return True


def spans(frames, gap=SPAN_GAP):
    out = []
    for f in frames:
        if out and f - out[-1][1] <= gap:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


# ---------------------------------------------------------------- numbers_full candidates

def number_frames(d, L, okframes, lab):
    """Frames of crops with kept number boxes that also pass the car guard and where every other labelled car
    >= NUM_MIN_CAR_H px tall has number boxes too. -> {frame: [number boxes (x1,y1,x2,y2,track,label)]}, stats"""
    if L is None or "number_boxes_src" not in L.columns:
        return {}, dict(crops_with_boxes=0, tracks_with_boxes=0)
    C = pd.read_parquet(d / "crops.parquet", columns=["path", "track_id", "frame_idx"]).set_index("path")
    nb = {}
    ntr = 0
    for r in L.itertuples():
        if not r.number_boxes_src or r.number_boxes_src == "{}" or pd.isna(r.label):
            continue
        m = json.loads(r.number_boxes_src)
        if m:
            ntr += 1
        for p, bl in m.items():
            if p not in C.index:
                continue
            f = int(C.loc[p, "frame_idx"])
            for b in bl:
                nb.setdefault(f, []).append((*b, int(r.track_id), int(r.label)))
    st = dict(crops_with_boxes=sum(1 for r in L.itertuples() for _ in json.loads(r.number_boxes_src or "{}")),
              tracks_with_boxes=ntr, frames_with_boxes=len(nb))
    out = {}
    rej = 0
    for f, nbx in nb.items():
        if f not in okframes:
            rej += 1
            continue
        have = {x[4] for x in nbx}
        bad = [b for b in lab.get(f, []) if b[4] not in have and (b[3] - b[1]) >= NUM_MIN_CAR_H]
        if bad:
            rej += 1
            continue
        out[f] = nbx
    st.update(frames_ok=len(out), frames_rejected=rej)
    return out, st


# ---------------------------------------------------------------- pool stage

def input_stamp(d):
    st = {"export_version": POOL_VERSION}
    for n in ("tracks.parquet", "dets.parquet", "labels.parquet", "negatives.parquet", "shots.csv", "verify.done",
              "grounded.done", "labels.done"):
        p = d / n
        st[n] = round(p.stat().st_mtime, 1) if p.exists() else None
    return st


def pool_video(row, allow_unlabelled=False):
    vid = row["id"]
    d = wdir(vid)
    W, H = row["width"], row["height"]
    t0 = time.time()
    pdir = POOL / vid
    if pdir.exists():
        shutil.rmtree(pdir)
    (pdir / "frames").mkdir(parents=True)
    S, rule = track_status(d, row)
    S.to_parquet(pdir / "status.parquet", index=False)
    F, shots, lab = frame_table(d, row, S)
    okframes = set(F.frame_idx[F.ok])
    L = pd.read_parquet(d / "labels.parquet") if (d / "labels.parquet").exists() else None
    NB, nstat = number_frames(d, L, okframes, lab)
    P = sample_frames(F)
    P = P.assign(src="sample")
    # numbers_full frames are always decoded (even if not picked by the 1/s sampling)
    extra = F[F.frame_idx.isin(set(NB) - set(P.frame_idx)) & F.ok].assign(src="numbers")
    P = pd.concat([P, extra]).sort_values("frame_idx")
    # negatives (no car within +-1 s)
    N = pd.read_parquet(d / "negatives.parquet") if (d / "negatives.parquet").exists() else None
    if N is not None and len(N):
        tag = shots.set_index("shot_id").tag.to_dict()
        N = pd.DataFrame(dict(frame_idx=N.frame_idx.astype(int), shot_id=N.shot_id.astype(int), ok=True, kind="neg",
                              why="", score=0.0, boxes=[[] for _ in range(len(N))], n_bg=0, n_bad=0, n_nest_drop=0,
                              n_interp=0, dropped=[[] for _ in range(len(N))], tag=N.shot_id.map(tag),
                              src="negatives"))
        N = N[~N.frame_idx.isin(P.frame_idx)]
        P = pd.concat([P, N]).sort_values("frame_idx")
    P = P.reset_index(drop=True)
    want = {int(f): i for i, f in enumerate(P.frame_idx)}
    hashes = np.full(len(P), -1, np.int64)
    hv = np.zeros(len(P), np.uint64)
    keep = np.zeros(len(P), bool)
    dup_of = np.full(len(P), -1, np.int64)
    kept_idx = {"pos": [], "neg": []}
    n_dec = 0
    path = ROOT / row["path"]
    for f0, f1 in spans(sorted(want)):
        for kk, img in frame_pipe(path, FPS, W, H, start=f0 / FPS, dur=(f1 - f0 + 1) / FPS + 0.5):
            f = f0 + kk
            if f > f1:
                break
            n_dec += 1
            i = want.get(f)
            if i is None:
                continue
            h = np.uint64(phash(img))
            hv[i] = h
            r = P.iloc[i]
            grp = "pos" if r.kind == "pos" else "neg"
            is_dup = False
            if r.src != "numbers" and kept_idx[grp]:
                ki = np.array(kept_idx[grp])
                dist = popcount64(hv[ki] ^ h)
                for j in ki[dist <= PHASH_T]:
                    if grp == "neg" or boxes_match([b[:4] for b in r.boxes], [b[:4] for b in P.boxes.iloc[j]]):
                        is_dup, dup_of[i] = True, int(P.frame_idx.iloc[j])
                        break
            if is_dup:
                continue
            keep[i] = True
            kept_idx[grp].append(i)
            name = f"{vid}_f{f:06d}.jpg"
            cv2.imwrite(str(pdir / "frames" / name), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    P["phash"] = hv.astype(np.int64)
    P["kept"] = keep
    P["dup_of"] = dup_of
    P["pts"] = P.frame_idx / FPS
    P["image"] = [f"{vid}_f{int(f):06d}.jpg" if k else None for f, k in zip(P.frame_idx, keep)]
    P["numbers"] = [[list(map(float, x[:4])) + [int(x[4]), int(x[5])] for x in NB.get(int(f), [])]
                    for f in P.frame_idx]
    P["boxes"] = P.boxes.map(json.dumps)
    P["dropped"] = P.dropped.map(json.dumps)
    P["numbers"] = P.numbers.map(json.dumps)
    P.to_parquet(pdir / "frames.parquet", index=False)
    whys = F[~F.ok].why.value_counts().to_dict()
    info = dict(video=vid, event=row["event"], rule=rule, W=W, H=H,
                tracks={k: int(v) for k, v in S.status.value_counts().items()},
                track_why={k: int(v) for k, v in S.why.value_counts().items()},
                frames_considered=int(len(F)), frames_ok=int(F.ok.sum()),
                frames_ok_pos=int((F.ok & (F.kind == "pos")).sum()),
                frames_ok_hardneg=int((F.ok & (F.kind == "hardneg")).sum()),
                skipped={k: int(v) for k, v in whys.items()},
                sampled={k: int(v) for k, v in P.kind.value_counts().items()},
                kept={k: int(v) for k, v in P[P.kept].kind.value_counts().items()},
                dups=int((P.dup_of >= 0).sum()), decoded=n_dec, numbers=nstat,
                inputs=input_stamp(d), runtime_s=round(time.time() - t0, 1))
    (pdir / "pool.json").write_text(json.dumps(info, indent=1))
    return info


def pool_done_ok(vid):
    p = STATE / f"{vid}.pool.done"
    if not p.exists():
        return False
    try:
        old = json.loads(p.read_text())["inputs"]
    except Exception:  # noqa: BLE001
        return False
    return old == input_stamp(wdir(vid))


def _pool_one(row, allow_unlabelled):
    vid = row["id"]
    t = time.time()
    try:
        info = pool_video(row, allow_unlabelled)
        (STATE / f"{vid}.pool.done").write_text(json.dumps(info))
        return vid, True, time.time() - t, info
    except Exception as e:  # noqa: BLE001 - one bad video must not stop the others
        with open(STATE / "errors.log", "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {vid} pool: {traceback.format_exc()}\n")
        return vid, False, time.time() - t, repr(e)


def pool(ids=None, workers=3, force=False, allow_unlabelled=False):
    STATE.mkdir(parents=True, exist_ok=True)
    todo = []
    for r in select_videos(ids or None):
        d = wdir(r["id"])
        if not (d / "track.done").exists():
            log(f"{r['id']}: no track.done, skip")
            continue
        if not (d / "labels.done").exists() and not allow_unlabelled:
            log(f"{r['id']}: no labels.done yet, skip (use --allow-unlabelled for a fallback-only cars pool)")
            continue
        if not force and pool_done_ok(r["id"]):
            log(f"{r['id']}: pool up to date, skip")
            continue
        (STATE / f"{r['id']}.pool.done").unlink(missing_ok=True)
        todo.append(r)
    if not todo:
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=max(1, min(workers, 3))) as ex:
        futs = [ex.submit(_pool_one, r, allow_unlabelled) for r in todo]
        for fu in as_completed(futs):
            vid, ok, dt, info = fu.result()
            if ok:
                log(f"{vid}: pool ok in {dt:.0f}s rule={info['rule']} tracks={info['tracks']} "
                    f"frames ok {info['frames_ok']}/{info['frames_considered']} kept={info['kept']} "
                    f"dups={info['dups']} numbers={info['numbers']}")
            else:
                log(f"{vid}: pool FAILED in {dt:.0f}s {info} (datasets/_state/errors.log)")


# ---------------------------------------------------------------- build

def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def yolo_line(cls, b, W, H):
    x1, y1, x2, y2 = max(0.0, b[0]), max(0.0, b[1]), min(float(W), b[2]), min(float(H), b[3])
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def splits_of(val_events):
    ev = pd.read_csv(DATA / "events.csv")
    sp = {}
    for r in ev.itertuples():
        sp[r.event] = "test" if r.split == "test" else ("val" if r.event in val_events else "train")
    return sp


def share_cap(counts, share):
    """Largest c with c <= share * sum(min(n, c)): capping every group at c keeps each group <= share of the total.
    When that is impossible (too few groups), the cap is the second-largest count (no group above the next one)."""
    n = sorted(counts, reverse=True)
    if not n:
        return 0
    if len(n) * share <= 1:
        return n[1] if len(n) > 1 else n[0]
    lo, hi = 0, n[0]
    while lo < hi:
        c = (lo + hi + 1) // 2
        if c <= share * sum(min(x, c) for x in n):
            lo = c
        else:
            hi = c - 1
    return max(lo, 1)


def balance_train(P, max_share, tag_share, max_train, rng):
    """P: pos rows of the train split -> the selected subset. 1) per video, no shot tag above tag_share of the video
    (only when >= 2 tags hold >= 5% each; the cap never goes below 25% of the video); 2) no video above max_share of
    the split; 3) optional global size cap, proportional. Random within each (video, tag) cell."""
    parts = []
    for vid, g in P.groupby("video"):
        tc = g.tag.fillna("none").value_counts()
        if (tc >= 0.05 * len(g)).sum() >= 2:
            cap = max(share_cap(list(tc), tag_share), int(0.25 * len(g)))
        else:
            cap = len(g)
        for t, gg in g.groupby(g.tag.fillna("none")):
            parts.append(gg.sample(min(len(gg), cap), random_state=rng.randint(0, 1 << 30)))
    P = pd.concat(parts)
    vc = P.video.value_counts()
    cap = share_cap(list(vc), max_share)
    P = pd.concat([g.sample(min(len(g), cap), random_state=rng.randint(0, 1 << 30)) for _, g in P.groupby("video")])
    if max_train and len(P) > max_train:
        P = P.sample(max_train, random_state=rng.randint(0, 1 << 30))
    return P


def load_pools(sp):
    rows = []
    infos = {}
    for r in select_videos():
        vid = r["id"]
        p = POOL / vid / "frames.parquet"
        if not (STATE / f"{vid}.pool.done").exists() or not p.exists():
            continue
        X = pd.read_parquet(p)
        X = X[X.kept]
        X["video"] = vid
        X["event"] = r["event"]
        X["split"] = sp.get(r["event"], "train")
        X["W"], X["H"] = r["width"], r["height"]
        rows.append(X)
        infos[vid] = json.loads((POOL / vid / "pool.json").read_text())
    if not rows:
        raise SystemExit("no pooled videos: run `python teacher/export.py pool` first")
    return pd.concat(rows, ignore_index=True), infos


def reset_dir(p):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)


def write_yaml(root, names):
    lines = [f"path: {root.as_posix()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_cars(A, neg_frac, max_share, tag_share, max_train, seed):
    rng = random.Random(seed)
    root = DS / "cars"
    reset_dir(root)
    sel = []
    for split in ("train", "val", "test"):
        X = A[A.split == split]
        Pp = X[(X.kind == "pos") & (X.src != "numbers")]
        if split == "train" and len(Pp):
            Pp = balance_train(Pp, max_share, tag_share, max_train, rng)
        n_neg = int(round(neg_frac / (1 - neg_frac) * len(Pp)))
        Hn = X[X.kind == "hardneg"]
        Nn = X[X.kind == "neg"]
        nh = min(len(Hn), n_neg // 2)
        nn_ = min(len(Nn), n_neg - nh)
        nh = min(len(Hn), n_neg - nn_)
        sel += [Pp, Hn.sample(nh, random_state=seed), Nn.sample(nn_, random_state=seed)]
    S = pd.concat(sel, ignore_index=True)
    recs = []
    for r in S.itertuples():
        src = POOL / r.video / "frames" / r.image
        link(src, root / "images" / r.split / r.image)
        lines = [yolo_line(0, b, r.W, r.H) for b in json.loads(r.boxes)] if r.kind == "pos" else []
        lines = [x for x in lines if x]
        (root / "labels" / r.split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / r.split / (Path(r.image).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        recs.append(dict(image=f"images/{r.split}/{r.image}", split=r.split, event=r.event, video=r.video,
                         frame_idx=int(r.frame_idx), pts=float(r.pts), shot_id=int(r.shot_id), tag=r.tag, kind=r.kind,
                         n_boxes=len(lines), n_interp=int(r.n_interp)))
    write_yaml(root, ["car"])
    I = pd.DataFrame(recs)
    I.to_csv(root / "index.csv", index=False)
    # box size distribution (h / H)
    hs = []
    for r in S[S.kind == "pos"].itertuples():
        for b in json.loads(r.boxes):
            hs.append((r.split, (min(b[3], r.H) - max(b[1], 0)) / r.H))
    Hd = pd.DataFrame(hs, columns=["split", "h"])
    Hd["bin"] = pd.cut(Hd.h, HBINS).astype(str)
    return I, Hd


def build_numbers(A, sp, seed):
    """numbers_crop (stored WP4 crops + kept number boxes) and numbers_full (full frames, car + number)."""
    stats = {}
    # --- numbers_crop
    root = DS / "numbers_crop"
    reset_dir(root)
    recs = []
    for r in select_videos():
        vid = r["id"]
        d = wdir(vid)
        if not (STATE / f"{vid}.pool.done").exists() or not (d / "labels.parquet").exists():
            continue
        L = pd.read_parquet(d / "labels.parquet")
        if "number_boxes" not in L.columns:
            continue
        C = pd.read_parquet(d / "crops.parquet").set_index("path")
        split = sp.get(r["event"], "train")
        for t in L.itertuples():
            if pd.isna(t.label) or not t.number_boxes or t.number_boxes == "{}":
                continue
            for p, bl in json.loads(t.number_boxes).items():
                if not bl or p not in C.index:
                    continue
                c = C.loc[p]
                cw, ch = int(c.cw), int(c.ch)
                name = Path(p).stem
                img = f"{vid}_{name}.jpg"
                lines = [yolo_line(0, b, cw, ch) for b in bl]
                lines = [x for x in lines if x]
                if not lines:
                    continue
                link(ROOT / p, root / "images" / split / img)
                (root / "labels" / split).mkdir(parents=True, exist_ok=True)
                (root / "labels" / split / (Path(img).stem + ".txt")).write_text("\n".join(lines) + "\n")
                recs.append(dict(image=f"images/{split}/{img}", split=split, event=r["event"], video=vid,
                                 track_id=int(t.track_id), frame_idx=int(c.frame_idx), number=int(t.label),
                                 source=t.source, n_boxes=len(lines), crop_w=cw, crop_h=ch,
                                 box_h_px=round(float(np.median([b[3] - b[1] for b in bl])), 1)))
    write_yaml(root, ["number"])
    NC = pd.DataFrame(recs, columns=["image", "split", "event", "video", "track_id", "frame_idx", "number", "source",
                                     "n_boxes", "crop_w", "crop_h", "box_h_px"])
    NC.to_csv(root / "index.csv", index=False)
    stats["numbers_crop"] = NC
    # --- numbers_full
    root = DS / "numbers_full"
    reset_dir(root)
    recs = []
    X = A[A.numbers.map(lambda s: s not in (None, "[]"))]
    for r in X.itertuples():
        link(POOL / r.video / "frames" / r.image, root / "images" / r.split / r.image)
        lines = [yolo_line(0, b, r.W, r.H) for b in json.loads(r.boxes)]
        nums = json.loads(r.numbers)
        lines += [yolo_line(1, b, r.W, r.H) for b in nums]
        lines = [x for x in lines if x]
        (root / "labels" / r.split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / r.split / (Path(r.image).stem + ".txt")).write_text("\n".join(lines) + "\n")
        recs.append(dict(image=f"images/{r.split}/{r.image}", split=r.split, event=r.event, video=r.video,
                         frame_idx=int(r.frame_idx), n_cars=len(json.loads(r.boxes)), n_numbers=len(nums),
                         numbers=" ".join(str(n[5]) for n in nums)))
    write_yaml(root, ["car", "number"])
    NF = pd.DataFrame(recs, columns=["image", "split", "event", "video", "frame_idx", "n_cars", "n_numbers", "numbers"])
    NF.to_csv(root / "index.csv", index=False)
    stats["numbers_full"] = NF
    return stats


def build_digits(sp):
    root = DS / "digits"
    reset_dir(root)
    recs = []
    for r in select_videos():
        vid = r["id"]
        d = wdir(vid)
        if not (STATE / f"{vid}.pool.done").exists() or not (d / "labels.parquet").exists():
            continue
        L = pd.read_parquet(d / "labels.parquet")
        if "number_boxes" not in L.columns:
            continue
        split = sp.get(r["event"], "train")
        C = pd.read_parquet(d / "crops.parquet").set_index("path")
        for t in L.itertuples():
            if pd.isna(t.label) or not t.number_boxes or t.number_boxes == "{}":
                continue
            for p, bl in json.loads(t.number_boxes).items():
                if p not in C.index:
                    continue
                im = cv2.imread(str(ROOT / p))
                if im is None:
                    continue
                hh, ww = im.shape[:2]
                for k, (x1, y1, x2, y2) in enumerate(bl):
                    pw, ph = DIGIT_PAD * (x2 - x1), DIGIT_PAD * (y2 - y1)
                    a, b = int(max(0, np.floor(x1 - pw))), int(max(0, np.floor(y1 - ph)))
                    c, e = int(min(ww, np.ceil(x2 + pw))), int(min(hh, np.ceil(y2 + ph)))
                    if c - a < 4 or e - b < 4:
                        continue
                    name = f"{vid}_{Path(p).stem}_n{k}.jpg"
                    (root / "images" / split).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(root / "images" / split / name), im[b:e, a:c], [cv2.IMWRITE_JPEG_QUALITY, 95])
                    recs.append(dict(image=f"images/{split}/{name}", text=str(int(t.label)), split=split,
                                     event=r["event"], video=vid, track_id=int(t.track_id),
                                     frame_idx=int(C.loc[p, "frame_idx"]), source=t.source, w=c - a, h=e - b,
                                     crop_scale=round(float(C.loc[p, "scale"]), 3)))
    D = pd.DataFrame(recs, columns=["image", "text", "split", "event", "video", "track_id", "frame_idx", "source", "w",
                                    "h", "crop_scale"])
    D.to_csv(root / "labels.csv", index=False)
    return D


def build_ids(sp):
    root = DS / "ids"
    reset_dir(root)
    recs = []
    for r in select_videos():
        vid = r["id"]
        d = wdir(vid)
        if not (STATE / f"{vid}.pool.done").exists() or not (d / "labels.parquet").exists():
            continue
        S = pd.read_parquet(POOL / vid / "status.parquet").set_index("track_id")
        L = pd.read_parquet(d / "labels.parquet")
        L = L[L.label.notna() & L.source.isin(ID_SOURCES)]
        split = sp.get(r["event"], "train")
        C = pd.read_parquet(d / "crops.parquet")
        V = pd.read_parquet(d / "vcrops.parquet") if (d / "vcrops.parquet").exists() else C.iloc[:0]
        Cg = {t: g for t, g in C.groupby("track_id")}
        Vg = {t: g for t, g in V.groupby("track_id")}
        for t in L.itertuples():
            tid = int(t.track_id)
            st = S.status.get(tid, "unres")
            if st == "bg":
                continue   # verified not-a-car / ego / graphic: never an identity
            g = Cg.get(tid)
            kind = "wp4"
            if g is None or not len(g):
                g, kind = Vg.get(tid), "verify"
            if g is None:
                continue
            for c in g.itertuples():
                src = ROOT / c.path
                if not src.exists():
                    continue
                name = f"{vid}_t{tid}_f{int(c.frame_idx)}.jpg"
                rel = Path("images") / split / r["event"] / str(int(t.label)) / name
                link(src, root / rel)
                recs.append(dict(image=rel.as_posix(), split=split, event=r["event"], number=int(t.label),
                                 source=t.source, video=vid, track=tid, shot=int(t.shot_id),
                                 time=round(int(c.frame_idx) / FPS, 2), crop_kind=kind, car_status=st,
                                 h_px=round(float(c.h_px), 1)))
    D = pd.DataFrame(recs, columns=["image", "split", "event", "number", "source", "video", "track", "shot", "time",
                                    "crop_kind", "car_status", "h_px"])
    D.to_csv(root / "labels.csv", index=False)
    return D


def du(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def build(val_events=DEFAULT_VAL, neg_frac=0.12, max_share=0.3, tag_share=0.5, max_train=0, seed=0):
    sp = splits_of(set(val_events))
    A, infos = load_pools(sp)
    I, Hd = build_cars(A, neg_frac, max_share, tag_share, max_train, seed)
    NS = build_numbers(A, sp, seed)
    DG = build_digits(sp)
    ID = build_ids(sp)
    st = dict(splits=sp, val_events=list(val_events), neg_frac=neg_frac, max_share=max_share, tag_share=tag_share,
              max_train=max_train, videos=infos)
    st["cars"] = dict(
        per_split=I.groupby(["split", "kind"]).size().unstack(fill_value=0).to_dict("index"),
        per_event=I.groupby(["split", "event"]).size().unstack(fill_value=0).to_dict("index"),
        per_video=I.groupby(["video", "kind"]).size().unstack(fill_value=0).to_dict("index"),
        per_tag=I.groupby(["split", "tag"]).size().unstack(fill_value=0).to_dict("index"),
        boxes=I.groupby("split").n_boxes.sum().to_dict(),
        box_h=Hd.groupby(["split", "bin"]).size().unstack(fill_value=0).to_dict("index") if len(Hd) else {},
        box_h_quantiles={s: [round(float(x), 3) for x in g.h.quantile([0.05, 0.25, 0.5, 0.75, 0.95])]
                         for s, g in Hd.groupby("split")})
    NC, NF = NS["numbers_crop"], NS["numbers_full"]
    st["numbers_crop"] = dict(images=NC.groupby("split").size().to_dict() if len(NC) else {},
                              tracks=int(NC.groupby(["video", "track_id"]).ngroups) if len(NC) else 0,
                              boxes=int(NC.n_boxes.sum()) if len(NC) else 0)
    st["numbers_full"] = dict(images=NF.groupby("split").size().to_dict() if len(NF) else {},
                              numbers=int(NF.n_numbers.sum()) if len(NF) else 0)
    st["digits"] = dict(images=DG.groupby("split").size().to_dict() if len(DG) else {},
                        strings=int(DG.text.nunique()) if len(DG) else 0,
                        h_quantiles=[int(x) for x in DG.h.quantile([0.05, 0.5, 0.95])] if len(DG) else [])
    st["ids"] = dict(images=ID.groupby("split").size().to_dict() if len(ID) else {},
                     identities=int(ID.groupby(["event", "number"]).ngroups) if len(ID) else 0,
                     tracks=int(ID.groupby(["video", "track"]).ngroups) if len(ID) else 0,
                     by_source=ID.groupby("source").track.nunique().to_dict() if len(ID) else {})
    st["disk_bytes"] = {k: du(DS / k) for k in ("_pool", "cars", "numbers_crop", "numbers_full", "digits", "ids")}
    (DS / "stats.json").write_text(json.dumps(st, indent=1, default=str))
    write_readme(st, I, Hd, NC, NF, DG, ID)
    print_summary(st)
    return st


def print_summary(st):
    c = st["cars"]
    print("cars per split/kind:", json.dumps(c["per_split"]))
    print("cars per split/event:", c["per_event"])
    print("cars boxes:", c["boxes"], "box h/H quantiles (5/25/50/75/95%):", c["box_h_quantiles"])
    for k in ("numbers_crop", "numbers_full", "digits", "ids"):
        print(k, json.dumps(st[k]))
    print("disk MB:", {k: round(v / 2 ** 20) for k, v in st["disk_bytes"].items()})


def md_table(df):
    if df is None or not len(df):
        return "(empty)\n"
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df.itertuples(index=False):
        out += "| " + " | ".join(str(x) for x in r) + " |\n"
    return out


def write_readme(st, I, Hd, NC, NF, DG, ID):
    rules = {v: i["rule"] for v, i in st["videos"].items()}
    vt = pd.DataFrame({v: dict(rule=i["rule"], **{f"tracks_{k}": n for k, n in i["tracks"].items()},
                               frames_ok=i["frames_ok"], frames_considered=i["frames_considered"],
                               **{f"kept_{k}": n for k, n in i["kept"].items()}, dups=i["dups"])
                       for v, i in st["videos"].items()}).T.fillna(0)
    sk = pd.DataFrame({v: i["skipped"] for v, i in st["videos"].items()}).T.fillna(0).astype(int)
    split_kind = I.groupby(["split", "kind"]).size().unstack(fill_value=0)
    ev = I.groupby(["event", "split"]).size().unstack(fill_value=0)
    tag = I.groupby(["tag", "split"]).size().unstack(fill_value=0)
    vid = I.groupby(["video", "kind"]).size().unstack(fill_value=0)
    hb = Hd.groupby(["bin", "split"]).size().unstack(fill_value=0) if len(Hd) else None
    txt = f"""# datasets/ (WP8 export, generated by `teacher/export.py build` on {time.strftime('%Y-%m-%d %H:%M')})

Generated, gitignored. Rebuild: `python teacher/export.py pool --workers 3` (decode, resumable per video) then
`python teacher/export.py build` (seconds) and `python teacher/export.py qa`. All images are hardlinks into
`datasets/_pool/` (full frames) or `work/<id>/crops/` (car crops), so deleting a dataset folder frees nothing; delete
`_pool` to free the frames.

## Provenance

* Source: the 9 YouTube streams in `youtube/` (720p30 H.264, `data/videos.csv`), teacher outputs in `work/<id>/`:
  YOLO-World v2-x detections at 4 fps (`dets_raw.parquet`, WP3), per-shot tracks + flags (`tracks.parquet`), shot tags
  (`shots.csv`, CLIP, ~70% right: stratification only), WP4 crops + Qwen3-VL-4B reads, WP5 `labels.parquet` (identity,
  number boxes from grounded reads kept only when their label equals the consensus, `car_verified` VLM car check) and
  `negatives.parquet`.
* Frames are re-decoded with `teacher/detect.py`'s `frame_pipe` (ffmpeg CPU `fps=4`, seek to frame_idx / 4 s), the same
  decode the detector saw, so stored boxes line up (checked by eye on drawn boxes). JPEG q90, native 1280x720.
* Splits by whole event (`data/events.csv`): test = {sorted(k for k, v in st['splits'].items() if v == 'test')},
  val = {sorted(k for k, v in st['splits'].items() if v == 'val')} (whole event: END + DIN, chosen so that val covers
  both prova types, is not the largest event, and 19SU, the only fixed start-line camera, stays in train),
  train = {sorted(k for k, v in st['splits'].items() if v == 'train')}. Never random frames.
* Car-status rule per video (`verify` = WP5 `car_verified`; `fallback` = no verification yet, see below): {rules}

## A. cars (`cars/`, YOLO, class 0 = car)

Rules (details in the `teacher/export.py` docstring):
* Labels = every detection of a track whose status is `car` (verified Baja car; fallback: WP5 identity or
  conf_median >= 0.40, >= 0.50 for long-static tracks), plus linear interpolation over gaps <= 0.5 s inside the track.
  Longer gaps inside a car track: the frames are dropped (car probably visible but unlabelled).
* Never labelled: screen_fixed (onboard ego car, graphics), ego residue, verified not-a-car; these boxes stay in the
  picture as background (hard negatives). Overlays are not masked; graphic boxes are simply not labelled.
* Poison guard: a frame is exported only if every buggy detection >= 0.25 in it is a `car` / `bg` track or is covered by
  a label box (IoU >= 0.5, or >= 80% inside: duplicate / partial box). Unsure / unchecked / in-overlay-but-yes tracks
  and stray detections block the frame. Transition shots and +-0.5 s around shot boundaries are skipped.
* Nested label boxes: a box containing >= 2 other label boxes is a group box (dropped); a box inside exactly one
  bigger label box is a partial box (dropped).
* Sampling: <= 1 positive frame per second per shot (best label confidence in the bin), hard negatives <= 1 per 5 s per
  shot, pHash dedupe per video (<= {PHASH_T} of 64 bits AND same label boxes, IoU >= {DUP_IOU}). Train is balanced: no video
  above {st['max_share']:.0%} of train, no shot tag above {st['tag_share']:.0%} of a video's frames. Negatives
  = {st['neg_frac']:.0%} of each split, half hard negatives (only verified not-car boxes in frame: road cars, props, ego
  car, logos) and half `negatives.parquet` frames (no detection >= 0.10 within +-1 s).

### Pool per video (before balancing)
{md_table(vt)}
Frames skipped by the guard (counted over all 4 fps frames with car content):
{md_table(sk)}
### Images per split and kind
{md_table(split_kind)}
### Images per event
{md_table(ev)}
### Images per shot tag
{md_table(tag)}
### Images per video
{md_table(vid)}
### Box height / frame height (label boxes)
{md_table(hb)}
Quantiles (5/25/50/75/95%): {st['cars']['box_h_quantiles']}

## B. numbers_crop / numbers_full

* `numbers_crop/` (YOLO, class 0 = number): the stored WP4 car crops (<= 640 px, overlay regions painted gray as the VLM
  saw them) that have >= 1 kept grounded number box. Images: {st['numbers_crop']['images']},
  tracks: {st['numbers_crop']['tracks']}, boxes: {st['numbers_crop']['boxes']}.
* `numbers_full/` (YOLO, 0 = car, 1 = number; design (i)): the full frame of those crops, only if it passes the car guard
  and every other labelled car >= {NUM_MIN_CAR_H} px tall in the frame also has number boxes. Images:
  {st['numbers_full']['images']}, number boxes: {st['numbers_full']['numbers']}.

## C. digits (`digits/images/<split>/*.jpg` + `digits/labels.csv`)

Number-box crops (+10% padding, cut from the stored crop) with `text` = the track's consensus number. Images:
{st['digits']['images']}, distinct strings: {st['digits']['strings']}, height px (5/50/95%): {st['digits']['h_quantiles']}.

## D. ids (`ids/images/<split>/<event>/<number>/*.jpg` + `ids/labels.csv`)

WP4 car crops (or the verify crops when a track has none) of tracks with an identity label (source in {list(ID_SOURCES)}:
`both-agree` = VLM consensus and broadcast overlay agree, `vlm` = VLM consensus only (vote share >= 0.5, >= 3 votes,
snapped to the event roster), `overlay` = lower third with one car tracked, `heat-vlm-weak` = SPP heat pair + weak
VLM votes). Tracks with status bg are excluded. Images: {st['ids']['images']}, identities (event, number):
{st['ids']['identities']}, tracks: {st['ids']['tracks']}, tracks by source: {st['ids']['by_source']}.

## Known label-noise sources

* Teacher misses: a car the detector never saw (no box at any conf) is unlabelled in the frame. The guard only checks
  boxes >= 0.25; recovery scenes with marshals, cars behind fences in the pits and small cars ahead in onboard views were
  the known recall gaps (WP3).
* Car check: verify precision ~97.5% at score >= 0.5 on the eye-labelled set, so ~2-3% of 'car' tracks are not cars
  (road cars, props). In the fallback rule ~5%. Verified-'no' tracks that are real cars appear unlabelled (rare at
  the <= 0.3 threshold; 'no' on crops < 60 px is turned into unsure).
* Boxes: detector boxes (not hand-drawn); interpolated boxes in fast PTZ moves can lag by a few px; nested-box resolution
  can drop the wrong box of a pair. Partially visible cars entering the frame before the first detection are unlabelled.
* Numbers: the grounded read is precise (25/25 boxes on real numbers) but not exhaustive: ~1 in 5 secondary number
  instances (side plate / firewall of the same car) has no box, and other cars' numbers in a crop are unlabelled.
* Identity (ids, digits): VLM consensus ~98% precise in session 1 (vote share >= 0.5); overlay-only labels depend on one
  car being tracked during the lower third. Mud cars rarely reach a consensus, so they are under-represented.
* Shot tags are CLIP zero-shot (~70% right): balancing by tag is approximate.
"""
    (DS / "README.md").write_text(txt, encoding="utf-8")


# ---------------------------------------------------------------- QA sheets

def qa(n=50, seed=1):
    QA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    cols = {0: (0, 255, 0), 1: (0, 0, 255)}

    def tile(img, lines_or_text, cap, size=(400, 225)):
        h, w = img.shape[:2]
        im = img.copy()
        if isinstance(lines_or_text, list):
            for ln in lines_or_text:
                c, x, y, bw, bh = ln
                p1 = (int((x - bw / 2) * w), int((y - bh / 2) * h))
                p2 = (int((x + bw / 2) * w), int((y + bh / 2) * h))
                cv2.rectangle(im, p1, p2, cols.get(int(c), (255, 0, 255)), max(1, w // 400))
        s = min(size[0] / w, size[1] / h)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        t = np.zeros((size[1] + 18, size[0], 3), np.uint8)
        t[18:18 + im.shape[0], :im.shape[1]] = im
        cv2.putText(t, cap[:60], (2, 13), 0, 0.42, (0, 255, 255), 1)
        return t

    def sheet(tiles, name, per_row=5, rows_per=5):
        while len(tiles) % per_row:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
        for k in range(0, len(rows), rows_per):
            cv2.imwrite(str(QA / f"{name}_sheet{k // rows_per}.jpg"), np.vstack(rows[k:k + rows_per]),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])

    def read_lines(p):
        if not p.exists():
            return []
        return [list(map(float, x.split())) for x in p.read_text().split("\n") if x.strip()]

    for f in QA.glob("*.jpg"):
        f.unlink()
    for ds in ("cars", "numbers_full", "numbers_crop"):
        root = DS / ds
        if not (root / "index.csv").exists():
            continue
        try:
            I = pd.read_csv(root / "index.csv")
        except pd.errors.EmptyDataError:
            continue
        I = I[I.split == "train"] if (I.split == "train").any() else I
        if not len(I):
            continue
        pick = I.sample(min(n, len(I)), random_state=seed)
        tiles = []
        for r in pick.itertuples():
            ip = root / r.image
            lp = root / r.image.replace("images/", "labels/", 1).rsplit(".", 1)[0]
            lp = Path(str(lp) + ".txt")
            img = cv2.imread(str(ip))
            if img is None:
                continue
            cap = f"{Path(r.image).stem} {getattr(r, 'kind', '')} {getattr(r, 'tag', '') or ''}"
            if ds == "numbers_crop":
                cap = f"{Path(r.image).stem} #{r.number}"
            tiles.append(tile(img, read_lines(lp), cap, (640, 360) if ds != "numbers_crop" else (200, 200)))
        if tiles:
            if ds == "numbers_crop":
                sheet(tiles, ds, 10)
            else:
                sheet(tiles, ds, 3, 3)   # 9 frames per sheet at 640x360: small cars stay checkable
    for ds, csvn in (("digits", "labels.csv"), ("ids", "labels.csv")):
        p = DS / ds / csvn
        if not p.exists():
            continue
        I = pd.read_csv(p)
        if not len(I):
            continue
        I = I[I.split == "train"] if (I.split == "train").any() else I
        pick = I.sample(min(n, len(I)), random_state=seed)
        tiles = []
        for r in pick.itertuples():
            img = cv2.imread(str(DS / ds / r.image))
            if img is None:
                continue
            cap = f"'{r.text}'" if ds == "digits" else f"{r.event} #{r.number} {r.source}"
            tiles.append(tile(img, cap, cap, (160, 120) if ds == "digits" else (200, 200)))
        if tiles:
            sheet(tiles, ds, 10)
    print("QA sheets ->", sorted(p.name for p in QA.glob("*.jpg")))


# ---------------------------------------------------------------- CLI

def main(argv):
    if not argv:
        print(__doc__)
        return 2
    mode, a = argv[0], list(argv[1:])

    def opt(name, default, typ=str):
        if name in a:
            i = a.index(name)
            v = typ(a[i + 1])
            del a[i:i + 2]
            return v
        return default

    workers = opt("--workers", 3, int)
    val = opt("--val", ",".join(DEFAULT_VAL))
    neg_frac = opt("--neg-frac", 0.12, float)
    max_share = opt("--max-share", 0.3, float)
    tag_share = opt("--tag-share", 0.5, float)
    max_train = opt("--max-train", 0, int)
    n = opt("--n", 50, int)
    force = "--force" in a
    allow = "--allow-unlabelled" in a
    ids = [x for x in a if not x.startswith("--")]
    if mode in ("pool", "all"):
        pool(ids, workers, force, allow)
    if mode in ("build", "all"):
        build(tuple(x for x in val.split(",") if x), neg_frac, max_share, tag_share, max_train)
    if mode in ("qa", "all"):
        qa(n)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
