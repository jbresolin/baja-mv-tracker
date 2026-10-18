"""WP4 step 1 (CPU): pick K crops per readable car track and cut them from the video -> work/<id>/crops/,
work/<id>/crops.parquet, work/<id>/crop_tracks.parquet, work/<id>/ctx/ (context frames), crops.done

Readable tracks (tracks.parquet from WP3): cls buggy, not screen_fixed, not nested in a larger
coexisting box (same car, the outer track is read), not in a transition shot, and >= MIN_CAND detections whose box is
>= H_MIN px high (720p source). If WP3 left in_overlay null (track.py ran before overlay.done), it is computed here
with track.py's own overlay_flag().
Crop choice per track (as lapvision/passes.pick_crops): candidates = detections with h >= H_MIN and area >= 0.4 x the
track's largest candidate; split in time into K chunks; per chunk the PRE best by area x conf are decoded and the one
with the best rank(sharpness) + rank(area) is kept. Sharpness = variance of the Laplacian of the crop at 128 px height.
Crops: det box + 5% padding per side, clipped, cut at native resolution from the SAME frames detect.py saw (ffmpeg
fps=4 pipe, frame_idx k = k-th output frame; seeking to frame_idx / 4 s reproduces the frames bit for bit, checked),
JPEG q92, downscaled to <= 640 px on the long side. Context frame per track: the frame of its largest crop at 640x360
with the box drawn (for the QA page).

crops.parquet: track_id, shot_id, frame_idx, pts, x1..y2 (det box, src px), cx1..cy2 (crop window, src px),
  scale (file px per src px), cw, ch (file size), h_px, conf, sharp, chunk, path (repo-relative, forward slashes)
crop_tracks.parquet: every car track with readable (bool), reason (why not), n_cand, n_crops, h_cand_max, ctx path

usage (repo root): python teacher/crops.py [id ...] [--workers 2] [--force] [--hmin 60] [--k 8] [--update]
                   (--update: add crops for tracks that became readable since the last run -> crops_update.done)
                   python teacher/crops.py verify <id> [--n 12]    (redraw boxes on freshly decoded frames -> sheet)
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # "" does not hide the GPU on Windows
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, cli_ids, run_stage, select_videos, wdir  # noqa: E402
from detect import frame_pipe  # noqa: E402  (the exact decode detect.py used)

FPS = 4.0
H_MIN = 60
K = 8
MIN_CAND = 3
PRE = 3            # decoded candidates per chunk
PAD = 0.05
MAX_SIDE = 640
SPAN_GAP = 24      # frames: decode spans merged when closer than this (6 s)
CTX_W, CTX_H = 640, 360
MASK_SKIP = {"inset_or_logo_box", "bracket_or_fullscreen_graphic"}  # loose boxes around live video: never masked
MASK_MAX_AREA = 0.25
MASK_DROP = 0.5    # candidates whose crop is more than this share masked are dropped
GRAY = 114


def mask_regions(d, W, H):
    """Overlay regions (overlays.csv, after overrides) that are painted gray inside crops while visible, so the VLM never
    reads tower / ticker / lower-third numbers as the car's number. -> list of (box, intervals)."""
    p = d / "overlays.csv"
    if not p.exists():
        return []
    o = pd.read_csv(p)
    out = []
    for r in o.itertuples():
        if str(r.kind) in MASK_SKIP or (r.x2 - r.x1) * (r.y2 - r.y1) > MASK_MAX_AREA * W * H:
            continue
        iv = [tuple(map(float, x.split("-"))) for x in str(r.intervals).split(";") if "-" in x]
        out.append(((int(r.x1), int(r.y1), int(r.x2), int(r.y2)), iv))
    return out


def visible(regs, t):
    return [b for b, iv in regs if any(a - 2.5 <= t <= e + 2.5 for a, e in iv)]


def apply_mask(c, win, boxes):
    """Paint the visible overlay boxes gray inside crop c (window win in source px). Returns masked share."""
    cx1, cy1, cx2, cy2 = win
    m = 0
    for x1, y1, x2, y2 in boxes:
        ix1, iy1, ix2, iy2 = max(x1, cx1), max(y1, cy1), min(x2, cx2), min(y2, cy2)
        if ix2 > ix1 and iy2 > iy1:
            c[iy1 - cy1:iy2 - cy1, ix1 - cx1:ix2 - cx1] = GRAY
            m = 1
    if not m:
        return 0.0
    return float(np.mean(np.all(c == GRAY, axis=2)))


def ensure_overlay_flag(d, tr, W, H):
    if tr["in_overlay"].notna().any() or not (d / "overlay.done").exists():
        return tr, False
    import track as T  # noqa: E402
    regs = T.load_overlays(d, W, H)
    if regs is None:
        return tr, False
    dets = pd.read_parquet(d / "dets.parquet", columns=["pts", "track_id", "x1", "y1", "x2", "y2"])
    flag, share = T.overlay_flag(dets, tr, regs)
    tr = tr.copy()
    tr["in_overlay"] = flag.to_numpy()
    tr["overlay_share"] = share.to_numpy() if share is not None else None
    return tr, True


def select(d, row, hmin=H_MIN, k=K):
    W, H = row["width"], row["height"]
    tr = pd.read_parquet(d / "tracks.parquet")
    tr, recomputed = ensure_overlay_flag(d, tr, W, H)
    car = tr[tr.cls == "buggy"].copy()
    shots = pd.read_csv(d / "shots.csv")
    trans = set(shots.loc[shots.is_transition == 1, "shot_id"].astype(int))
    reason = np.full(len(car), "", object)
    reason[car.screen_fixed.to_numpy(bool)] = "screen_fixed"
    io = car.in_overlay.map(lambda v: bool(v) if v is not None and v == v else False).to_numpy(bool)
    car["in_overlay_soft"] = io  # soft flag (WP3: ~half of the in_overlay tracks are real cars): carried, not excluded
    reason[(reason == "") & (car.nested_in.to_numpy() >= 0)] = "nested"
    from verify import ego_residue  # noqa: E402
    ego, _ = ego_residue(d, H)
    reason[(reason == "") & car.track_id.isin(ego).to_numpy()] = "ego_residue"
    reason[(reason == "") & car.shot_id.isin(trans).to_numpy()] = "transition"
    car["reason"] = reason
    dets = pd.read_parquet(d / "dets.parquet")
    dets = dets[(dets.cls == "buggy") & (dets.track_id >= 0)]
    dets = dets[dets.track_id.isin(car.track_id[car.reason == ""])].copy()
    dets["h"] = dets.y2 - dets.y1
    dets["area"] = (dets.x2 - dets.x1) * dets.h
    cand = dets[dets.h >= hmin]
    ncand = cand.groupby("track_id").size()
    car["n_cand"] = car.track_id.map(ncand).fillna(0).astype(int)
    car["h_cand_max"] = car.track_id.map(cand.groupby("track_id").h.max()).fillna(0.0)
    car.loc[(car.reason == "") & (car.n_cand < MIN_CAND), "reason"] = f"lt{MIN_CAND}_boxes_ge{hmin}px"
    car["readable"] = car.reason == ""
    cand = cand[cand.track_id.isin(car.track_id[car.readable])]
    picks = []
    for tid, g in cand.groupby("track_id", sort=False):
        g = g[g.area >= 0.4 * g.area.max()].sort_values("frame_idx")
        for ci, idx in enumerate(np.array_split(np.arange(len(g)), min(k, len(g)))):
            c = g.iloc[idx]
            c = c.assign(score=c.area * c.conf).nlargest(PRE, "score")
            for r in c.itertuples():
                picks.append((int(tid), ci, int(r.frame_idx), float(r.pts), int(r.shot_id), float(r.conf),
                              float(r.x1), float(r.y1), float(r.x2), float(r.y2)))
    P = pd.DataFrame(picks, columns=["track_id", "chunk", "frame_idx", "pts", "shot_id", "conf", "x1", "y1", "x2",
                                     "y2"])
    return car, P, recomputed


def spans(frames, gap=SPAN_GAP):
    out = []
    for f in frames:
        if out and f - out[-1][1] <= gap:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def crop_window(r, W, H):
    w, h = r.x2 - r.x1, r.y2 - r.y1
    cx1 = int(max(0, np.floor(r.x1 - PAD * w)))
    cy1 = int(max(0, np.floor(r.y1 - PAD * h)))
    cx2 = int(min(W, np.ceil(r.x2 + PAD * w)))
    cy2 = int(min(H, np.ceil(r.y2 + PAD * h)))
    return cx1, cy1, cx2, cy2


def sharpness(c):
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    s = 128.0 / max(g.shape[0], 1)
    g = cv2.resize(g, (max(8, int(g.shape[1] * s)), 128), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def finalize(tid, cands, thumbs, d, vid, W, H):
    """cands: list of dicts (with 'img'); choose one per chunk, write files. Returns (crop rows, ctx path)."""
    df = pd.DataFrame([{k: v for k, v in c.items() if k != "img"} for c in cands])
    df["img_i"] = range(len(cands))
    df["sc"] = df.sharp.rank(pct=True) + df.area.rank(pct=True)
    best = df.sort_values("sc").groupby("chunk").tail(1).sort_values("frame_idx")
    rows = []
    cdir = d / "crops"
    for r in best.itertuples():
        img = cands[r.img_i]["img"]
        s = min(1.0, MAX_SIDE / max(img.shape[:2]))
        if s < 1:
            img = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        name = f"s{int(r.shot_id)}_t{tid}_f{int(r.frame_idx)}.jpg"
        cv2.imwrite(str(cdir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append(dict(track_id=tid, shot_id=int(r.shot_id), frame_idx=int(r.frame_idx), pts=r.pts, x1=r.x1,
                         y1=r.y1, x2=r.x2, y2=r.y2, cx1=r.cx1, cy1=r.cy1, cx2=r.cx2, cy2=r.cy2, scale=s,
                         cw=img.shape[1], ch=img.shape[0], h_px=r.y2 - r.y1, conf=r.conf, sharp=r.sharp,
                         chunk=int(r.chunk), masked=round(float(r.masked), 3), path=f"work/{vid}/crops/{name}"))
    big = best.sort_values("area").iloc[-1]
    th = thumbs[int(big.frame_idx)].copy()
    sx, sy = CTX_W / W, CTX_H / H
    cv2.rectangle(th, (int(big.x1 * sx), int(big.y1 * sy)), (int(big.x2 * sx), int(big.y2 * sy)), (0, 255, 255), 2)
    ctx = d / "ctx" / f"t{tid}.jpg"
    cv2.imwrite(str(ctx), th, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return rows, f"work/{vid}/ctx/t{tid}.jpg"


def process(row, hmin=H_MIN, k=K, update=False):
    """update=True: keep the existing crops and only add tracks that became readable since (re-flagged tracks.parquet,
    in_overlay made soft); crops.parquet / crop_tracks.parquet are rewritten with the union."""
    vid = row["id"]
    d = wdir(vid)
    W, H = row["width"], row["height"]
    t0 = time.time()
    car, P, recomputed = select(d, row, hmin, k)
    old = None
    if update and (d / "crops.parquet").exists():
        old = pd.read_parquet(d / "crops.parquet")
        old_ct = pd.read_parquet(d / "crop_tracks.parquet")
        P = P[~P.track_id.isin(old.track_id)]
    else:
        for sub in ("crops", "ctx"):  # a re-run replaces the previous crop set entirely
            (d / sub).mkdir(exist_ok=True)
            for f in (d / sub).glob("*.jpg"):
                f.unlink()
    last = P.groupby("track_id").frame_idx.max()
    by_frame = {f: g for f, g in P.groupby("frame_idx")}
    ends = {}
    for tid, f in last.items():
        ends.setdefault(int(f), []).append(int(tid))
    live, thumbs = {}, {}
    rows, ctx = [], {}
    frames = sorted(by_frame)
    n_dec = n_masked_drop = 0
    regs = mask_regions(d, W, H)
    for f0, f1 in spans(frames):
        for kk, img in frame_pipe(ROOT / row["path"], FPS, W, H, start=f0 / FPS, dur=(f1 - f0 + 1) / FPS + 0.5):
            f = f0 + kk
            if f > f1:
                break
            n_dec += 1
            if f not in by_frame:
                continue
            g = by_frame[f]
            thumbs[f] = cv2.resize(img, (CTX_W, CTX_H), interpolation=cv2.INTER_AREA)
            vis = visible(regs, f / FPS)
            for r in g.itertuples():
                cx1, cy1, cx2, cy2 = crop_window(r, W, H)
                c = img[cy1:cy2, cx1:cx2].copy()
                mfrac = apply_mask(c, (cx1, cy1, cx2, cy2), vis) if vis else 0.0
                if mfrac > MASK_DROP:
                    n_masked_drop += 1
                    continue
                live.setdefault(int(r.track_id), []).append(dict(
                    track_id=int(r.track_id), chunk=int(r.chunk), frame_idx=f, pts=r.pts, shot_id=int(r.shot_id),
                    conf=r.conf, x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2, cx1=cx1, cy1=cy1, cx2=cx2, cy2=cy2,
                    area=(r.x2 - r.x1) * (r.y2 - r.y1), sharp=sharpness(c), masked=mfrac, img=c))
            for tid in ends.get(f, []):
                cands = live.pop(tid, [])
                if cands:
                    rr, ctx[tid] = finalize(tid, cands, thumbs, d, vid, W, H)
                    rows.extend(rr)
            # free context thumbnails no live track can still use
            minf = min(v[0]["frame_idx"] for v in live.values()) if live else f + 1
            for fr in [x for x in thumbs if x < minf]:
                del thumbs[fr]
    if live:
        raise RuntimeError(f"{len(live)} tracks not finalised (decode ended early?)")
    C = pd.DataFrame(rows)
    n_new = len(C)
    if old is not None:
        C = pd.concat([old, C], ignore_index=True)
        ctx = {**old_ct.dropna(subset=["ctx"]).set_index("track_id").ctx.to_dict(), **ctx}
    C.to_parquet(d / "crops.parquet", index=False)
    car["ctx"] = car.track_id.map(ctx)
    car["n_crops"] = car.track_id.map(C.groupby("track_id").size() if len(C) else {}).fillna(0).astype(int)
    keep = ["track_id", "shot_id", "t0", "t1", "n", "conf_max", "h_px_median", "h_px_max", "h_cand_max", "n_cand",
            "n_crops", "readable", "reason", "screen_fixed", "in_overlay", "in_overlay_soft", "nested_in", "small", "long_static", "ctx"]
    ct = car[keep].copy()
    ct["in_overlay"] = ct.in_overlay.map(lambda v: None if v is None or v != v else bool(v))
    ct.to_parquet(d / "crop_tracks.parquet", index=False)
    tm = (d / "tracks.parquet").stat().st_mtime
    info = dict(video=vid, hmin=hmin, k=k, car_tracks=int(len(car)), readable=int(car.readable.sum()),
                reasons={str(a): int(b) for a, b in car.reason.value_counts().items()}, crops=int(len(C)), new_crops=n_new,
                decoded_frames=n_dec, mask_regions=len(regs), masked_dropped=n_masked_drop, in_overlay_recomputed=recomputed, tracks_mtime=tm,
                runtime_s=round(time.time() - t0, 1))
    (d / "crops.json").write_text(json.dumps(info, indent=1))
    return json.dumps(info)


def verify(vid, n=12, seed=0):
    """Redraw the stored det boxes on frames decoded independently (ffmpeg accurate seek to pts, 1280x720) and on the
    stored crop -> out/crops_verify_<id>.jpg."""
    import subprocess
    from common import tool
    row = select_videos([vid])[0]
    d = wdir(vid)
    C = pd.read_parquet(d / "crops.parquet").sample(n, random_state=seed)
    tiles = []
    for r in C.itertuples():
        # independent decode: all source frames, pick the one nearest pts (no fps filter)
        cmd = [tool("ffmpeg"), "-v", "info", "-ss", f"{max(r.pts - 0.5, 0):.3f}", "-i", str(ROOT / row["path"]),
               "-t", "1.0", "-vf", "showinfo", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        p = subprocess.run(cmd, capture_output=True)
        fs = row["width"] * row["height"] * 3
        frames = [np.frombuffer(p.stdout[i:i + fs], np.uint8).reshape(row["height"], row["width"], 3)
                  for i in range(0, len(p.stdout) - fs + 1, fs)]
        ts = [float(x.split("pts_time:")[1].split()[0]) for x in p.stderr.decode().splitlines() if "pts_time:" in x]
        t_rel = r.pts - max(r.pts - 0.5, 0)
        j = int(np.argmin([abs(t - t_rel) for t in ts[:len(frames)]])) if frames else None
        crop = cv2.imread(str(ROOT / r.path))
        fp = next(frame_pipe(ROOT / row["path"], FPS, row["width"], row["height"], start=r.frame_idx / FPS,
                             dur=0.5))[1].copy()
        for img, col in ((fp, (0, 255, 255)),) + (((frames[j].copy(), (255, 0, 255)),) if j is not None else ()):
            cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, 3)
        a = cv2.resize(fp, (480, 270))
        b = cv2.resize(frames[j], (480, 270)) if j is not None else np.zeros_like(a)
        if j is not None:
            cv2.rectangle(b, (int(r.x1 * 0.375), int(r.y1 * 0.375)), (int(r.x2 * 0.375), int(r.y2 * 0.375)),
                          (255, 0, 255), 2)
        cs = 270 / crop.shape[0]
        c = cv2.resize(crop, (max(1, int(crop.shape[1] * cs)), 270))[:, :320]
        c = cv2.copyMakeBorder(c, 0, 0, 0, 320 - c.shape[1], cv2.BORDER_CONSTANT)
        tile = np.hstack([a, b, c])
        cv2.putText(tile, f"t{r.track_id} f{r.frame_idx} pts {r.pts:.2f}", (5, 20), 0, 0.6, (0, 0, 255), 2)
        tiles.append(tile)
    sheet = np.vstack(tiles)
    out = ROOT / "out" / f"crops_verify_{vid}.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


def main(argv):
    if argv and argv[0] == "verify":
        n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 12
        verify(argv[1], n)
        return
    argv = list(argv)
    opt = {"--hmin": H_MIN, "--k": K}
    for key in opt:
        if key in argv:
            i = argv.index(key)
            opt[key] = int(argv[i + 1])
            del argv[i:i + 2]
    hmin, k = opt["--hmin"], opt["--k"]
    workers, force, ids, rest = cli_ids(argv)
    rows = select_videos(ids or None)
    if "--update" in rest:
        run_stage("crops_update", process, rows, workers=workers, force=force, requires=("crops",), hmin=hmin, k=k,
                  update=True)
        return
    run_stage("crops", process, rows, workers=workers, force=force, requires=("track",), hmin=hmin, k=k)


if __name__ == "__main__":
    main(sys.argv[1:])
