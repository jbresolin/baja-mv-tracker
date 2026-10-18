"""WP5b: per-track "is this really a Baja car?" check (detector dataset A must be built from verified tracks only).

WP3 measured YOLO-World's car detections on the streams at ~71-83% real cars (road sedans / pickups, a crop sprayer, a
tyre and props under 19SU's fixed camera, TV commercials, full-screen maps). The plain number read does not reject them.

Pool: every car track that is not screen_fixed and not an ego residue (see below), with median box height >= POOL_H px
(in_overlay is soft: kept, flag carried). Per track up to 2 crops: reuse the 2 largest WP4 crops when the track has
them, else cut 1-2 detections (largest conf >= 0.25 box, and a second one >= 5 s away for long tracks) from the same
4 fps frames as detect.py (5% padding, overlays masked as in crops.py) -> work/<id>/vcrops/, vcrops.parquet.
VLM (GPU): one yes/no question per crop; score = P(yes) / (P(yes) + P(no)) from the first output token's logits
-> work/<id>/verify.parquet (+ verify.done). labels.py turns the per-track mean score into car_verified yes/no/unsure.

Ego residue (WP3 open problem): in a shot that contains a screen_fixed track, any car track whose boxes are mostly
taller than EGO_H x frame height and bottom-anchored (y2 >= 0.97 H) is the onboard ego car missed by WP3's rules:
flag ego_residue, never a car label, not verified.

usage (repo root): python teacher/verify.py crops [id ...] [--workers 2] [--force]       (CPU)
                   python teacher/verify.py run [id ...] [--batch 6] [--second]          (GPU, PAUSE honoured)
                   (first pass: largest crop per track; --second: the 2nd crop of tracks with an unsure first score)
                   python teacher/verify.py valset [--n 160]       (validation crops + sheets for labelling by eye)
                   python teacher/verify.py valeval <labels.csv>   (precision / recall of the prompt vs eye labels)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORK, cli_ids, log_error, run_stage, select_videos, wdir  # noqa: E402

POOL_H = 30
EGO_H = 0.8
PROMPT = ("Is this an off-road Baja SAE race buggy (a small single-seat off-road car with an exposed tubular roll "
          "cage)? Answer yes or no.")
YES_T, NO_T = 0.7, 0.3


def ego_residue(d, H):
    """-> set of car track ids that are ego residue (see module doc)."""
    tr = pd.read_parquet(d / "tracks.parquet")
    car = tr[tr.cls == "buggy"]
    ego_shots = set(car.shot_id[car.screen_fixed.astype(bool)])
    if not ego_shots:
        return set(), 0
    D = pd.read_parquet(d / "dets.parquet", columns=["shot_id", "track_id", "cls", "y1", "y2"])
    D = D[(D.cls == "buggy") & (D.track_id >= 0) & D.shot_id.isin(ego_shots)]
    D["ego"] = ((D.y2 - D.y1) > EGO_H * H) & (D.y2 >= 0.97 * H)
    share = D.groupby("track_id").ego.mean()
    cand = set(share[share >= 0.5].index) - set(car.track_id[car.screen_fixed.astype(bool)])
    return cand, int(D.ego.sum())


def pool(d, H):
    tr = pd.read_parquet(d / "tracks.parquet")
    car = tr[tr.cls == "buggy"].copy()
    ego, nbox = ego_residue(d, H)
    car["ego_residue"] = car.track_id.isin(ego)
    keep = (~car.screen_fixed.astype(bool)) & (~car.ego_residue) & (car.h_px_median >= POOL_H)
    return car, car[keep], nbox


def crops_stage(row):
    import cv2
    import crops as CR
    from detect import frame_pipe
    vid = row["id"]
    d = wdir(vid)
    W, H = row["width"], row["height"]
    car, P, nbox = pool(d, H)
    (d / "vcrops").mkdir(exist_ok=True)
    for f in (d / "vcrops").glob("*.jpg"):
        f.unlink()
    have = pd.read_parquet(d / "crops.parquet") if (d / "crops.parquet").exists() else pd.DataFrame(
        columns=["track_id", "path", "h_px", "frame_idx"])
    reuse = have[have.track_id.isin(P.track_id)].sort_values("h_px", ascending=False).groupby("track_id").head(2)
    rows = [dict(track_id=int(r.track_id), frame_idx=int(r.frame_idx), h_px=float(r.h_px), path=r.path, reused=True)
            for r in reuse.itertuples()]
    need = P[~P.track_id.isin(reuse.track_id)]
    D = pd.read_parquet(d / "dets.parquet")
    D = D[(D.cls == "buggy") & D.track_id.isin(need.track_id)].copy()
    D["h"] = D.y2 - D.y1
    D["area"] = D.h * (D.x2 - D.x1)
    picks = []
    for tid, g in D.groupby("track_id"):
        g2 = g[g.conf >= 0.25] if (g.conf >= 0.25).any() else g
        a = g2.loc[g2.area.idxmax()]
        picks.append(a)
        far = g2[(g2.pts - a.pts).abs() >= 5.0]
        if len(far):
            picks.append(far.loc[far.area.idxmax()])
    Pk = pd.DataFrame(picks)
    regs = CR.mask_regions(d, W, H)
    if len(Pk):
        by_frame = {f: g for f, g in Pk.groupby("frame_idx")}
        for f0, f1 in CR.spans(sorted(by_frame)):
            for kk, img in frame_pipe(ROOT / row["path"], CR.FPS, W, H, start=f0 / CR.FPS,
                                      dur=(f1 - f0 + 1) / CR.FPS + 0.5):
                f = f0 + kk
                if f > f1:
                    break
                if f not in by_frame:
                    continue
                vis = CR.visible(regs, f / CR.FPS)
                for r in by_frame[f].itertuples():
                    win = CR.crop_window(r, W, H)
                    c = img[win[1]:win[3], win[0]:win[2]].copy()
                    if vis:
                        CR.apply_mask(c, win, vis)
                    s = min(1.0, CR.MAX_SIDE / max(c.shape[:2]))
                    if s < 1:
                        c = cv2.resize(c, (max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s))),
                                       interpolation=cv2.INTER_AREA)
                    name = f"t{int(r.track_id)}_f{f}.jpg"
                    cv2.imwrite(str(d / "vcrops" / name), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    rows.append(dict(track_id=int(r.track_id), frame_idx=f, h_px=float(r.h),
                                     path=f"work/{vid}/vcrops/{name}", reused=False))
    V = pd.DataFrame(rows)
    V.to_parquet(d / "vcrops.parquet", index=False)
    info = dict(pool=len(P), ego_residue=int(car.ego_residue.sum()), ego_boxes=nbox, crops=len(V),
                reused=int(V.reused.sum()) if len(V) else 0)
    (d / "vcrops.json").write_text(json.dumps(info))
    return json.dumps(info)


# ---------------------------------------------------------------- VLM yes/no

_IDS = None


def yes_no_scores(paths, prompt=PROMPT, max_side=640):
    import torch
    import idlib
    r = idlib.reader()
    global _IDS
    if _IDS is None:
        tok = r.proc.tokenizer
        ys = {tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes", "YES")}
        ns = {tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No", "NO")}
        _IDS = (sorted(ys), sorted(ns))
    ims, inp = r._inputs(paths, prompt, max_side)
    with torch.inference_mode():
        out = r.model.generate(**inp, max_new_tokens=1, do_sample=False, output_scores=True,
                               return_dict_in_generate=True)
    lp = torch.log_softmax(out.scores[0].float(), dim=-1)
    py = lp[:, _IDS[0]].exp().sum(-1)
    pn = lp[:, _IDS[1]].exp().sum(-1)
    sc = (py / (py + pn + 1e-9)).cpu().numpy()
    mass = (py + pn).cpu().numpy()
    return [float(x) for x in sc], [float(x) for x in mass]


def run(rows, batch=6, second=False):
    """First pass: the largest crop of every pool track. second=True: the other crop of tracks whose first score
    is unsure (NO_T < score < YES_T). verify.done is written after the first pass; the second pass updates it."""
    import idlib
    for row in rows:
        vid = row["id"]
        d = wdir(vid)
        if not (d / "vcrops.parquet").exists() or ((d / "verify.done").exists() and not second):
            continue
        try:
            V = pd.read_parquet(d / "vcrops.parquet").sort_values("h_px", ascending=False)
            parts = d / "verify_parts"
            parts.mkdir(exist_ok=True)
            done = pd.concat([pd.read_parquet(f) for f in sorted(parts.glob("*.parquet"))]) if any(
                parts.glob("*.parquet")) else pd.DataFrame(columns=["path", "track_id", "score"])
            first = V.groupby("track_id").head(1)
            if second:
                f1 = done[done.path.isin(first.path)]
                unsure = f1[(f1.score > NO_T) & (f1.score < YES_T)].track_id
                V = V[V.track_id.isin(unsure) & ~V.path.isin(first.path)]
            else:
                V = first
            todo = V[~V.path.isin(done.path)].to_dict("records")
            print(f"{vid}: {len(todo)} crops to verify", flush=True)
            t0, buf, nb = time.time(), [], 0
            for bj in idlib.batched(todo, batch):
                sc, mass = yes_no_scores([str(ROOT / r["path"]) for r in bj])
                for r, s, m in zip(bj, sc, mass):
                    buf.append(dict(track_id=r["track_id"], path=r["path"], h_px=r["h_px"], score=s, mass=m))
                nb += 1
                if nb % 100 == 0 or idlib.paused():
                    pd.DataFrame(buf).to_parquet(parts / f"part_{len(list(parts.glob('*.parquet'))):05d}.parquet")
                    buf = []
                    print(f"{vid}: {nb * batch}/{len(todo)} {(time.time() - t0) / (nb * batch):.3f} s/crop", flush=True)
                    if idlib.paused():
                        print("PAUSE: stopping cleanly")
                        return 3
            if buf:
                pd.DataFrame(buf).to_parquet(parts / f"part_{len(list(parts.glob('*.parquet'))):05d}.parquet")
            el = time.time() - t0
            R = pd.concat([pd.read_parquet(f) for f in sorted(parts.glob("*.parquet"))]).drop_duplicates("path")
            R.to_parquet(d / "verify.parquet", index=False)
            (d / "verify.done").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {len(R)} crops {el:.0f}s\n")
            gpu = d / "gpu_time.json"
            g = json.loads(gpu.read_text()) if gpu.exists() else {}
            g["verify_s"] = g.get("verify_s", 0) + el
            g["verify_n"] = g.get("verify_n", 0) + len(todo)
            gpu.write_text(json.dumps(g))
            print(f"{vid}: verified {len(todo)} crops in {el:.0f}s", flush=True)
        except Exception:  # noqa: BLE001
            import traceback
            log_error(vid, "verify", traceback.format_exc())
            print(f"{vid}: verify FAILED", flush=True)
    return 0


def verdict(scores):
    """per-track verdict from its crop scores (mean)."""
    if not len(scores):
        return None, None
    m = float(np.mean(scores))
    return ("yes" if m >= YES_T else "no" if m <= NO_T else "unsure"), round(m, 3)


# ---------------------------------------------------------------- validation set

VAL = WORK / "verify_val"


def valset(n=160, seed=0):
    """Sample n pool tracks over all videos (stratified: long_static, small boxes, height bins, videos), take one
    existing crop (vcrops) each, write sheets of 40 numbered crops for labelling by eye + a CSV to fill in."""
    import cv2
    VAL.mkdir(exist_ok=True)
    allr = []
    for row in select_videos():
        d = wdir(row["id"])
        if not (d / "vcrops.parquet").exists():
            continue
        V = pd.read_parquet(d / "vcrops.parquet").sort_values("h_px", ascending=False).groupby("track_id").head(1)
        tr = pd.read_parquet(d / "tracks.parquet").set_index("track_id")
        V["video"] = row["id"]
        V["long_static"] = V.track_id.map(tr.long_static).astype(bool)
        V["conf_max"] = V.track_id.map(tr.conf_max)
        allr.append(V)
    A = pd.concat(allr, ignore_index=True)
    A["hb"] = pd.cut(A.h_px, [0, 45, 60, 100, 170, 2000])
    A["stratum"] = A.hb.astype(str) + "|" + np.where(A.long_static, "static", "moving")
    per = max(1, n // A.stratum.nunique())
    S = A.groupby("stratum", group_keys=False).apply(lambda g: g.sample(min(len(g), per), random_state=seed))
    S = S.sample(frac=1, random_state=seed).reset_index(drop=True)
    S["i"] = range(len(S))
    S.to_csv(VAL / "valset.csv", index=False)
    for k in range(0, len(S), 40):
        tiles = []
        for r in S.iloc[k:k + 40].itertuples():
            im = cv2.imread(str(ROOT / r.path))
            s = 200 / max(im.shape[:2])
            im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))),
                            interpolation=cv2.INTER_CUBIC)
            t = np.zeros((220, 210, 3), np.uint8)
            t[20:20 + im.shape[0], 5:5 + im.shape[1]] = im
            cv2.putText(t, f"{r.i} h{int(r.h_px)}", (3, 15), 0, 0.5, (0, 255, 255), 1)
            tiles.append(t)
        while len(tiles) % 8:
            tiles.append(np.zeros((220, 210, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[j:j + 8]) for j in range(0, len(tiles), 8)])
        cv2.imwrite(str(VAL / f"sheet_{k // 40}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(S)} crops, height bins {S.hb.astype(str).value_counts().to_dict()} -> {VAL}")


def valeval(labels_csv, prompt_i=None):
    """labels_csv: i,label (1 = real baja car, 0 = not, 2 = can't tell). Runs the prompt (GPU) on the val crops."""
    S = pd.read_csv(VAL / "valset.csv")
    L = pd.read_csv(labels_csv)
    S = S.merge(L, on="i")
    sc = []
    for k in range(0, len(S), 6):
        sc += yes_no_scores([str(ROOT / p) for p in S.path.iloc[k:k + 6]])[0]
    S["score"] = sc
    S.to_csv(VAL / "valset_scored.csv", index=False)
    J = S[S.label.isin([0, 1])]
    for thr in (0.3, 0.5, 0.7, 0.9):
        pred = J.score >= thr
        tp = int((pred & (J.label == 1)).sum())
        fp = int((pred & (J.label == 0)).sum())
        fn = int((~pred & (J.label == 1)).sum())
        tn = int((~pred & (J.label == 0)).sum())
        print(f"thr {thr}: precision {tp / max(tp + fp, 1):.3f} recall {tp / max(tp + fn, 1):.3f} "
              f"(tp {tp} fp {fp} fn {fn} tn {tn}); not-car rejected {tn}/{tn + fp}")
    J = J.assign(hb=pd.cut(J.h_px, [0, 45, 60, 100, 170, 2000]))
    for hb, g in J.groupby("hb", observed=True):
        pred = g.score >= 0.5
        print(f"  h {hb}: n {len(g)} cars {int((g.label == 1).sum())} acc {(pred == (g.label == 1)).mean():.2f} "
              f"recall {(pred & (g.label == 1)).sum()}/{(g.label == 1).sum()} "
              f"FP {(pred & (g.label == 0)).sum()}/{(g.label == 0).sum()}")
    print(J.sort_values("score")[["i", "video", "h_px", "label", "score"]].to_string())


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    mode, argv = argv[0], list(argv[1:])
    if mode == "valset":
        valset(int(argv[argv.index("--n") + 1]) if "--n" in argv else 160)
        return 0
    if mode == "valeval":
        valeval(argv[0])
        return 0
    if mode == "crops":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # "" does not hide the GPU on Windows
        workers, force, ids, rest = cli_ids(argv)
        run_stage("vcrops", crops_stage, select_videos(ids or None), workers=workers, force=force,
                  requires=("track",))
        return 0
    if mode == "run":
        batch = 6
        if "--batch" in argv:
            i = argv.index("--batch")
            batch = int(argv[i + 1])
            del argv[i:i + 2]
        return run(select_videos([a for a in argv if not a.startswith("--")] or None), batch,
                   second="--second" in argv)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
