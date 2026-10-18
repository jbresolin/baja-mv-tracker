"""WP10b: pick the panel-chain abstain rule on val (25SU) with a crop-count-independent vote, and check it on test (24SU).

Chain as in students/panel_val.py / the 26SE harness: WP4 car crops of teacher-labelled tracks (all of a track's read
crops, up to 8) -> number-panel detector (num_n320, conf 0.1, imgsz 320, boxes enlarged by 5% per side: WP9's val
choice) -> recognizer on each box (+10%) -> crop posterior = max over its boxes.
Track rules compared:
  mean  : common.vote (WP10/WP9) - mean over ALL crops, box-less crops add zeros (score falls with the crop count)
  boxed : common.vote_boxed(score="mean") - mean over crops WITH a box, answer only if >= k boxed crops support the
          winner (P >= 0.5)
  share : common.vote_boxed(score="share") - posterior-weighted vote share over boxed crops, same support rule
Crop-count simulation: every track is scored with random subsets of n crops (n = 2, 4, 6 and all; 5 draws per track),
so the curves show whether a tau picked at one crop count holds at another (deployment reads ~10-30 crops per pass;
val tracks have at most 8). The chosen rule (boxed or share; tau, k) = max coverage at precision >= --target on val with all
crops, required to keep >= target - 1 point at n = 4; written to <recog run>/eval/panel_tau.json, which
harness_26se.py --vote boxed reads. Never uses 26SE.

usage: CUDA_VISIBLE_DEVICES=-1 python students/recog/panel_tau.py --recog gx_base_full [--ckpt last.pt] [--target 0.97]
Detector boxes are cached in students/runs/recog/panel_boxes_<split>.csv, crop posteriors in
<run>/eval/panel_crops_<split>_<ckpt>.pkl.
"""
import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import torch  # before ultralytics (OMP_NUM_THREADS, LESSONS WP9)

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RUNS, RosterScorer, load_model, read_roster, vote, vote_boxed  # noqa: E402
from evaluate import wilson  # noqa: E402
from harness_26se import read_panels  # noqa: E402

VIDEOS = {"val": ["25SU_DIN", "25SU_END"], "test": ["24SU_END", "24SU_SPP"]}
EVENT = {"val": "25SU", "test": "24SU"}
DET = ROOT / "students" / "runs" / "num_n320" / "weights" / "best.pt"
TAUS = [round(x, 2) for x in np.arange(0.3, 0.991, 0.05)] + [0.97, 0.99]
KS = [1, 2, 3]
NS = [2, 4, 6, 0]          # 0 = all crops


def tracks(split):
    rows = []
    for vid in VIDEOS[split]:
        L = pd.read_parquet(ROOT / "work" / vid / "labels.parquet")
        L = L[L.label.notna() & L.source.isin(["vlm", "both-agree", "overlay"])]
        R = pd.read_parquet(ROOT / "work" / vid / "reads.parquet")
        for t in L.itertuples():
            g = R[R.track_id == t.track_id].sort_values("h_px", ascending=False)
            if len(g):
                rows.append((vid, int(t.track_id), int(t.label), g.path.tolist()))
    return rows


def boxes(split, rows, conf, pad, threads):
    f = RUNS / f"panel_boxes_{split}.csv"
    if f.exists():
        B = pd.read_csv(f)
    else:
        from ultralytics import YOLO
        torch.set_num_threads(threads)
        det = YOLO(str(DET))
        out = []
        for k, (_, _, _, paths) in enumerate(rows):
            for p in paths:
                im = cv2.imread(str(ROOT / p))
                if im is None:
                    continue
                H, W = im.shape[:2]
                r = det.predict(im, conf=conf, imgsz=320, device="cpu", verbose=False)[0]
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                    bw, bh = x2 - x1, y2 - y1
                    out.append(dict(path=p, x1=max(0, x1 - pad * bw), y1=max(0, y1 - pad * bh),
                                    x2=min(W, x2 + pad * bw), y2=min(H, y2 + pad * bh), conf=c))
            if k % 100 == 0:
                print("boxes", split, k, len(rows), flush=True)
        B = pd.DataFrame(out, columns=["path", "x1", "y1", "x2", "y2", "conf"])
        B.to_csv(f, index=False)
    return {p: g[["x1", "y1", "x2", "y2"]].values.tolist() for p, g in B.groupby("path")}


def crop_posteriors(run, ckpt, split, rows, bx, threads):
    f = RUNS / run / "eval" / f"panel_crops_{split}_{Path(ckpt).stem}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    torch.set_num_threads(threads)
    model, _ = load_model(RUNS / run / ckpt)
    sc = RosterScorer(read_roster(EVENT[split]))
    out = []
    for vid, tid, lab, paths in rows:
        Ps, HB = [], []
        for p in paths:
            im = cv2.imread(str(ROOT / p))
            if im is None:
                continue
            b = bx.get(p, [])
            Ps.append(read_panels(model, im, b, sc))
            HB.append(len(b) > 0)
        if Ps:
            out.append(dict(video=vid, track=tid, gt=lab, P=np.array(Ps, np.float32), hb=np.array(HB)))
    f.parent.mkdir(exist_ok=True)
    pickle.dump(out, open(f, "wb"))
    return out


def simulate(T, roster, n, draws=5, seed=0):
    """-> list of dicts per (track, draw): gt, mean-vote (pred, score), boxed-vote (pred, score, support)"""
    rng = random.Random(seed)
    res = []
    for t in T:
        m = len(t["P"])
        if n and m < n:
            continue
        for d in range(draws if n and n < m else 1):
            idx = sorted(rng.sample(range(m), n)) if n and n < m else list(range(m))
            P, hb = t["P"][idx], t["hb"][idx]
            j, s, _ = vote(P)
            jb, sb, _, nb, sup = vote_boxed(P, hb, score="mean")
            _, ss, _, _, _ = vote_boxed(P, hb, score="share")
            res.append(dict(gt=t["gt"], pm=roster[j], sm=s, pb=roster[jb] if jb >= 0 else None, sb=sb, ss=ss, sup=sup,
                            w=1.0 / (draws if n and n < m else 1)))
    return res


def curve(S, rule, k=1):
    out = []
    W = sum(x["w"] for x in S)
    for t in TAUS:
        if rule == "mean":
            ans = [x for x in S if x["sm"] >= t]
            ok = sum(x["w"] for x in ans if x["pm"] == x["gt"])
        else:
            key = "sb" if rule == "boxed" else "ss"
            ans = [x for x in S if x[key] >= t and x["sup"] >= k]
            ok = sum(x["w"] for x in ans if x["pb"] == x["gt"])
        wa = sum(x["w"] for x in ans)
        out.append(dict(tau=t, coverage=round(wa / max(W, 1e-9), 4), precision=round(ok / max(wa, 1e-9), 4),
                        n=round(wa, 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recog", required=True)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--target", type=float, default=0.97)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--pad", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    res = dict(recog=a.recog, ckpt=a.ckpt, detector=str(DET.relative_to(ROOT)), conf=a.conf, pad=a.pad, splits={})
    for split in ("val", "test"):
        rows = tracks(split)
        bx = boxes(split, rows, a.conf, a.pad, a.threads)
        T = crop_posteriors(a.recog, a.ckpt, split, rows, bx, a.threads)
        roster = read_roster(EVENT[split])
        sp = dict(tracks=len(T), crops=int(sum(len(t["P"]) for t in T)),
                  crops_without_box=int(sum((~t["hb"]).sum() for t in T)), by_n={})
        for n in NS:
            S = simulate(T, roster, n)
            top1_b = sum(x["w"] for x in S if x["pb"] == x["gt"]) / max(sum(x["w"] for x in S), 1e-9)
            sp["by_n"][str(n or "all")] = dict(
                tracks=round(sum(x["w"] for x in S)),
                top1_mean=round(sum(x["w"] for x in S if x["pm"] == x["gt"]) / max(sum(x["w"] for x in S), 1e-9), 4),
                top1_boxed=round(top1_b, 4), mean=curve(S, "mean"), **{f"boxed_k{k}": curve(S, "boxed", k) for k in KS},
                **{f"share_k{k}": curve(S, "share", k) for k in KS})
        res["splits"][split] = sp
    # choose (tau, k) on val, all crops, max coverage with precision >= target, and >= target - 0.01 at n = 4
    v = res["splits"]["val"]["by_n"]
    best = None
    for rule in ("boxed", "share"):
        for k in KS:
            for ca, c4 in zip(v["all"][f"{rule}_k{k}"], v["4"][f"{rule}_k{k}"]):
                if ca["precision"] >= a.target and c4["precision"] >= a.target - 0.01 and ca["n"] >= 20:
                    if best is None or ca["coverage"] > best["val_coverage"]:
                        best = dict(score="mean" if rule == "boxed" else "share", rule=rule, tau=ca["tau"], min_support=k,
                                    val_coverage=ca["coverage"], val_precision=ca["precision"])
    if best is None:
        print("no (tau, k) reaches the target on val; lower --target")
    else:
        t = res["splits"]["test"]["by_n"]
        for n in ("2", "4", "6", "all"):
            c = [x for x in t[n][f"{best['rule']}_k{best['min_support']}"] if x["tau"] == best["tau"]][0]
            best[f"test_n{n}"] = dict(coverage=c["coverage"], precision=c["precision"])
            cv = [x for x in v[n][f"{best['rule']}_k{best['min_support']}"] if x["tau"] == best["tau"]][0]
            best[f"val_n{n}"] = dict(coverage=cv["coverage"], precision=cv["precision"])
    res["chosen"] = best
    res["target"] = a.target
    out = RUNS / a.recog / "eval" / "panel_tau.json"
    json.dump(res, open(out, "w"), indent=1, default=float)
    for split in ("val", "test"):
        s = res["splits"][split]
        print(f"== {split}: {s['tracks']} tracks, {s['crops']} crops, {s['crops_without_box']} without a box")
        for n, d in s["by_n"].items():
            def at(c, t):
                x = [y for y in c if y["tau"] == t][0]
                return f"{x['coverage']:.2f}/{x['precision']:.3f}"
            print(f"  n={n:>3}: top1 mean {d['top1_mean']:.3f} boxed {d['top1_boxed']:.3f} | mean t0.6 {at(d['mean'], 0.6)} "
                  f"t0.8 {at(d['mean'], 0.8)} | boxed k1 t0.6 {at(d['boxed_k1'], 0.6)} t0.8 {at(d['boxed_k1'], 0.8)} "
                  f"| boxed k2 t0.8 {at(d['boxed_k2'], 0.8)} | share k2 t0.8 {at(d['share_k2'], 0.8)} "
                  f"t0.9 {at(d['share_k2'], 0.9)} k3 t0.9 {at(d['share_k3'], 0.9)}")
    print("chosen:", json.dumps(best))


if __name__ == "__main__":
    main()
