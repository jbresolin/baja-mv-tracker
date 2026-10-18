"""WP10: 26SE identity harness for the number recognizer (evaluation only: never train or tune on 26SE).

Passes = out/passes_all.csv (session-1 YOLO-World passes after 09:18:30, gt_car = time-matched human lap's car).
Crops  = the 10 crops per pass the VLM read (out/world_*/reads.csv, paths relative to lapvision/), or with
--all-crops every saved crop of the pass's tracks (out/world_*/<event>/t<track>_f*.jpg, subsampled to --max-crops).

Modes (the final chain is: number-panel detector -> recognizer -> vote -> roster):
  --mode window                       interim, no panel detector: windows over the car crop (window.py)
  --mode panels --boxes boxes.csv     panel boxes from any source: csv path,x1,y1,x2,y2[,conf] (car-crop px; path as
                                      in reads.csv or absolute). Crop posterior = max over its boxes.
  --mode panels --panel-weights W.pt  run a YOLO number-panel net (WP9, datasets/numbers_crop) on each crop first,
                                      writes the boxes csv, then as above. (--panel-conf 0.25, --panel-imgsz 640)

Vote (WP10b): --vote mean (WP10: mean over all crops, crops without a box add zeros, so the score depends on the
crop count) or --vote boxed (mean over crops with a box + a minimum number of supporting crops, common.vote_boxed;
tau / --min-support default from students/runs/recog/<name>/eval/panel_tau.json, picked on 25SU by panel_tau.py).
The abstain threshold comes from the val split (--tau, default: read from students/runs/recog/<name>/eval/
window_val.json or metrics.json for the target precision --target), never from 26SE. The full curve is printed for
information only.
usage: CUDA_VISIBLE_DEVICES=-1 python students/recog/harness_26se.py --name real --mode window
Output: out/eval_recog_<name>_<mode>/{passes.csv, metrics.json}
"""
import argparse
import ast
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment import plain  # noqa: E402
from common import ROOT, RUNS, RosterScorer, batch_tensor, load_any, read_roster, vote, vote_boxed  # noqa: E402
from evaluate import TAUS, wilson  # noqa: E402
from window import read_car_crop  # noqa: E402

LAPV = ROOT / "lapvision"


def crop_paths(P, all_crops, max_crops):
    """dict (src, pass_id) -> list of absolute crop paths"""
    out = {}
    for src in P.src.unique():
        R = pd.read_csv(ROOT / "out" / src / "reads.csv")
        for pid, g in R.groupby("pass_id"):
            out[(src, int(pid))] = [str((LAPV / p.replace("\\", "/")).resolve()) for p in g.path]
    if all_crops:
        for r in P.itertuples():
            paths = []
            for tk in ast.literal_eval(r.tracks):
                ev, tid = tk.rsplit(":", 1)
                paths += sorted(str(p) for p in (ROOT / "out" / r.src / ev).glob(f"t{int(tid):03d}_f*.jpg"))
            if paths:
                idx = np.linspace(0, len(paths) - 1, min(max_crops, len(paths))).round().astype(int)
                out[(r.src, int(r.pass_id))] = [paths[i] for i in sorted(set(idx))]
    return out


def panel_boxes_from_weights(weights, paths, conf, imgsz, out_csv):
    import torch as _t  # noqa: F401  (torch before ultralytics: it sets OMP_NUM_THREADS=1 otherwise, LESSONS WP9)
    from ultralytics import YOLO
    m = YOLO(weights)
    rows = []
    for p in paths:
        r = m.predict(p, conf=conf, imgsz=imgsz, device="cpu", verbose=False)[0]
        names = r.names
        for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            if names[int(k)] == "number" or len(names) == 1:
                rows.append(dict(path=p, x1=b[0], y1=b[1], x2=b[2], y2=b[3], conf=c))
    pd.DataFrame(rows, columns=["path", "x1", "y1", "x2", "y2", "conf"]).to_csv(out_csv, index=False)
    return out_csv


@torch.no_grad()
def read_panels(model, im, boxes, sc):
    if not boxes:
        return np.zeros(len(sc.numbers))
    X = batch_tensor([plain(im, b) for b in boxes])
    P, _ = sc(model(X))
    return P.numpy().max(0)


def default_tau(name, mode, target):
    d = RUNS / name / "eval"
    f = d / ("window_val.json" if mode == "window" else "metrics.json")
    if not f.exists():
        return None
    m = json.load(open(f))
    curve = m["curve"] if mode == "window" else m["val"]["track"]["curve"]
    ok = [c["tau"] for c in curve if c["precision"] >= target]
    return min(ok) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["window", "panels"], default="window")
    ap.add_argument("--boxes", default=None)
    ap.add_argument("--panel-weights", default=None)
    ap.add_argument("--panel-conf", type=float, default=0.25)
    ap.add_argument("--panel-imgsz", type=int, default=640)
    ap.add_argument("--all-crops", action="store_true")
    ap.add_argument("--max-crops", type=int, default=30)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--target", type=float, default=0.97)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="first N passes only (smoke test)")
    ap.add_argument("--vote", choices=["mean", "boxed"], default="mean")
    ap.add_argument("--min-support", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = load_any(RUNS / a.name, a.threads)
    roster = read_roster("26SE")
    sc = RosterScorer(roster)
    P = pd.read_csv(ROOT / "out" / "passes_all.csv")
    if a.limit:
        P = P.head(a.limit)
    cp = crop_paths(P, a.all_crops, a.max_crops)
    tag = f"{a.mode}{'_all' if a.all_crops else ''}{'_boxed' if a.vote == 'boxed' else ''}{a.tag}"
    od = ROOT / "out" / f"eval_recog_{a.name}_{tag}"
    od.mkdir(parents=True, exist_ok=True)
    boxes = None
    if a.mode == "panels":
        bcsv = a.boxes
        if a.panel_weights:
            allp = [p for v in cp.values() for p in v]
            bcsv = panel_boxes_from_weights(a.panel_weights, allp, a.panel_conf, a.panel_imgsz, od / "panel_boxes.csv")
        B = pd.read_csv(bcsv)
        B["path"] = [str((LAPV / p.replace("\\", "/")).resolve()) if not Path(p).is_absolute() else str(Path(p).resolve())
                     for p in B.path]
        boxes = {p: g[["x1", "y1", "x2", "y2"]].values.tolist() for p, g in B.groupby("path")}
    rows, t0, nc = [], time.time(), 0
    min_support, tau_boxed, score_kind = 1, None, "share"
    if a.vote == "boxed":
        f = RUNS / a.name / "eval" / "panel_tau.json"
        if f.exists():
            pt = json.load(open(f))["chosen"]
            min_support, tau_boxed, score_kind = pt["min_support"], pt["tau"], pt.get("score", "mean")
        if a.min_support is not None:
            min_support = a.min_support
    for i, r in enumerate(P.itertuples()):
        Ps, HB = [], []
        for p in cp.get((r.src, int(r.pass_id)), []):
            im = cv2.imread(p)
            if im is None:
                continue
            Ps.append(read_car_crop(model, im, sc) if a.mode == "window" else read_panels(model, im, boxes.get(p, []), sc))
            HB.append(a.mode == "window" or len(boxes.get(p, [])) > 0)
            nc += 1
        nb, sup = sum(HB), 0
        if Ps and a.vote == "boxed":
            j, s, s2, nb, sup = vote_boxed(Ps, HB, score=score_kind)
            pred = roster[j] if j >= 0 else None
            if j >= 0 and sup < min_support:
                s = 0.0                      # abstains at any tau
        elif Ps:
            j, s, s2 = vote(Ps)
            pred = roster[j]
        else:
            pred, s, s2 = None, 0.0, 0.0
        rows.append(dict(src=r.src, pass_id=r.pass_id, time=r.time, gt_car=r.gt_car, vlm_car=r.vlm_car,
                         vlm_share=r.vlm_share, n_crops=len(Ps), n_boxed=nb, support=sup, pred=pred, score=s,
                         margin=s - s2))
        if i % 20 == 0:
            print(i, len(P), f"{(time.time() - t0) / max(nc, 1):.2f} s/crop", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(od / "passes.csv", index=False)
    G = D[D.gt_car.notna()]
    tau = a.tau if a.tau is not None else (tau_boxed if a.vote == "boxed" else default_tau(a.name, a.mode, a.target))
    res = dict(name=a.name, mode=tag, vote=a.vote, min_support=min_support if a.vote == "boxed" else None, passes=len(D), with_gt=len(G), crops=nc, s_per_crop=(time.time() - t0) / max(nc, 1),
               tau=tau, tau_source="--tau" if a.tau is not None else f"val, target precision {a.target}")
    vl = G[G.vlm_car.notna()]
    res["vlm_teacher"] = dict(coverage=len(vl) / max(len(G), 1), precision=float((vl.vlm_car == vl.gt_car).mean()))
    res["curve"] = []
    for t in TAUS + ([tau] if tau is not None and tau not in TAUS else []):
        ans = G[G.score >= t]
        ok = int((ans.pred == ans.gt_car).sum())
        res["curve"].append(dict(tau=t, coverage=len(ans) / max(len(G), 1), precision=ok / max(len(ans), 1), n=len(ans),
                                 ci=[round(x, 3) for x in wilson(ok, len(ans))]))
    # margin (top - runner-up vote) is the better abstain rule for the window mode (max over windows saturates)
    res["margin_curve"] = []
    for t in (0.1, 0.2, 0.3, 0.4):
        ans = G[G.margin >= t]
        ok = int((ans.pred == ans.gt_car).sum())
        res["margin_curve"].append(dict(margin=t, coverage=len(ans) / max(len(G), 1), precision=ok / max(len(ans), 1),
                                        n=len(ans), ci=[round(x, 3) for x in wilson(ok, len(ans))]))
    print("margin:", " ".join(f"m{c['margin']}:{c['coverage']:.2f}/{c['precision']:.3f}" for c in res["margin_curve"]))
    json.dump(res, open(od / "metrics.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}, default=float))
    print(" ".join(f"t{c['tau']}:{c['coverage']:.2f}/{c['precision']:.3f}" for c in res["curve"]))


if __name__ == "__main__":
    main()
