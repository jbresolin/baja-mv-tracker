"""WP10: track-level evaluation of the interim window mode (no panel detector) on whole WP4 car crops.

usage: CUDA_VISIBLE_DEVICES=-1 python students/recog/eval_window.py --name real --split val --tracks 250 [--crops 4]

Tracks = teacher-labelled tracks (labels.parquet label not null, sources vlm / both-agree / overlay) of the split's
videos (val 25SU, test 24SU). Per track up to --crops WP4 crops (the crops the VLM read, largest first), each read by
window.read_car_crop(), soft-voted. Output: students/runs/recog/<name>/eval/window_<split>.json (+ per-track csv).
Caveat: labelled tracks are the ones the teacher could read, so coverage here is on teacher-readable tracks.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RUNS, RosterScorer, load_any, read_roster, vote  # noqa: E402
from evaluate import TAUS, wilson  # noqa: E402
from window import read_car_crop  # noqa: E402

VIDEOS = {"val": ["25SU_DIN", "25SU_END"], "test": ["24SU_END", "24SU_SPP"]}
EVENT = {"val": "25SU", "test": "24SU"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--tracks", type=int, default=250)
    ap.add_argument("--crops", type=int, default=4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = load_any(RUNS / a.name, a.threads)
    roster = read_roster(EVENT[a.split])
    sc = RosterScorer(roster)
    rows = []
    for vid in VIDEOS[a.split]:
        L = pd.read_parquet(ROOT / "work" / vid / "labels.parquet")
        L = L[L.label.notna() & L.source.isin(["vlm", "both-agree", "overlay"])]
        R = pd.read_parquet(ROOT / "work" / vid / "reads.parquet")
        for t in L.itertuples():
            g = R[R.track_id == t.track_id].sort_values("h_px", ascending=False)
            if len(g):
                rows.append((vid, int(t.track_id), int(t.label), t.source, g.path.tolist()[:a.crops], float(g.h_px.max())))
    random.Random(a.seed).shuffle(rows)
    rows = rows[:a.tracks]
    out, t0, n_crops = [], time.time(), 0
    for i, (vid, tid, lab, src, paths, hmax) in enumerate(rows):
        Ps = []
        for p in paths:
            im = cv2.imread(str(ROOT / p))
            if im is not None:
                Ps.append(read_car_crop(model, im, sc))
                n_crops += 1
        if not Ps:
            continue
        j, s, s2 = vote(Ps)
        crop_ok = [roster[int(np.argmax(P))] == lab for P in Ps]
        out.append(dict(video=vid, track=tid, gt=lab, source=src, h_max=hmax, n=len(Ps), pred=roster[j], score=s,
                        margin=s - s2, crop_acc=float(np.mean(crop_ok)), crop_pmax=[float(P.max()) for P in Ps]))
        if i % 25 == 0:
            print(i, len(rows), f"{(time.time() - t0) / max(n_crops, 1):.2f} s/crop", flush=True)
    D = pd.DataFrame(out)
    res = dict(split=a.split, tracks=len(D), crops=n_crops, s_per_crop=(time.time() - t0) / max(n_crops, 1),
               crop_acc_top1=float(D.crop_acc.mean()), curve=[])
    for t in TAUS:
        ans = D[D.score >= t]
        ok = int((ans["pred"] == ans["gt"]).sum())
        res["curve"].append(dict(tau=t, coverage=len(ans) / max(len(D), 1), precision=ok / max(len(ans), 1), n=len(ans),
                                 ci=[round(x, 3) for x in wilson(ok, len(ans))]))
    res["margin_curve"] = []
    for t in (0.1, 0.2, 0.3, 0.4):
        ans = D[D.margin >= t]
        ok = int((ans["pred"] == ans["gt"]).sum())
        res["margin_curve"].append(dict(margin=t, coverage=len(ans) / max(len(D), 1), precision=ok / max(len(ans), 1),
                                        n=len(ans), ci=[round(x, 3) for x in wilson(ok, len(ans))]))
    print("margin:", " ".join(f"m{c['margin']}:{c['coverage']:.2f}/{c['precision']:.3f}" for c in res["margin_curve"]))
    d = RUNS / a.name / "eval"
    d.mkdir(exist_ok=True)
    D.to_csv(d / f"window_{a.split}_tracks.csv", index=False)
    json.dump(res, open(d / f"window_{a.split}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}))
    print(" ".join(f"t{c['tau']}:{c['coverage']:.2f}/{c['precision']:.3f}" for c in res["curve"]))


if __name__ == "__main__":
    main()
