"""WP9 x WP10: the number-panel detector + recognizer chain on teacher-labelled tracks of a held-out event, to pick the
abstain threshold tau (and the extra box padding) on val, never on 26SE.

usage:
  CUDA_VISIBLE_DEVICES=-1 python students/panel_val.py --weights students/runs/num_n320/weights/best.pt --split val
  ... --split test --tau 0.6 --pad 0.05      # report the val-picked settings on 24SU

Tracks = teacher-labelled tracks (labels.parquet label not null, sources vlm / both-agree / overlay) of the split's videos
(val 25SU, test 24SU), up to --crops WP4 car crops each (largest first). Per crop: panel detector (conf --conf, imgsz
--imgsz) -> every box enlarged by --pad on each side (on top of the recognizer's own +10% in augment.plain) -> recognizer
roster posterior, crop posterior = max over its boxes -> soft vote per track (students/recog/common.vote).
Writes <run>/panel_<split>.json (curves per pad, box stats) and panel_<split>_tracks.csv; with --sheet a contact sheet of
detections on random crops (<run>/panel_<split>_sheet.jpg) to check for boxes on broadcast graphics.
Caveat (as in WP10): labelled tracks are the ones the teacher could read.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch  # before ultralytics (OMP_NUM_THREADS, LESSONS WP9)

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "students" / "recog"))
from common import RUNS as RECOG_RUNS, RosterScorer, load_any, read_roster, vote  # noqa: E402
from evaluate import TAUS, wilson  # noqa: E402
from harness_26se import read_panels  # noqa: E402

VIDEOS = {"val": ["25SU_DIN", "25SU_END"], "test": ["24SU_END", "24SU_SPP"]}
EVENT = {"val": "25SU", "test": "24SU"}


def enlarge(b, pad, W, H):
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    return [max(0, x1 - pad * bw), max(0, y1 - pad * bh), min(W, x2 + pad * bw), min(H, y2 + pad * bh)]


def curve(D, target):
    out = []
    for t in TAUS:
        ans = D[D.score >= t]
        ok = int((ans["pred"] == ans["gt"]).sum())
        out.append(dict(tau=t, coverage=len(ans) / max(len(D), 1), precision=ok / max(len(ans), 1), n=len(ans),
                        ci=[round(x, 3) for x in wilson(ok, len(ans))]))
    ok = [c["tau"] for c in out if c["precision"] >= target and c["n"] >= 20]
    return out, (min(ok) if ok else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--recog", default="syn_ft")
    ap.add_argument("--tracks", type=int, default=400)
    ap.add_argument("--crops", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--pads", default="0,0.05,0.1", help="extra padding per side, comma list")
    ap.add_argument("--target", type=float, default=0.97)
    ap.add_argument("--tau", type=float, default=None, help="report this tau (picked on val)")
    ap.add_argument("--min-h", type=float, default=0, help="drop boxes shorter than this (px) before reading")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from ultralytics import YOLO
    det = YOLO(a.weights)
    model = load_any(RECOG_RUNS / a.recog, a.threads)
    roster = read_roster(EVENT[a.split])
    sc = RosterScorer(roster)
    pads = [float(x) for x in a.pads.split(",")]

    rows = []
    for vid in VIDEOS[a.split]:
        L = pd.read_parquet(ROOT / "work" / vid / "labels.parquet")
        L = L[L.label.notna() & L.source.isin(["vlm", "both-agree", "overlay"])]
        R = pd.read_parquet(ROOT / "work" / vid / "reads.parquet")
        for t in L.itertuples():
            g = R[R.track_id == t.track_id].sort_values("h_px", ascending=False)
            if len(g):
                rows.append((vid, int(t.track_id), int(t.label), g.path.tolist()[:a.crops]))
    random.Random(a.seed).shuffle(rows)
    rows = rows[:a.tracks]

    out, boxes_h, n_crops, n_empty, t0, t_det = [], [], 0, 0, time.time(), 0.0
    sheet = []
    for i, (vid, tid, lab, paths) in enumerate(rows):
        Ps = {p: [] for p in pads}
        for path in paths:
            im = cv2.imread(str(ROOT / path))
            if im is None:
                continue
            n_crops += 1
            td = time.time()
            r = det.predict(im, conf=a.conf, imgsz=a.imgsz, device="cpu", verbose=False)[0]
            t_det += time.time() - td
            bx = [b for b in r.boxes.xyxy.tolist() if b[3] - b[1] >= a.min_h]
            boxes_h += [b[3] - b[1] for b in r.boxes.xyxy.tolist()]
            n_empty += not bx
            H, W = im.shape[:2]
            for p in pads:
                Ps[p].append(read_panels(model, im, [enlarge(b, p, W, H) for b in bx], sc))
            if a.sheet and len(sheet) < 48 and random.Random(i).random() < 0.3:
                v = im.copy()
                for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                    cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                    cv2.putText(v, "%.2f" % c, (int(b[0]), max(12, int(b[1]) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)
                s = 240 / max(v.shape[:2])
                v = cv2.resize(v, (int(v.shape[1] * s), int(v.shape[0] * s)))
                tile = np.zeros((240, 240, 3), np.uint8)
                tile[:v.shape[0], :v.shape[1]] = v
                sheet.append(tile)
        if not Ps[pads[0]]:
            continue
        row = dict(video=vid, track=tid, gt=lab, n=len(Ps[pads[0]]))
        for p in pads:
            j, s, s2 = vote(Ps[p])
            row["pred_%g" % p], row["score_%g" % p] = roster[j], s
        out.append(row)
        if i % 50 == 0:
            print(i, len(rows), "%.3f s/crop (det %.3f)" % ((time.time() - t0) / max(n_crops, 1), t_det / max(n_crops, 1)),
                  flush=True)
    D = pd.DataFrame(out)
    h = np.array(boxes_h)
    res = dict(split=a.split, weights=a.weights, recog=a.recog, conf=a.conf, imgsz=a.imgsz, min_h=a.min_h,
               tracks=len(D), crops=n_crops, crops_without_box=n_empty,
               boxes_per_crop=round(len(h) / max(n_crops, 1), 3),
               box_h_px=dict(p10=float(np.percentile(h, 10)) if len(h) else None, median=float(np.median(h)) if len(h) else None,
                             share_lt16=float((h < 16).mean()) if len(h) else None),
               det_ms_per_crop=round(1000 * t_det / max(n_crops, 1), 1), per_pad={})
    for p in pads:
        Dp = D.assign(pred=D["pred_%g" % p], score=D["score_%g" % p])
        cv, tau = curve(Dp, a.target)
        res["per_pad"][str(p)] = dict(top1=float((Dp["pred"] == Dp["gt"]).mean()), tau_for_target=tau, curve=cv)
        if a.tau is not None:
            ans = Dp[Dp.score >= a.tau]
            ok = int((ans["pred"] == ans["gt"]).sum())
            res["per_pad"][str(p)]["at_tau"] = dict(tau=a.tau, coverage=len(ans) / max(len(Dp), 1),
                                                   precision=ok / max(len(ans), 1), n=len(ans),
                                                   ci=[round(x, 3) for x in wilson(ok, len(ans))])
        print("pad %g: top1 %.3f, tau for %.2f: %s | %s" % (p, res["per_pad"][str(p)]["top1"], a.target, tau,
              " ".join("t%g:%.2f/%.3f" % (c["tau"], c["coverage"], c["precision"]) for c in cv)), flush=True)
    rd = Path(a.weights).resolve().parent.parent
    D.to_csv(rd / ("panel_%s_tracks.csv" % a.split), index=False)
    (rd / ("panel_%s.json" % a.split)).write_text(json.dumps(res, indent=1))
    if sheet:
        while len(sheet) % 8:
            sheet.append(np.zeros((240, 240, 3), np.uint8))
        cv2.imwrite(str(rd / ("panel_%s_sheet.jpg" % a.split)),
                    np.vstack([np.hstack(sheet[k:k + 8]) for k in range(0, len(sheet), 8)]))
    print(json.dumps({k: v for k, v in res.items() if k != "per_pad"}))


if __name__ == "__main__":
    main()
