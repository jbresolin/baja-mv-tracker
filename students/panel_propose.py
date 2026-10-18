"""Second-round number-box proposals: run the panel detector on the crops of teacher-labelled tracks that have no
grounded number box, read every box with the WP10 recognizer, and keep a box only if its read equals the track's
consensus number (roster posterior argmax == label and P >= --min-p). Report only (no retraining here).

usage:
  CUDA_VISIBLE_DEVICES=-1 python students/panel_propose.py --weights students/runs/num_n320/weights/best.pt \
      [--videos 19SU_END,25BR_DIN,...] --out out/panel_proposals.csv
Default videos = the train events of datasets/numbers_crop (not 25SU val / 24SU test).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "students" / "recog"))
from augment import plain  # noqa: E402
from common import RUNS as RECOG_RUNS, RosterScorer, batch_tensor, load_any, read_roster  # noqa: E402
sys.path.insert(0, str(ROOT / "students"))
from panel_val import enlarge  # noqa: E402

TRAIN_VIDEOS = ["19SU_END", "25BR_DIN", "25BR_END", "26BR_DIN", "26BR_END"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--videos", default=",".join(TRAIN_VIDEOS))
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--pad", type=float, default=0.05)
    ap.add_argument("--min-p", type=float, default=0.5)
    ap.add_argument("--min-h", type=float, default=16)
    ap.add_argument("--recog", default="syn_ft")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from ultralytics import YOLO
    det = YOLO(a.weights)
    rec = load_any(RECOG_RUNS / a.recog, a.threads)
    rows, stats, t0 = [], {}, time.time()
    for vid in a.videos.split(","):
        L = pd.read_parquet(ROOT / "work" / vid / "labels.parquet")
        L = L[L.label.notna()]
        R = pd.read_parquet(ROOT / "work" / vid / "reads.parquet")
        ev = str(L.event.iloc[0])
        roster = read_roster(ev)
        sc = RosterScorer(roster)
        st = dict(tracks=len(L), crops=0, proposed=0, too_small=0, accepted=0, crops_gaining=0, tracks_gaining=0)
        for t in L.itertuples():
            grounded = set(json.loads(t.number_boxes)) if isinstance(t.number_boxes, str) else set()
            todo = [p for p in R[R.track_id == t.track_id].path if p not in grounded]
            gained = False
            for p in todo:
                im = cv2.imread(str(ROOT / p))
                if im is None:
                    continue
                st["crops"] += 1
                r = det.predict(im, conf=a.conf, imgsz=a.imgsz, device="cpu", verbose=False)[0]
                bx = r.boxes.xyxy.tolist()
                st["proposed"] += len(bx)
                keep = [b for b in bx if b[3] - b[1] >= a.min_h]
                st["too_small"] += len(bx) - len(keep)
                if not keep:
                    continue
                H, W = im.shape[:2]
                with torch.no_grad():
                    P, _ = sc(rec(batch_tensor([plain(im, enlarge(b, a.pad, W, H)) for b in keep])))
                P = P.numpy()
                ok = 0
                for b, pr in zip(keep, P):
                    j = int(np.argmax(pr))
                    if roster[j] == int(t.label) and pr[j] >= a.min_p:
                        rows.append(dict(video=vid, track_id=int(t.track_id), path=p, label=int(t.label),
                                         x1=round(b[0], 1), y1=round(b[1], 1), x2=round(b[2], 1), y2=round(b[3], 1),
                                         p=round(float(pr[j]), 3)))
                        ok += 1
                st["accepted"] += ok
                st["crops_gaining"] += ok > 0
                gained |= ok > 0
            st["tracks_gaining"] += gained
        stats[vid] = st
        print(vid, st, "%.0f s" % (time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    tot = {k: sum(s[k] for s in stats.values()) for k in next(iter(stats.values()))}
    print("total", tot)
    Path(a.out).with_suffix(".json").write_text(json.dumps(dict(per_video=stats, total=tot, args=vars(a)), indent=1))


if __name__ == "__main__":
    main()
