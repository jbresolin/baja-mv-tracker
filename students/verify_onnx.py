"""Check that an exported ONNX detector gives the same boxes as its .pt through ultralytics predict.

usage: CUDA_VISIBLE_DEVICES=-1 python students/verify_onnx.py --pt <best.pt> --onnx <x.onnx> --imgsz 320 \
          --images "out/world_1129/*/*.jpg" [--n 500]
Per image: same number of boxes (conf >= --conf), and the max |coordinate difference| after matching boxes greedily by
IoU. Prints the share of images with the same count and the coordinate-difference percentiles; exits 1 if the count
agreement is below --min-agree.
"""
import argparse
import glob
import json
import random
import sys

import torch  # noqa: F401  (before ultralytics)

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--imgsz", nargs="+", type=int, default=[320])
    ap.add_argument("--images", nargs="+", required=True, help="glob(s)")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--min-agree", type=float, default=0.99)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from ultralytics import YOLO
    files = sorted({f for g in a.images for f in glob.glob(g)})
    random.Random(0).shuffle(files)
    files = files[:a.n]
    imgsz = a.imgsz[0] if len(a.imgsz) == 1 else a.imgsz
    P, O = YOLO(a.pt), YOLO(a.onnx, task="detect")
    same, diffs, nb = 0, [], 0
    for f in files:
        bp = P.predict(f, imgsz=imgsz, conf=a.conf, device="cpu", verbose=False)[0].boxes.xyxy.numpy()
        bo = O.predict(f, imgsz=imgsz, conf=a.conf, device="cpu", verbose=False)[0].boxes.xyxy.numpy()
        nb += len(bp)
        same += len(bp) == len(bo)
        used = set()
        for b in bp:
            if not len(bo):
                break
            d = np.abs(bo - b).max(1)
            j = int(np.argmin(d))
            if j not in used:
                used.add(j)
                diffs.append(float(d[j]))
    d = np.array(diffs) if diffs else np.zeros(1)
    res = dict(pt=a.pt, onnx=a.onnx, imgsz=imgsz, images=len(files), boxes_pt=nb, same_count=round(same / max(len(files), 1), 4),
               max_px_p50=round(float(np.percentile(d, 50)), 4), max_px_p99=round(float(np.percentile(d, 99)), 4),
               max_px_max=round(float(d.max()), 4))
    print(json.dumps(res))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))
    sys.exit(0 if res["same_count"] >= a.min_agree else 1)


if __name__ == "__main__":
    main()
