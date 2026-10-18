"""Panel boxes for students/recog/harness_26se.py (--boxes), with the val-picked conf and extra padding.

usage:
  CUDA_VISIBLE_DEVICES=-1 python students/panel_boxes.py --weights students/runs/num_n320/weights/best.pt \
      --imgsz 320 --conf 0.1 --pad 0.05 [--all-crops --max-crops 30] --out out/panel_boxes_num_n320.csv

Crop list = exactly the harness's (out/passes_all.csv passes, the 10 VLM crops per pass, or --all-crops). Boxes are in
car-crop pixels, enlarged by --pad per side (the recognizer adds its own +10% in augment.plain).
"""
import argparse
import sys
from pathlib import Path

import torch  # noqa: F401  (before ultralytics)

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "students" / "recog"))
from harness_26se import crop_paths  # noqa: E402
sys.path.insert(0, str(ROOT / "students"))
from panel_val import enlarge  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--pad", type=float, default=0.05)
    ap.add_argument("--all-crops", action="store_true")
    ap.add_argument("--max-crops", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(a.weights)
    P = pd.read_csv(ROOT / "out" / "passes_all.csv")
    paths = sorted({p for v in crop_paths(P, a.all_crops, a.max_crops).values() for p in v})
    rows = []
    for i in range(0, len(paths), 32):
        chunk = paths[i:i + 32]
        ims = [cv2.imread(p) for p in chunk]
        res = m.predict([im for im in ims if im is not None], conf=a.conf, imgsz=a.imgsz, device=a.device, verbose=False)
        res = iter(res)
        for p, im in zip(chunk, ims):
            if im is None:
                continue
            r = next(res)
            H, W = im.shape[:2]
            for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                x1, y1, x2, y2 = enlarge(b, a.pad, W, H)
                rows.append(dict(path=p, x1=x1, y1=y1, x2=x2, y2=y2, conf=c, h_raw=b[3] - b[1]))
    D = pd.DataFrame(rows, columns=["path", "x1", "y1", "x2", "y2", "conf", "h_raw"])
    D.to_csv(a.out, index=False)
    print(len(paths), "crops,", len(D), "boxes,", D.path.nunique(), "crops with a box ->", a.out)


if __name__ == "__main__":
    main()
