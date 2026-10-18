"""WP10 data prep: build an in-RAM cache of number crops with context, for on-the-fly augmentation.

For every row of datasets/digits/labels.csv the number box is re-cut from the stored WP4 car crop
(work/<video>/crops/<stem>.jpg) with 50% context on each side, so the augmentation can jitter the box
(cut or extend it by +-10% and more) and warp it without black borders. Negatives ("no number", empty target) are
random windows of the same car crops that do not touch any grounded number box (expanded by 25%). About one crop
in three shows a second, unlabelled number of the same car (WP8 QA), so a few negatives contain a number: accepted
label noise (a window covers a small part of the crop).

usage: python students/recog/prep.py            -> students/runs/recog/cache_{train,val,test}.pkl
Label: text_read (the grounded string, keeps the side plates' leading zero "06") when it equals text up to
leading zeros, else the row is dropped (2 rows).
"""
import json
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "datasets" / "digits"
OUT = ROOT / "students" / "runs" / "recog"
CTX = 0.5          # context kept around the box, fraction of box w / h on each side
NEG_PER_CROP = 2


def cut(im, box, ctx=CTX):
    """patch around box (x1,y1,x2,y2 in im px) with ctx context; returns patch, box in patch coords."""
    H, W = im.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    a, b = int(max(0, np.floor(x1 - ctx * bw))), int(max(0, np.floor(y1 - ctx * bh)))
    c, d = int(min(W, np.ceil(x2 + ctx * bw))), int(min(H, np.ceil(y2 + ctx * bh)))
    return im[b:d, a:c].copy(), (x1 - a, y1 - b, x2 - a, y2 - b)


def load_boxes(vid):
    L = pd.read_parquet(ROOT / "work" / vid / "labels.parquet", columns=["track_id", "label", "number_boxes"])
    out = {}
    for t in L.itertuples():
        if pd.isna(t.label) or not t.number_boxes or t.number_boxes == "{}":
            continue
        for p, bl in json.loads(t.number_boxes).items():
            out[p] = bl
    return out


def overlaps(w, boxes, grow=0.25):
    x1, y1, x2, y2 = w
    for (a, b, c, d) in boxes:
        gw, gh = grow * (c - a), grow * (d - b)
        if x1 < c + gw and x2 > a - gw and y1 < d + gh and y2 > b - gh:
            return True
    return False


def main():
    rng = random.Random(0)
    D = pd.read_csv(DS / "labels.csv", dtype={"text": str, "text_read": str}, keep_default_na=False)
    OUT.mkdir(parents=True, exist_ok=True)
    boxes_by_vid = {v: load_boxes(v) for v in D.video.unique()}
    cache = {s: [] for s in ("train", "val", "test")}
    neg_done = set()
    dropped = 0
    for r in D.itertuples():
        tr = r.text_read if r.text_read else r.text
        if not tr.isdigit() or tr.lstrip("0") != r.text.lstrip("0"):
            dropped += 1
            continue
        stem_k = Path(r.image).stem[len(r.video) + 1:]
        stem, k = stem_k.rsplit("_n", 1)
        p = f"work/{r.video}/crops/{stem}.jpg"
        bl = boxes_by_vid[r.video].get(p)
        im = cv2.imread(str(ROOT / p))
        if bl is None or im is None:
            dropped += 1
            continue
        box = bl[int(k)]
        patch, pb = cut(im, box)
        cache[r.split].append(dict(img=patch, box=pb, text=tr, number=r.text, bucket=r.bucket, video=r.video,
                                   track=int(r.track_id), frame=int(r.frame_idx), box_h=float(r.box_h),
                                   name=Path(r.image).name, crop=p, neg=False))
        if p in neg_done:
            continue
        neg_done.add(p)
        H, W = im.shape[:2]
        hs = [b[3] - b[1] for b in bl]
        for _ in range(NEG_PER_CROP):
            for _try in range(30):
                h = float(np.clip(rng.choice(hs) * rng.uniform(0.6, 1.6), 12, H * 0.5))
                w = h * rng.uniform(0.6, 2.0)
                if w >= W - 2 or h >= H - 2:
                    continue
                x1, y1 = rng.uniform(0, W - w), rng.uniform(0, H - h)
                win = (x1, y1, x1 + w, y1 + h)
                if overlaps(win, bl):
                    continue
                patch, pb = cut(im, win)
                cache[r.split].append(dict(img=patch, box=pb, text="", number="", bucket="neg", video=r.video,
                                           track=int(r.track_id), frame=int(r.frame_idx), box_h=h,
                                           name=f"neg_{Path(p).stem}", crop=p, neg=True))
                break
    for s, items in cache.items():
        with open(OUT / f"cache_{s}.pkl", "wb") as f:
            pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)
        n_pos = sum(not i["neg"] for i in items)
        n_ok = sum(i["bucket"] == "ok" for i in items)
        print(s, "pos", n_pos, "(ok", n_ok, ") neg", len(items) - n_pos)
    print("dropped", dropped)


if __name__ == "__main__":
    sys.exit(main())
