"""WP10b: training cache from WP9's second-round panel proposals (out/panel_proposals_num_n320.csv, train events only).

Each proposal is a num_n320 box (conf 0.1, >= 16 px) on a WP4 crop of a teacher-labelled track whose roster posterior
from syn_ft matched the track consensus (P >= 0.5). Here: box enlarged by 5% per side (the deployment input: harness /
panel_val pad 0.05), cut with 50% context like prep.py, label = syn_ft's greedy string on that box when it equals the
consensus up to leading zeros (keeps side-plate "06"; otherwise the box is dropped). Capped at --per-track boxes per
track (spread over the track's crops) against confirmation bias / near-duplicate views. Patches stored as JPEG q95.
Every video must be a train-split video (asserted). -> students/runs/recog/cache_props.pkl
usage: CUDA_VISIBLE_DEVICES=-1 python students/recog/prep_props.py [--per-track 6] [--threads 2]
"""
import argparse
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment import plain  # noqa: E402
from common import ROOT, RUNS, batch_tensor, greedy, load_model, strip0  # noqa: E402
from prep import cut  # noqa: E402

TRAIN_VIDEOS = {"19SU_END", "25BR_DIN", "25BR_END", "26BR_DIN", "26BR_END"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "out" / "panel_proposals_num_n320.csv"))
    ap.add_argument("--per-track", type=int, default=6)
    ap.add_argument("--pad", type=float, default=0.05)
    ap.add_argument("--model", default="syn_ft")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    D = pd.read_csv(a.csv)
    assert set(D.video) <= TRAIN_VIDEOS, set(D.video) - TRAIN_VIDEOS
    rng = random.Random(0)
    keep = []
    for (vid, tid), g in D.groupby(["video", "track_id"]):
        per_crop = defaultdict(list)
        for r in g.itertuples():
            per_crop[r.path].append(r)
        lists = list(per_crop.values())
        rng.shuffle(lists)
        rr = []                                    # round-robin over the track's crops: one box per crop first
        for i in range(max(len(x) for x in lists)):
            rr += [x[i] for x in lists if i < len(x)]
        keep += rr[:a.per_track]
    model, _ = load_model(RUNS / a.model / "best.pt")
    items, dropped = [], 0
    by_path = defaultdict(list)
    for r in keep:
        by_path[r.path].append(r)
    batch = []

    def flush():
        nonlocal dropped
        if not batch:
            return
        with torch.no_grad():
            g = greedy(model(batch_tensor([plain(it["_im"], it["_b"]) for it in batch])))
        for it, (s, _) in zip(batch, g):
            if strip0(s) != it["number"] or not s:
                dropped += 1
                continue
            patch, pb = cut(it.pop("_im"), it.pop("_b"))
            ok, enc = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
            it.update(img_jpg=enc.tobytes(), box=pb, text=s)
            items.append(it)
        batch.clear()

    for k, (p, rs) in enumerate(by_path.items()):
        im = cv2.imread(str(ROOT / p))
        if im is None:
            continue
        H, W = im.shape[:2]
        for r in rs:
            bw, bh = r.x2 - r.x1, r.y2 - r.y1
            b = (max(0, r.x1 - a.pad * bw), max(0, r.y1 - a.pad * bh), min(W, r.x2 + a.pad * bw), min(H, r.y2 + a.pad * bh))
            batch.append(dict(_im=im, _b=b, number=str(int(r.label)), bucket="ok", video=r.video, track=int(r.track_id),
                              box_h=float(bh), name=f"prop_{Path(p).stem}", crop=p, neg=False, p=float(r.p)))
        if len(batch) >= 256:
            flush()
        if k % 2000 == 0:
            print(k, len(by_path), len(items), flush=True)
    flush()
    pickle.dump(items, open(RUNS / "cache_props.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    nums = {it["number"] for it in items}
    print(f"proposals {len(D)}, kept by cap {len(keep)}, label mismatch dropped {dropped}, cached {len(items)}, "
          f"tracks {len({(i['video'], i['track']) for i in items})}, numbers {len(nums)}, "
          f"leading-zero labels {sum(i['text'].startswith('0') for i in items)}")


if __name__ == "__main__":
    sys.exit(main())
