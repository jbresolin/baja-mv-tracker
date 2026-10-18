"""Build datasets/numbers_crop_r2 = datasets/numbers_crop + second-round panel proposals (train events only).

usage: python students/build_numbers_r2.py [--proposals out/panel_proposals_num_n320.csv] [--per-track 6]

* numbers_crop is not modified: its images are hard-linked (they are already hard links of work/<vid>/crops), labels are
  copied, val (25SU) and test (24SU) are exactly numbers_crop's, so metrics stay comparable.
* Proposals (students/panel_propose.py: num_n320 boxes whose recognizer read equals the track consensus) are added to train
  only, at most --per-track crops per track (spread evenly over the track's frames) so long tracks don't dominate. All
  accepted boxes of a chosen crop go into its label file. Crops already in numbers_crop are skipped.
* index.csv gets the new rows with source = 'proposal'.
"""
import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "numbers_crop"
DST = ROOT / "datasets" / "numbers_crop_r2"
VAL_TEST_EVENTS = {"25SU", "24SU"}


def link(a, b):
    b.parent.mkdir(parents=True, exist_ok=True)
    if b.exists():
        return
    try:
        os.link(a, b)
    except OSError:
        shutil.copy2(a, b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proposals", default=str(ROOT / "out" / "panel_proposals_num_n320.csv"))
    ap.add_argument("--per-track", type=int, default=6)
    a = ap.parse_args()
    if DST.exists():
        raise SystemExit("%s exists; delete it to rebuild" % DST)
    idx = pd.read_csv(SRC / "index.csv")
    for r in idx.itertuples():
        link(SRC / r.image, DST / r.image)
        lab = Path(r.image.replace("images/", "labels/", 1)).with_suffix(".txt")
        (DST / lab).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / lab, DST / lab)
    have = set(idx.image.str.split("/").str[-1])

    P = pd.read_csv(a.proposals)
    P["event"] = P.video.str[:4]
    P = P[~P.event.isin(VAL_TEST_EVENTS)]
    P["name"] = P.video + "_" + P.path.str.split("/").str[-1]
    P = P[~P.name.isin(have)]
    P["frame"] = P.path.str.extract(r"_f(\d+)\.jpg$")[0].astype(int)
    rows = []
    for (vid, tid), g in P.groupby(["video", "track_id"]):
        crops = g.drop_duplicates("path").sort_values("frame")
        if len(crops) > a.per_track:
            crops = crops.iloc[np.linspace(0, len(crops) - 1, a.per_track).round().astype(int)]
        for c in crops.itertuples():
            im = cv2.imread(str(ROOT / c.path))
            if im is None:
                continue
            H, W = im.shape[:2]
            img = Path("images/train") / c.name
            link(ROOT / c.path, DST / img)
            bx = g[g.path == c.path]
            lines = []
            for b in bx.itertuples():
                x1, y1, x2, y2 = max(0, b.x1), max(0, b.y1), min(W, b.x2), min(H, b.y2)
                lines.append("0 %.6f %.6f %.6f %.6f" % ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
            lab = DST / "labels" / "train" / (img.stem + ".txt")
            lab.write_text("\n".join(lines) + "\n")
            rows.append(dict(image=img.as_posix(), split="train", event=c.event, video=vid, track_id=int(tid),
                             frame_idx=int(c.frame), number=int(c.label), source="proposal", n_boxes=len(lines),
                             crop_w=W, crop_h=H, box_h_px=round(float((bx.y2 - bx.y1).median()), 1)))
    new = pd.DataFrame(rows)
    pd.concat([idx, new], ignore_index=True).to_csv(DST / "index.csv", index=False)
    (DST / "data.yaml").write_text("path: %s\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: number\n"
                                   % DST.as_posix())
    print("numbers_crop: %d images (train %d); added %d proposal crops / %d boxes from %d tracks"
          % (len(idx), (idx.split == "train").sum(), len(new), int(new.n_boxes.sum()), new.groupby(["video", "track_id"]).ngroups))
    print(new.groupby("video").agg(crops=("image", "size"), boxes=("n_boxes", "sum")).to_string())


if __name__ == "__main__":
    main()
