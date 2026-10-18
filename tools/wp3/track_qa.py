"""WP3 QA: frame strips with car tracks and flags drawn.

usage: python tools/wp3/track_qa.py <video_id> <dets.parquet> <tracks.parquet> <t0> <t1> <out.jpg> [--n=12]
                                    [--cols=3] [--people] [--tracks=id,id,...]
  --tracks: one frame per listed track (its middle detection) instead of the time window.
  n frames evenly spaced in [t0, t1] (seconds, snapped to the 4 fps detection grid).
  Car tracks: red = screen_fixed (P persistent/ego, B static over moving background, R recurrent), magenta =
  in_overlay, orange = small, green = other. Label: track id, flags, conf. Untracked car dets >= 0.25: thin gray.
  Shot id + time printed top-left.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "teacher"))
from common import select_videos  # noqa: E402

COLS = {"fixed": (0, 0, 255), "ovl": (255, 0, 255), "small": (0, 165, 255), "ok": (0, 255, 0)}


def arg(name, default):
    for a in sys.argv:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    vid, dp, tp, t0, t1, out = a[0], a[1], a[2], float(a[3]), float(a[4]), a[5]
    d = pd.read_parquet(dp)
    tr = pd.read_parquet(tp).set_index("track_id")
    n, cols = arg("n", 12), arg("cols", 3)
    if arg("tracks", ""):  # one frame per listed track: its middle detection (t0/t1 ignored)
        ids = [int(x) for x in arg("tracks", "").split(",")]
        cd = d[d.track_id.isin(ids)].sort_values("frame_idx")
        fis = [int(g.frame_idx.iloc[len(g) // 2]) for _, g in cd.groupby("track_id")]
    else:
        fis = np.unique(np.round(np.linspace(t0, t1, n) * 4).astype(int))
    row = select_videos([vid], active_only=False)[0]
    cap = cv2.VideoCapture(str(ROOT / row["path"]))
    tiles = []
    classes = ["buggy", "person"] if "--people" in sys.argv else ["buggy"]
    for fi in fis:
        cap.set(cv2.CAP_PROP_POS_MSEC, fi * 250.0)
        ok, f = cap.read()
        if not ok:
            continue
        g = d[(d.frame_idx == fi) & d.cls.isin(classes)]
        for r in g[g.track_id < 0].itertuples():
            if r.conf >= 0.25:
                cv2.rectangle(f, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (160, 160, 160), 1)
        for r in g[g.track_id >= 0].itertuples():
            t = tr.loc[r.track_id]
            fl = ""
            if t.get("fixed_persistent", False):
                fl += "P"
            if t.get("fixed_bg_moving", False):
                fl += "B"
            if t.get("fixed_recurrent", False):
                fl += "R"
            if t.get("in_overlay") in (True, 1) and t.get("in_overlay") is not None:
                fl += "O"
            if t.get("small", False):
                fl += "s"
            if t.get("nested_in", -1) >= 0:
                fl += "n"
            c = COLS["fixed"] if t.get("screen_fixed", False) else COLS["ovl"] if t.get("in_overlay") in (True, 1) and t.get("in_overlay") is not None else \
                COLS["small"] if t.get("small", False) else COLS["ok"]
            if r.cls == "person":
                c = (255, 200, 0)
            cv2.rectangle(f, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, 3 if r.conf >= 0.25 else 1)
            lab = f"{r.track_id}{(' ' + fl) if fl else ''} {r.conf:.2f}"
            y = int(r.y1) + 22 if r.y1 < 30 else int(r.y1) - 6
            cv2.putText(f, lab, (int(r.x1) + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(f, lab, (int(r.x1) + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
        sid = int(g.shot_id.iat[0]) if len(g) else int(d.loc[(d.frame_idx - fi).abs().idxmin(), "shot_id"])
        lab = f"shot {sid}  t={fi / 4:.2f}s"
        cv2.putText(f, lab, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(f, lab, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA))
    if not tiles:
        raise SystemExit("no frames")
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rows * 360, cols * 640, 3), np.uint8)
    for i, t in enumerate(tiles):
        grid[(i // cols) * 360:(i // cols + 1) * 360, (i % cols) * 640:(i % cols + 1) * 640] = t
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out, len(tiles), "frames")


if __name__ == "__main__":
    main()
