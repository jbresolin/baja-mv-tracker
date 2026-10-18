"""QA sheet for one overlay region: random keyframes with the region outlined and its on/off state + lift.

usage: python teacher/overlay_qa.py <id> <region> [n=24] [seed=0]   -> work/<id>/qa/overlay_r<region>.jpg
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, frame_at, select_videos, wdir  # noqa: E402
import overlay as O  # noqa: E402,F401


def main(vid, region, n=24, seed=0):
    row = select_videos([vid])[0]
    d = wdir(vid)
    z = np.load(d / "overlay_vis.npz")
    with open(d / "overlays.csv", newline="", encoding="utf-8") as f:
        r = list(csv.DictReader(f))[region]
    ts, lift = z["ts"], z["vis"][region]
    on = z["on"][region] if "on" in z else (lift >= O.VIS_LIFT)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(ts), size=min(n, len(ts)), replace=False))
    x1, y1, x2, y2 = (int(r[k]) // 2 for k in ("x1", "y1", "x2", "y2"))
    tiles = []
    for i in pick:
        f = frame_at(ROOT / row["path"], ts[i] + 0.01, (640, 360), "bgr24")
        f = f.copy() if f is not None else np.zeros((360, 640, 3), np.uint8)
        c = (0, 255, 0) if on[i] else (0, 0, 255)
        cv2.rectangle(f, (x1, y1), (x2 - 1, y2 - 1), c, 2)
        cv2.putText(f, f"{ts[i]:.0f}s {'ON' if on[i] else 'off'} lift {lift[i]:.1f}", (5, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        tiles.append(cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA))
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    (d / "qa").mkdir(exist_ok=True)
    p = d / "qa" / f"overlay_r{region}.jpg"
    cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(p, f"region {region} {r['kind']} on {on.mean():.0%} of keyframes")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), *(int(x) for x in a[2:]))
