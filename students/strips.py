"""Frame strips of missed human laps (or any pass list) for an eval_26se.py run, to see why a detector missed them.

usage:
  python students/strips.py out/eval_<name> [--n 16] [--offsets -1,0,1,2,3] [--csv missed.csv|passes.csv]

For every missed lap (out/eval_<name>/missed.csv: GT time already in video clock) it grabs frames at t + offsets (CPU
ffmpeg), draws the run's own 5 fps timeline detections (green, conf; out/eval_<name>/tl_<file>.csv) and the YOLO-World
teacher's timeline (magenta, out/tlw_<file>.csv) at the nearest 5 fps timestamp, and writes one strip per lap plus a
contact sheet to out/eval_<name>/strips/. The foreground rule of the event stage (vehicle conf > 0.3, h > 0.2 H,
bottom > 0.4 H) is drawn as a thick box; thin boxes did not qualify as an event.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_26se import FILES, file_offset  # noqa: E402

sys.path.insert(0, str(ROOT / "lapvision"))
import clock  # noqa: E402

W = 640


def grab(video, t):
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.3f" % max(0, t), "-i", str(video), "-frames:v", "1",
                          "-vf", "scale=1920:1080", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR) if out else np.zeros((1080, 1920, 3), np.uint8)


def draw(img, tl, tod, color, label):
    if tl is None or not len(tl):
        return
    ts = tl.tod.values
    j = np.argmin(np.abs(ts - tod))
    if abs(ts[j] - tod) > 0.15:
        return
    for r in tl[tl.tod == ts[j]].itertuples():
        if r.cls == "person":
            continue
        fgq = r.conf > 0.3 and (r.y2 - r.y1) > 0.2 * 1080 and r.y2 > 0.4 * 1080
        cv2.rectangle(img, (r.x1, r.y1), (r.x2, r.y2), color, 8 if fgq else 2)
        cv2.putText(img, "%s %.2f" % (label, r.conf), (r.x1 + 4, max(30, r.y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    color, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("evaldir", type=Path)
    ap.add_argument("--csv", default="missed.csv")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--offsets", default="-1,0,1,2,3")
    a = ap.parse_args()
    offs = [float(x) for x in a.offsets.split(",")]
    m = pd.read_csv(a.evaldir / a.csv)
    if "t" not in m:  # a passes.csv
        m = m.assign(t=m.t0, car=m.get("gt_car", -1), lap=-1)
    if "file" not in m:
        m["file"] = m.t.map(lambda t: next((k for k, v in FILES.items() if v[1] <= t <= v[2]), "gap"))
    m["file"] = m.file.astype(str).str.zfill(4)  # read back as int (0909 -> 909)
    m = m[m.file.isin(list(FILES))].head(a.n)
    timing = json.loads((a.evaldir / "timing.json").read_text()) if (a.evaldir / "timing.json").exists() else {}
    out = a.evaldir / "strips"
    out.mkdir(exist_ok=True)
    cache, strips = {}, []
    for i, r in enumerate(m.itertuples()):
        k = r.file
        if k not in cache:
            video = ROOT / FILES[k][0]
            off = timing.get("tl_" + k, {}).get("offset") or file_offset(video)
            own = a.evaldir / ("tl_%s.csv" % k)
            tw = ROOT / "out" / ("tlw_%s.csv" % k)
            cache[k] = (video, off, pd.read_csv(own) if own.exists() else None, pd.read_csv(tw) if tw.exists() else None)
        video, off, own, tw = cache[k]
        tiles = []
        for o in offs:
            tod = r.t + o
            img = grab(video, tod - off)
            draw(img, tw, tod, (255, 0, 255), "YW")
            draw(img, own, tod, (0, 255, 0), "S")
            cv2.putText(img, "%s (%+.0fs)" % (clock.fmt(tod), o), (20, 1060), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                        (0, 255, 255), 4)
            tiles.append(cv2.resize(img, (W, W * 9 // 16), interpolation=cv2.INTER_AREA))
        strip = np.hstack(tiles)
        head = np.zeros((40, strip.shape[1], 3), np.uint8)
        cv2.putText(head, "missed lap: car %s lap %s  human click %s (video clock)  file %s" %
                    (r.car, r.lap, clock.fmt(r.t), k), (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        strip = np.vstack([head, strip])
        cv2.imwrite(str(out / ("miss_%02d_car%s.jpg" % (i, r.car))), strip, [cv2.IMWRITE_JPEG_QUALITY, 85])
        strips.append(strip)
        print(i, r.car, clock.fmt(r.t), flush=True)
    if strips:
        cv2.imwrite(str(out / "sheet.jpg"), np.vstack(strips[:8]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("->", out)


if __name__ == "__main__":
    main()
