"""Render a human-review video: boxes, pass ids, voted numbers, human GT laps.

usage: python render_review.py <video> <crops_dir> <out.mp4> [--condensed]
  --condensed: only keep frames within +-WIN s of a machine pass or a human GT lap.
"""
import ast
import sys
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import clock
import evaluate as E
import gt
from detect_timeline import file_offset
from extract import probe

FPS = 30
WIN = 4.0
OUT = Path(__file__).resolve().parent.parent / "out"
PALETTE = [(0, 200, 255), (255, 120, 0), (80, 255, 80), (255, 0, 200), (0, 255, 255), (200, 200, 0),
           (120, 120, 255), (255, 255, 255)]


def text(img, s, org, scale=0.9, color=(255, 255, 255), thick=2, bg=(0, 0, 0)):
    (w, h), b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 3, y - h - 6), (x + w + 3, y + b + 2), bg, -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    video, cdir, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    condensed = "--condensed" in sys.argv
    w, h, nfps = probe(video)
    off = file_offset(video)
    tracks = pd.read_csv(cdir / "tracks.csv")
    allp = pd.read_csv(OUT / "passes_all.csv")
    ps = allp[allp.src == cdir.name].copy()
    ps["tracks"] = ps.tracks.map(ast.literal_eval)
    owner = {}
    for _, p in ps.iterrows():
        for t in p.tracks:
            owner[t] = p
    reads = pd.read_csv(cdir / "reads.csv")
    reads["pass_id"] = (reads.event + ":" + reads.track.astype(str)).map(lambda k: owner[k].pass_id if k in owner else None)
    votes = {}
    for pid, s in reads.dropna(subset=["pass_id", "vlm"]).groupby("pass_id").vlm:
        votes[int(pid)] = s.astype(int).value_counts().to_dict()
    # boxes keyed by output frame index (pts*FPS)
    boxes = {}
    for r in tracks.itertuples():
        boxes.setdefault(int(round(r.pts * FPS)), []).append(r)

    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(video)], capture_output=True, text=True).stdout)
    g = E.gt_laps(off, off + dur)
    mp = ps.sort_values("tcross")
    keep = None
    if condensed:
        ts = np.concatenate([g.t.values, mp.tcross.values, mp.t0.values])
        keep = lambda tod: np.any(np.abs(ts - tod) < WIN)

    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-i", str(video), "-vf",
                            f"fps={FPS},scale=1920:1080", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE, bufsize=1920 * 1080 * 3 * 2)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080",
                            "-r", str(FPS), "-i", "-", "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "26",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    sx = 1920 / w
    last = {}  # hold boxes between processed frames (tracks are at ~30fps already)
    fi = 0
    while True:
        buf = dec.stdout.read(1920 * 1080 * 3)
        if len(buf) < 1920 * 1080 * 3:
            break
        pts = fi / FPS
        tod = off + pts
        fi += 1
        if keep is not None and not keep(tod):
            continue
        img = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3).copy()
        cur = boxes.get(int(round(pts * FPS)))
        if cur is not None:
            last = {"t": pts, "b": cur}
        elif last and pts - last["t"] > 0.1:
            last = {}
        for r in (last.get("b") or []):
            p = owner.get("%s:%d" % (r.event, r.track))
            if p is None:
                col, lab = (90, 90, 90), "bg t%d" % r.track
            else:
                col = PALETTE[int(p.pass_id) % len(PALETTE)]
                v = votes.get(int(p.pass_id), {})
                car = "#%d" % p.vlm_car if pd.notna(p.vlm_car) else "#?"
                vs = " ".join("%s:%d" % kv for kv in list(v.items())[:3]) or "no read"
                lab = "P%d %s  [%s]" % (p.pass_id, car, vs)
            x1, y1, x2, y2 = (int(v * sx) for v in (r.x1, r.y1, r.x2, r.y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
            text(img, lab, (x1 + 4, max(110, y1 - 8)), 0.9, col)

        # panel: human GT laps near now, machine passes near now
        y = 110
        text(img, "HUMAN (GT+80s)", (1500, y), 0.7, (180, 255, 180)); y += 34
        for q in g[(g.t > tod - 6) & (g.t < tod + 6)].itertuples():
            act = abs(q.t - tod) < 1.0
            text(img, "#%d lap %d  %+.1fs" % (q.car, q.lap, q.t - tod), (1500, y), 0.8,
                 (0, 255, 0) if act else (200, 200, 200), 2, (0, 90, 0) if act else (0, 0, 0)); y += 34
        y += 10
        text(img, "MACHINE", (1500, y), 0.7, (180, 220, 255)); y += 34
        for q in mp[(mp.tcross > tod - 6) & (mp.tcross < tod + 6)].itertuples():
            act = abs(q.tcross - tod) < 1.0
            car = "#%d" % q.vlm_car if pd.notna(q.vlm_car) else "#?"
            gtc = " (gt #%d)" % q.gt_car if pd.notna(q.gt_car) else " (no gt match)"
            text(img, "P%d %s%s %+.1fs" % (q.pass_id, car, gtc, q.tcross - tod), (1500 - 150, y), 0.8,
                 (0, 200, 255) if act else (200, 200, 200), 2, (0, 60, 120) if act else (0, 0, 0)); y += 34
        n_m = int((mp.tcross <= tod).sum())
        n_g = int((g.t <= tod).sum())
        text(img, "count so far: machine %d | human %d" % (n_m, n_g), (20, 1060), 0.9, (255, 255, 255))
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
