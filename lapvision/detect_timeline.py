"""Stage 1: coarse detection timeline. Decodes a file at low fps (NVDEC via ffmpeg),
runs YOLO (COCO) and writes vehicle/person boxes with wall-clock time to CSV.

usage: python detect_timeline.py <video> <out.csv> [fps] [model]
"""
import sys
import csv
import subprocess
from pathlib import Path

import numpy as np
from det import Detector

import clock

VEH = {2: "car", 3: "motorcycle", 7: "truck", 0: "person"}
W, H = 1920, 1080


def file_offset(video):
    """Seconds-of-day at pts=0 (all files proved linear, see LESSONS.md)."""
    f = clock.grab(video, 0.5)
    s, _ = clock.read(f)
    return s - 0.5


def frames(video, fps):
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-i", str(video),
           "-vf", f"fps={fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    n = 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        yield n / fps, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        n += 1


def main():
    video, out = Path(sys.argv[1]), Path(sys.argv[2])
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 5
    det = Detector(sys.argv[4] if len(sys.argv) > 4 else "yolo11x.pt")
    off = file_offset(video)
    print("offset", clock.fmt(off), flush=True)
    batch, meta = [], []
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "pts", "tod", "cls", "conf", "x1", "y1", "x2", "y2"])

        def flush():
            res = det.predict(batch, imgsz=1280, conf=0.15, half=True)
            for (pts, ), r in zip(meta, res):
                for b in r.boxes:
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                    w.writerow([video.name, round(pts, 3), round(off + pts, 3), det.names[int(b.cls)],
                                round(float(b.conf), 3), int(x1), int(y1), int(x2), int(y2)])
            batch.clear()
            meta.clear()

        for pts, f in frames(video, fps):
            batch.append(f)
            meta.append((pts,))
            if len(batch) == 16:
                flush()
                if int(pts) % 300 == 0:
                    print("pts", int(pts), flush=True)
        if batch:
            flush()


if __name__ == "__main__":
    main()
