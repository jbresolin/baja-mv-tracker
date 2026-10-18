"""Stage 2: for each vehicle event in a stage-1 timeline, decode the window at native
frame rate, run YOLO + ByteTrack, and save per-track vehicle crops (native resolution)
plus a tracks.csv. OCR is a separate stage so it can be iterated on cheaply.

usage: python extract.py <video> <timeline.csv> <outdir> [t_from_tod|- t_to_tod|- [model]]
"""
import sys
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from det import Detector

import clock
from detect_timeline import file_offset
from align import fg

VEH = [2, 3, 7]
PAD = 1.5          # seconds before/after the coarse event
GAP = 2.5          # merge coarse detections closer than this
MIN_CONF = 0.3
TARGET_FPS = 30    # process ~30 fps regardless of native rate


def events(tl, t0=None, t1=None):
    v = fg(tl, conf=MIN_CONF)  # only events with a big, near vehicle (skips background traffic)
    if t0 is not None:
        v = v[(v.tod >= t0) & (v.tod <= t1)]
    t = np.sort(v.tod.unique())
    if not len(t):
        return []
    ev, s, p = [], t[0], t[0]
    for x in t[1:]:
        if x - p > GAP:
            ev.append((s, p))
            s = x
        p = x
    ev.append((s, p))
    return ev


def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "json", str(video)],
                         capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    a, b = s["r_frame_rate"].split("/")
    return s["width"], s["height"], float(a) / float(b)


def decode(video, start, dur, w, h):
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-ss", "%.3f" % start, "-i", str(video),
           "-t", "%.3f" % dur, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 2)
    while True:
        buf = p.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.wait()


def sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    video, tlf, outdir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    t0 = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
    t1 = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "-" else None
    tl = pd.read_csv(tlf)
    tl = tl[tl.file == video.name]
    w, h, fps = probe(video)
    off = file_offset(video)
    step = max(1, round(fps / TARGET_FPS))
    det = Detector(sys.argv[6] if len(sys.argv) > 6 else "yolo11x.pt")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    evs = events(tl, t0, t1)
    print(len(evs), "events", flush=True)
    for ei, (a, b) in enumerate(evs):
        start = max(0.0, a - off - PAD)
        dur = (b - a) + 2 * PAD
        eid = "%s_%s" % (video.stem[-8:].replace("-", ""), clock.fmt(a).replace(":", "").replace(".", "_"))
        edir = outdir / eid
        edir.mkdir(exist_ok=True)
        det.reset()  # reset tracker state between events
        for fi, frame in enumerate(decode(video, start, dur, w, h)):
            if fi % step:
                continue
            pts = start + fi / fps
            r = det.track(frame, imgsz=1280, conf=0.15)[0]
            if r.boxes.id is None:
                continue
            for bx, tid, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().numpy(),
                                   r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = bx
                bw, bh = x2 - x1, y2 - y1
                X1, Y1 = int(max(0, x1 - 0.05 * bw)), int(max(0, y1 - 0.05 * bh))
                X2, Y2 = int(min(w, x2 + 0.05 * bw)), int(min(h, y2 + 0.05 * bh))
                crop = frame[Y1:Y2, X1:X2]
                name = "t%03d_f%05d.jpg" % (tid, fi)
                cv2.imwrite(str(edir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append(dict(event=eid, track=int(tid), frame=fi, pts=round(pts, 3),
                                 tod=round(off + pts, 3), conf=round(float(cf), 3),
                                 x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                                 W=w, H=h, sharp=round(sharpness(crop), 1), crop=name))
        print(ei, eid, "%.1fs" % dur, flush=True)
        pd.DataFrame(rows).to_csv(outdir / "tracks.csv", index=False)


if __name__ == "__main__":
    main()
