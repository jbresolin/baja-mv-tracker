"""Shared bits of the offline tools (replay, detection check): reading a recording frame by frame, picking the
detector, writing small plots without matplotlib."""
import csv
from pathlib import Path

import cv2
import numpy as np

from zone.source import ZoneFrame


def video_frames(rtcfg, video, start=None, duration=None, frame_times=None, stats=None):
    """Every sampled frame of a recording (at rt capture.fps), as fast as possible.

    Without `frame_times`: t = seconds into the file (pts) and t_wall = rt's wall clock for the frame (overlay /
    creation_time / mtime, see rt/capture.py). That's right for camera files, whose pts are real time.

    With `frame_times` (from `load_frame_times`, the `.frames.csv` sidecar our own SpotterRecorder writes): t and
    t_wall are the grab times the tracker saw live. Our recordings are encoded at a fixed frame rate, so their pts
    are only nominal (a frame the webcam delivered late, or a dropped one, shifts everything after it); the sidecar
    has the real times. Frames missing from the sidecar fall back to pts and are counted in `stats["missing"]`."""
    from rt.capture import FileCapture
    cap = FileCapture(rtcfg, video, start, duration, realtime=False, log=lambda m: None)
    fps = float(cap.info["fps"]) or 1.0
    if stats is not None:
        stats.update(matched=0, missing=0)
    cap.start()
    try:
        for fr in cap.frames():
            t, t_wall = fr.pts, fr.t
            if frame_times is not None:
                hit = frame_times.get(file_frame_number(fr.pts, fps))
                if hit is not None:
                    t_wall, t = hit
                if stats is not None:
                    stats["matched" if hit is not None else "missing"] += 1
            yield ZoneFrame(fr.img, t, t_wall, fr.idx)
    finally:
        cap.stop()


def file_frame_number(pts, fps):
    """The frame number in the file (0, 1, 2...) of a frame shown at `pts` seconds, for a constant-rate file."""
    return int(round(float(pts) * float(fps)))


def sidecar_for(video):
    """`spotter_X.mkv` -> `spotter_X.frames.csv` if that file exists next to it, else None."""
    p = Path(video).with_suffix(".frames.csv")
    return p if p.exists() else None


def load_frame_times(path):
    """Read a SpotterRecorder sidecar: -> {frame number in the file: (t_wall, t_mono)}."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["frame"])] = (float(row["t_wall"]), float(row["t_mono"]))
    return out


def pick_detector(kind, rtcfg):
    """-> (detector, (input_w, input_h)). kind "rt" = the deployed ONNX car detector; "blobs" = the white-box
    detector of zone/testing.py (only for synthetic test videos)."""
    if kind == "blobs":
        from zone.testing import BlobDetector
        return BlobDetector(), (640, 384)
    from zone.spotter import make_detector
    return make_detector(rtcfg)


def line_plot(path, xs, ys, title, xlabel, ylabel, size=(800, 400), xlim=None, ylim=None, marks=()):
    """A minimal line chart as a PNG (cv2 only). `marks`: vertical lines at these x values (keyframes)."""
    w, h = size
    left, right, top, bottom = 70, 20, 40, 50
    img = np.full((h, w, 3), 255, np.uint8)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    x0, x1 = xlim if xlim else ((xs.min(), xs.max()) if len(xs) else (0, 1))
    y0, y1 = ylim if ylim else (0.0, max(1.0, float(ys.max()) * 1.1 if len(ys) else 1.0))
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1

    def px(x, y):
        return (int(left + (x - x0) / (x1 - x0) * (w - left - right)),
                int(h - bottom - (y - y0) / (y1 - y0) * (h - top - bottom)))

    cv2.rectangle(img, px(x0, y0), px(x1, y1), (0, 0, 0), 1)
    for k in range(5):
        yv = y0 + (y1 - y0) * k / 4
        p = px(x0, yv)
        cv2.line(img, p, px(x1, yv), (225, 225, 225), 1)
        cv2.putText(img, "%.1f" % yv, (5, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
    for k in range(6):
        xv = x0 + (x1 - x0) * k / 5
        p = px(xv, y0)
        cv2.putText(img, "%.2f" % xv, (p[0] - 15, h - bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1,
                    cv2.LINE_AA)
    for m in marks:
        if x0 <= m <= x1:
            cv2.line(img, px(m, y0), px(m, y1), (255, 180, 120), 1)
    if len(xs) > 1:
        cv2.polylines(img, [np.array([px(x, y) for x, y in zip(xs, ys)], np.int32)], False, (200, 60, 20), 2,
                      cv2.LINE_AA)
    cv2.putText(img, title, (left, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, xlabel, (w // 2 - 20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, ylabel, (5, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
