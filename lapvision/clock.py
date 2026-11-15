"""Reads the burned-in OBS clock overlay ("2026-10-17 11:39:49.146 UTC-03:00").

Monospace font, white on a translucent bar. At 1920 width, cell i spans
x = 14 + 24.2*i .. +24, y = 16..48. Templates are bootstrapped from a few hand
labelled crops (see templates/). Classification = nearest mean template.
"""
import sys
import datetime as dt
import subprocess
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TPL_FILE = HERE / "templates" / "clock_digits.npy"
X0, PITCH, CW, Y0, Y1 = 14.0, 24.2, 24, 16, 48
TIME_IDX = [11, 12, 14, 15, 17, 18, 20, 21, 22]  # HH MM SS mmm
DATE_IDX = [0, 1, 2, 3, 5, 6, 8, 9]  # 2026-10-17, used for templates only
DATE = dt.date(2026, 10, 17)
TZ = dt.timezone(dt.timedelta(hours=-3))


def _cell(img1080, i):
    x = int(round(X0 + PITCH * i))
    c = img1080[Y0:Y1, x:x + CW]
    m = (c.min(2) > 200).astype(np.float32)
    return m


def _norm(frame):
    """Return the overlay strip at 1920-wide scale."""
    h, w = frame.shape[:2]
    if w != 1920:
        s = 1920.0 / w
        strip = frame[0:int(70 / s)]
        return cv2.resize(strip, (1920, int(strip.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return frame[0:70]


def build_templates(labelled):
    """labelled: list of (image_path_or_array, 'HH:MM:SS.mmm')."""
    acc = {d: [] for d in "0123456789"}
    for img, txt in labelled:
        im = cv2.imread(str(img)) if not isinstance(img, np.ndarray) else img
        im = _norm(im)
        digits = "20261017" + txt.replace(":", "").replace(".", "")
        for i, ch in zip(DATE_IDX + TIME_IDX, digits):
            acc[ch].append(_cell(im, i))
    missing = [d for d, v in acc.items() if not v]
    assert not missing, "no samples for digits %s" % missing
    tpl = np.stack([np.mean(acc[d], 0) for d in "0123456789"])
    TPL_FILE.parent.mkdir(exist_ok=True)
    np.save(TPL_FILE, tpl)
    return tpl


_TPL = None


def read(frame):
    """Returns (seconds_since_local_midnight, min_margin) or (None, 0)."""
    global _TPL
    if _TPL is None:
        _TPL = np.load(TPL_FILE)
    im = _norm(frame)
    ds, margin = [], 1e9
    for i in TIME_IDX:
        c = _cell(im, i)
        d = ((_TPL - c[None]) ** 2).reshape(10, -1).mean(1)
        o = np.argsort(d)
        ds.append(int(o[0]))
        margin = min(margin, d[o[1]] - d[o[0]])
    hh, mm, ss = ds[0] * 10 + ds[1], ds[2] * 10 + ds[3], ds[4] * 10 + ds[5]
    ms = ds[6] * 100 + ds[7] * 10 + ds[8]
    if hh > 23 or mm > 59 or ss > 59:
        return None, 0
    return hh * 3600 + mm * 60 + ss + ms / 1000.0, float(margin)


def to_epoch(sec_of_day):
    base = dt.datetime(DATE.year, DATE.month, DATE.day, tzinfo=TZ).timestamp()
    return base + sec_of_day


def fmt(sec_of_day):
    h = int(sec_of_day // 3600)
    m = int(sec_of_day % 3600 // 60)
    s = sec_of_day % 60
    return "%02d:%02d:%06.3f" % (h, m, s)


def grab(video, t, crop_top=True, ffmpeg="ffmpeg"):
    """Grab a single frame at t seconds via ffmpeg (accurate seek). `ffmpeg`: the program to run. The default (a bare
    name, looked up on PATH) is only for the offline command line below; runtime callers (rt/capture.py) must pass
    the configured path (capture.ffmpeg): the installed app's bundled ffmpeg is not on PATH."""
    vf = "crop=iw:ih*70/1080:0:0" if crop_top else "null"
    out = subprocess.run([ffmpeg, "-v", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1",
                          "-vf", vf, "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    if not out:
        return None
    return cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)


def survey(video, step=60.0, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    """Sample the clock along a file, print wall - pts offset to find jumps/freezes."""
    dur = float(subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of",
                                "csv=p=0", str(video)], capture_output=True, text=True).stdout)
    rows = []
    t = 0.5
    while t < dur:
        f = grab(video, t, ffmpeg=ffmpeg)
        if f is not None:
            s, mg = read(f)
            if s is not None:
                rows.append((t, s, s - t, mg))
        t += step
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "build":
        pairs = [(a, b) for a, b in (x.split("=") for x in sys.argv[2:])]
        build_templates(pairs)
        print("templates saved", TPL_FILE)
    elif sys.argv[1] == "survey":
        step = float(sys.argv[3]) if len(sys.argv) > 3 else 60
        for t, s, off, mg in survey(sys.argv[2], step):
            print("pts %8.2f  clock %s  offset %s  margin %.3f" % (t, fmt(s), fmt(off), mg))
