"""QA sheets for shots.csv (CPU). Writes JPEG contact sheets into work/<id>/qa/.

usage:
  python teacher/shots_qa.py cuts <id> [n=20] [seed=0]      random boundaries: frames at t-0.4 | (mid) | t+0.4
  python teacher/shots_qa.py seq <id> <first_shot> [n=10]  boundaries in order from shot first_shot
  python teacher/shots_qa.py long <id> [n=6]                the n longest shots: 12 frames evenly spread over each
  python teacher/shots_qa.py window <id> <t0> <t1> [step=1]  every `step` s in [t0, t1] with the shot id burned in
"""
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, fmt_t, frame_at, select_videos, wdir  # noqa: E402

SZ = (320, 180)


def label(img, txt, color=(255, 255, 0)):
    img = np.array(img)  # writable copy
    cv2.putText(img, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return img


def grab(path, t):
    f = frame_at(path, t, SZ, "bgr24")
    return f.copy() if f is not None else np.zeros((SZ[1], SZ[0], 3), np.uint8)


def load(vid):
    row = select_videos([vid])[0]
    with open(wdir(vid) / "shots.csv", newline="") as f:
        shots = list(csv.DictReader(f))
    for s in shots:
        for k in ("t0", "t1", "dur"):
            s[k] = float(s[k])
    return row, shots


def save(vid, name, rows):
    w = max(len(r) for r in rows)
    rows = [r + [np.zeros_like(r[0])] * (w - len(r)) for r in rows]
    sheet = np.vstack([np.hstack(r) for r in rows])
    out = wdir(vid) / "qa"
    out.mkdir(exist_ok=True)
    p = out / name
    cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(p)


def cuts(vid, n=20, seed=0, first=None):
    """n random boundaries (seed), or with `first` the n boundaries starting at shot `first`, in order."""
    row, shots = load(vid)
    path = ROOT / row["path"]
    b = shots[1:]
    if first is None:
        random.Random(seed).shuffle(b)
        b = sorted(b[:n], key=lambda s: s["t0"])
        name = f"cuts_s{seed}.jpg"
    else:
        b = [s for s in b if int(s["shot_id"]) >= first][:n]
        name = f"seq_{first}.jpg"
    rows = []
    for s in b:
        i = int(s["shot_id"])
        prev = shots[i - 1]
        if int(s["is_transition"]):
            ts = [s["t0"] - 0.4, (s["t0"] + s["t1"]) / 2, s["t1"] + 0.4]
        else:
            ts = [s["t0"] - 0.4, s["t0"] + 0.4]
        r = [label(grab(path, t), f"{fmt_t(t)} {t:.1f}") for t in ts]
        cv2.putText(r[0], f"prev {prev['dur']:.1f}s", (4, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(r[-1], f"#{i} {s['cut_in']}{' T' if int(s['is_transition']) else ''} {s['dur']:.1f}s",
                    (4, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        if len(r) == 2:
            r.insert(1, np.zeros_like(r[0]))
        rows.append(r)
    # 2 boundaries per sheet row -> 6 tiles of 320 px
    pairs = [rows[i] + (rows[i + 1] if i + 1 < len(rows) else []) for i in range(0, len(rows), 2)]
    save(vid, name, pairs)


def long(vid, n=6):
    row, shots = load(vid)
    path = ROOT / row["path"]
    for s in sorted(shots, key=lambda s: -s["dur"])[:n]:
        ts = np.linspace(s["t0"] + 0.5, s["t1"] - 0.5, 12)
        tiles = [label(grab(path, t), f"{fmt_t(t)}") for t in ts]
        save(vid, f"long_{int(s['shot_id'])}.jpg", [tiles[:6], tiles[6:]])
        print(f"  shot {s['shot_id']} {fmt_t(s['t0'])}-{fmt_t(s['t1'])} {s['dur']:.0f}s")


def window(vid, t0, t1, step=1.0):
    row, shots = load(vid)
    path = ROOT / row["path"]
    tiles = []
    for t in np.arange(t0, t1, step):
        sid = next((s["shot_id"] for s in shots if s["t0"] <= t < s["t1"]), "?")
        tiles.append(label(grab(path, t), f"{t:.1f} s{sid}"))
    save(vid, f"win_{int(t0)}_{int(t1)}.jpg", [tiles[i:i + 6] for i in range(0, len(tiles), 6)])


if __name__ == "__main__":
    cmd, vid, *a = sys.argv[1:]
    if cmd == "cuts":
        cuts(vid, *(int(x) for x in a))
    elif cmd == "seq":
        cuts(vid, int(a[1]) if len(a) > 1 else 10, 0, int(a[0]))
    elif cmd == "long":
        long(vid, *(int(x) for x in a))
    elif cmd == "window":
        window(vid, *(float(x) for x in a))
