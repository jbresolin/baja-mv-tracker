"""Bootstraps clock templates from a file whose wall clock is linear in pts.
Label = OFFSET + pts (we only trust HH MM SS and the 100ms digit away from boundaries)."""
import sys, random
import numpy as np, cv2
import clock

video, offset = sys.argv[1], float(sys.argv[2])  # offset in seconds-of-day at pts 0
dur = float(sys.argv[3]); n = int(sys.argv[4])
random.seed(0)
acc = {d: [] for d in "0123456789"}
samples = []
for k in range(n):
    t = random.uniform(1, dur - 1)
    f = clock.grab(video, t)
    if f is None: continue
    im = clock._norm(f)
    s = offset + t
    frac = s % 0.1
    if frac < 0.03 or frac > 0.07: continue  # too close to a 100ms boundary
    lab = clock.fmt(s).replace(":", "").replace(".", "")[:7]  # HHMMSS + 100ms digit
    for i, ch in zip(clock.TIME_IDX[:7], lab):
        acc[ch].append(clock._cell(im, i))
    for i, ch in zip(clock.DATE_IDX, "20261017"):
        acc[ch].append(clock._cell(im, i))
    samples.append((im, lab))
print({d: len(v) for d, v in acc.items()})
tpl = np.stack([np.mean(acc[d], 0) for d in "0123456789"])
np.save(clock.TPL_FILE, tpl)
clock._TPL = None
bad = 0
for im, lab in samples:
    s, mg = clock.read(im)
    got = clock.fmt(s).replace(":", "").replace(".", "")[:7] if s is not None else None
    bad += got != lab
print("train errors", bad, "/", len(samples))
