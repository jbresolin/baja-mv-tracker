"""Thumbnail every N s across files (for framing-change survey). usage: scenes.py out.jpg step"""
import sys, cv2, numpy as np, subprocess, clock
from pathlib import Path
files = ["09-09-26", "09-56-32", "11-09-03", "11-29-49"]
step = float(sys.argv[2]); ims = []
for f in files:
    v = Path("..") / ("2026-10-17 %s.mkv" % f)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(v)], capture_output=True, text=True).stdout)
    t = 1.0
    while t < dur:
        out = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(v), "-frames:v", "1", "-vf", "scale=320:180", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
        im = cv2.imdecode(np.frombuffer(out, np.uint8), 1)
        big = clock.grab(v, t); s, _ = clock.read(big)
        cv2.rectangle(im, (0, 0), (110, 22), (0, 0, 0), -1)
        cv2.putText(im, clock.fmt(s)[:8], (3, 17), 0, 0.55, (0, 255, 255), 1)
        ims.append(im); t += step
cols = 10
while len(ims) % cols: ims.append(np.zeros_like(ims[0]))
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)]))
print(len(ims))
