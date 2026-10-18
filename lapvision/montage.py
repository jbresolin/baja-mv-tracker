"""Montage of crops of one track: python montage.py <crops_dir> <event> <track> <out.jpg> [n]"""
import sys, cv2, numpy as np, pandas as pd
from pathlib import Path
d = Path(sys.argv[1]); ev, tr, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 12
t = pd.read_csv(d / "tracks.csv")
t = t[(t.event == ev) & (t.track == tr)].sort_values("frame")
idx = np.linspace(0, len(t) - 1, min(n, len(t))).astype(int)
ims = []
for _, r in t.iloc[idx].iterrows():
    im = cv2.imread(str(d / ev / r.crop))
    s = 360 / im.shape[0]
    im = cv2.resize(im, (int(im.shape[1] * s), 360))
    im = cv2.resize(im, (480, 360)) if im.shape[1] > 480 else cv2.copyMakeBorder(im, 0, 0, 0, 480 - im.shape[1], cv2.BORDER_CONSTANT)
    cv2.putText(im, "f%d s%.0f" % (r.frame, r.sharp), (5, 25), 0, 0.8, (0, 255, 255), 2)
    ims.append(im)
while len(ims) % 4: ims.append(np.zeros_like(ims[0]))
cv2.imwrite(out, np.vstack([np.hstack(ims[i:i + 4]) for i in range(0, len(ims), 4)]))
