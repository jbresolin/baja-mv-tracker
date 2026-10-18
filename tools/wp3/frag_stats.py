"""WP3: tracker fragmentation proxy on a dets/tracks pair: car tracks that start <= GAP s after another car track of
the same shot ended, near where it ended (centre distance < 1.0 x box height). Prints counts and examples.

usage: python tools/wp3/frag_stats.py <dets.parquet> <tracks.parquet> [--gap=3]
"""
import sys

import numpy as np
import pandas as pd

d = pd.read_parquet(sys.argv[1])
tr = pd.read_parquet(sys.argv[2])
gap = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--gap=")), 3.0)
c = d[(d.track_id >= 0) & (d.cls == "buggy")].sort_values("frame_idx")
first = c.groupby("track_id").first()
last = c.groupby("track_id").last()
car = tr[(tr.cls == "buggy") & ~tr.screen_fixed]
n, ex = 0, []
for s, g in car.groupby("shot_id"):
    ids = g.track_id.to_numpy()
    for b in ids:
        fb = first.loc[b]
        for a in ids:
            if a == b:
                continue
            la = last.loc[a]
            dt = fb.pts - la.pts
            if 0 < dt <= gap:
                h = max(la.y2 - la.y1, fb.y2 - fb.y1)
                dc = np.hypot((fb.x1 + fb.x2 - la.x1 - la.x2) / 2, (fb.y1 + fb.y2 - la.y1 - la.y2) / 2)
                if dc < 1.0 * h:
                    n += 1
                    ex.append((a, b, round(la.pts, 2), round(fb.pts, 2), round(dc / h, 2)))
                    break
print(f"{len(car)} car tracks (not screen_fixed); {n} start near where another ended within {gap} s "
      f"({n / max(len(car), 1):.0%}); per-track dets median {car.n.median():.0f}, mean {car.n.mean():.1f}")
for e in ex[:15]:
    print("  ", e)
