"""WP3: sample car track ids for QA strips by category.

usage: python tools/wp3/qa_sample.py <tracks.parquet> <category> [--n=12] [--seed=0]
categories: ego (fixed_persistent), bgfix (fixed_bg_moving), recur (fixed_recurrent), bigbottom (unflagged, bottom-anchored,
            h > 300), onboard (unflagged tracks in shots that have an ego track), overlay (in_overlay), static (long_static),
            normal (unflagged, h >= 100)
"""
import sys

import pandas as pd

t = pd.read_parquet(sys.argv[1])
cat = sys.argv[2]
n = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--n=")), 12)
seed = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--seed=")), 0)
c = t[t.cls == "buggy"]
H = 720
sel = {
    "ego": lambda: c[c.fixed_persistent],
    "bgfix": lambda: c[c.fixed_bg_moving],
    "recur": lambda: c[c.fixed_recurrent],
    "bigbottom": lambda: c[(c.y2_med >= 0.97 * H) & (c.h_px_median > 300) & ~c.screen_fixed],
    "onboard": lambda: c[c.shot_id.isin(set(c[c.fixed_persistent].shot_id)) & ~c.screen_fixed & (c.n >= 6)],
    "overlay": lambda: c[c.in_overlay == True],  # noqa: E712
    "static": lambda: c[c.long_static],
    "normal": lambda: c[~c.screen_fixed & (c.h_px_median >= 100)],
}[cat]()
s = sel.sample(min(n, len(sel)), random_state=seed) if len(sel) else sel
print(",".join(str(i) for i in sorted(s.track_id)))
