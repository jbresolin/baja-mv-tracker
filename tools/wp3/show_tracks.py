"""WP3: print car tracks of a tracks.parquet (filter: big boxes or flagged), for eyeballing the flag features.

usage: python tools/wp3/show_tracks.py <tracks.parquet> [--minh=250] [--all]
"""
import sys

import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 1000)
t = pd.read_parquet(sys.argv[1])
minh = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--minh=")), 250)
c = t[t.cls == "buggy"]
if "--all" not in sys.argv:
    c = c[(c.h_px_median > minh) | c.screen_fixed]
cols = ["track_id", "shot_id", "t0", "n", "conf_median", "x1_med", "y1_med", "x2_med", "y2_med", "center_mad_rel",
        "pos_persist", "inbox_change", "bg_motion", "recur_shots", "fixed_persistent", "fixed_bg_moving",
        "fixed_recurrent", "in_overlay"]
print(c[[x for x in cols if x in c.columns]].round(3).to_string())
