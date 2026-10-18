"""WP3: per-video summary of detect + track outputs (work/<id>/detect_settings.json, track_summary.json).

usage: python tools/wp3/summary.py [--md]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "teacher"))
from common import WORK, select_videos  # noqa: E402

rows = []
for r in select_videos():
    d = WORK / r["id"]
    ds = json.loads((d / "detect_settings.json").read_text()) if (d / "detect_settings.json").exists() else {}
    ts = json.loads((d / "track_summary.json").read_text()) if (d / "track_summary.json").exists() else {}
    det_min = None
    if (d / "detect.done").exists() and ds.get("wall_s_last_session"):
        det_min = ds["wall_s_last_session"] / 60
    rows.append(dict(
        video=r["id"], hours=round(r["duration_s"] / 3600, 2),
        detect="done" if (d / "detect.done").exists() else "-", dets=ds.get("n_dets"),
        car_dets_025=ds.get("n_buggy_025"), car_tracks=ts.get("car_tracks"), tracks_per_h=ts.get("car_tracks_per_h"),
        person_tracks=ts.get("person_tracks"),
        screen_fixed=ts.get("screen_fixed"), in_overlay=ts.get("in_overlay"), small=ts.get("small"),
        long_static=ts.get("long_static"), detect_min=round(det_min, 1) if det_min else None,
        track_min=round(ts["runtime_s"] / 60, 1) if ts.get("runtime_s") else None))
cols = list(rows[0])
if "--md" in sys.argv:
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for x in rows:
        print("| " + " | ".join("" if x[c] is None else (f"{x[c]:.1%}" if c in ("screen_fixed", "in_overlay", "small",
              "long_static") and isinstance(x[c], float) else str(x[c])) for c in cols) + " |")
else:
    import pandas as pd
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))
