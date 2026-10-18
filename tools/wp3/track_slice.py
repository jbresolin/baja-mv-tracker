"""WP3: run teacher/track.py's tracking + flags on a calibration slice (work/calib/<name>.parquet from detect.py calib).

usage: python tools/wp3/track_slice.py <video_id> <dets_slice.parquet> <out_prefix>
       -> <out_prefix>_dets.parquet, <out_prefix>_tracks.parquet (in_overlay from overlays.csv if present)
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "teacher"))
import track as T  # noqa: E402
from common import select_videos, wdir  # noqa: E402

vid, src, pre = sys.argv[1:4]
row = select_videos([vid])[0]
meta = json.loads(Path(src).with_suffix(".json").read_text())
d = wdir(vid)
shots = pd.read_csv(d / "shots.csv").sort_values("t0").reset_index(drop=True)
dets = T.run_tracking(pd.read_parquet(src), shots)
tr = T.box_stats(dets, row["height"])
tr["nested_in"] = T.nested(dets, tr).to_numpy()
feats, sig, _ = T.decode_feats(ROOT / row["path"], dets, tr, row["width"], row["height"], shots,
                               start=meta["start"], dur=meta["dur"])
tr = T.flags(dets, tr, feats, sig, T.load_overlays(d, row["width"], row["height"]), shots, row["height"])
dets.to_parquet(pre + "_dets.parquet", index=False)
tr.to_parquet(pre + "_tracks.parquet", index=False)
c = tr[tr.cls == "buggy"]
print(f"{vid}: {len(c)} car tracks, {(tr.cls == 'person').sum()} person tracks; screen_fixed {c.screen_fixed.sum()} "
      f"(persist {c.fixed_persistent.sum()}, bg {c.fixed_bg_moving.sum()}, recur {c.fixed_recurrent.sum()}), static_box {c.static_box.sum()}, "
      f"small {c.small.sum()}, nested {(c.nested_in >= 0).sum()}")
