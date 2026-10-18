"""Per-video summary of WP2 outputs -> data/wp2_summary.csv (+ a markdown table on stdout).

usage: python teacher/wp2_summary.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, select_videos, wdir  # noqa: E402
from shot_tags import TAG_NAMES  # noqa: E402


def main():
    rows = []
    for v in select_videos():
        d = wdir(v["id"])
        r = {"id": v["id"], "hours": round(v["duration_s"] / 3600, 2)}
        if (d / "shots.done").exists():
            with open(d / "shots.csv", newline="", encoding="utf-8") as f:
                sh = list(csv.DictReader(f))
            norm = [float(s["dur"]) for s in sh if s["is_transition"] == "0"]
            r.update(n_shots=len(sh), n_transitions=sum(s["is_transition"] == "1" for s in sh),
                     median_shot_s=round(float(np.median(norm)), 1) if norm else 0,
                     shots_per_hour=round(len(sh) / max(v["duration_s"] / 3600, 1e-6), 1))
            if (d / "shot_tags.done").exists():
                for t in TAG_NAMES:
                    r[f"h_{t}"] = round(sum(float(s["dur"]) for s in sh if s.get("tag") == t) / 3600, 2)
        if (d / "overlay.done").exists():
            with open(d / "overlays.csv", newline="", encoding="utf-8") as f:
                ov = list(csv.DictReader(f))
            r["n_overlays"] = len(ov)
            r["overlays"] = "; ".join(f"{o['kind']} ({o['x1']},{o['y1']},{o['x2']},{o['y2']}) {float(o['coverage']):.0%}"
                                      for o in ov)
        rows.append(r)
    cols = ["id", "hours", "n_shots", "n_transitions", "median_shot_s", "shots_per_hour"] + \
        [f"h_{t}" for t in TAG_NAMES] + ["n_overlays", "overlays"]
    with open(DATA / "wp2_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    short = ["id", "hours", "n_shots", "median_shot_s"] + [f"h_{t}" for t in TAG_NAMES] + ["n_overlays"]
    print("| " + " | ".join(c.replace("h_", "") for c in short) + " |")
    print("|" + "---|" * len(short))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in short) + " |")


if __name__ == "__main__":
    main()
