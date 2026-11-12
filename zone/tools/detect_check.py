"""AUTOPTZ-16 detection check: does each car get ONE continuous track from the approach to the exit?

Runs the spotter (rt car detector + ByteTrack) on a wide-angle recording in several ROI modes, and reports for each:

* detections per frame (all, and with the ground point inside the corridor),
* tracks inside the corridor, how many run continuously from the approach (s <= 0) to the exit (s >= 1), and how
  many are fragments; with --passes N (the number of cars João counted by hand) also tracks per real pass,
* time per frame on this machine (detector + tracker), and inferences per frame (tiles).

    python -m zone.tools.detect_check --video "out/oct_wide.mp4" --zone zones/oct.json --passes 23
    python -m zone.tools.detect_check --video x.mp4 --zone z.json --modes full,downscale,tile --start 600 --duration 300

Writes detect_check.json, a tracks_<mode>.csv per mode (one row per track: frames, s from/to, continuous?), and
prints a table. The zone file is needed for the corridor (draw it on the recording first: the teach page works with a
video file as the spotter and the PTZ simulated).
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from zone import config as C  # noqa: E402
from zone import zonefile  # noqa: E402
from zone.path import PathModel  # noqa: E402
from zone.progress import ProgressTracker  # noqa: E402
from zone.spotter import RoiDetector, Spotter, ground_point, make_tracker, rt_config  # noqa: E402


def run_mode(frames, det, inwh, mode, path, cfg, rtcfg):
    """One ROI mode over all frames -> (per-mode stats, per-track rows)."""
    s = cfg["spotter"]
    roi = path.corridor_bbox(float(s["roi_margin_px"]), None)
    rd = RoiDetector(det, mode, roi, inwh, s["tile_overlap"], s["tile_scale"], rtcfg["detection"]["conf"],
                     rtcfg["detection"]["nms_iou"])
    spotter = Spotter(rd, make_tracker(rtcfg))
    settings = C.build(cfg)
    pt = ProgressTracker(path, settings.progress)
    seen = {}
    n_frames, ms, n_det, n_det_in, n_inf = 0, [], [], [], []
    for fr in frames:
        t0 = time.perf_counter()
        tr = spotter.process(fr.img)
        ms.append((time.perf_counter() - t0) * 1000.0)
        boxes, _ = spotter.last_dets
        n_det.append(len(boxes))
        n_det_in.append(sum(1 for b in boxes if path.in_corridor(path.project(*ground_point(b)))))
        n_inf.append(rd.last_n_inferences)
        pt.update(fr.t, tr)
        seen.update(pt.tracks)
        n_frames += 1
    tracks = []
    for tid, tp in sorted(seen.items()):
        ss = [x[1] for x in tp.samples]
        tracks.append(dict(track_id=tid, frames=tp.n_updates, t_first=round(tp.first_t, 2),
                           t_last=round(tp.last_update_t, 2), s_first=round(ss[0], 3), s_last=round(ss[-1], 3),
                           s_min=round(min(ss), 3), s_max=round(max(ss), 3),
                           continuous=bool(min(ss) <= 0.0 and max(ss) >= 1.0),
                           crosses_middle=bool(min(ss) <= 0.5 <= max(ss))))
    stats = dict(mode=mode, frames=n_frames, roi=list(roi) if mode != "full" else None,
                 inferences_per_frame=round(float(np.mean(n_inf)), 2) if n_inf else 0,
                 ms_per_frame_median=round(float(np.median(ms)), 1) if ms else None,
                 ms_per_frame_p90=round(float(np.percentile(ms, 90)), 1) if ms else None,
                 det_per_frame=round(float(np.mean(n_det)), 3) if n_det else 0,
                 det_in_corridor_per_frame=round(float(np.mean(n_det_in)), 3) if n_det_in else 0,
                 frames_with_corridor_det=int(sum(1 for n in n_det_in if n)),
                 corridor_tracks=len(tracks), continuous_tracks=sum(t["continuous"] for t in tracks),
                 tracks_crossing_middle=sum(t["crosses_middle"] for t in tracks))
    return stats, tracks


def summarize(stats, passes):
    if passes:
        for st in stats:
            st["tracks_per_pass"] = round(st["corridor_tracks"] / passes, 2)
            st["continuous_share_of_passes"] = round(st["continuous_tracks"] / passes, 3)
    return stats


def print_table(stats, passes):
    cols = ["mode", "ms_per_frame_median", "inferences_per_frame", "det_per_frame", "det_in_corridor_per_frame",
            "corridor_tracks", "continuous_tracks", "tracks_crossing_middle"]
    if passes:
        cols += ["tracks_per_pass", "continuous_share_of_passes"]
    print(" | ".join(cols))
    for st in stats:
        print(" | ".join(str(st.get(c)) for c in cols))
    if passes:
        print("real passes (hand count): %d. Goal: continuous_tracks == passes and tracks_per_pass == 1." % passes)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--modes", default="full,downscale,tile")
    ap.add_argument("--passes", type=int, default=None, help="cars that really went through (hand count)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--start", default=None)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--detector", default="rt", choices=["rt", "blobs"])
    a = ap.parse_args(argv)
    cfg = C.load(a.config, a.set)
    rtcfg = rt_config(cfg)
    zone = zonefile.load(a.zone)
    path = PathModel.build(zone)
    if path is None:
        raise SystemExit("the zone needs at least 2 keyframes (s = 0 and s = 1) to measure continuity")
    from zone.tools.common import pick_detector, video_frames
    det, inwh = pick_detector(a.detector, rtcfg)
    out_dir = Path(a.out) if a.out else C.resolve_path(cfg["zone"]["out_dir"]) / "detect_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        st, tracks = run_mode(video_frames(rtcfg, a.video, a.start, a.duration), det, inwh, mode, path, cfg, rtcfg)
        stats.append(st)
        with open(out_dir / ("tracks_%s.csv" % mode), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(tracks[0]) if tracks else ["track_id"])
            w.writeheader()
            w.writerows(tracks)
        print("mode %s done: %d frames, %d corridor tracks" % (mode, st["frames"], st["corridor_tracks"]), flush=True)
    stats = summarize(stats, a.passes)
    (out_dir / "detect_check.json").write_text(json.dumps(dict(video=a.video, zone=a.zone, passes=a.passes,
                                                               start=a.start, duration=a.duration, modes=stats),
                                                          indent=1), encoding="utf-8")
    print_table(stats, a.passes)
    print("written to %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
