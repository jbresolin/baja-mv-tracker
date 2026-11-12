"""Offline replay: run the real zone pipeline (detector, tracker, progress, director, controller) on a recorded spotter
video, driving the simulated PTZ on a fake clock, and score how well it framed each car.

    python -m zone.tools.replay --video out/zone/spotter_20261115-101500.mkv --zone zones/zone.json \\
                                --report reports/ptz1/report.json [--out out/zone/replay_x] [--set controller.branch=A]

Frame times: our own spotter recordings are encoded at a fixed frame rate, so their pts are nominal. The real grab
time of every frame is in the `.frames.csv` sidecar next to the video; it's picked up by itself when it's there
(`--frames <sidecar.csv>` to point at one, `--no-frames` to use the pts anyway). Other videos (camera files, the
October recording) have no sidecar and use their pts.

Scoring: for every control cycle while a car is locked, the PTZ's true pointing (from the simulator) is compared with
pose(s_true), where s_true is the same track's progress *without* lookahead, latency or filtering: its raw projected
positions, interpolated (both sides) at that instant. The error is angular (degrees, when the report gives units
per degree; raw units otherwise).

Outputs (in --out):
  cycles.csv      one row per control cycle
  passes.csv      one row per followed car: lock/release time, time locked, drop reason, mean / max error, s range
  pass_NN.png     error over s for each pass (keyframes as vertical lines)
  summary.json    the per-pass table plus the settings used
  tracks.json     the detector + tracker output per frame (re-used with --tracks to re-run control quickly)

Detection runs once; `--tracks tracks.json` skips it (useful to compare controller settings). `--warm-profile`
first learns the speed profile from all tracks of the video, as passive mode would have before the test.
"""
import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from zone import config as C  # noqa: E402
from zone import director as D  # noqa: E402
from zone import zonefile  # noqa: E402
from zone.path import PathModel  # noqa: E402
from zone.progress import ProgressTracker, SpeedProfile  # noqa: E402
from zone.runtime import ZoneRuntime  # noqa: E402
from zone.source import ZoneFrame  # noqa: E402


# ============================================================================================================ steps
def detect(frames, spotter):
    """frames -> [(t, idx, t_wall, [(id, [x1,y1,x2,y2], score), ...]), ...]"""
    out = []
    for fr in frames:
        tr = spotter.process(fr.img)
        out.append((float(fr.t), int(fr.idx), float(fr.t_wall),
                    [(int(i), [round(float(v), 2) for v in b], round(float(s), 3)) for i, b, s in tr]))
    return out


def truth_tracks(dets, path: PathModel, progress_settings):
    """Raw projected s of every track over time: {track_id: (ts, s_raw)} (no filter, no lookahead)."""
    pt = ProgressTracker(path, progress_settings)
    seen = {}
    for t, _, _, tr in dets:
        pt.update(t, tr)
        seen.update(pt.tracks)
    return {tid: (np.array([x[0] for x in tp.samples]), np.array([x[1] for x in tp.samples]))
            for tid, tp in seen.items()}


def s_true_at(truth, tid, t):
    if tid not in truth:
        return None
    ts, ss = truth[tid]
    if len(ts) == 0 or t < ts[0] or t > ts[-1]:
        return None
    return float(np.interp(t, ts, ss))


def learn_profile(dets, path, settings):
    sp = settings.cfg["speed_profile"]
    prof = SpeedProfile(s_lo=min(-0.5, path.s_min), s_hi=max(1.2, path.s_max), bin_width=float(sp["bin_width"]),
                        max_passes=int(sp["max_passes"]), default_v=float(sp["default_v"]))
    pt = ProgressTracker(path, settings.progress, prof)
    for t, _, _, tr in dets:
        pt.update(t, tr)
    pt.flush()
    return prof


def teach_presets(ptz, clock, zone):
    """The simulated PTZ starts empty: store each keyframe's preset (directly when the simulator allows it, else the
    way the operator did: by moving there)."""
    if hasattr(ptz, "define_preset"):
        for k in zone.keyframes():
            ptz.define_preset(int(k.preset), k.pan, k.tilt, k.zoom)
        return
    for k in zone.keyframes():
        ptz.absolute_position(int(k.pan), int(k.tilt), 24, 20)
        ptz.zoom_direct(int(k.zoom))
        for _ in range(200):           # up to 20 s per keyframe for the simulated move to finish
            clock.advance(0.1)
            if hasattr(ptz, "tick"):
                ptz.tick()
        ptz.preset_store(int(k.preset))


def angular_error(pan, tilt, want_pan, want_tilt, units_per_degree=None):
    """Distance between where the PTZ points and where it should, in degrees when the units per degree are known
    ((pan_units, tilt_units): each axis is converted with its own factor), else in raw units."""
    dp, dt = pan - want_pan, tilt - want_tilt
    if units_per_degree:
        dp, dt = dp / units_per_degree[0], dt / units_per_degree[1]
    return math.hypot(dp, dt)


def control(dets, settings, zone, model, truth, proc_delay=0.05, profile=None, units_per_degree=None):
    """Drive SimulatedPtz with the pipeline on a fake clock. -> (cycle rows, pass rows)."""
    from zone.clock import FakeClock
    from zone.ptz.sim import SimulatedPtz
    t0 = dets[0][0] if dets else 0.0
    clock = FakeClock(start=t0 - 1000.0, wall_start=(dets[0][2] if dets else 0.0) - 1000.0)
    ptz = SimulatedPtz(model, clock)
    teach_presets(ptz, clock, zone)
    clock.advance(max(0.0, (t0 - 1.0) - clock.monotonic()))
    events = []
    rt = ZoneRuntime(settings, zone, ptz, clock, None, profile=profile, on_event=events.append)
    rt.set_mode(D.AUTO, why="replay")
    curve = rt.curve
    upd = units_per_degree
    rows, passes, cur = [], [], None
    for t, idx, t_wall, tr in dets:
        target_t = t + proc_delay
        if target_t > clock.monotonic():
            clock.advance(target_t - clock.monotonic())
        n_ev = len(events)
        res = rt.cycle(ZoneFrame(None, t, t_wall, idx), [(i, np.array(b, np.float32), s) for i, b, s in tr])
        now = clock.monotonic()
        st = rt.director.state()
        for ev in events[n_ev:]:
            if ev.get("event") == "lock":
                cur = dict(track_id=ev["track_id"], t_lock=now, errs=[], ss=[], reason="", t_release=None)
            elif ev.get("event") == "release" and cur is not None:
                cur.update(reason=ev["reason"], t_release=now)
                passes.append(cur)
                cur = None
        pan, tilt = ptz.true_pan_tilt()
        zoom = ptz.true_zoom()
        row = dict(t=round(now, 3), frame=idx, mode=st["mode"], sub=st["sub"], locked=st["locked_id"],
                   pan=round(pan, 1), tilt=round(tilt, 1), zoom=round(zoom, 1), s_est="", v_est="", s_true="",
                   target_pan="", target_tilt="", err="", pan_step="", tilt_step="")
        if res.command is not None:
            row.update(s_est=round(res.command.s, 4), v_est=round(res.command.v, 4), pan_step=res.command.pan_step,
                       tilt_step=res.command.tilt_step)
        if st["locked_id"] is not None and cur is not None:
            s_true = s_true_at(truth, st["locked_id"], now)
            if s_true is not None:
                want = curve.pose(s_true)
                e = angular_error(pan, tilt, want.pan, want.tilt, upd)
                row.update(s_true=round(s_true, 4), target_pan=round(want.pan, 1), target_tilt=round(want.tilt, 1),
                           err=round(e, 3))
                cur["errs"].append(e)
                cur["ss"].append(s_true)
        rows.append(row)
    if cur is not None:
        cur.update(reason="end_of_video", t_release=clock.monotonic())
        passes.append(cur)
    out = []
    for k, p in enumerate(passes):
        e = np.array(p["errs"]) if p["errs"] else np.array([np.nan])
        out.append(dict(pass_no=k + 1, track_id=p["track_id"], t_lock=round(p["t_lock"], 2),
                        t_release=round(p["t_release"], 2), locked_s=round(p["t_release"] - p["t_lock"], 2),
                        drop_reason=p["reason"], n_scored=len(p["errs"]),
                        err_mean=round(float(np.nanmean(e)), 3), err_max=round(float(np.nanmax(e)), 3),
                        err_p90=round(float(np.nanpercentile(e, 90)), 3),
                        s_from=round(min(p["ss"]), 3) if p["ss"] else None,
                        s_to=round(max(p["ss"]), 3) if p["ss"] else None,
                        _ss=p["ss"], _errs=p["errs"]))
    return rows, out


def write_outputs(out_dir, rows, passes, keyframe_s, unit, meta):
    from zone.tools.common import line_plot
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_dir / "cycles.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    cols = [k for k in (passes[0] if passes else {}) if not k.startswith("_")]
    with open(out_dir / "passes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols or ["pass_no"])
        w.writeheader()
        for p in passes:
            w.writerow({k: p[k] for k in cols})
    for p in passes:
        if p["_ss"]:
            line_plot(out_dir / ("pass_%02d.png" % p["pass_no"]), p["_ss"], p["_errs"],
                      "pass %d, car track %d: %s, mean %.2f max %.2f %s" % (
                          p["pass_no"], p["track_id"], p["drop_reason"], p["err_mean"], p["err_max"], unit),
                      "s (0 = entry keyframe, 1 = exit)", "error (%s)" % unit, xlim=(min(-0.1, min(p["_ss"])),
                                                                                     max(1.1, max(p["_ss"]))),
                      marks=keyframe_s)
    summary = dict(meta, unit=unit, passes=[{k: p[k] for k in cols} for p in passes])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


# ============================================================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="spotter recording (not needed with --tracks)")
    ap.add_argument("--zone", required=True)
    ap.add_argument("--report", default="", help="characterization report JSON (default: config ptz.report)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--start", default=None, help="seconds into the video, or HH:MM[:SS] wall time")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--frames", default=None,
                    help="frame-time sidecar (.frames.csv) of a zone recording; default: the one next to the video")
    ap.add_argument("--no-frames", action="store_true", help="ignore the sidecar: frame times from the pts")
    ap.add_argument("--tracks", default=None, help="tracks.json from an earlier replay: skip detection")
    ap.add_argument("--detector", default="rt", choices=["rt", "blobs"], help="blobs: synthetic test videos only")
    ap.add_argument("--proc-delay", type=float, default=None,
                    help="simulated detection time per frame (s); default: measured ms per frame")
    ap.add_argument("--warm-profile", action="store_true", help="learn the speed profile from the video first")
    a = ap.parse_args(argv)

    sets = list(a.set) + (["ptz.report=%s" % a.report] if a.report else [])
    cfg = C.load(a.config, sets)
    settings = C.build(cfg)
    zone = zonefile.load(a.zone)
    path = PathModel.build(zone)
    if path is None:
        raise SystemExit("the zone needs at least 2 keyframes")
    out_dir = Path(a.out) if a.out else C.resolve_path(cfg["zone"]["out_dir"]) / (
        "replay_%s" % dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    det_ms = None
    if a.tracks:
        saved = json.loads(Path(a.tracks).read_text(encoding="utf-8"))
        dets = [tuple(d) for d in saved["frames"]]
        det_ms = saved.get("det_ms_median")          # same simulated processing delay as the run that saved them
    else:
        if not a.video:
            raise SystemExit("--video or --tracks is needed")
        from zone.spotter import RoiDetector, Spotter, make_tracker, rt_config
        from zone.tools.common import load_frame_times, pick_detector, sidecar_for, video_frames
        rtcfg = rt_config(cfg)
        frames_csv = None if a.no_frames else (Path(a.frames) if a.frames else sidecar_for(a.video))
        frame_times = load_frame_times(frames_csv) if frames_csv else None
        print("frame times: %s" % (("sidecar %s (%d frames)" % (frames_csv, len(frame_times))) if frame_times
                                   else "video pts"))
        ft_stats = {}
        det, inwh = pick_detector(a.detector, rtcfg)
        s = cfg["spotter"]
        size = zone.image_size
        roi = path.corridor_bbox(float(s["roi_margin_px"]), size)
        spotter = Spotter(RoiDetector(det, s["roi_mode"], roi, inwh, s["tile_overlap"], s["tile_scale"],
                                      rtcfg["detection"]["conf"], rtcfg["detection"]["nms_iou"]), make_tracker(rtcfg))
        ms = []
        dets = []
        for fr in video_frames(rtcfg, a.video, a.start, a.duration, frame_times, ft_stats):
            dets.extend(detect([fr], spotter))
            ms.append(spotter.last_ms)
        det_ms = float(np.median(ms)) if ms else None
        (out_dir / "tracks.json").write_text(json.dumps(dict(video=str(a.video), roi_mode=s["roi_mode"], roi=roi,
                                                             det_ms_median=det_ms,
                                                             frame_times=str(frames_csv) if frames_csv else "pts",
                                                             frames=dets)), encoding="utf-8")
        print("detection: %d frames, %.1f ms per frame (median)" % (len(dets), det_ms or 0))
        if frame_times and ft_stats.get("missing"):
            print("warning: %d frames not in the sidecar (their time comes from the pts)" % ft_stats["missing"])
    proc_delay = a.proc_delay if a.proc_delay is not None else (det_ms / 1000.0 if det_ms else 0.05)
    truth = truth_tracks(dets, path, settings.progress)
    profile = learn_profile(dets, path, settings) if a.warm_profile else None
    model = settings.report.model
    pan_upd, tilt_upd = getattr(model, "pan_units_per_degree", None), getattr(model, "tilt_units_per_degree", None)
    upd = (pan_upd, tilt_upd or pan_upd) if pan_upd else None
    rows, passes = control(dets, settings, zone, model, truth, proc_delay, profile, upd)
    meta = dict(video=a.video, zone=a.zone, report=settings.report.summary(), branch=settings.controller.branch,
                latency_s=settings.controller.latency_s, proc_delay_s=proc_delay, frames=len(dets),
                warm_profile=a.warm_profile)
    summary = write_outputs(out_dir, rows, passes, path.keyframe_s(), "deg" if upd else "raw units", meta)
    print("%d passes followed; branch %s; results in %s" % (len(passes), settings.controller.branch, out_dir))
    for p in summary["passes"]:
        print("  pass %(pass_no)d track %(track_id)d: %(locked_s).1f s, %(drop_reason)s, error mean %(err_mean).2f "
              "max %(err_max).2f, s %(s_from)s..%(s_to)s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
