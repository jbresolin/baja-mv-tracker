"""Real-time lap-pass prototype (WP11): capture -> car detector (10 fps) -> ByteTrack -> online pass logic ->
async identity (panel detector -> recognizer -> boxed vote -> roster) -> JSONL events + passes CSV (+ sinks).

usage:
  python rt/run.py --source "2026-10-17 11-29-49.mkv" --event 26SE --out out/rt/1129.jsonl            # file, fast
  python rt/run.py --source "2026-10-17 11-29-49.mkv" --event 26SE --start 11:38 --duration 600 --realtime --stats
  python rt/run.py --source 0 --event 27SE --out passes.jsonl                                       # camera index 0
  python rt/run.py --source rtsp://192.168.0.10/stream1 --event 27SE --out passes.jsonl
options: --config FILE (only the keys it changes; default rt/config.toml), --set section.key=value (repeatable),
  --threads N (= runtime.threads_detector and threads_identity), --affinity 0,2 (CPU list), --quiet, --as-live
  (read a file through the live-stream path, at real speed), --no-identity.
CPU only by default: CUDA is never used (onnxruntime CPU / OpenVINO CPU or Intel iGPU).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def early_threads(argv):
    """thread env vars must be set before numpy / onnxruntime / cv2 load their thread pools."""
    n = None
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            n = argv[i + 1]
        elif a.startswith("--threads="):
            n = a.split("=", 1)[1]
    if n:
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[k] = str(n)
    else:
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(k, "1")


early_threads(sys.argv)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from rt import config as C  # noqa: E402
from rt.capture import open_source, set_affinity  # noqa: E402
from rt.models import YoloOnnx  # noqa: E402
from rt.passlogic import PassLogic  # noqa: E402
from rt.sinks import ConsoleSink, CsvSink, Emitter, JsonlSink, LapAppSink  # noqa: E402
from rt.stats import Monitor, summary  # noqa: E402
from rt.tracker import ByteTracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, action="append",
                    help="video file | camera index | rtsp/http url; repeat for a playlist of files")
    ap.add_argument("--event", required=True, help="event code: roster rosters/<EV>.csv")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    ap.add_argument("--out", default="passes.jsonl")
    ap.add_argument("--realtime", action="store_true", help="files: pace at real speed, drop frames when late")
    ap.add_argument("--as-live", action="store_true", help="files: go through the live-stream path (ffmpeg -re)")
    ap.add_argument("--start", default=None, help="files: seconds into the file, or HH:MM[:SS] wall time")
    ap.add_argument("--duration", type=float, default=None, help="files: seconds to process")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--affinity", default=None, help="comma list of CPUs for this process + ffmpeg")
    ap.add_argument("--stats", action="store_true", help="sample CPU / RAM every second (needs psutil)")
    ap.add_argument("--no-identity", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    sets = list(a.set)
    if a.threads:
        sets += ["runtime.threads_detector=%d" % a.threads, "runtime.threads_identity=%d" % a.threads]
    if a.affinity:
        sets += ["runtime.affinity=[%s]" % a.affinity]
    cfg = C.load(a.config, sets)
    r = cfg["runtime"]
    cv2.setNumThreads(int(r["cv2_threads"]))
    if r.get("affinity"):
        set_affinity(os.getpid(), r["affinity"])
    if r.get("priority") == "high":
        try:
            import psutil
            psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS if hasattr(psutil, "ABOVE_NORMAL_PRIORITY_CLASS")
                                  else -5)
        except Exception as e:  # noqa: BLE001
            print("priority not set:", e)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(cfg["output"]["csv"]) if cfg["output"]["csv"] else out.with_suffix(".csv")
    sinks = [JsonlSink(out), CsvSink(csv_path), ConsoleSink(a.quiet)]
    if cfg["lapapp"]["enabled"]:
        ob = cfg["lapapp"]["outbox"] or str(out.with_suffix(".lapapp.jsonl"))
        sinks.append(LapAppSink(cfg, ob, a.event, Path(a.source[0]).name))
    em = Emitter(sinks)
    log = lambda m: em.emit(dict(type="info", msg=m))  # noqa: E731
    warn = lambda m: em.emit(dict(type="warning", msg=m))  # noqa: E731

    roster, teams = C.read_roster(a.event)
    m = cfg["models"]
    det = YoloOnnx(m["car_detector"], m["car_imgsz"], int(r["threads_detector"]), spinning=bool(r["ort_spinning"]),
                   backend=r["car_backend"], device=r["car_device"], log=warn)
    trk = ByteTracker(**{k: cfg["tracker"][k] for k in ("track_high_thresh", "track_low_thresh", "new_track_thresh",
                                                         "track_buffer", "match_thresh", "fuse_score")})
    idw = None
    if not a.no_identity:
        from rt.identity import IdentityWorker
        idw = IdentityWorker(cfg, roster, teams, em.emit, log=warn)
        idw.start()
    pl = PassLogic(cfg, em.emit, idw, log=log)
    if idw:
        idw.now_fn = lambda: pl.now
    vr = C.vote_rule(cfg)
    log("rt start: event %s (%d roster numbers), car detector %s [%s], panel %s [%s], recognizer %s [%s], vote %s" % (
        a.event, len(roster), Path(m["car_detector"]).parent.parent.name + "/" + Path(m["car_detector"]).name, det.desc,
        Path(m["panel_detector"]).parent.parent.name, idw.panel.desc if idw else "-",
        Path(m["recognizer"]).parent.name, idw.rec.desc if idw else "-", vr))

    mon = Monitor() if a.stats else None
    tdet, tpost, lags = [], [], []
    n_frames = dropped = 0
    t_wall0 = time.time()
    vid_t0 = vid_t1 = None
    caps = []
    try:
        for si, src in enumerate(a.source):
            cap = open_source(cfg, src, a.start, a.duration, a.realtime, a.as_live, log=log)
            caps.append(cap)
            log(cap.describe())
            if cap.live:
                from rt import timesync
                for w in timesync.check(cfg, log):
                    warn("CLOCK: " + w)
            if si:
                trk.reset()
                pl.new_source()
            if mon:
                mon.child = cap
                if not mon.is_alive():
                    mon.start()
            cap.start()
            for fr in cap.frames():
                t = time.perf_counter()
                b, s, _ = det(fr.img, conf=float(cfg["detection"]["conf"]), iou=float(cfg["detection"]["nms_iou"]))
                t2 = time.perf_counter()
                outs = trk.update(b, s)
                pl.step(fr, outs, trk.alive_ids())
                t3 = time.perf_counter()
                tdet.append(t2 - t)
                tpost.append(t3 - t2)
                lags.append(time.monotonic() - fr.t_cap)
                if len(tdet) > 200000:  # bounded: keep a decimated history
                    tdet, tpost, lags = tdet[::2], tpost[::2], lags[::2]
                n_frames += 1
                vid_t0 = fr.t if vid_t0 is None else vid_t0
                vid_t1 = fr.t
                if mon:
                    mon.note(open_passes=pl.open_count(), crop_mb=pl.crop_bytes() / 2**20 if n_frames % 10 == 0 else None,
                             idq=idw.q.qsize() if idw else 0)
            dropped += cap.dropped
            log("source done: %d frames captured, %d dropped" % (cap.captured, cap.dropped))
    except KeyboardInterrupt:
        warn("interrupted")
        for c in caps:
            c.stop()
    pl.flush()
    t_flush = time.time()
    if idw:
        idw.finish()
    wall = time.time() - t_wall0
    st = summary(a, cfg, det, idw, mon, n_frames, dropped, tdet, tpost, lags, wall, t_flush - t_wall0, vid_t0, vid_t1,
                 em.counts)
    if mon:
        mon.stop()
    log("rt end: " + json.dumps({k: st[k] for k in ("frames", "dropped", "video_s", "wall_s", "speed_x",
                                                    "det_ms_p50", "passes", "cpu")}, default=str))
    Path(str(out.with_suffix("")) + ".stats.json").write_text(json.dumps(st, indent=1, default=str))
    em.close()


if __name__ == "__main__":
    main()
