"""Real-time lap-pass prototype (WP11): capture -> car detector (10 fps) -> ByteTrack -> online pass logic ->
async identity (panel detector -> recognizer -> boxed vote -> roster) -> JSONL events + passes CSV (+ sinks).

usage:
  python rt/run.py --source "2026-10-17 11-29-49.mkv" --event 26SE --out out/rt/1129.jsonl            # file, fast
  python rt/run.py --source "2026-10-17 11-29-49.mkv" --event 26SE --start 11:38 --duration 600 --realtime --stats
  python rt/run.py --source camera --event 27SE --config rt/configs/lapline.toml --out out/lapline/run.jsonl
  python rt/run.py --source 0 --event 27SE --out passes.jsonl                                       # camera index 0
  python rt/run.py --list-cameras                                         # camera indices + device paths (Windows)
  python rt/run.py --source rtsp://192.168.0.10/stream1 --event 27SE --out passes.jsonl
options: --config FILE (only the keys it changes; default rt/config.toml), --set section.key=value (repeatable),
  --threads N (= runtime.threads_detector and threads_identity), --affinity 0,2 (CPU list), --quiet, --as-live
  (read a file through the live-stream path, at real speed), --no-identity.
Live camera (WP-L1): records the video to <folder of --out>/rec/ ([recording]), serves the status page on
http://<this PC>:8765/ ([status]). For an unattended race day, start it through rt/supervise.py (rt\\run_lapline.bat).
Beam input (WP-L2): [beam] driver = modem-line / byte-stream / fake adds the timing sensor's crossing times to the
same JSONL (rt/beam/). CPU only by default: CUDA is never used (onnxruntime CPU / OpenVINO CPU or Intel iGPU).
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
    n_threads = None
    for i, arg in enumerate(argv):
        if arg == "--threads" and i + 1 < len(argv):
            n_threads = argv[i + 1]
        elif arg.startswith("--threads="):
            n_threads = arg.split("=", 1)[1]
    if n_threads:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = str(n_threads)
    else:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")


early_threads(sys.argv)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from rt import config as C  # noqa: E402
from rt.capture import open_source, set_affinity  # noqa: E402
from rt.recorder import Recorder, check_recording_config, software_version  # noqa: E402
from rt.models import YoloOnnx  # noqa: E402
from rt.passlogic import PassLogic  # noqa: E402
from rt.sinks import ConsoleSink, CsvSink, Emitter, JsonlSink, LapAppSink  # noqa: E402
from rt.stats import Monitor, summary  # noqa: E402
from rt.tracker import ByteTracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append",
                    help="video file | camera (the [camera] config) | camera index | rtsp/http url; repeat for a "
                         "playlist of files")
    ap.add_argument("--event", help="event code: roster rosters/<EV>.csv")
    ap.add_argument("--list-cameras", action="store_true", help="print the cameras (index, name, device path), exit")
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
    ap.add_argument("--require-status", action="store_true",
                    help="exit (code 3) if the status page cannot open its port (the launcher passes this)")
    a = ap.parse_args()
    if a.list_cameras:
        return list_cameras(C.load(a.config, a.set))
    if not a.source or not a.event:
        ap.error("--source and --event are required")

    sets = list(a.set)
    if a.threads:
        sets += ["runtime.threads_detector=%d" % a.threads, "runtime.threads_identity=%d" % a.threads]
    if a.affinity:
        sets += ["runtime.affinity=[%s]" % a.affinity]
    cfg = C.load(a.config, sets)
    runtime_cfg = cfg["runtime"]
    cv2.setNumThreads(int(runtime_cfg["cv2_threads"]))
    if runtime_cfg.get("affinity"):
        set_affinity(os.getpid(), runtime_cfg["affinity"])
    if runtime_cfg.get("priority") == "high":
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
    board = None
    if cfg["status"]["enabled"]:
        from rt.status import StatusBoard, StatusSink
        board = StatusBoard(cfg)
        sinks.append(StatusSink(board))
    if cfg["lapapp"]["enabled"]:
        ob = cfg["lapapp"]["outbox"] or str(out.with_suffix(".lapapp.jsonl"))
        sinks.append(LapAppSink(cfg, ob, a.event, Path(a.source[0]).name))
    em = Emitter(sinks)
    log = lambda m: em.emit(dict(type="info", msg=m))  # noqa: E731
    warn = lambda m: em.emit(dict(type="warning", msg=m))  # noqa: E731

    version = software_version()
    from rt.wintimer import python_check, raise_timer_resolution
    python_info, python_warning = python_check()
    log(python_info)
    if python_warning:
        warn(python_warning)
    timer_note = raise_timer_resolution()           # Windows: 1 ms sleeps for the beam's 1 kHz poll
    if timer_note:
        log(timer_note)
    live_camera = any(is_camera(s) for s in a.source)
    server = timechecker = recorder = None
    if board is not None:
        board.info = dict(event=a.event, version=version, source=" ".join(a.source), out=str(out))
        from rt.status import TimeChecker, start_status_server
        status_cfg = cfg["status"]
        required = a.require_status or bool(status_cfg.get("required", False))
        server = start_status_server(board, status_cfg["bind"], status_cfg["port"], required, log=log, warn=warn)
        if live_camera:
            timechecker = TimeChecker(cfg, board, float(status_cfg["time_check_s"]))
            timechecker.start()
    if live_camera:
        check_recording_config(cfg["recording"])       # a clear error for settings that cannot work
    if live_camera and cfg["recording"]["mode"] != "off":
        recording_cfg = cfg["recording"]
        recorder = Recorder(cfg, out.parent / recording_cfg["subdir"], float(cfg["capture"]["fps"]), version,
                            log=log)
        recorder.choose_encoder()                     # probe QSV / AMF / x264 once, now, and log the one used
        if recorder.unavailable:                      # ffmpeg missing: run without recording, detection unaffected
            warn("RECORDING: " + recorder.unavailable)
            if board is not None:
                board.notices.append(recorder.unavailable)
            recorder = None
        else:
            recorder.start()
            if board is not None:
                board.recorder = recorder

    beam = start_beam(cfg, em, board, live_camera, log, warn, config_path=a.config or C.DEFAULT)

    # A missing roster never stops rt (the launcher would restart it forever): run in open-set mode instead.
    roster_info = C.load_roster(a.event, int(cfg["identity"].get("open_set_max", C.OPEN_SET_MAX)),
                                override_dir=cfg["identity"].get("rosters_dir") or None)
    roster, teams = roster_info["numbers"], roster_info["teams"]
    if roster_info["open_set"]:
        warn("ROSTER: " + roster_info["warning"])
        if board is not None:
            board.notices.append(roster_info["warning"])
    m = cfg["models"]
    det = YoloOnnx(m["car_detector"], m["car_imgsz"], int(runtime_cfg["threads_detector"]),
                   spinning=bool(runtime_cfg["ort_spinning"]), backend=runtime_cfg["car_backend"],
                   device=runtime_cfg["car_device"], log=warn)
    trk = ByteTracker(**{key: cfg["tracker"][key] for key in ("track_high_thresh", "track_low_thresh",
                                                               "new_track_thresh", "track_buffer", "match_thresh",
                                                               "fuse_score")})
    idw = None
    if not a.no_identity:
        from rt.identity import IdentityWorker
        idw = IdentityWorker(cfg, roster, teams, em.emit, log=warn, open_set=roster_info["open_set"])
        idw.start()
    pl = PassLogic(cfg, em.emit, idw, log=log)
    if idw:
        idw.now_fn = lambda: pl.now
    vr = C.vote_rule(cfg)
    roster_desc = ("NO ROSTER, open set 1-%d" % len(roster)) if roster_info["open_set"] else (
        "%d roster numbers" % len(roster))
    log("rt start (%s): event %s (%s), car detector %s [%s], panel %s [%s], recognizer %s [%s], "
        "vote %s" % (version, a.event, roster_desc, Path(m["car_detector"]).parent.parent.name + "/" + Path(m["car_detector"]).name, det.desc,
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
            live_kw = dict(recorder=recorder, emit=em.emit) if is_camera(src) else {}
            cap = open_source(cfg, src, a.start, a.duration, a.realtime, a.as_live, log=log, **live_kw)
            caps.append(cap)
            if board is not None:
                board.capture = cap
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
                if board is not None:
                    board.on_frame(fr)
                if beam is not None:
                    beam.on_frame(fr)           # LED latency ROI (only when beam.led_roi is set)
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
    stats = summary(a, cfg, det, idw, mon, n_frames, dropped, tdet, tpost, lags, wall, t_flush - t_wall0, vid_t0,
                    vid_t1, em.counts)
    stats["version"] = version
    if recorder is not None:
        recorder.stop()
        stats["recording"] = recorder.status()
        stats["recording"]["segments"] = [str(path) for path in recorder.segments]
    for c in caps:
        if hasattr(c, "status"):
            stats["camera"] = c.status()
    if beam is not None:
        beam.stop()
        stats["beam"] = beam.status()
    if timechecker is not None:
        timechecker.halt.set()
    if server is not None:
        server.stop()
    if mon:
        mon.stop()
    log("rt end: " + json.dumps({key: stats[key] for key in ("frames", "dropped", "video_s", "wall_s", "speed_x",
                                                              "det_ms_p50", "passes", "cpu")}, default=str))
    Path(str(out.with_suffix("")) + ".stats.json").write_text(json.dumps(stats, indent=1, default=str))
    em.close()


def start_beam(cfg, em, board, live_camera, log, warn, config_path=None):
    """The beam input (rt/beam/, WP-L2) behind a BeamController: the controller always exists (so the status page's
    Beam setup panel can turn the beam on on site); the BeamService and its two threads only run when beam.driver
    is not "off". It never touches the pass logic. Returns the BeamController."""
    from rt.beam.control import BeamController, ControllerSink
    controller = BeamController(cfg, em.emit, config_path=config_path)
    with em.lock:
        em.sinks.append(ControllerSink(controller))   # pass_end -> pass_beam (live association), via the controller
    if board is not None:
        board.beam = controller
    if not live_camera and str(cfg["beam"].get("driver", "off")).lower() != "off":
        warn("beam input is on but the source is not a live camera: beam and frame times will not line up")
    controller.start_from_config()
    return controller


def is_camera(source):
    s = str(source)
    return s == "camera" or s.isdigit()


def list_cameras(cfg):
    """Print the cameras this PC sees, with the device path to put in camera.device."""
    from rt.camsource import default_backend, list_dshow_video_devices
    backend = cfg["camera"]["backend"]
    backend = default_backend() if backend == "auto" else backend
    if backend == "dshow":
        devs = list_dshow_video_devices(cfg["capture"]["ffmpeg"])
        if not devs:
            print("no DirectShow video devices found")
        for i, (name, path) in enumerate(devs):
            print("%d  %s\n   device = '%s'" % (i, name, path))
        print("Put the device path (or a unique part of it, e.g. the usb#vid_..._pid_...#<serial> piece) in "
              "camera.device. Names are not used: identical webcams share a name.")
    else:
        by_id = Path("/dev/v4l/by-id")
        for link in sorted(by_id.glob("*")) if by_id.exists() else []:
            print("%s -> %s" % (link, link.resolve()))
        print("Linux: put a /dev/v4l/by-id/... path in camera.device (stable across replugs).")


if __name__ == "__main__":
    main()
