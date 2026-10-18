"""Shared helpers for the teacher stages: paths, manifest access, ffmpeg frame pipes, resumable per-video stages.

Every stage writes its outputs into work/<video_id>/ plus a `<stage>.done` marker, skips videos that already have the
marker, and logs a failure to work/<video_id>/errors.log without stopping the other videos.
"""
import csv
import datetime as _dt
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WORK = ROOT / "work"
YOUTUBE = ROOT / "youtube"
ROSTERS = ROOT / "rosters"

_FF_DIRS = [r"C:\Users\baja\AppData\Local\Microsoft\WinGet\Links"]
for _d in _FF_DIRS:
    if Path(_d).is_dir() and _d not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")


def tool(name):
    p = shutil.which(name)
    if not p:
        raise RuntimeError(f"{name} not found on PATH")
    return p


def load_manifest(active_only=True):
    path = DATA / "videos.csv"
    if not path.exists():
        raise SystemExit("data/videos.csv missing: run teacher/manifest.py first")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if active_only:
        rows = [r for r in rows if r["status"] == "active"]
    for r in rows:
        for k in ("duration_s", "fps", "bitrate"):
            r[k] = float(r[k]) if r[k] else 0.0
        for k in ("width", "height"):
            r[k] = int(r[k]) if r[k] else 0
    return rows


def select_videos(ids=None, active_only=True):
    rows = load_manifest(active_only)
    if ids:
        want = set(ids)
        rows = [r for r in rows if r["id"] in want]
        missing = want - {r["id"] for r in rows}
        if missing:
            print("unknown / inactive ids:", ", ".join(sorted(missing)), file=sys.stderr)
    return rows


def wdir(vid):
    d = WORK / vid
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_error(vid, stage, msg):
    with open(wdir(vid) / "errors.log", "a", encoding="utf-8") as f:
        f.write(f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S}] {stage}: {msg}\n")


def _run_one(stage, fn, row, kwargs):
    vid = row["id"]
    t0 = time.time()
    try:
        info = fn(row, **kwargs)
        (wdir(vid) / f"{stage}.done").write_text(
            f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S} {time.time() - t0:.0f}s {info or ''}\n", encoding="utf-8")
        return vid, True, time.time() - t0, info
    except Exception as e:  # noqa: BLE001 - one bad video must not stop the others
        log_error(vid, stage, traceback.format_exc())
        return vid, False, time.time() - t0, repr(e)


def run_stage(stage, fn, rows, workers=1, force=False, requires=(), **kwargs):
    """Run fn(row, **kwargs) for every video row without `<stage>.done`, in `workers` processes."""
    todo = []
    for r in rows:
        d = wdir(r["id"])
        if (d / f"{stage}.done").exists() and not force:
            print(f"{r['id']}: {stage} already done, skip")
            continue
        miss = [q for q in requires if not (d / f"{q}.done").exists()]
        if miss:
            print(f"{r['id']}: needs {', '.join(miss)} first, skip")
            continue
        todo.append(r)
    if not todo:
        return []
    results = []
    if workers <= 1:
        for r in todo:
            res = _run_one(stage, fn, r, kwargs)
            _report(stage, res)
            results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_one, stage, fn, r, kwargs) for r in todo]
            for fu in as_completed(futs):
                res = fu.result()
                _report(stage, res)
                results.append(res)
    return results


def _report(stage, res):
    vid, ok, dt, info = res
    print(f"{vid}: {stage} {'ok' if ok else 'FAILED'} in {dt:.0f}s {info or ''}", flush=True)


def ffmpeg_frames(path, fps=None, size=(320, 180), pix="rgb24", threads=3, keyframes_only=False, start=None,
                  duration=None):
    """Yield (t_seconds, frame ndarray) decoded on the CPU through an ffmpeg pipe.

    fps: resample to this rate (frame i is at t = i / fps). keyframes_only: decode I-frames only (very fast; this
    stream set has a fixed 5 s GOP), timestamps from showinfo-free pts via -fps_mode passthrough + a pts side file.
    """
    w, h = size
    ch = {"rgb24": 3, "gray": 1, "bgr24": 3}[pix]
    cmd = [tool("ffmpeg"), "-v", "error", "-nostdin", "-threads", str(threads)]
    if keyframes_only:
        cmd += ["-skip_frame", "nokey"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(path), "-an", "-sn"]
    vf = []
    if fps and not keyframes_only:
        vf.append(f"fps={fps}")
    vf.append(f"scale={w}:{h}:flags=area")
    cmd += ["-vf", ",".join(vf)]
    if keyframes_only:
        cmd += ["-fps_mode", "passthrough"]
        # keyframe pts from a separate (fast) ffprobe pass; frame i of the pipe = keyframe i
        pts = keyframe_times(path, start, duration)
    cmd += ["-pix_fmt", pix, "-f", "rawvideo", "-"]
    nbytes = w * h * ch
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=nbytes * 4)
    i = 0
    off = start or 0.0
    try:
        while True:
            buf = p.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            a = np.frombuffer(buf, np.uint8).reshape((h, w, ch) if ch > 1 else (h, w))
            if keyframes_only:
                t = pts[i] if i < len(pts) else (pts[-1] + 5.0 if pts else 0.0)
            else:
                t = off + i / fps
            yield t, a
            i += 1
    finally:
        p.stdout.close()
        err = p.stderr.read().decode("utf-8", "replace")
        rc = p.wait()
        if rc not in (0, None) and i == 0:
            raise RuntimeError(f"ffmpeg failed ({rc}): {err[-500:]}")


def keyframe_times(path, start=None, duration=None):
    cmd = [tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
           "-show_entries", "frame=pts_time", "-of", "csv=p=0"]
    if start is not None or duration is not None:
        s = start or 0.0
        cmd += ["-read_intervals", f"{s:.3f}%" + (f"+{duration:.3f}" if duration is not None else "")]
    cmd.append(str(path))
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    ts = []
    for line in out.split():
        line = line.strip().strip(",")
        if line and line != "N/A":
            ts.append(float(line))
    return ts


def frame_at(path, t, size=(640, 360), pix="rgb24"):
    """Decode a single frame at time t (accurate seek)."""
    w, h = size
    ch = 3 if pix != "gray" else 1
    cmd = [tool("ffmpeg"), "-v", "error", "-nostdin", "-threads", "2", "-ss", f"{max(t, 0):.3f}", "-i", str(path),
           "-frames:v", "1", "-an", "-vf", f"scale={w}:{h}:flags=area", "-pix_fmt", pix, "-f", "rawvideo", "-"]
    buf = subprocess.run(cmd, capture_output=True, check=True).stdout
    if len(buf) < w * h * ch:
        return None
    return np.frombuffer(buf[: w * h * ch], np.uint8).reshape((h, w, ch) if ch > 1 else (h, w))


def fmt_t(t):
    t = int(round(t))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}"


def cli_ids(argv):
    """Parse `[--workers N] [--force] [id ...]` from argv."""
    workers, force, ids, rest = 1, False, [], []
    it = iter(argv)
    for a in it:
        if a == "--workers":
            workers = int(next(it))
        elif a == "--force":
            force = True
        elif a.startswith("--"):
            rest.append(a)
        else:
            ids.append(a)
    return workers, force, ids, rest
