"""Capture: a thread that reads frames, samples them at the detection rate and stamps them with the video PC's wall
clock (epoch seconds). Sources:

* video file: ffmpeg pipe, `framestep` = round(fps / det_fps) (exactly the offline evaluation's sampling; LESSONS
  WP9: never `select`, it duplicates frames), frame time = file start + pts. File start from the OBS clock overlay
  when present (26SE), else the container creation_time, else mtime - duration. `realtime=True` paces frames at real
  speed and drops frames when detection falls behind (as a live source would); otherwise every frame is processed.
* camera index (OpenCV), or rtsp/http/udp URL (ffmpeg pipe, `fps` filter): frame time = system clock at arrival
  minus a configured latency; a full queue drops the frame (never blocks the camera).
"""
import datetime as dt
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Frame:
    img: np.ndarray
    t: float           # wall clock of the video PC (epoch s), incl. clock.offset_s
    pts: float         # seconds into the source (files) or since start (live)
    idx: int           # sampled frame number
    t_cap: float       # time.monotonic() when captured (latency measurement)


def probe(ffprobe, src):
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,avg_frame_rate:format=duration:format_tags=creation_time",
                          "-of", "json", str(src)], capture_output=True, text=True, timeout=60).stdout
    j = json.loads(out or "{}")
    if not j.get("streams"):
        raise SystemExit("ffprobe found no video stream in %s" % src)
    s = j["streams"][0]
    a, b = (s.get("avg_frame_rate") or s["r_frame_rate"]).split("/")
    fps = float(a) / float(b) if float(b) else 0.0
    if fps <= 0 or fps > 240:
        a, b = s["r_frame_rate"].split("/")
        fps = float(a) / float(b)
    fmt = j.get("format", {})
    return dict(w=int(s["width"]), h=int(s["height"]), fps=fps, duration=float(fmt.get("duration", 0) or 0),
                creation=(fmt.get("tags") or {}).get("creation_time"))


def out_size(w, h, max_width):
    if max_width and w > max_width:
        return int(max_width), int(round(h * max_width / w / 2) * 2)
    return w, h


# ------------------------------------------------------------------------------------------------ file clock
def overlay_start(path, log=print):
    """OBS clock overlay (lapvision/clock.py): read at pts 0.5 and 2.5; accepted when both read with a margin and the
    difference is 2 s (+-0.2). Returns the time-of-day at pts 0 (UTC-3 per the overlay), or None."""
    sys.path.insert(0, str(ROOT / "lapvision"))
    try:
        import clock
    except Exception as e:  # noqa: BLE001
        log("clock overlay reader unavailable: %s" % e)
        return None
    r = []
    for t in (0.5, 2.5):
        f = clock.grab(path, t)
        if f is None:
            return None
        s, mg = clock.read(f)
        if s is None or mg < 0.03:
            return None
        r.append(s - t)
    if abs(r[0] - r[1]) > 0.2:
        return None
    return r[0]


def file_start_epoch(path, info, mode, log=print):
    """-> (epoch at pts 0, how)"""
    path = Path(path)
    modes = ["overlay", "creation", "mtime"] if mode == "auto" else [mode]
    for m in modes:
        if m == "overlay":
            tod = overlay_start(path, log)
            if tod is None:
                continue
            md = re.search(r"(\d{4})-(\d{2})-(\d{2})", path.name)
            day = dt.date(*map(int, md.groups())) if md else \
                dt.datetime.fromtimestamp(os.path.getmtime(path) - info["duration"]).date()
            tz = dt.timezone(dt.timedelta(hours=-3))  # the overlay prints UTC-03:00
            return dt.datetime(day.year, day.month, day.day, tzinfo=tz).timestamp() + tod, "overlay"
        if m == "creation" and info.get("creation"):
            c = info["creation"].replace("Z", "+00:00")
            try:
                return dt.datetime.fromisoformat(c).timestamp(), "creation_time"
            except ValueError:
                continue
        if m == "mtime":
            return os.path.getmtime(path) - info["duration"], "mtime-duration"
        if m == "pts":
            return 0.0, "pts"
    if mode == "auto":
        return os.path.getmtime(path) - info["duration"], "mtime-duration"
    raise SystemExit("clock mode %s not available for %s" % (mode, path))


def parse_start(s, base_epoch):
    """'645' / '645.5' = seconds into the file; 'HH:MM[:SS]' = wall time of day (local) -> seconds into the file."""
    if s is None:
        return None
    if ":" not in str(s):
        return float(s)
    p = [float(x) for x in s.split(":")]
    tod = p[0] * 3600 + p[1] * 60 + (p[2] if len(p) > 2 else 0)
    d0 = dt.datetime.fromtimestamp(base_epoch).astimezone()
    midnight = d0.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    return midnight + tod - base_epoch


def set_affinity(pid, cpus, log=print):
    if not cpus:
        return
    try:
        import psutil
        psutil.Process(pid).cpu_affinity(list(cpus))
    except Exception as e:  # noqa: BLE001
        log("WARNING: could not set CPU affinity %s on pid %d: %s" % (cpus, pid, e))


# ------------------------------------------------------------------------------------------------ sources
class Capture(threading.Thread):
    """Common part: the frame queue, drop counting, the end sentinel."""

    def __init__(self, cfg, log=print):
        super().__init__(daemon=True)
        self.cfg, self.log = cfg, log
        c = cfg["capture"]
        self.det_fps = float(c["fps"])
        self.q = queue.Queue(maxsize=int(c["queue_size"]))
        self.dropped = 0
        self.captured = 0
        self.proc = None            # ffmpeg child (CPU accounting)
        self.stop_ev = threading.Event()
        self.error = None
        self.offset = float(cfg["clock"]["offset_s"])
        self.info = {}

    def _push(self, fr, block):
        self.captured += 1
        if block:
            while not self.stop_ev.is_set():
                try:
                    self.q.put(fr, timeout=0.5)
                    return
                except queue.Full:
                    continue
        else:
            try:
                self.q.put_nowait(fr)
            except queue.Full:
                self.dropped += 1

    def _end(self):
        while True:
            try:
                self.q.put(None, timeout=0.5)
                return
            except queue.Full:
                if self.stop_ev.is_set():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass

    def frames(self):
        while True:
            fr = self.q.get()
            if fr is None:
                if self.error:
                    raise RuntimeError("capture failed: %s" % self.error)
                return
            yield fr

    def stop(self):
        self.stop_ev.set()
        if self.proc and self.proc.poll() is None:
            self.proc.kill()

    def run(self):
        try:
            self._run()
        except Exception as e:  # noqa: BLE001
            self.error = "%s: %s" % (type(e).__name__, e)
            self.log("capture error: %s" % self.error)
        finally:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
            self._end()


class FileCapture(Capture):
    def __init__(self, cfg, path, start=None, duration=None, realtime=False, log=print):
        super().__init__(cfg, log)
        c = cfg["capture"]
        self.path = Path(path)
        self.realtime = realtime
        self.info = probe(c["ffprobe"], self.path)
        self.base, self.clock_how = file_start_epoch(self.path, self.info, cfg["clock"]["mode"], log)
        self.ss = max(0.0, parse_start(start, self.base) or 0.0)
        self.duration = float(duration) if duration else None
        self.step = max(1, int(round(self.info["fps"] / self.det_fps)))
        self.w, self.h = out_size(self.info["w"], self.info["h"], int(c["max_width"]))
        self.live = False

    def describe(self):
        i = self.info
        return ("file %s: %dx%d @ %.3g fps, %.0f s; sampling every %d frame(s) = %.2f fps, decoded to %dx%d; clock %s "
                "(pts 0 = %s); start %.1f s%s%s" % (self.path.name, i["w"], i["h"], i["fps"], i["duration"], self.step,
                                                   i["fps"] / self.step, self.w, self.h, self.clock_how,
                                                   fmt_t(self.base), self.ss,
                                                   (", duration %.0f s" % self.duration) if self.duration else "",
                                                   ", paced in real time" if self.realtime else ", as fast as possible"))

    def _run(self):
        c = self.cfg["capture"]
        cmd = [c["ffmpeg"], "-v", "error", "-nostdin"]
        if int(c["decode_threads"]):
            cmd += ["-threads", str(int(c["decode_threads"]))]
        if c.get("hwaccel", "none") not in ("", "none"):
            cmd += ["-hwaccel", c["hwaccel"]]
        if self.ss:
            cmd += ["-ss", "%.3f" % self.ss]
        cmd += ["-i", str(self.path)]
        if self.duration:
            cmd += ["-t", "%.3f" % self.duration]
        vf = "framestep=%d" % self.step
        if (self.w, self.h) != (self.info["w"], self.info["h"]):
            vf += ",scale=%d:%d" % (self.w, self.h)
        cmd += ["-vf", vf, "-fps_mode", "passthrough", "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        n = self.w * self.h * 3
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=n * 2)
        if self.cfg["runtime"].get("affinity_decoder", True):
            set_affinity(self.proc.pid, self.cfg["runtime"].get("affinity"), self.log)
        t0 = time.monotonic()
        k = 0
        while not self.stop_ev.is_set():
            buf = self.proc.stdout.read(n)
            if len(buf) < n:
                break
            pts = self.ss + k * self.step / self.info["fps"]
            if self.realtime:
                due = t0 + (pts - self.ss)
                d = due - time.monotonic()
                if d > 0:
                    time.sleep(d)
            img = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
            self._push(Frame(img, self.base + pts + self.offset, pts, k, time.monotonic()), block=not self.realtime)
            k += 1
        self.proc.stdout.close()
        self.proc.wait()


class StreamCapture(Capture):
    """rtsp/http/udp URL through ffmpeg (or a file with as_live=True: ffmpeg -re, to exercise the live path)."""

    def __init__(self, cfg, url, as_live=False, log=print):
        super().__init__(cfg, log)
        c = cfg["capture"]
        self.url, self.as_live = url, as_live
        self.info = probe(c["ffprobe"], url)
        self.w, self.h = out_size(self.info["w"], self.info["h"], int(c["max_width"]))
        self.live = True
        self.clock_how = "system clock at arrival - %.2f s" % float(c["live_latency_s"])

    def describe(self):
        return "stream %s: %dx%d -> %dx%d @ %.3g fps (fps filter); clock %s" % (
            self.url, self.info["w"], self.info["h"], self.w, self.h, self.det_fps, self.clock_how)

    def _run(self):
        c = self.cfg["capture"]
        cmd = [c["ffmpeg"], "-v", "error", "-nostdin"]
        if int(c["decode_threads"]):
            cmd += ["-threads", str(int(c["decode_threads"]))]
        if c.get("hwaccel", "none") not in ("", "none"):
            cmd += ["-hwaccel", c["hwaccel"]]
        if self.as_live:
            cmd += ["-re"]
        if str(self.url).startswith("rtsp"):
            cmd += ["-rtsp_transport", "tcp"]
        cmd += ["-i", str(self.url), "-vf", "fps=%g,scale=%d:%d" % (self.det_fps, self.w, self.h),
                "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        n = self.w * self.h * 3
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=n * 2)
        if self.cfg["runtime"].get("affinity_decoder", True):
            set_affinity(self.proc.pid, self.cfg["runtime"].get("affinity"), self.log)
        lat = float(c["live_latency_s"])
        t_start, k = time.time(), 0
        while not self.stop_ev.is_set():
            buf = self.proc.stdout.read(n)
            if len(buf) < n:
                break
            now = time.time()
            img = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
            self._push(Frame(img, now - lat + self.offset, now - t_start, k, time.monotonic()), block=False)
            k += 1
        self.proc.stdout.close()
        self.proc.wait()


class CameraCapture(Capture):
    """OpenCV camera index. Frames are kept at det_fps by time (an accumulator, so 30 -> 10 fps keeps every 3rd)."""

    def __init__(self, cfg, index, log=print):
        super().__init__(cfg, log)
        import cv2
        c = cfg["capture"]
        api = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
               "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)}[c["camera_backend"]]
        self.cap = cv2.VideoCapture(int(index), api)
        if not self.cap.isOpened():
            raise SystemExit("cannot open camera %s" % index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(c["camera_width"]))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(c["camera_height"]))
        self.cap.set(cv2.CAP_PROP_FPS, float(c["camera_fps"]))
        self.index = index
        self.live = True
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.info = dict(w=self.w, h=self.h, fps=self.cap.get(cv2.CAP_PROP_FPS))
        self.clock_how = "system clock at arrival - %.2f s" % float(c["live_latency_s"])
        self.max_width = int(c["max_width"])

    def describe(self):
        return "camera %s: %dx%d @ %.3g fps, sampled to %.3g fps; clock %s" % (
            self.index, self.w, self.h, self.info["fps"], self.det_fps, self.clock_how)

    def _run(self):
        import cv2
        lat = float(self.cfg["capture"]["live_latency_s"])
        period = 1.0 / self.det_fps
        t_start = time.time()
        next_t, k = t_start, 0
        while not self.stop_ev.is_set():
            ok, img = self.cap.read()
            if not ok:
                break
            now = time.time()
            if now < next_t:
                continue
            next_t = max(next_t + period, now - period)  # keep the cadence, never burst after a stall
            ow, oh = out_size(img.shape[1], img.shape[0], self.max_width)
            if ow != img.shape[1]:
                img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
            self._push(Frame(img, now - lat + self.offset, now - t_start, k, time.monotonic()), block=False)
            k += 1
        self.cap.release()


def open_source(cfg, source, start=None, duration=None, realtime=False, as_live=False, log=print):
    s = str(source)
    if re.fullmatch(r"\d+", s):
        return CameraCapture(cfg, int(s), log)
    if re.match(r"^(rtsp|rtmp|http|https|udp|tcp|srt)://", s, re.I):
        return StreamCapture(cfg, s, log=log)
    if not Path(s).exists():
        raise SystemExit("source not found: %s" % s)
    if as_live:
        return StreamCapture(cfg, s, as_live=True, log=log)
    return FileCapture(cfg, s, start, duration, realtime, log)


def fmt_t(epoch):
    return dt.datetime.fromtimestamp(epoch).astimezone().isoformat(timespec="milliseconds")
