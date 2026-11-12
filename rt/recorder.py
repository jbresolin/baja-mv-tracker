"""Recording of the camera's video next to the JSONL (WP-L1), so an event can be replayed and scored afterwards.

The capture thread hands each frame to `Recorder.offer()`, which never blocks: frames wait in a bounded queue, and a
full queue drops the frame from the recording (counted), never from detection. The recorder thread pipes frames into
an ffmpeg process that writes Matroska (.mkv: still playable if the laptop loses power mid-file).

`recording.mode`:
  * "h264" (default): H.264 with the first encoder that works on this machine: Intel QuickSync (h264_qsv), AMD AMF
    (h264_amf), then x264 `veryfast` (CPU). ~4 GB per hour at 10 fps on 26SE footage. ffmpeg gets the camera's JPEG
    bytes (small in RAM and in the pipe) and decodes them itself, or raw pixels when the camera backend decodes.
  * "mjpeg": the camera's JPEG bytes are copied into the file as they are (`-c:v copy`): no encode, no decode, no
    CPU; but 26SE-like scenes (grass, trees) are 600-800 KB per 1080p JPEG = 20-30 GB per hour at 10 fps, too much
    for a race day on a laptop disk. If the backend only gives decoded frames, this falls back to H.264.
  * "off".
Only the processed frames (10 fps) are recorded: exactly what rt saw. `recording.all_frames = true` records every
camera frame (30 fps) when the camera passes MJPEG through (3x the disk).

Segments: a new file every `segment_min` minutes. The file name carries the wall-clock start
(rec_2026-11-21_09-15-02.mkv) and has two companions:
  * rec_....json: the start time on both clocks, the camera settings read back, the software version, the encoder;
    rewritten at close with the end time and the frame count.
  * rec_....frames.csv: one row per recorded frame: n, t_wall, t_mono, grab_no. The video's own frame rate is
    nominal; this file holds the true time of every frame.

Disk guard: under `warn_free_gb` free the status page warns; under `stop_free_gb` recording stops (detection goes on)
and resumes once free space is back above `warn_free_gb`.
"""
import csv
import datetime as dt
import json
import os
import queue
import shutil
import subprocess
import threading
from pathlib import Path

from .backoff import SystemClock

H264_OPTS = {
    "h264_qsv": ["-preset", "veryfast", "-global_quality", "25"],
    "h264_amf": ["-quality", "speed", "-rc", "cqp", "-qp_i", "23", "-qp_p", "25"],
    "libx264": ["-preset", "veryfast", "-crf", "23"],
}


def probe_encoder(ffmpeg, name, runner=subprocess.run):
    """True if ffmpeg can encode one small frame with this encoder on this machine (hardware encoders are listed
    by ffmpeg even when the GPU or its driver is missing, so listing is not enough)."""
    try:
        r = runner([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=0.2", "-frames:v", "1",
                    "-c:v", name, "-f", "null", "-"], capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def pick_h264_encoder(ffmpeg, candidates, runner=subprocess.run):
    for name in candidates:
        if probe_encoder(ffmpeg, name, runner):
            return name
    return None


def segment_name(t_wall):
    return "rec_" + dt.datetime.fromtimestamp(t_wall).strftime("%Y-%m-%d_%H-%M-%S")


def iso(t):
    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


class Segment:
    """One recording file: the ffmpeg process, the frames CSV and the JSON sidecar."""

    def __init__(self, path, cmd, sidecar, popen):
        self.path = Path(path)
        self.sidecar = dict(sidecar)
        self.frames = 0
        self.start_mono = sidecar["start_mono"]
        self.json_path = self.path.with_suffix(".json")
        self.csv_f = open(self.path.with_suffix(".frames.csv"), "w", encoding="utf-8", newline="")
        self.csv = csv.writer(self.csv_f)
        self.csv.writerow(["n", "t_wall", "t_mono", "grab_no"])
        self._write_sidecar()
        self.proc = popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def _write_sidecar(self):
        self.json_path.write_text(json.dumps(self.sidecar, indent=1, default=str), encoding="utf-8")

    def write(self, data, t_wall, t_mono, grab_no):
        self.proc.stdin.write(data.tobytes() if hasattr(data, "tobytes") else data)
        self.csv.writerow([self.frames, "%.6f" % t_wall, "%.6f" % t_mono, grab_no])
        self.frames += 1

    def close(self, t_wall, t_mono, reason):
        try:
            self.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        err = b""
        try:
            self.proc.wait(timeout=30)
            if self.proc.stderr is not None:
                err = self.proc.stderr.read() or b""
        except Exception:  # noqa: BLE001
            self.proc.kill()
        self.csv_f.close()
        self.sidecar.update(end_wall=round(t_wall, 6), end_wall_iso=iso(t_wall), end_mono=round(t_mono, 6),
                            frames=self.frames, closed_because=reason, ffmpeg_returncode=self.proc.returncode)
        if err:
            self.sidecar["ffmpeg_stderr"] = err.decode("utf-8", "replace")[-2000:]
        self._write_sidecar()


class Recorder(threading.Thread):
    """See the module docstring. `out_dir` = the folder for the files; `fps` = the nominal frame rate written into
    the files (the processing rate, or the camera rate with all_frames). Tests inject `clock`, `disk_free_gb`
    (a function: folder -> GB free), `popen` (a fake ffmpeg) and `encoder` (skip the encoder probe)."""

    def __init__(self, cfg, out_dir, fps, version="unknown", log=print, clock=None, disk_free_gb=None, popen=None,
                 encoder=None):
        super().__init__(daemon=True)
        r = cfg["recording"]
        self.mode = r["mode"]                      # "auto" | "h264" | "off"
        self.all_frames = bool(r["all_frames"])
        self.segment_s = float(r["segment_min"]) * 60
        self.warn_gb, self.stop_gb = float(r["warn_free_gb"]), float(r["stop_free_gb"])
        self.disk_check_s = float(r.get("disk_check_s", 10))
        self.encoders = list(r["h264_encoders"])
        self.ffmpeg = cfg["capture"]["ffmpeg"]
        self.out_dir = Path(out_dir)
        self.fps = float(fps)
        self.version = version
        self.log = log
        self.clock = clock or SystemClock()
        self.disk_free_gb = disk_free_gb or (lambda p: shutil.disk_usage(p).free / 1e9)
        self.popen = popen or subprocess.Popen
        self.encoder = encoder
        self.q = queue.Queue(maxsize=int(r["queue_frames"]))
        self.camera_info = {}
        self.seg = None
        self.segments = []          # paths of every segment started
        self.frames = self.dropped = self.skipped_low_disk = self.errors = 0
        self.free_gb = None
        self.disk_state = "ok"      # ok | low (warning) | stopped
        self.last_disk_check = None
        self.retry_after = None     # after an ffmpeg failure, wait before opening a new segment
        self.last_error = ""
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ called from the capture thread
    def wants_all_frames(self):
        return self.all_frames

    def set_camera_info(self, info):
        self.camera_info = dict(info)

    def offer(self, data, t_wall, t_mono, grab_no):
        """Queue one frame for recording. Never blocks; returns False if the frame was dropped."""
        try:
            self.q.put_nowait((data, t_wall, t_mono, grab_no))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    # ------------------------------------------------------------------ the recorder thread
    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            self.handle(*item)
        self.close_segment("stopped")

    def stop(self):
        """Write what is queued, close the file. Call once at the end of the run."""
        self.q.put(None)
        self.join(timeout=60)

    def handle(self, data, t_wall, t_mono, grab_no):
        """Write one frame (the recorder thread calls this; tests call it directly)."""
        self.check_disk(t_mono)
        if self.disk_state == "stopped":
            self.skipped_low_disk += 1
            return
        kind = "jpeg" if getattr(data, "ndim", 3) == 1 else "bgr"
        size = None if kind == "jpeg" else (int(data.shape[1]), int(data.shape[0]))
        if self.seg is not None and (t_mono - self.seg.start_mono >= self.segment_s
                                     or (kind, size) != (self.seg.sidecar["input"], self.seg.sidecar.get("size"))):
            self.close_segment("segment time" if t_mono - self.seg.start_mono >= self.segment_s else "format change")
        if self.seg is None:
            if self.retry_after is not None and t_mono < self.retry_after:
                return
            try:
                self.open_segment(kind, size, t_wall, t_mono)
            except Exception as e:  # noqa: BLE001
                self._fail("cannot start a recording segment: %s" % e, t_mono)
                return
        try:
            self.seg.write(data, t_wall, t_mono, grab_no)
            self.frames += 1
        except (BrokenPipeError, OSError, ValueError) as e:
            self._fail("recording ffmpeg failed: %s" % e, t_mono)
            self.close_segment("ffmpeg error")

    def _fail(self, msg, t_mono):
        self.errors += 1
        self.last_error = msg
        self.retry_after = t_mono + 10.0
        self.log("WARNING " + msg)

    def check_disk(self, t_mono):
        if self.last_disk_check is not None and t_mono - self.last_disk_check < self.disk_check_s:
            return
        self.last_disk_check = t_mono
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.free_gb = float(self.disk_free_gb(self.out_dir))
        except Exception as e:  # noqa: BLE001
            self.log("WARNING free disk check failed: %s" % e)
            return
        before = self.disk_state
        if self.free_gb < self.stop_gb:
            self.disk_state = "stopped"
        elif self.disk_state == "stopped" and self.free_gb < self.warn_gb:
            self.disk_state = "stopped"     # hysteresis: resume only above the warning level
        elif self.free_gb < self.warn_gb:
            self.disk_state = "low"
        else:
            self.disk_state = "ok"
        if self.disk_state != before:
            self.log("%s recording disk: %.1f GB free -> %s" % ("WARNING" if self.disk_state != "ok" else "INFO",
                                                                self.free_gb, self.disk_state))
            if self.disk_state == "stopped":
                self.close_segment("low disk")

    def ffmpeg_cmd(self, kind, size, path, t_wall):
        cmd = [self.ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-y"]
        if kind == "jpeg":
            cmd += ["-f", "mjpeg", "-framerate", "%g" % self.fps, "-i", "pipe:0"]
        else:
            cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % size, "-framerate", "%g" % self.fps,
                    "-i", "pipe:0"]
        if kind == "jpeg" and self.mode != "h264":
            codec, enc = ["-c:v", "copy"], "copy (MJPEG)"
        else:
            if self.encoder is None:
                self.encoder = pick_h264_encoder(self.ffmpeg, self.encoders) or "libx264"
            enc = self.encoder
            codec = ["-c:v", enc] + H264_OPTS.get(enc, []) + ["-g", str(int(round(self.fps * 2)))]
        utc = dt.datetime.fromtimestamp(t_wall, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        cmd += codec + ["-an", "-metadata", "creation_time=" + utc, "-f", "matroska", str(path)]
        return cmd, enc

    def open_segment(self, kind, size, t_wall, t_mono):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        base = segment_name(t_wall)
        path = self.out_dir / (base + ".mkv")
        n = 1
        while path.exists():
            path = self.out_dir / ("%s_%d.mkv" % (base, n))
            n += 1
        cmd, enc = self.ffmpeg_cmd(kind, size, path, t_wall)
        sidecar = dict(file=path.name, start_wall=round(t_wall, 6), start_wall_iso=iso(t_wall),
                       start_mono=round(t_mono, 6), nominal_fps=self.fps, all_frames=self.all_frames, input=kind,
                       size=size, encoder=enc, segment_min=self.segment_s / 60, camera=self.camera_info,
                       software_version=self.version, frames_csv=path.with_suffix(".frames.csv").name,
                       note="frame n's true time is in frames_csv; the file's frame rate is nominal")
        self.seg = Segment(path, cmd, sidecar, self.popen)
        self.segments.append(path)
        self.retry_after = None
        self.log("recording -> %s (%s)" % (path, enc))

    def close_segment(self, reason):
        if self.seg is None:
            return
        seg, self.seg = self.seg, None
        seg.close(self.clock.wall(), self.clock.mono(), reason)

    # ------------------------------------------------------------------ status page
    def status(self):
        seg = self.seg
        if self.disk_state == "stopped":
            state = "stopped: low disk"
        elif seg is not None:
            state = "recording"
        elif self.retry_after is not None:
            state = "error: " + self.last_error
        else:
            state = "waiting for frames"
        return dict(state=state, file=seg.path.name if seg else None, segment_frames=seg.frames if seg else 0,
                    frames=self.frames, dropped=self.dropped, skipped_low_disk=self.skipped_low_disk,
                    errors=self.errors, free_gb=None if self.free_gb is None else round(self.free_gb, 1),
                    disk_state=self.disk_state, warn_free_gb=self.warn_gb, stop_free_gb=self.stop_gb,
                    encoder=seg.sidecar["encoder"] if seg else None, folder=str(self.out_dir))


def software_version(root=None):
    """`git describe` of the repo, or "unknown" (a copied folder without .git)."""
    root = root or Path(__file__).resolve().parent.parent
    try:
        r = subprocess.run(["git", "-C", str(root), "describe", "--always", "--dirty", "--tags"],
                           capture_output=True, text=True, timeout=10)
        v = r.stdout.strip()
        return v or "unknown"
    except Exception:  # noqa: BLE001
        return os.environ.get("RT_VERSION", "unknown")
