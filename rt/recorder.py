"""Recording of the camera's video next to the JSONL (WP-L1), so an event can be replayed and scored afterwards.

The capture thread hands each processed frame to `Recorder.offer()`, which never blocks: frames wait in a small
queue (about 1 s of frames; a 1080p BGR frame is ~6 MB), and when it is full the OLDEST frame is dropped from the
recording (counted), never from detection. The recorder thread pipes frames into an ffmpeg process that writes
Matroska (.mkv: still playable if the laptop loses power mid-file).

Encoding: H.264 with the first encoder that works on this machine, probed once at start and logged: Intel
QuickSync (h264_qsv), AMD AMF (h264_amf), then x264 `veryfast` (CPU). ~4 GB per hour at 10 fps on 26SE footage
(x264 CRF 23). On DirectShow the camera frames arrive decoded (BGR), so there are no camera JPEGs to copy: the old
`mode = "mjpeg"` (copy) is refused with an explanation. `recording.mode` is "h264" or "off".
Only the processed frames (10 fps) are recorded: exactly what rt saw.

Segments: a new file every `segment_min` minutes, and also whenever the frames stop for a while (the camera was
lost, or more than `split_gap_periods` frame periods passed between two frames). The file's own frame rate is
nominal (10 fps), so a file must never span a gap: the frames after it would play (and replay) too early. The file
name carries the wall-clock start (rec_2026-11-21_09-15-02.mkv) and has three companions:
  * rec_....json: the start time on both clocks, the camera settings read back at the last (re)open, the software
    version, the encoder. Written when the file opens (marked provisional) and finalised at close with the end
    time, the frame count and why it closed.
  * rec_....frames.csv: one row per recorded frame: n, t_wall, t_mono, camera_frame_no. THIS is the authoritative
    time of every frame (the replay, `run.py --source rec_....mkv`, uses it when it is there). Flushed every
    `csv_flush_rows` rows, so a power cut loses at most a few seconds of it.
  * rec_....ffmpeg.log: ffmpeg's own messages (stderr). A file, not a pipe: an unread pipe fills up and freezes
    ffmpeg and the writer.

Watchdog: if ffmpeg does not accept a frame for `write_timeout_s` seconds, or exits, the segment is marked failed
(status page), ffmpeg is killed, and a new segment is tried after `retry_wait_s`. After `max_failures` failures in a
row recording gives up (status page "failed"); detection never stops.

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
RECORDING_MODES = ("h264", "off")


def check_recording_config(recording_cfg):
    """Refuse settings that cannot work with the OpenCV camera source, with a message that says why."""
    mode = recording_cfg.get("mode", "h264")
    if mode == "mjpeg":
        raise SystemExit(
            "recording.mode = \"mjpeg\" is not possible: on Windows the camera is read through OpenCV's DirectShow "
            "backend, which always decodes the camera's JPEGs, so rt only has decoded images to record and there "
            "is nothing to copy. Use recording.mode = \"h264\" (hardware encoder if the laptop has one, else x264) "
            "or \"off\".")
    if mode not in RECORDING_MODES:
        raise SystemExit("recording.mode = %r: use \"h264\" or \"off\"" % mode)
    if recording_cfg.get("all_frames"):
        raise SystemExit("recording.all_frames was removed: only the processed frames (what rt saw) are recorded. "
                         "Delete the line from your config.")


def probe_encoder(ffmpeg, name, runner=subprocess.run):
    """True if ffmpeg can encode one small frame with this encoder on this machine (hardware encoders are listed
    by ffmpeg even when the GPU or its driver is missing, so listing is not enough)."""
    try:
        result = runner([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=0.2", "-frames:v",
                         "1", "-c:v", name, "-f", "null", "-"], capture_output=True, timeout=30)
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def pick_h264_encoder(ffmpeg, candidates, runner=subprocess.run, tried=None):
    """The first encoder of `candidates` that works here, or None. `tried` (a list) receives (name, worked)."""
    for name in candidates:
        worked = probe_encoder(ffmpeg, name, runner)
        if tried is not None:
            tried.append((name, worked))
        if worked:
            return name
    return None


def segment_name(t_wall):
    return "rec_" + dt.datetime.fromtimestamp(t_wall).strftime("%Y-%m-%d_%H-%M-%S")


def iso(t):
    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


class Segment:
    """One recording file: the ffmpeg process, its log file, the frames CSV and the JSON sidecar."""

    def __init__(self, path, cmd, sidecar, popen, csv_flush_rows=30):
        self.path = Path(path)
        self.sidecar = dict(sidecar, status="open (provisional: rewritten when the file is closed)")
        self.frames = 0
        self.start_mono = sidecar["start_mono"]
        self.last_frame_mono = None
        self.csv_flush_rows = max(1, int(csv_flush_rows))
        self.json_path = self.path.with_suffix(".json")
        self.log_path = self.path.with_suffix(".ffmpeg.log")
        self.csv_f = open(self.path.with_suffix(".frames.csv"), "w", encoding="utf-8", newline="")
        self.csv = csv.writer(self.csv_f)
        self.csv.writerow(["n", "t_wall", "t_mono", "camera_frame_no"])
        self.csv_f.flush()
        self._write_sidecar()
        self.log_f = open(self.log_path, "wb")
        # Watchdog fields (read by Recorder.watchdog_check from another thread):
        self.busy_since = None      # monotonic time a write to ffmpeg started; None = not writing now
        self.failed = None          # why this segment failed (set by the watchdog or the writer), or None
        self.proc = popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log_f)

    def _write_sidecar(self):
        self.json_path.write_text(json.dumps(self.sidecar, indent=1, default=str), encoding="utf-8")

    def write(self, data, t_wall, t_mono, camera_frame_no, now_mono):
        self.busy_since = now_mono
        try:
            self.proc.stdin.write(data.tobytes() if hasattr(data, "tobytes") else data)
        finally:
            self.busy_since = None
        self.csv.writerow([self.frames, "%.6f" % t_wall, "%.6f" % t_mono, camera_frame_no])
        self.frames += 1
        self.last_frame_mono = t_mono
        if self.frames % self.csv_flush_rows == 0:
            self.csv_f.flush()

    def kill(self):
        try:
            self.proc.kill()
        except Exception:  # noqa: BLE001
            pass

    def close(self, t_wall, t_mono, reason, now_mono=None):
        if self.failed:
            self.kill()                         # do not wait for a stuck or dead ffmpeg
        self.busy_since = now_mono              # closing stdin flushes: the watchdog also covers a stuck close
        try:
            self.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        self.busy_since = None
        try:
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.kill()
        self.csv_f.close()
        self.log_f.close()
        self.sidecar.update(status="closed", end_wall=round(t_wall, 6), end_wall_iso=iso(t_wall),
                            end_mono=round(t_mono, 6), frames=self.frames, closed_because=reason,
                            ffmpeg_returncode=self.proc.returncode, ffmpeg_log=self.log_path.name)
        try:
            tail = self.log_path.read_bytes()[-2000:]
        except OSError:
            tail = b""
        if tail:
            self.sidecar["ffmpeg_log_tail"] = tail.decode("utf-8", "replace")
        self._write_sidecar()


class Recorder(threading.Thread):
    """See the module docstring. `out_dir` = the folder for the files; `fps` = the nominal frame rate written into
    the files (the processing rate). Tests inject `clock`, `disk_free_gb` (a function: folder -> GB free), `popen`
    (a fake ffmpeg), `encoder` (skip the encoder probe) and `watchdog_poll_s=0` (no watchdog thread: the test calls
    watchdog_check() itself)."""

    def __init__(self, cfg, out_dir, fps, version="unknown", log=print, clock=None, disk_free_gb=None, popen=None,
                 encoder=None, watchdog_poll_s=1.0):
        super().__init__(daemon=True)
        recording_cfg = cfg["recording"]
        check_recording_config(recording_cfg)
        self.mode = recording_cfg["mode"]
        self.segment_s = float(recording_cfg["segment_min"]) * 60
        self.warn_gb, self.stop_gb = float(recording_cfg["warn_free_gb"]), float(recording_cfg["stop_free_gb"])
        self.disk_check_s = float(recording_cfg.get("disk_check_s", 10))
        self.encoders = list(recording_cfg["h264_encoders"])
        self.split_gap_periods = float(recording_cfg.get("split_gap_periods", 2))
        self.csv_flush_rows = int(recording_cfg.get("csv_flush_rows", 30))
        self.write_timeout_s = float(recording_cfg.get("write_timeout_s", 10))
        self.retry_wait_s = float(recording_cfg.get("retry_wait_s", 10))
        self.max_failures = int(recording_cfg.get("max_failures", 5))
        self.ffmpeg = cfg["capture"]["ffmpeg"]
        self.out_dir = Path(out_dir)
        self.fps = float(fps)
        self.version = version
        self.log = log
        self.clock = clock or SystemClock()
        self.disk_free_gb = disk_free_gb or (lambda folder: shutil.disk_usage(folder).free / 1e9)
        self.popen = popen or subprocess.Popen
        self.encoder = encoder
        self.encoder_probe = []                  # (encoder, worked) from choose_encoder()
        self.q = queue.Queue(maxsize=max(1, int(recording_cfg["queue_frames"])))
        self.camera_info = {}
        self.seg = None
        self.segments = []          # paths of every segment started
        self.frames = self.dropped = self.skipped_low_disk = self.errors = 0
        self.free_gb = None
        self.disk_state = "ok"      # ok | low (warning) | stopped
        self.last_disk_check = None
        self.retry_after = None     # after an ffmpeg failure, wait before opening a new segment
        self.failures_in_a_row = 0
        self.gave_up = False
        self.last_error = ""
        self.close_request = None   # set by camera_lost() (capture thread); acted on by the writer thread
        self.stopping = threading.Event()
        self.watchdog_poll_s = float(watchdog_poll_s)
        self.seg_lock = threading.Lock()         # guards self.seg between the writer and the watchdog

    # ------------------------------------------------------------------ set-up (main thread)
    def choose_encoder(self):
        """Probe the H.264 encoders once (at start, before any frame) and log which one is used."""
        if self.encoder is None:
            self.encoder = pick_h264_encoder(self.ffmpeg, self.encoders, tried=self.encoder_probe)
            tried = ", ".join("%s %s" % (name, "ok" if worked else "failed") for name, worked in self.encoder_probe)
            if self.encoder is None:
                self.encoder = "libx264"
                self.log("WARNING recording: no H.264 encoder passed the probe (%s); trying libx264 anyway" % tried)
            else:
                self.log("recording encoder: %s (probe: %s)" % (self.encoder, tried))
        return self.encoder

    # ------------------------------------------------------------------ called from the capture thread
    def set_camera_info(self, info):
        self.camera_info = dict(info)

    def camera_lost(self, reason="camera lost"):
        """The camera is gone: close the current file (the writer thread does it). The next frame starts a new
        file, with the camera settings read back at the reopen."""
        self.close_request = reason

    def offer(self, data, t_wall, t_mono, camera_frame_no):
        """Queue one frame for recording. Never blocks. When the queue is full the oldest queued frame is dropped
        (counted), so the queue never holds more than queue_frames frames. Returns False if recording gave up."""
        if self.gave_up:
            return False
        item = (data, t_wall, t_mono, camera_frame_no)
        while True:
            try:
                self.q.put_nowait(item)
                return True
            except queue.Full:
                try:
                    self.q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    # ------------------------------------------------------------------ the recorder thread
    def run(self):
        watchdog = None
        if self.watchdog_poll_s > 0:
            watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
            watchdog.start()
        while True:
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                if self.stopping.is_set():
                    break
                self._apply_close_request()
                self._check_failed_segment(self.clock.mono())
                continue
            if item is None:
                break
            self.handle(*item)
        self.close_segment("stopped")

    def stop(self):
        """Write what is queued, close the file. Call once at the end of the run."""
        self.stopping.set()
        try:
            self.q.put(None, timeout=5)
        except queue.Full:
            pass
        self.join(timeout=60)

    def _watchdog_loop(self):
        while not self.stopping.wait(self.watchdog_poll_s):
            self.watchdog_check()

    def watchdog_check(self):
        """Called every second from the watchdog thread (tests call it directly). Marks the segment failed and
        kills ffmpeg if ffmpeg exited, or if a write to it has not returned for write_timeout_s seconds."""
        with self.seg_lock:
            seg = self.seg
        if seg is None or seg.failed:
            return
        return_code = seg.proc.poll()
        if return_code is not None:
            seg.failed = "ffmpeg exited with code %s (see %s)" % (return_code, seg.log_path.name)
        elif seg.busy_since is not None and self.clock.mono() - seg.busy_since > self.write_timeout_s:
            seg.failed = "ffmpeg did not accept data for %.0f s (see %s)" % (self.write_timeout_s,
                                                                           seg.log_path.name)
        if seg.failed:
            self.log("WARNING recording: %s; killing it" % seg.failed)
            seg.kill()          # unblocks a stuck write (it raises BrokenPipeError)

    def _apply_close_request(self):
        reason, self.close_request = self.close_request, None
        if reason:
            self.close_segment(reason)

    def _check_failed_segment(self, t_mono):
        seg = self.seg
        if seg is not None and seg.failed:
            self._segment_failed(seg.failed, t_mono)

    def handle(self, data, t_wall, t_mono, camera_frame_no):
        """Write one frame (the recorder thread calls this; tests call it directly)."""
        self._apply_close_request()
        if self.gave_up:
            return
        self.check_disk(t_mono)
        if self.disk_state == "stopped":
            self.skipped_low_disk += 1
            return
        self._check_failed_segment(t_mono)
        kind = "jpeg" if getattr(data, "ndim", 3) == 1 else "bgr"
        size = None if kind == "jpeg" else (int(data.shape[1]), int(data.shape[0]))
        seg = self.seg
        if seg is not None:
            gap_s = None if seg.last_frame_mono is None else t_mono - seg.last_frame_mono
            if t_mono - seg.start_mono >= self.segment_s:
                self.close_segment("segment time")
            elif (kind, size) != (seg.sidecar["input"], seg.sidecar.get("size")):
                self.close_segment("format change")
            elif gap_s is not None and gap_s > self.split_gap_periods / self.fps:
                self.close_segment("frame gap of %.2f s" % gap_s)
        if self.seg is None:
            if self.retry_after is not None and t_mono < self.retry_after:
                return
            try:
                self.open_segment(kind, size, t_wall, t_mono)
            except Exception as e:  # noqa: BLE001
                self._segment_failed("cannot start a recording segment: %s" % e, t_mono)
                return
        seg = self.seg
        try:
            seg.write(data, t_wall, t_mono, camera_frame_no, self.clock.mono())
        except (BrokenPipeError, OSError, ValueError) as e:
            self._segment_failed(seg.failed or "recording ffmpeg failed: %s" % e, t_mono)
            return
        if seg.failed:                                   # the watchdog fired while this write was stuck
            self._segment_failed(seg.failed, t_mono)
            return
        self.frames += 1
        if seg.frames >= 60 * self.fps:                  # a minute of good writing: the failure streak is over
            self.failures_in_a_row = 0

    def _segment_failed(self, msg, t_mono):
        self.errors += 1
        self.failures_in_a_row += 1
        self.last_error = msg
        self.log("WARNING recording failed: " + msg)
        if self.seg is not None:
            self.seg.failed = self.seg.failed or msg
            self.close_segment("failed: " + msg)
        if self.failures_in_a_row >= self.max_failures:
            self.gave_up = True
            self.log("WARNING recording: %d failures in a row; recording is off for this run (detection goes on)"
                     % self.failures_in_a_row)
            while True:                                  # free the queued frames
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
        else:
            self.retry_after = t_mono + self.retry_wait_s

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
        if kind == "jpeg":           # a backend that passes the camera's JPEGs through (V4L2): ffmpeg decodes them
            cmd += ["-f", "mjpeg", "-framerate", "%g" % self.fps, "-i", "pipe:0"]
        else:
            cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % size, "-framerate", "%g" % self.fps,
                    "-i", "pipe:0"]
        encoder = self.choose_encoder()
        codec = ["-c:v", encoder] + H264_OPTS.get(encoder, []) + ["-g", str(int(round(self.fps * 2)))]
        utc = dt.datetime.fromtimestamp(t_wall, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        cmd += codec + ["-an", "-metadata", "creation_time=" + utc, "-f", "matroska", str(path)]
        return cmd, encoder

    def open_segment(self, kind, size, t_wall, t_mono):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        base = segment_name(t_wall)
        path = self.out_dir / (base + ".mkv")
        suffix_no = 1
        while path.exists():
            path = self.out_dir / ("%s_%d.mkv" % (base, suffix_no))
            suffix_no += 1
        cmd, encoder = self.ffmpeg_cmd(kind, size, path, t_wall)
        sidecar = dict(file=path.name, start_wall=round(t_wall, 6), start_wall_iso=iso(t_wall),
                       start_mono=round(t_mono, 6), nominal_fps=self.fps, input=kind, size=size, encoder=encoder,
                       segment_min=self.segment_s / 60, camera=self.camera_info, software_version=self.version,
                       frames_csv=path.with_suffix(".frames.csv").name,
                       note="frame n's true time is in frames_csv; the file's frame rate is nominal")
        segment = Segment(path, cmd, sidecar, self.popen, self.csv_flush_rows)
        with self.seg_lock:
            self.seg = segment
        self.segments.append(path)
        self.retry_after = None
        self.log("recording -> %s (%s)" % (path, encoder))

    def close_segment(self, reason):
        with self.seg_lock:
            seg, self.seg = self.seg, None
        if seg is None:
            return
        seg.close(self.clock.wall(), self.clock.mono(), reason, self.clock.mono())

    # ------------------------------------------------------------------ status page
    def status(self):
        seg = self.seg
        if self.gave_up:
            state = "failed: gave up after %d ffmpeg failures (last: %s)" % (self.failures_in_a_row, self.last_error)
        elif self.disk_state == "stopped":
            state = "stopped: low disk"
        elif seg is not None and seg.failed:
            state = "error: " + seg.failed
        elif seg is not None:
            state = "recording"
        elif self.retry_after is not None:
            state = "error: %s; retrying" % self.last_error
        else:
            state = "waiting for frames"
        return dict(state=state, file=seg.path.name if seg else None, segment_frames=seg.frames if seg else 0,
                    frames=self.frames, dropped=self.dropped, skipped_low_disk=self.skipped_low_disk,
                    errors=self.errors, failures_in_a_row=self.failures_in_a_row, gave_up=self.gave_up,
                    free_gb=None if self.free_gb is None else round(self.free_gb, 1),
                    disk_state=self.disk_state, warn_free_gb=self.warn_gb, stop_free_gb=self.stop_gb,
                    encoder=self.encoder, folder=str(self.out_dir))


def software_version(root=None):
    """`git describe` of the repo, or "unknown" (a copied folder without .git)."""
    root = root or Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(["git", "-C", str(root), "describe", "--always", "--dirty", "--tags"],
                                capture_output=True, text=True, timeout=10)
        version = result.stdout.strip()
        return version or "unknown"
    except Exception:  # noqa: BLE001
        return os.environ.get("RT_VERSION", "unknown")
