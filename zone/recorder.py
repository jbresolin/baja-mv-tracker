"""Spotter recording, without opening the webcam twice: the frames the tracker already grabbed are piped to an
ffmpeg encoder in a background thread.

* The file name carries the wall-clock start time: `spotter_YYYYmmdd-HHMMSS.mkv`. Matroska, like the PTZ recording,
  because it stays readable up to the last written cluster when the process dies or the laptop loses power; an mp4
  is unreadable until ffmpeg writes its index at the very end.
* A sidecar `spotter_YYYYmmdd-HHMMSS.frames.csv` lists every written frame: index, grab wall time, grab monotonic
  time. That's what lines the video up with the control log, and what the replay harness reads (`--frames`).
* The encoder never slows the tracker down: frames go through a bounded queue, and when it's full the frame is
  dropped and counted (shown on the page).

The frames are the processed ones (rt.capture.fps, 10 fps by default), which is also what the replay needs.
"""
import csv
import datetime as dt
import json
import queue
import subprocess
import threading
from pathlib import Path


def stamp(epoch):
    return dt.datetime.fromtimestamp(epoch).strftime("%Y%m%d-%H%M%S")


class SpotterRecorder:
    def __init__(self, out_dir, fps, size, t_wall0, ffmpeg="ffmpeg", codec_args=(), queue_size=30, log=print,
                 popen=subprocess.Popen):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps, self.size = float(fps), (int(size[0]), int(size[1]))
        base = "spotter_%s" % stamp(t_wall0)
        self.path = self.out_dir / (base + ".mkv")
        self.frames_path = self.out_dir / (base + ".frames.csv")
        self.meta_path = self.out_dir / (base + ".json")
        self.q = queue.Queue(maxsize=int(queue_size))
        self.dropped = 0
        self.written = 0
        self.error = None
        self.log = log
        w, h = self.size
        cmd = [ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h),
               "-r", "%g" % self.fps, "-i", "-"] + list(codec_args) + ["-f", "matroska", str(self.path)]
        self.cmd = cmd
        self.meta_path.write_text(json.dumps(dict(start_wall=t_wall0, start_iso=dt.datetime.fromtimestamp(
            t_wall0).astimezone().isoformat(timespec="milliseconds"), fps=self.fps, size=list(self.size),
            ffmpeg=cmd, frames_csv=self.frames_path.name), indent=1), encoding="utf-8")
        self.proc = popen(cmd, stdin=subprocess.PIPE)
        self._csv_f = open(self.frames_path, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow(["frame", "t_wall", "t_mono", "src_idx"])
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @property
    def alive(self):
        return self.error is None and self.proc.poll() is None

    def write(self, frame):
        """frame: a ZoneFrame. Never blocks."""
        if frame.img.shape[1] != self.size[0] or frame.img.shape[0] != self.size[1]:
            self.dropped += 1
            return
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while True:
            fr = self.q.get()
            if fr is None:
                break
            try:
                self.proc.stdin.write(fr.img.tobytes())
            except Exception as e:  # noqa: BLE001 (ffmpeg died: stop recording, keep tracking)
                self.error = "%s: %s" % (type(e).__name__, e)
                self.log("spotter recording stopped: %s" % self.error)
                break
            self._csv.writerow([self.written, "%.3f" % fr.t_wall, "%.4f" % fr.t, fr.idx])
            self.written += 1
            if self.written % 50 == 0:
                self._csv_f.flush()

    def close(self):
        while self.thread.is_alive():
            try:
                self.q.put(None, timeout=0.5)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=10)
        self._csv_f.close()
        try:
            self.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=20)
        except Exception:  # noqa: BLE001
            self.proc.kill()

    def status(self):
        return dict(file=self.path.name, written=self.written, dropped=self.dropped, alive=self.alive,
                    error=self.error)
