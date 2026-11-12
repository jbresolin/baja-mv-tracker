"""Live USB camera capture for rt (WP-L1): grab every frame, retrieve only the processed ones, stamp each frame on two
clocks, record, and survive the camera disappearing.

Per grabbed frame (30 fps from the camera):
  1. grab() (no decode) and read both clocks right away: t_wall (time.time(), the clock the lap app and the JSONL use)
     and t_mono (time.monotonic(), never jumps; the beam input of WP-L2 stamps its edges on the same two clocks).
  2. The Decimator says whether this frame is processed (10 fps by default). Frames that are not processed are never
     retrieved, so never decoded.
  3. retrieve(): raw JPEG bytes if the camera's MJPEG passthrough works, else a decoded image.
  4. The recorder (rt/recorder.py) gets the frame (JPEG bytes are written as they are, no re-encoding).
  5. Processed frames are decoded (if raw) and queued for detection with t = t_wall - live_latency + offset.
Once a minute a `clock_map` event logs the pair (t_wall, t_mono) of a grab, so the two clocks can be related later.

If the camera disappears (grab() keeps failing) or cannot be opened, the loop releases it, waits with backoff (1 s,
doubling, max 30 s) and opens it again. The process stays alive; the status page shows the camera state.
"""
import cv2

from .backoff import Backoff, SystemClock
from .camsource import CameraError, OpenCvCamera, is_jpeg
from .capture import Capture, Frame, out_size


class Decimator:
    """Picks which frames are processed: about `fps` per second of grab time.

    It keeps a grid of due times (0, 0.1, 0.2, ... s at 10 fps). A frame is processed when its time reaches the next
    due time minus a tolerance, which absorbs the jitter of USB frame times. The tolerance is 40% of the camera's
    frame interval (13 ms at 30 fps), never more than a quarter of the processing period, so a frame that arrives a
    whole camera frame early is never taken. At exactly 30 fps this is every 3rd frame. After a stall longer than
    one period the grid restarts at the next frame instead of processing a burst of late frames."""

    def __init__(self, fps, source_fps=30.0):
        self.period = 1.0 / float(fps)
        self.tol = min(0.25 * self.period, 0.4 / max(float(source_fps or 30.0), 1.0))
        self.next_due = None

    def due(self, t):
        if self.next_due is None:
            self.next_due = t + self.period
            return True
        if t < self.next_due - self.tol:
            return False
        if t - self.next_due > self.period:   # stall: restart the grid here
            self.next_due = t + self.period
        else:
            self.next_due += self.period
        return True


def cc_fps(cam_cfg):
    return float(cam_cfg.get("fps", 30) or 30)


class ClockMapper:
    """Every `period_s` (of monotonic time) returns a `clock_map` event relating the two clocks of one grab."""

    def __init__(self, period_s=60.0):
        self.period = float(period_s)
        self.last = None

    def maybe(self, t_wall, t_mono, frame_no):
        if self.last is not None and t_mono - self.last < self.period:
            return None
        self.last = t_mono
        return dict(type="clock_map", grab_wall=round(t_wall, 6), grab_mono=round(t_mono, 6),
                    wall_minus_mono=round(t_wall - t_mono, 6), grab_no=frame_no)


class CameraCapture(Capture):
    """A live camera (rt/camsource.py) as an rt capture source. `camera_factory` builds the camera object (default:
    OpenCvCamera from the [camera] config); tests pass a FakeCamera and a FakeClock. `recorder` (optional) receives
    the frames to record; `emit` (optional) receives `clock_map` and camera events."""

    def __init__(self, cfg, device=None, log=print, camera_factory=None, clock=None, recorder=None, emit=None,
                 sleep=None):
        super().__init__(cfg, log)
        cc = cfg["camera"]
        self.cam_cfg = cc
        self.device = cc.get("device", 0) if device is None else device
        self.clock = clock or SystemClock()
        self.factory = camera_factory or (lambda: OpenCvCamera(cc, self.device, cfg["capture"]["ffmpeg"], log))
        self.recorder = recorder
        self.emit = emit
        self.wait = sleep or self.stop_ev.wait     # tests replace it to avoid real sleeps
        self.live = True
        self.latency = float(cfg["capture"]["live_latency_s"])
        self.max_width = int(cfg["capture"]["max_width"])
        self.decimator = Decimator(self.det_fps, cc_fps(cc))
        self.clockmap = ClockMapper(float(cc.get("clock_map_every_s", 60)))
        self.backoff = Backoff(float(cc.get("reopen_first_s", 1)), 2.0, float(cc.get("reopen_max_s", 30)))
        self.lost_after = int(cc.get("lost_after_failed_grabs", 30))
        self.state, self.state_detail, self.state_since = "starting", "", self.clock.wall()
        self.cam_info = {}
        self.raw = False
        self.grabbed = self.retrieved = self.corrupt = self.reopens = 0
        self.w = self.h = None
        self.info = dict(device=str(self.device))
        self.clock_how = "system clock at grab - %.2f s" % self.latency

    # ------------------------------------------------------------------ status
    def _set_state(self, state, detail=""):
        if (state, detail) != (self.state, self.state_detail):
            self.state, self.state_detail, self.state_since = state, detail, self.clock.wall()
            if self.emit:
                self.emit(dict(type="camera", state=state, detail=detail))

    def status(self):
        return dict(state=self.state, detail=self.state_detail, since=self.state_since, device=str(self.device),
                    info=self.cam_info, raw_mjpeg=self.raw, grabbed=self.grabbed, retrieved=self.retrieved,
                    processed=self.captured, dropped=self.dropped, corrupt=self.corrupt, reopens=self.reopens)

    def describe(self):
        return ("camera %s (%s): opened by the capture thread; processing %.3g fps via grab()/retrieve(); clock %s" %
                (self.device, self.cam_cfg.get("backend", "auto"), self.det_fps, self.clock_how))

    # ------------------------------------------------------------------ loop
    def _run(self):
        t_start = self.clock.wall()
        k = 0
        while not self.stop_ev.is_set():
            cam = self.factory()
            try:
                self._set_state("opening")
                self.cam_info = cam.open()
            except CameraError as e:
                cam.release()
                wait = self.backoff.next()
                self._set_state("retrying", "%s; next try in %.0f s" % (e, wait))
                self.log("camera: %s; retrying in %.0f s" % (e, wait))
                self.wait(wait)
                continue
            self.raw = bool(self.cam_info.get("raw_accepted"))
            self.decimator = Decimator(self.det_fps, self.cam_info.get("fps") or cc_fps(self.cam_cfg))
            self.info = dict(w=self.cam_info.get("width"), h=self.cam_info.get("height"), fps=self.cam_info.get("fps"))
            self.log("camera open: " + ", ".join("%s=%s" % (key, v) for key, v in self.cam_info.items()
                                                 if key != "settings"))
            for name, s in (self.cam_info.get("settings") or {}).items():
                self.log("camera setting %s: asked %s, accepted %s, reads back %s" % (
                    name, s["requested"], s["accepted"], s["read_back"]))
            if self.recorder is not None:
                self.recorder.set_camera_info(self.cam_info)
            self._set_state("streaming")
            k = self._stream(cam, t_start, k)
            cam.release()
            if self.stop_ev.is_set():
                break
            self.reopens += 1
            wait = self.backoff.next()
            self._set_state("lost", "no frames from the camera; reopening in %.0f s" % wait)
            self.log("camera lost after %d grabs; reopening in %.0f s" % (self.grabbed, wait))
            self.wait(wait)
        self._set_state("stopped")

    def _stream(self, cam, t_start, k):
        """Grab until the camera stops delivering. Returns the next processed-frame number."""
        fails = 0
        first = True
        while not self.stop_ev.is_set():
            if not cam.grab():
                fails += 1
                if fails >= self.lost_after:   # 30 x 50 ms = 1.5 s without a frame: the camera is gone
                    return k
                self.wait(0.05)
                continue
            fails = 0
            t_wall, t_mono = self.clock.wall(), self.clock.mono()
            n = self.grabbed
            self.grabbed += 1
            ev = self.clockmap.maybe(t_wall, t_mono, n)
            if ev and self.emit:
                self.emit(ev)
            process = self.decimator.due(t_mono)
            record_all = self.recorder is not None and self.recorder.wants_all_frames() and self.raw
            if not (process or record_all):
                continue                       # never retrieved: no copy, no decode
            data = cam.retrieve()
            self.retrieved += 1
            if data is None:
                self.corrupt += 1
                continue
            jpeg = data if is_jpeg(data) else None
            if first:
                first = False
                self.backoff.reset()           # the camera works again: next failure starts at the short wait
                if self.raw and jpeg is None:  # the backend accepted raw mode but does not deliver JPEG bytes
                    self.raw = False
                    if data.ndim != 3:         # undecodable bytes: ask OpenCV to decode, drop this frame
                        cam.disable_raw()
                        self.log("camera: raw mode returned non-JPEG data; switched to decoded frames")
                        continue
                    self.log("camera: raw MJPEG passthrough not supported by this backend; frames are decoded by "
                             "OpenCV and recordings are re-encoded")
            img = None if jpeg is not None else data
            if self.recorder is not None and (process or jpeg is not None):
                self.recorder.offer(jpeg if jpeg is not None else img, t_wall, t_mono, n)
            if not process:
                continue
            if img is None:
                img = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
                if img is None:
                    self.corrupt += 1
                    continue
            if self.w is None:
                self.w, self.h = out_size(img.shape[1], img.shape[0], self.max_width)
            ow, oh = out_size(img.shape[1], img.shape[0], self.max_width)
            if ow != img.shape[1]:
                img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
            self._push(Frame(img, t_wall - self.latency + self.offset, t_wall - t_start, k, t_mono), block=False)
            k += 1
        return k
