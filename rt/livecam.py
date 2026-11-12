"""Live USB camera capture for rt (WP-L1): read the camera, stamp each frame on two clocks, keep ~10 fps for
detection, record them, and survive the camera disappearing.

Two read loops, chosen per backend by what its grab() really does (rt/camsource.py explains why):

  * "retrieve_every_frame" (DirectShow, the Windows default, and any backend not verified otherwise):
      grab() returns at once, so it cannot pace the loop; retrieve() waits for the next frame (up to ~1 s) and
      decodes it. So every camera frame is retrieved (30 fps: the decode is paid for every frame anyway), stamped
      right after retrieve() returns, and the Decimator then keeps ~10 fps; the other frames are simply dropped.
      One grab() + one retrieve() per camera frame: no busy loop.
  * "grab_then_retrieve" (V4L2, and the FakeCamera of the tests): grab() waits for the next frame, so every frame is
      grabbed and stamped at grab(), and only the frames the Decimator keeps are retrieved (decoded).

Per camera frame:
  1. Read both clocks: t_wall (time.time(), the clock the lap app and the JSONL use) and t_mono (time.monotonic(),
     never jumps; the beam input of WP-L2 stamps its edges on the same two clocks).
  2. The Decimator says whether this frame is processed (10 fps by default).
  3. Processed frames go to the recorder (rt/recorder.py) and, decoded, to detection with
     t = t_wall - live_latency + offset.
Once a minute a `clock_map` event logs the pair (t_wall, t_mono) of one frame, so the two clocks can be related.

Camera lost: DirectShow reports an unplug only once (one grab() returns False) and then its retrieve() just times
out, so a lost camera is recognised by time, not by an error: no good frame for `camera.lost_after_s` seconds
(default 3) on the monotonic clock, or `camera.lost_after_failures` failed reads in a row. Then the state becomes
"lost" (status page), the recorder closes its file, the camera is released and opened again with backoff (1 s,
doubling, max 30 s) until it is back. The process stays alive the whole time.
"""
import cv2

from .backoff import Backoff, SystemClock
from .camsource import CameraError, OpenCvCamera, is_jpeg
from .capture import Capture, Frame, out_size

READ_MODES = ("auto", "retrieve_every_frame", "grab_then_retrieve")
FAILED_READ_WAIT_S = 0.05      # pause after a failed read, so a backend that fails instantly cannot spin a core


class Decimator:
    """Picks which frames are processed: about `fps` per second of frame time.

    It keeps a grid of due times (0, 0.1, 0.2, ... s at 10 fps). A frame is processed when its time reaches the next
    due time minus a tolerance, which absorbs the jitter of USB frame times. The tolerance is 40% of the camera's
    frame interval (13 ms at 30 fps), never more than a quarter of the processing period, so a frame that arrives a
    whole camera frame early is never taken. At exactly 30 fps this is every 3rd frame. After a stall longer than
    one period the grid restarts at the next frame instead of processing a burst of late frames."""

    def __init__(self, fps, source_fps=30.0):
        self.period = 1.0 / float(fps)
        self.tolerance = min(0.25 * self.period, 0.4 / max(float(source_fps or 30.0), 1.0))
        self.next_due = None

    def due(self, t):
        if self.next_due is None:
            self.next_due = t + self.period
            return True
        if t < self.next_due - self.tolerance:
            return False
        if t - self.next_due > self.period:   # stall: restart the grid here
            self.next_due = t + self.period
        else:
            self.next_due += self.period
        return True


def configured_camera_fps(camera_cfg):
    return float(camera_cfg.get("fps", 30) or 30)


class ClockMapper:
    """Every `period_s` (of monotonic time) returns a `clock_map` event relating the two clocks of one frame."""

    def __init__(self, period_s=60.0):
        self.period = float(period_s)
        self.last = None

    def maybe(self, t_wall, t_mono, camera_frame_no, stamped_at="grab"):
        if self.last is not None and t_mono - self.last < self.period:
            return None
        self.last = t_mono
        return dict(type="clock_map", frame_wall=round(t_wall, 6), frame_mono=round(t_mono, 6),
                    wall_minus_mono=round(t_wall - t_mono, 6), camera_frame_no=camera_frame_no,
                    stamped_at=stamped_at)


class CameraCapture(Capture):
    """A live camera (rt/camsource.py) as an rt capture source. `camera_factory` builds the camera object (default:
    OpenCvCamera from the [camera] config); tests pass a fake camera and a FakeClock. `recorder` (optional) receives
    the frames to record; `emit` (optional) receives `clock_map` and camera events; `sleep` (optional) replaces the
    real waits (tests make it advance the fake clock instead)."""

    def __init__(self, cfg, device=None, log=print, camera_factory=None, clock=None, recorder=None, emit=None,
                 sleep=None):
        super().__init__(cfg, log)
        camera_cfg = cfg["camera"]
        self.cam_cfg = camera_cfg
        self.device = camera_cfg.get("device", 0) if device is None else device
        self.clock = clock or SystemClock()
        self.factory = camera_factory or (lambda: OpenCvCamera(camera_cfg, self.device, cfg["capture"]["ffmpeg"],
                                                               log))
        self.recorder = recorder
        self.emit = emit
        self.wait = sleep or self.stop_ev.wait     # tests replace it to avoid real sleeps
        self.live = True
        self.latency = float(cfg["capture"]["live_latency_s"])
        self.max_width = int(cfg["capture"]["max_width"])
        self.read_mode_setting = str(camera_cfg.get("read_mode", "auto"))
        if self.read_mode_setting not in READ_MODES:
            raise SystemExit("camera.read_mode = %r: use one of %s" % (self.read_mode_setting, ", ".join(READ_MODES)))
        self.read_mode = None                      # resolved when the camera opens
        self.decimator = Decimator(self.det_fps, configured_camera_fps(camera_cfg))
        self.clockmap = ClockMapper(float(camera_cfg.get("clock_map_every_s", 60)))
        self.backoff = Backoff(float(camera_cfg.get("reopen_first_s", 1)), 2.0, float(camera_cfg.get("reopen_max_s",
                                                                                                    30)))
        self.lost_after_s = float(camera_cfg.get("lost_after_s", 3.0))
        self.lost_after_failures = int(camera_cfg.get("lost_after_failures", 10))
        self.state, self.state_detail, self.state_since = "starting", "", self.clock.wall()
        self.cam_info = {}
        self.raw = False
        self.ever_streamed = False
        # Counters (status page / stats). camera_frames = frames the camera delivered (successful retrieves in
        # retrieve_every_frame mode, successful grabs in grab_then_retrieve mode).
        self.camera_frames = self.grab_calls = self.retrieve_calls = 0
        self.failed_reads = self.corrupt = self.reopens = 0
        # Open attempts (first open and every reopen): the launcher (rt/supervise.py) watches this counter while the
        # camera is lost. If it stops going up, the reopen loop itself is stuck (e.g. an open() that never returns)
        # and only a restart of rt can help.
        self.open_attempts = 0
        self.last_attempt_mono = None
        self.last_good_mono = None
        self.w = self.h = None
        self.info = dict(device=str(self.device))
        self.clock_how = "system clock when the frame is received - %.2f s" % self.latency

    # ------------------------------------------------------------------ status
    def _set_state(self, state, detail=""):
        if (state, detail) != (self.state, self.state_detail):
            self.state, self.state_detail, self.state_since = state, detail, self.clock.wall()
            if self.emit:
                self.emit(dict(type="camera", state=state, detail=detail))

    def status(self):
        age = None if self.last_good_mono is None else round(self.clock.mono() - self.last_good_mono, 2)
        return dict(state=self.state, detail=self.state_detail, since=self.state_since, device=str(self.device),
                    info=self.cam_info, read_mode=self.read_mode, raw_mjpeg=self.raw,
                    camera_frames=self.camera_frames, grab_calls=self.grab_calls,
                    retrieve_calls=self.retrieve_calls, failed_reads=self.failed_reads, processed=self.captured,
                    dropped=self.dropped, corrupt=self.corrupt, reopens=self.reopens, last_frame_age_s=age,
                    open_attempts=self.open_attempts,
                    last_attempt_age_s=None if self.last_attempt_mono is None else round(
                        self.clock.mono() - self.last_attempt_mono, 2))

    def describe(self):
        return ("camera %s (%s): opened by the capture thread; processing %.3g fps; read mode %s; clock %s" %
                (self.device, self.cam_cfg.get("backend", "auto"), self.det_fps, self.read_mode_setting,
                 self.clock_how))

    def _resolve_read_mode(self, cam):
        if self.read_mode_setting != "auto":
            return self.read_mode_setting
        return "grab_then_retrieve" if getattr(cam, "grab_blocks", False) else "retrieve_every_frame"

    # ------------------------------------------------------------------ loop
    def _run(self):
        t_start = self.clock.wall()
        processed_no = 0
        while not self.stop_ev.is_set():
            cam = self.factory()
            self.open_attempts += 1
            self.last_attempt_mono = self.clock.mono()
            try:
                self._set_state("opening")
                self.cam_info = cam.open()
            except CameraError as e:
                cam.release()
                wait_s = self.backoff.next()
                # Before the first frame ever: "retrying" (still starting). After a loss: still "lost".
                self._set_state("lost" if self.ever_streamed else "retrying",
                                "%s; next try in %.0f s" % (e, wait_s))
                self.log("camera: %s; retrying in %.0f s" % (e, wait_s))
                self.wait(wait_s)
                continue
            self.raw = bool(self.cam_info.get("raw_accepted"))
            self.read_mode = self._resolve_read_mode(cam)
            self.decimator = Decimator(self.det_fps, self.cam_info.get("fps") or configured_camera_fps(self.cam_cfg))
            self.info = dict(w=self.cam_info.get("width"), h=self.cam_info.get("height"), fps=self.cam_info.get("fps"))
            self.log("camera open (read mode %s): " % self.read_mode + ", ".join(
                "%s=%s" % (key, value) for key, value in self.cam_info.items() if key != "settings"))
            for name, setting in (self.cam_info.get("settings") or {}).items():
                self.log("camera setting %s: asked %s, accepted %s, reads back %s" % (
                    name, setting["requested"], setting["accepted"], setting["read_back"]))
            if self.recorder is not None:
                self.recorder.set_camera_info(self.cam_info)
            self._set_state("streaming")
            processed_no, why = self._stream(cam, t_start, processed_no)
            cam.release()
            if self.stop_ev.is_set():
                break
            self.reopens += 1
            if self.recorder is not None and hasattr(self.recorder, "camera_lost"):
                self.recorder.camera_lost("camera lost")      # close the recording file; a new one starts later
            wait_s = self.backoff.next()
            self._set_state("lost", "%s; reopening in %.0f s" % (why, wait_s))
            self.log("camera lost (%s) after %d frames; reopening in %.0f s" % (why, self.camera_frames, wait_s))
            self.wait(wait_s)
        self._set_state("stopped")

    def _stream(self, cam, t_start, processed_no):
        """Read frames until the camera counts as lost. Returns (next processed-frame number, why it was lost)."""
        self.last_good_mono = self.clock.mono()       # the loss timer starts at the open
        failures_in_a_row = 0
        first_frame = True
        stamped_at = "after retrieve" if self.read_mode == "retrieve_every_frame" else "grab"
        while not self.stop_ev.is_set():
            # ---- read one camera frame
            self.grab_calls += 1
            data = None
            ok = cam.grab()
            if ok and self.read_mode == "retrieve_every_frame":
                self.retrieve_calls += 1
                data = cam.retrieve()                 # waits for the frame (DirectShow) and decodes it
                ok = data is not None
            if not ok:
                failures_in_a_row += 1
                self.failed_reads += 1
                silent_s = self.clock.mono() - self.last_good_mono
                if silent_s >= self.lost_after_s:
                    return processed_no, "no frame for %.1f s" % silent_s
                if failures_in_a_row >= self.lost_after_failures:
                    return processed_no, "%d failed reads in a row" % failures_in_a_row
                self.wait(FAILED_READ_WAIT_S)
                continue
            failures_in_a_row = 0
            t_wall, t_mono = self.clock.wall(), self.clock.mono()   # stamp as soon as the frame is in hand
            camera_frame_no = self.camera_frames
            self.camera_frames += 1
            clock_event = self.clockmap.maybe(t_wall, t_mono, camera_frame_no, stamped_at)
            if clock_event and self.emit:
                self.emit(clock_event)
            process = self.decimator.due(t_mono)
            if not process:
                self.last_good_mono = t_mono
                continue                              # grab_then_retrieve: never retrieved, never decoded
            if data is None:                          # grab_then_retrieve: now fetch (decode) this one
                self.retrieve_calls += 1
                data = cam.retrieve()
                if data is None:
                    self.corrupt += 1
                    continue
            jpeg = data if is_jpeg(data) else None
            if first_frame:
                first_frame = False
                self.ever_streamed = True
                self.backoff.reset()           # the camera works again: next failure starts at the short wait
                if self.raw and jpeg is None:  # the backend accepted raw mode but does not deliver JPEG bytes
                    self.raw = False
                    if data.ndim != 3:         # undecodable bytes: ask OpenCV to decode, drop this frame
                        cam.disable_raw()
                        self.log("camera: raw mode returned non-JPEG data; switched to decoded frames")
                        continue
                    self.log("camera: raw MJPEG passthrough not supported by this backend; frames are decoded by "
                             "OpenCV")
            if self.recorder is not None:
                self.recorder.offer(data, t_wall, t_mono, camera_frame_no)
            img = data if jpeg is None else cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
            if img is None:
                self.corrupt += 1
                continue
            self.last_good_mono = t_mono
            if self.w is None:
                self.w, self.h = out_size(img.shape[1], img.shape[0], self.max_width)
            out_w, out_h = out_size(img.shape[1], img.shape[0], self.max_width)
            if out_w != img.shape[1]:
                img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
            self._push(Frame(img, t_wall - self.latency + self.offset, t_wall - t_start, processed_no, t_mono),
                       block=False)
            processed_no += 1
        return processed_no, "stopped"
