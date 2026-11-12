"""Camera devices behind a small interface, so the capture loop (rt/livecam.py) can run on a real webcam or on a fake.

A camera source has these methods and one attribute:

    info = cam.open()      # open the device and apply the settings; returns what the driver actually gave (dict).
                           # Raises CameraError if the camera is missing or unusable.
    ok = cam.grab()        # ask the driver for the next frame. False = no frame (or, on DirectShow, "the device
                           # was just disconnected").
    data = cam.retrieve()  # the frame: a BGR image (H, W, 3), or the raw JPEG bytes (1-D uint8) when the camera
                           # sends MJPEG and raw passthrough works. None = no usable frame.
    cam.release()
    cam.grab_blocks        # True if grab() really waits for the next camera frame (see below).

`OpenCvCamera` is the real one (OpenCV, DirectShow on Windows). `FakeCamera` and `FakeDshowCamera` are what the
tests use; no hardware is ever needed for tests.

Why `grab_blocks` matters (found by reading OpenCV 4.x modules/videoio/src/cap_dshow.cpp):
  * DirectShow: grab() returns True AT ONCE (it only checks that the device is not disconnected). The wait for a
    new frame (up to ~1 s) happens inside retrieve(), and retrieve() always decodes the MJPEG frame (DirectShow
    ignores CONVERT_RGB = 0 for MJPG). So "grab every frame, retrieve only some" does not save the decode there; it
    only spins grab() in a busy loop. On DirectShow rt therefore retrieves every frame (livecam.py,
    "retrieve_every_frame") and drops the unneeded ones after the decode.
  * V4L2 (Linux): grab() waits for the driver's next buffer (select + dequeue) and retrieve() decodes it, so
    "grab every frame, retrieve only the processed ones" really saves the decode of the others.
  * Media Foundation (msmf) and anything else: not verified, so treated like DirectShow (always correct, maybe
    slower).
Possible future sources that would avoid decoding unprocessed frames on Windows (not implemented): the MSMF
backend if its grab() is confirmed to block, or an ffmpeg `-f dshow -vcodec mjpeg` process that copies the
camera's JPEGs (decode only the processed ones, record without re-encoding).

Camera identity: identical webcams have the same name, so a camera is selected by index or by device path, never by
name. On Windows the device path is DirectShow's "alternative name" (`python rt/run.py --list-cameras` prints it).
"""
import os
import re
import subprocess
import sys

import numpy as np

JPEG_MAGIC = b"\xff\xd8"


class CameraError(Exception):
    """The camera could not be opened or is unusable (missing, wrong format, ...). The capture loop retries."""


# Does grab() wait for the next frame on this OpenCV backend? Only backends verified to block are True; the others
# get the always-correct "retrieve every frame" loop. See the module docstring.
GRAB_BLOCKS = {"dshow": False, "v4l2": True, "msmf": False, "any": False}


def default_backend():
    return "dshow" if sys.platform == "win32" else ("v4l2" if sys.platform.startswith("linux") else "any")


def fourcc_str(v):
    """OpenCV's FOURCC property is a float holding 4 characters packed in an int: 1196444237.0 -> 'MJPG'."""
    try:
        packed = int(v)
    except (TypeError, ValueError):
        return ""
    if packed <= 0:
        return ""
    return "".join(chr((packed >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")


def is_jpeg(data):
    """True if `data` is raw JPEG bytes (what a raw MJPEG retrieve returns), not a decoded image."""
    if not isinstance(data, np.ndarray) or data.dtype != np.uint8:
        return False
    flat = data.reshape(-1) if (data.ndim == 1 or (data.ndim == 2 and data.shape[0] == 1)) else None
    return flat is not None and flat.size > 2 and flat[:2].tobytes() == JPEG_MAGIC


# ------------------------------------------------------------------------------------------------ device selection
def list_dshow_video_devices(ffmpeg="ffmpeg", runner=subprocess.run):
    """DirectShow video devices in enumeration order, as [(name, device_path)]. Uses ffmpeg's device list, which
    enumerates the same DirectShow category in the same order as OpenCV's CAP_DSHOW indices.

    ffmpeg >= 5 prints `"Name" (video)` then `  Alternative name "@device_pnp_\\\\?\\usb#vid_...` per device; older
    versions print a "DirectShow video devices" header and the names under it. Both are handled."""
    try:
        result = runner([ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                        capture_output=True, text=True, timeout=20)
    except Exception as e:  # noqa: BLE001
        raise CameraError("cannot list DirectShow devices with %s: %s" % (ffmpeg, e)) from e
    return parse_dshow_list((result.stderr or "") + (result.stdout or ""))


def parse_dshow_list(text):
    devices, section, current = [], None, None
    for line in text.splitlines():
        body = re.sub(r"^\[[^\]]*\]\s?", "", line)  # drop the "[dshow @ 0000...]" prefix
        if "DirectShow video devices" in body:
            section = "video"
            continue
        if "DirectShow audio devices" in body:
            section = "audio"
            continue
        m = re.match(r'^\s*"(.+)"\s*(\((video|audio|none)\))?\s*$', body)
        if m and "Alternative name" not in body:
            kind = m.group(3) or section
            current = [m.group(1), None] if kind == "video" else None
            if current is not None:
                devices.append(current)
            continue
        m = re.match(r'^\s*Alternative name\s+"(.+)"\s*$', body)
        if m and current is not None:
            current[1] = m.group(1)
    return [(name, path) for name, path in devices]


def resolve_device(spec, backend, ffmpeg="ffmpeg", lister=None):
    """camera.device from the config -> what cv2.VideoCapture() takes.

    * an int, or a string of digits: an index (0 = the first camera). Simple, but the order can change on replug.
    * DirectShow (Windows): a device path, or a unique part of it (e.g. "usb#vid_046d&pid_085e&mi_00#7&1a2b3c"):
      looked up in the device list and turned into its index. 0 or several matches is an error. Names are never
      matched: two identical webcams have the same name.
    * V4L2 (Linux): a device path such as /dev/v4l/by-id/usb-...-video-index0 (stable across replugs)."""
    if isinstance(spec, int) or re.fullmatch(r"\d+", str(spec).strip()):
        return int(spec)
    s = str(spec).strip()
    if backend == "dshow":
        devs = (lister or list_dshow_video_devices)(ffmpeg)
        hits = [i for i, (_, path) in enumerate(devs) if path and s.lower() in path.lower()]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise CameraError("no DirectShow camera with a device path containing %r (have: %s)" % (
                s, "; ".join("%d %s %s" % (i, name, path) for i, (name, path) in enumerate(devs)) or "none"))
        raise CameraError("%d DirectShow cameras match %r: use a longer part of the device path" % (len(hits), s))
    return os.path.realpath(s) if os.path.exists(s) else s


# ------------------------------------------------------------------------------------------------ image settings
# Settings are applied in this order: the auto modes are switched off first, otherwise some drivers ignore the
# manual values written before.
AUTO_FIRST = ("auto_exposure", "auto_wb", "autofocus")


def apply_settings(cap, cv2, settings, log=print):
    """Write each image setting to the driver, then read it back. `settings` maps a name to a value; the name is an
    OpenCV property without the CAP_PROP_ prefix, in lower case (exposure, gain, auto_wb, wb_temperature, focus, ...).

    Values are the backend's raw units: on DirectShow, exposure is log2(seconds), so -10 = 1/1024 s. The read-back
    is what counts: everything is logged. A setting the camera does not support gives a warning, never an error.
    Returns {name: {"requested", "accepted", "read_back"}}."""
    out = {}
    order = [name for name in AUTO_FIRST if name in settings] + [name for name in settings if name not in AUTO_FIRST]
    for name in order:
        want = settings[name]
        prop = getattr(cv2, "CAP_PROP_" + name.upper(), None)
        if prop is None:
            log("WARNING camera setting %r is not an OpenCV property; ignored" % name)
            out[name] = dict(requested=want, accepted=False, read_back=None)
            continue
        try:
            ok = bool(cap.set(prop, float(want)))
        except Exception as e:  # noqa: BLE001 (some drivers throw on unsupported properties)
            log("WARNING camera setting %s=%s failed: %s" % (name, want, e))
            ok = False
        out[name] = dict(requested=want, accepted=ok, read_back=None)
    for name in order:
        prop = getattr(cv2, "CAP_PROP_" + name.upper(), None)
        if prop is None:
            continue
        try:
            read_back = float(cap.get(prop))
        except Exception:  # noqa: BLE001
            read_back = None
        out[name]["read_back"] = read_back
        want = out[name]["requested"]
        if not out[name]["accepted"]:
            log("WARNING camera does not accept %s=%s (reads back %s)" % (name, want, read_back))
        elif read_back is None or abs(read_back - float(want)) > 1e-3 * max(1.0, abs(float(want))):
            log("WARNING camera setting %s: asked %s, reads back %s" % (name, want, read_back))
    return out


# ------------------------------------------------------------------------------------------------ OpenCV camera
class OpenCvCamera:
    """A USB camera through OpenCV. Config: the [camera] section of rt/config.toml."""

    def __init__(self, cam_cfg, device=None, ffmpeg="ffmpeg", log=print, cv2_module=None, lister=None):
        self.c = cam_cfg
        self.device = cam_cfg.get("device", 0) if device is None else device
        self.ffmpeg, self.log, self.lister = ffmpeg, log, lister
        self.backend = cam_cfg.get("backend", "auto")
        if self.backend == "auto":
            self.backend = default_backend()
        if cv2_module is None:
            import cv2 as cv2_module
        self.cv2 = cv2_module
        self.cap = None
        self.grab_blocks = GRAB_BLOCKS.get(self.backend, False)

    def open(self):
        cv2 = self.cv2
        api = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
               "any": cv2.CAP_ANY}[self.backend]
        target = resolve_device(self.device, self.backend, self.ffmpeg, self.lister)
        cap = cv2.VideoCapture(target, api)
        if not cap.isOpened():
            cap.release()
            raise CameraError("cannot open camera %r (%s, %s)" % (self.device, self.backend, target))
        want_fourcc = str(self.c.get("fourcc", "MJPG"))
        code = cv2.VideoWriter_fourcc(*want_fourcc)
        # FOURCC before and after the size: some DirectShow drivers only keep MJPG if it is set first, others reset
        # it when the size changes.
        cap.set(cv2.CAP_PROP_FOURCC, code)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.c["width"]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.c["height"]))
        cap.set(cv2.CAP_PROP_FPS, float(self.c["fps"]))
        cap.set(cv2.CAP_PROP_FOURCC, code)
        raw_requested = bool(self.c.get("raw_mjpeg", True)) and want_fourcc == "MJPG"
        raw = False
        if raw_requested:
            # CONVERT_RGB = 0 asks the backend for the compressed frame (raw JPEG bytes). Not every backend supports
            # it; livecam.py checks the first frame and falls back to decoded frames if it is not JPEG.
            try:
                raw = bool(cap.set(cv2.CAP_PROP_CONVERT_RGB, 0))
            except Exception:  # noqa: BLE001
                raw = False
        settings = apply_settings(cap, cv2, dict(self.c.get("settings", {}) or {}), self.log)
        actual = dict(backend=self.backend, device=str(self.device), target=str(target),
                      fourcc=fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=round(float(cap.get(cv2.CAP_PROP_FPS)), 3),
                      raw_requested=raw_requested, raw_accepted=raw, grab_blocks=self.grab_blocks,
                      settings=settings)
        if bool(self.c.get("require_fourcc", True)) and actual["fourcc"] != want_fourcc:
            cap.release()
            raise CameraError("camera %r gives %r, not %s (at %dx%d %s fps). MJPG is needed for 1080p30 over USB 2; "
                              "check the camera supports it, or set camera.require_fourcc = false" % (
                                  self.device, actual["fourcc"], want_fourcc, actual["width"], actual["height"],
                                  actual["fps"]))
        if (actual["width"], actual["height"]) != (int(self.c["width"]), int(self.c["height"])):
            self.log("WARNING camera gives %dx%d, asked %dx%d" % (actual["width"], actual["height"],
                                                                  int(self.c["width"]), int(self.c["height"])))
        self.cap = cap
        return actual

    def grab(self):
        return self.cap is not None and bool(self.cap.grab())

    def retrieve(self):
        ok, data = self.cap.retrieve()
        return data if ok and data is not None and data.size else None

    def disable_raw(self):
        """Raw passthrough did not give JPEG: ask the backend to decode frames itself."""
        try:
            self.cap.set(self.cv2.CAP_PROP_CONVERT_RGB, 1)
        except Exception:  # noqa: BLE001
            pass

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ------------------------------------------------------------------------------------------------ fake camera
class FakeCamera:
    """A camera for tests whose grab() blocks for the next frame (like V4L2). `frames` is a list of images (BGR
    arrays), or a function frame_number -> image, or None for small synthetic grey frames. Each grab() advances
    `clock` (a backoff.FakeClock) by 1/fps, like a real camera delivering frames in real time.

    jpeg=True makes retrieve() return JPEG bytes (a camera with working raw MJPEG passthrough). `fail_opens` = how
    many open() calls fail before one works (a missing camera). `lose_after` = after this many grabs the camera
    "disappears" (grab() returns False until it is opened again). `n_frames` ends the stream (grab() False forever)."""

    def __init__(self, frames=None, fps=30.0, clock=None, jpeg=False, fail_opens=0, lose_after=None, n_frames=None,
                 size=(64, 48), info=None):
        self.frames, self.fps, self.clock, self.jpeg = frames, float(fps), clock, jpeg
        self.fail_opens, self.lose_after, self.n_frames, self.size = fail_opens, lose_after, n_frames, size
        self.extra_info = dict(info or {})
        self.opens = 0          # open() calls, failed ones included
        self.grabs = 0          # successful grab() calls in total
        self.retrieves = 0      # retrieve() calls in total
        self.retrieved = []     # frame numbers that were retrieved (what the decimation chose)
        self.is_open = False
        self._since_open = 0
        self._current = None
        self.grab_blocks = True     # this fake's grab() waits for the frame (it advances the clock)

    def open(self):
        self.opens += 1
        if self.opens <= self.fail_opens:
            raise CameraError("fake camera missing (open %d)" % self.opens)
        self.is_open = True
        self._since_open = 0
        w, h = self.size
        return dict(dict(backend="fake", device="fake", fourcc="MJPG", width=w, height=h, fps=self.fps,
                         raw_requested=self.jpeg, raw_accepted=self.jpeg, settings={}), **self.extra_info)

    def _image(self, frame_no):
        return synthetic_image(self.frames, frame_no, self.size)

    def grab(self):
        if not self.is_open:
            return False
        if self.n_frames is not None and self.grabs >= self.n_frames:
            return False
        if self.lose_after is not None and self._since_open >= self.lose_after:
            return False
        if self.clock is not None:
            self.clock.advance(1.0 / self.fps)
        self._current = self.grabs
        self.grabs += 1
        self._since_open += 1
        return True

    def retrieve(self):
        self.retrieves += 1
        self.retrieved.append(self._current)
        img = self._image(self._current)
        if not self.jpeg:
            return img
        import cv2
        ok, buf = cv2.imencode(".jpg", img)
        return buf.reshape(-1)

    def disable_raw(self):
        self.jpeg = False

    def release(self):
        self.is_open = False

    @classmethod
    def from_video(cls, path, **kw):
        """All frames of a (short) video file, decoded once with OpenCV."""
        import cv2
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()
        if not frames:
            raise CameraError("no frames in %s" % path)
        h, w = frames[0].shape[:2]
        kw.setdefault("n_frames", len(frames))
        return cls(frames=frames, size=(w, h), **kw)


def synthetic_image(frames, frame_no, size):
    """Image number `frame_no` for the fakes: from a list (cycled), a function, or a flat grey image whose
    brightness is the frame number (so a test can tell frames apart)."""
    if callable(frames):
        return frames(frame_no)
    if frames is not None:
        return frames[frame_no % len(frames)]
    width, height = size
    return np.full((height, width, 3), frame_no % 256, np.uint8)


class FakeDshowCamera:
    """A fake camera that behaves exactly like OpenCV's DirectShow backend (cap_dshow.cpp), for tests:

      * grab() never waits and returns True. The only time it returns False is ONCE, right after the device was
        unplugged (DirectShow reports the disconnect as a one-time event); after that it says True again.
      * retrieve() is where the waiting happens: it waits (advances the fake clock) until the camera's next frame,
        and returns it decoded (BGR image; DirectShow decodes MJPEG even when raw frames were asked for).
        If no frame comes (unplugged, or the stream ended), it gives up after `retrieve_timeout_s` (1 s, as in
        cap_dshow.cpp) and returns None.
      * once unplugged, the open handle is dead for good, even after the camera is plugged back: only a new open()
        works again. open() fails (CameraError) while the camera is unplugged.

    Frames arrive every 1/fps seconds of the fake clock from open(). `unplug_after` = unplug the camera right after
    this many frames have been delivered in total; `unplugged_for_s` = it comes back that many seconds later (None =
    never). `n_frames` = the stream simply stops after that many frames (no disconnect event).

    Counters for the tests: grab_calls, retrieve_calls, delivered (frames returned), opens."""

    grab_blocks = False      # what OpenCvCamera says for DirectShow

    def __init__(self, fps=30.0, clock=None, frames=None, size=(64, 48), retrieve_timeout_s=1.0, n_frames=None,
                 unplug_after=None, unplugged_for_s=None, info=None):
        if clock is None:
            raise ValueError("FakeDshowCamera needs a FakeClock: its retrieve() 'waits' by advancing it")
        self.fps, self.clock, self.frames, self.size = float(fps), clock, frames, size
        self.retrieve_timeout_s = float(retrieve_timeout_s)
        self.n_frames, self.unplug_after, self.unplugged_for_s = n_frames, unplug_after, unplugged_for_s
        self.extra_info = dict(info or {})
        self.grab_calls = 0
        self.retrieve_calls = 0
        self.delivered = 0
        self.opens = 0
        self.is_open = False
        self.plugged = True
        self.replug_at_mono = None          # when an unplugged camera comes back (fake monotonic time)
        self.handle_dead = False            # the open handle saw the unplug: it never delivers again
        self.disconnect_event_pending = False
        self.next_frame_mono = None         # when the camera's next frame is ready
        self._auto_unplug_done = False

    # ---- plugging, for the tests
    def unplug(self, for_s=None):
        """Pull the USB cable. for_s = plug it back in after that many seconds (None = stays out)."""
        self.plugged = False
        self.replug_at_mono = None if for_s is None else self.clock.mono() + float(for_s)
        if self.is_open:
            self.handle_dead = True
            self.disconnect_event_pending = True

    def _maybe_replugged(self):
        if not self.plugged and self.replug_at_mono is not None and self.clock.mono() >= self.replug_at_mono:
            self.plugged = True

    # ---- the camera interface
    def open(self):
        self.opens += 1
        self._maybe_replugged()
        if not self.plugged:
            raise CameraError("fake DirectShow camera is unplugged")
        self.is_open = True
        self.handle_dead = False
        self.disconnect_event_pending = False
        self.next_frame_mono = self.clock.mono() + 1.0 / self.fps
        width, height = self.size
        return dict(dict(backend="dshow (fake)", device="fake", fourcc="MJPG", width=width, height=height,
                         fps=self.fps, raw_requested=False, raw_accepted=False, grab_blocks=False, settings={}),
                    **self.extra_info)

    def grab(self):
        self.grab_calls += 1                # never advances the clock: DirectShow's grab() does not wait
        if not self.is_open:
            return False
        if self.disconnect_event_pending:   # the one-time disconnect event
            self.disconnect_event_pending = False
            return False
        return True

    def retrieve(self):
        self.retrieve_calls += 1
        stream_ended = self.n_frames is not None and self.delivered >= self.n_frames
        if not self.is_open or self.handle_dead or stream_ended:
            self.clock.advance(self.retrieve_timeout_s)    # waits for a frame that never comes, then gives up
            return None
        period = 1.0 / self.fps
        now = self.clock.mono()
        while self.next_frame_mono + period <= now:          # the caller was late: those frames are gone
            self.next_frame_mono += period
        if self.next_frame_mono > now:
            self.clock.advance(self.next_frame_mono - now)   # wait for the next frame
        self.next_frame_mono += period
        image = synthetic_image(self.frames, self.delivered, self.size)
        self.delivered += 1
        if self.unplug_after is not None and self.delivered >= self.unplug_after and not self._auto_unplug_done:
            self._auto_unplug_done = True
            self.unplug(self.unplugged_for_s)
        return image

    def disable_raw(self):
        pass

    def release(self):
        self.is_open = False
