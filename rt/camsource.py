"""Camera devices behind a small interface, so the capture loop (rt/livecam.py) can run on a real webcam or on a fake.

A camera source has four methods:

    info = cam.open()      # open the device and apply the settings; returns what the driver actually gave (dict).
                           # Raises CameraError if the camera is missing or unusable.
    ok = cam.grab()        # take the next frame from the driver, WITHOUT decoding it (cheap). False = no frame.
    data = cam.retrieve()  # the last grabbed frame: a BGR image (H, W, 3), or the raw JPEG bytes (1-D uint8) when
                           # the camera sends MJPEG and raw passthrough works. None = no usable frame.
    cam.release()

`OpenCvCamera` is the real one (OpenCV, DirectShow on Windows). `FakeCamera` yields frames from a list, a function or
a video file and is what the tests use; no hardware is ever needed for tests.

Why grab()/retrieve(): a USB webcam sends 30 fps, rt processes 10. read() = grab() + retrieve() decodes every frame;
grab() alone only takes the compressed frame from the driver. Decoding 1080p MJPEG is the biggest CPU cost of the
capture, so only the frames that are processed are retrieved (and decoded).

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


def default_backend():
    return "dshow" if sys.platform == "win32" else ("v4l2" if sys.platform.startswith("linux") else "any")


def fourcc_str(v):
    """OpenCV's FOURCC property is a float holding 4 characters packed in an int: 1196444237.0 -> 'MJPG'."""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")


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
        r = runner([ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                   capture_output=True, text=True, timeout=20)
    except Exception as e:  # noqa: BLE001
        raise CameraError("cannot list DirectShow devices with %s: %s" % (ffmpeg, e)) from e
    return parse_dshow_list((r.stderr or "") + (r.stdout or ""))


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
    return [(n, p) for n, p in devices]


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
                s, "; ".join("%d %s %s" % (i, n, p) for i, (n, p) in enumerate(devs)) or "none"))
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
    order = [k for k in AUTO_FIRST if k in settings] + [k for k in settings if k not in AUTO_FIRST]
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
            got = float(cap.get(prop))
        except Exception:  # noqa: BLE001
            got = None
        out[name]["read_back"] = got
        want = out[name]["requested"]
        if not out[name]["accepted"]:
            log("WARNING camera does not accept %s=%s (reads back %s)" % (name, want, got))
        elif got is None or abs(got - float(want)) > 1e-3 * max(1.0, abs(float(want))):
            log("WARNING camera setting %s: asked %s, reads back %s" % (name, want, got))
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
        got = dict(backend=self.backend, device=str(self.device), target=str(target),
                   fourcc=fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=round(float(cap.get(cv2.CAP_PROP_FPS)), 3),
                   raw_requested=raw_requested, raw_accepted=raw, settings=settings)
        if bool(self.c.get("require_fourcc", True)) and got["fourcc"] != want_fourcc:
            cap.release()
            raise CameraError("camera %r gives %r, not %s (at %dx%d %s fps). MJPG is needed for 1080p30 over USB 2; "
                              "check the camera supports it, or set camera.require_fourcc = false" % (
                                  self.device, got["fourcc"], want_fourcc, got["width"], got["height"], got["fps"]))
        if (got["width"], got["height"]) != (int(self.c["width"]), int(self.c["height"])):
            self.log("WARNING camera gives %dx%d, asked %dx%d" % (got["width"], got["height"], int(self.c["width"]),
                                                                  int(self.c["height"])))
        self.cap = cap
        return got

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
    """A camera for tests. `frames` is a list of images (BGR arrays), or a function frame_number -> image, or None
    for small synthetic grey frames. Each grab() advances `clock` (a backoff.FakeClock) by 1/fps, like a real camera
    delivering frames in real time.

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

    def open(self):
        self.opens += 1
        if self.opens <= self.fail_opens:
            raise CameraError("fake camera missing (open %d)" % self.opens)
        self.is_open = True
        self._since_open = 0
        w, h = self.size
        return dict(dict(backend="fake", device="fake", fourcc="MJPG", width=w, height=h, fps=self.fps,
                         raw_requested=self.jpeg, raw_accepted=self.jpeg, settings={}), **self.extra_info)

    def _image(self, n):
        if callable(self.frames):
            return self.frames(n)
        if self.frames is not None:
            return self.frames[n % len(self.frames)]
        w, h = self.size
        img = np.full((h, w, 3), n % 256, np.uint8)
        return img

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
