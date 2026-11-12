"""Tests of the camera device layer (rt/camsource.py) with a fake OpenCV: no webcam needed.

What is checked:
  * choosing the camera by index or by device path, never by name (two identical webcams share a name);
  * reading the DirectShow device list printed by ffmpeg (old and new ffmpeg formats);
  * image settings: the auto modes are switched off first, every value is read back, and a setting the camera does
    not support gives a warning instead of a crash;
  * the camera must deliver MJPG: anything else fails loudly (CameraError), so it shows on the status page.
"""
import types

import cv2
import numpy as np
import pytest

from rt.camsource import (CameraError, OpenCvCamera, apply_settings, fourcc_str, is_jpeg, parse_dshow_list,
                          resolve_device)

# What ffmpeg 6 prints for `ffmpeg -list_devices true -f dshow -i dummy` with two identical webcams and a mic.
DSHOW_NEW = r'''[dshow @ 000001] "HD Pro Webcam C920" (video)
[dshow @ 000001]   Alternative name "@device_pnp_\\?\usb#vid_046d&pid_082d&mi_00#6&aaaa&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"
[dshow @ 000001] "HD Pro Webcam C920" (video)
[dshow @ 000001]   Alternative name "@device_pnp_\\?\usb#vid_046d&pid_082d&mi_00#6&bbbb&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"
[dshow @ 000001] "Microphone (HD Pro Webcam C920)" (audio)
[dshow @ 000001]   Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{1234}"
dummy: Immediate exit requested'''

# The same from an older ffmpeg (4.x): section headers instead of "(video)".
DSHOW_OLD = r'''[dshow @ 000002] DirectShow video devices (some may be both video and audio devices)
[dshow @ 000002]  "Integrated Camera"
[dshow @ 000002]     Alternative name "@device_pnp_\\?\usb#vid_04f2&pid_b6d0&mi_00#7&cccc&0&0000#{65e8773d}\global"
[dshow @ 000002] DirectShow audio devices
[dshow @ 000002]  "Microphone Array"
[dshow @ 000002]     Alternative name "@device_cm_{33D9A762}\wave_{5678}"'''


def test_dshow_device_list_new_ffmpeg():
    devs = parse_dshow_list(DSHOW_NEW)
    assert [n for n, _ in devs] == ["HD Pro Webcam C920", "HD Pro Webcam C920"]   # only video devices, in order
    assert "6&aaaa" in devs[0][1] and "6&bbbb" in devs[1][1]


def test_dshow_device_list_old_ffmpeg():
    devs = parse_dshow_list(DSHOW_OLD)
    assert len(devs) == 1 and devs[0][0] == "Integrated Camera" and "7&cccc" in devs[0][1]


def test_device_by_index():
    assert resolve_device(0, "dshow") == 0
    assert resolve_device("1", "dshow") == 1      # a number written as text in the config also works


def test_device_by_path_picks_the_right_one_of_two_identical_webcams():
    # Both webcams are called "HD Pro Webcam C920"; only the device path (it has the USB serial/port in it) differs.
    lister = lambda ffmpeg: parse_dshow_list(DSHOW_NEW)  # noqa: E731
    assert resolve_device("6&bbbb", "dshow", lister=lister) == 1
    assert resolve_device("6&aaaa", "dshow", lister=lister) == 0


def test_device_by_name_is_refused():
    # The name matches both webcams (and is not a device path): an error, not a guess.
    lister = lambda ffmpeg: parse_dshow_list(DSHOW_NEW)  # noqa: E731
    with pytest.raises(CameraError):
        resolve_device("HD Pro Webcam C920", "dshow", lister=lister)
    with pytest.raises(CameraError):                     # a part shared by both paths: ambiguous
        resolve_device("vid_046d&pid_082d", "dshow", lister=lister)


# ------------------------------------------------------------------------------------------------ fake OpenCV
class FakeVideoCapture:
    """Pretends to be cv2.VideoCapture for a camera that supports only the properties in `supported`.
    `fixed` holds properties the driver silently refuses to change (set() says ok, get() returns the fixed value)."""

    def __init__(self, supported, fixed=None, opened=True):
        self.values = dict(supported)
        self.fixed = dict(fixed or {})
        self.opened = opened
        self.calls = []                 # (property, value) in the order they were set

    def __call__(self, target, api):    # used as the "class": cv2.VideoCapture(target, api) returns self
        self.target, self.api = target, api
        return self

    def isOpened(self):  # noqa: N802 (OpenCV's name)
        return self.opened

    def set(self, prop, value):
        self.calls.append((prop, value))
        if prop not in self.values:
            return False
        self.values[prop] = value
        return True

    def get(self, prop):
        return self.fixed.get(prop, self.values.get(prop, 0.0))

    def release(self):
        self.opened = False


def fake_cv2(vc):
    """A stand-in for the cv2 module: the real constants, our fake VideoCapture."""
    ns = {k: getattr(cv2, k) for k in dir(cv2) if k.startswith("CAP_")}
    return types.SimpleNamespace(VideoCapture=vc, VideoWriter_fourcc=cv2.VideoWriter_fourcc, **ns)


def camera_props(**extra):
    p = {cv2.CAP_PROP_FOURCC: 0.0, cv2.CAP_PROP_FRAME_WIDTH: 640.0, cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
         cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_CONVERT_RGB: 1.0, cv2.CAP_PROP_AUTO_EXPOSURE: 1.0,
         cv2.CAP_PROP_EXPOSURE: -5.0, cv2.CAP_PROP_GAIN: 0.0}
    p.update(extra)
    return p


def test_settings_auto_modes_first_then_values_all_read_back():
    vc = FakeVideoCapture(camera_props())
    logs = []
    out = apply_settings(vc, fake_cv2(vc), {"exposure": -10, "gain": 64, "auto_exposure": 0}, logs.append)
    # auto_exposure was written before exposure, even though it came last in the config.
    props = [p for p, _ in vc.calls]
    assert props.index(cv2.CAP_PROP_AUTO_EXPOSURE) < props.index(cv2.CAP_PROP_EXPOSURE)
    # Every setting is read back.
    assert out["exposure"] == dict(requested=-10, accepted=True, read_back=-10.0)
    assert out["gain"]["read_back"] == 64.0
    assert logs == []                    # all accepted and equal: nothing to warn about


def test_unsupported_setting_is_a_warning_not_a_crash():
    vc = FakeVideoCapture(camera_props())              # this camera has no focus control
    logs = []
    out = apply_settings(vc, fake_cv2(vc), {"focus": 0, "not_a_property": 3}, logs.append)
    assert out["focus"]["accepted"] is False
    assert out["not_a_property"]["accepted"] is False
    assert len(logs) == 2 and all(m.startswith("WARNING") for m in logs)


def test_setting_that_reads_back_different_is_logged():
    # The driver accepts exposure -12 but clamps it to -11 (its fastest shutter). The read-back shows the truth.
    vc = FakeVideoCapture(camera_props(), fixed={cv2.CAP_PROP_EXPOSURE: -11.0})
    logs = []
    out = apply_settings(vc, fake_cv2(vc), {"exposure": -12}, logs.append)
    assert out["exposure"]["read_back"] == -11.0
    assert any("reads back -11" in m for m in logs)


CAM_CFG = dict(backend="dshow", device=0, width=1920, height=1080, fps=30, fourcc="MJPG", require_fourcc=True,
               raw_mjpeg=True, settings={"exposure": -10})


def test_open_sets_mjpg_size_fps_and_reads_them_back():
    vc = FakeVideoCapture(camera_props())
    cam = OpenCvCamera(CAM_CFG, cv2_module=fake_cv2(vc), log=lambda m: None)
    info = cam.open()
    assert info["fourcc"] == "MJPG"
    assert (info["width"], info["height"], info["fps"]) == (1920, 1080, 30.0)
    assert info["raw_accepted"] is True            # CONVERT_RGB = 0 was accepted: raw JPEG frames
    assert info["settings"]["exposure"]["read_back"] == -10.0
    assert vc.api == cv2.CAP_DSHOW


def test_camera_that_will_not_do_mjpg_fails_loudly():
    # The driver ignores the MJPG request and stays on YUY2 (uncompressed: 1080p30 does not fit USB 2).
    yuy2 = float(cv2.VideoWriter_fourcc(*"YUY2"))
    vc = FakeVideoCapture(camera_props(), fixed={cv2.CAP_PROP_FOURCC: yuy2})
    cam = OpenCvCamera(CAM_CFG, cv2_module=fake_cv2(vc), log=lambda m: None)
    with pytest.raises(CameraError, match="YUY2"):
        cam.open()
    assert vc.opened is False                       # and the device was released, so a retry can open it


def test_camera_that_cannot_be_opened():
    vc = FakeVideoCapture(camera_props(), opened=False)
    cam = OpenCvCamera(CAM_CFG, cv2_module=fake_cv2(vc), log=lambda m: None)
    with pytest.raises(CameraError, match="cannot open"):
        cam.open()


def test_small_helpers():
    assert fourcc_str(cv2.VideoWriter_fourcc(*"MJPG")) == "MJPG"
    assert fourcc_str(0) == ""
    ok, buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    assert is_jpeg(buf.reshape(-1)) and is_jpeg(buf.reshape(1, -1))       # raw JPEG bytes, 1-D or one row
    assert not is_jpeg(np.zeros((8, 8, 3), np.uint8))                     # a decoded image is not
