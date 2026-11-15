"""Tests for zone/source.py: the time stamp each spotter frame gets.

Why it matters: a webcam hands a frame over ~0.1 s after the picture was taken. If we stamped the frame with its
arrival time, the controller would believe the car was at that spot 0.1 s later than it really was, and the PTZ
would lag behind by v * 0.1 s. So the known capture latency (`rt.capture.live_latency_s`) is taken off the arrival
time of live frames. Recorded files have no capture latency: nothing is taken off.
"""
import shutil
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

from zone.controller import Controller
from zone.path import Pose, PoseCurve
from zone.source import SpotterSource, grab_time
from zone.tests.z2_fixtures import settings, sim


def test_grab_time_subtracts_the_capture_latency_of_live_sources_only():
    assert grab_time(100.0, live=True, capture_latency_s=0.1) == pytest.approx(99.9)
    assert grab_time(100.0, live=False, capture_latency_s=0.1) == pytest.approx(100.0)


class FakeCapture:
    """Stands in for rt's capture object: hands out two frames that 'arrived' at monotonic 50.0 and 50.1."""

    def __init__(self, live):
        self.live = live
        self.dropped = 0

    def describe(self):
        return "fake"

    def start(self):
        pass

    def stop(self):
        pass

    def frames(self):
        for k, t_arrival in enumerate((50.0, 50.1)):
            # t = wall time (rt has already taken the latency off it), t_cap = monotonic arrival time
            yield types.SimpleNamespace(img=np.zeros((4, 4, 3), np.uint8), t=1_800_000_000.0 + k / 10,
                                        t_cap=t_arrival)


@pytest.mark.parametrize("live, expected", [(True, [49.9, 50.0]), (False, [50.0, 50.1])])
def test_spotter_source_stamps_frames_with_the_picture_time(monkeypatch, live, expected):
    import rt.capture
    # replace rt's "open the camera" with our fake, so no camera or ffmpeg is needed
    monkeypatch.setattr(rt.capture, "open_source", lambda *a, **kw: FakeCapture(live))
    cfg = dict(spotter=dict(source="0", loop_file=False, realtime_file=True))
    src = SpotterSource(cfg, dict(capture=dict(live_latency_s=0.1)), log=lambda m: None)
    times = [fr.t for fr in src.frames()]
    assert times == pytest.approx(expected)


def test_the_lookahead_covers_capture_latency_processing_and_command_latency():
    # A straight zone: pan 0 -> 800 over s 0 -> 1. A car at v = 0.25 /s.
    st = settings("B", sets=["controller.latency_s=0.2", "controller.include_processing_delay=true"])
    ptz, clock = sim()
    curve = PoseCurve([0.0, 1.0], [Pose(0, 0, 1000), Pose(800, 0, 1000)], "linear")
    ctrl = Controller(ptz, clock, curve, st.pan_map, st.tilt_map, st.controller)
    clock.advance(10.0)
    arrival = clock.monotonic()                       # the frame reached us now...
    t_grab = grab_time(arrival, True, 0.1)            # ...but the picture was taken 0.1 s earlier
    clock.advance(0.05)                               # detection + tracking took 50 ms
    cmd = ctrl.follow(0.5, 0.25, t_grab)
    # lookahead = 0.1 (capture) + 0.05 (processing) + 0.2 (command latency) = 0.35 s
    assert cmd.lookahead_s == pytest.approx(0.35)
    # so the camera is aimed where the car will be 0.35 s after the picture: s = 0.5 + 0.25 * 0.35
    assert cmd.s_ahead == pytest.approx(0.5 + 0.25 * 0.35)


# ---------------------------------------------------------------------------------------------- ffmpeg off PATH
FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")    # full paths, found before PATH is hidden


@pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="ffmpeg/ffprobe not on PATH")
def test_spotter_file_source_works_with_ffmpeg_only_at_its_configured_path(tmp_path, monkeypatch):
    # The installed app's ffmpeg is in INSTALL\ffmpeg\bin, which is NOT on PATH: the launcher writes its full path
    # into the zone config (rt.capture.ffmpeg / rt.capture.ffprobe). The zone's spotter, reading a video FILE, used
    # to crash ("frame loop crashed: FileNotFoundError", exit code 1) because the clock overlay reader inside rt ran a
    # bare "ffmpeg". Here PATH is emptied, so only the configured full paths can work.
    from zone import config as C
    from zone.spotter import rt_config
    video = tmp_path / "spotter.mkv"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10", "-t", "2",
                    "-c:v", "mpeg4", str(video)], check=True)
    cfg = C.load(None, ["spotter.source=%s" % video.as_posix(), "spotter.loop_file=false",
                        "spotter.realtime_file=false",
                        "rt.capture.ffmpeg=%s" % Path(FFMPEG).as_posix(),
                        "rt.capture.ffprobe=%s" % Path(FFPROBE).as_posix()])
    empty = tmp_path / "empty_path_folder"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))           # a bare "ffmpeg" can no longer be found
    assert shutil.which("ffmpeg") is None

    source = SpotterSource(cfg, rt_config(cfg), log=lambda m: None)
    frames = list(source.frames())
    assert len(frames) >= 10                         # 2 s of a 10 fps file, processed at 10 fps
