"""Tests of the recorder (rt/recorder.py): files every 30 minutes with their sidecars, and the disk-space guard.

Most tests use a fake ffmpeg (it only collects the bytes it is sent) and a fake disk, so they run in milliseconds.
The last test runs the real ffmpeg, if it is installed, and checks the file it writes is a valid video.
"""
import csv
import io
import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from rt.backoff import FakeClock
from rt.recorder import Recorder


class FakeFfmpeg:
    """Stands in for subprocess.Popen(ffmpeg ...): remembers the command and collects what is written to stdin."""

    instances = []

    def __init__(self, cmd, stdin=None, stdout=None, stderr=None):
        self.cmd = cmd
        self.stdin = io.BytesIO()
        self.stdin.close = lambda: None         # keep the bytes readable after the recorder "closes" the pipe
        self.stderr = io.BytesIO(b"")
        self.returncode = None
        FakeFfmpeg.instances.append(self)

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        pass


@pytest.fixture
def fake_ffmpeg():
    FakeFfmpeg.instances = []
    return FakeFfmpeg


def jpeg_bytes(seed=0):
    img = np.full((48, 64, 3), seed % 255, np.uint8)
    return cv2.imencode(".jpg", img)[1].reshape(-1)


def make_recorder(cfg, tmp_path, clock, popen, free_gb=100.0, encoder=None):
    """A recorder writing into a temporary folder, with a fake clock and a fake 'GB free' reading (a list, so the
    test can change it while the recorder runs)."""
    free = [free_gb]
    rec = Recorder(cfg, tmp_path, fps=10, version="test-1", log=lambda m: None, clock=clock,
                   disk_free_gb=lambda p: free[0], popen=popen, encoder=encoder)
    rec.set_camera_info(dict(fourcc="MJPG", width=64, height=48, settings={"exposure": {"read_back": -10.0}}))
    return rec, free


STEP = 0.125   # 8 frames per second in these tests: 0.125 is exact in binary, so 1800 s is exactly 14400 steps


def feed(rec, clock, seconds, data_fn=jpeg_bytes):
    """Hand the recorder `seconds` of frames (8 per second), advancing the fake clock between frames."""
    for k in range(int(round(seconds / STEP))):
        rec.handle(data_fn(k), clock.wall(), clock.mono(), k)
        clock.advance(STEP)


def test_one_file_every_30_minutes_with_sidecars(cfg, tmp_path, fake_ffmpeg):
    cfg["recording"]["mode"] = "mjpeg"              # copy the camera's JPEGs as they are
    clock = FakeClock(wall=1_800_000_000.0, mono=100.0)
    rec, _ = make_recorder(cfg, tmp_path, clock, fake_ffmpeg)
    feed(rec, clock, 65 * 60)                       # 65 minutes of frames -> 3 segments (30 + 30 + 5 min)
    rec.close_segment("end of test")

    assert len(rec.segments) == 3
    # The file name is the wall-clock start time (local time), so files sort by time.
    names = [p.name for p in rec.segments]
    assert names == sorted(names) and all(n.startswith("rec_") and n.endswith(".mkv") for n in names)

    # The JSON sidecar of the second segment: start on both clocks, the camera settings, the software version.
    side = json.loads(rec.segments[1].with_suffix(".json").read_text())
    assert side["start_wall"] == 1_800_000_000.0 + 1800
    assert side["start_mono"] == 100.0 + 1800
    assert side["camera"]["settings"]["exposure"]["read_back"] == -10.0
    assert side["software_version"] == "test-1"
    assert side["frames"] == 30 * 60 * 8            # 30 min of 8 fps, written at close
    assert side["encoder"] == "copy (MJPEG)"        # JPEG input is copied, not re-encoded

    # The frames CSV has one row per frame with its true times.
    rows = list(csv.DictReader(open(rec.segments[2].with_suffix(".frames.csv"))))
    assert len(rows) == 5 * 60 * 8
    assert float(rows[1]["t_mono"]) - float(rows[0]["t_mono"]) == pytest.approx(STEP)

    # ffmpeg was asked to copy the MJPEG stream into Matroska, with the start time as creation_time.
    cmd = fake_ffmpeg.instances[0].cmd
    assert cmd[cmd.index("-c:v") + 1] == "copy" and cmd[cmd.index("-f", cmd.index("-c:v")) + 1] == "matroska"
    assert any(c.startswith("creation_time=") for c in cmd)
    # And it received exactly the JPEG bytes we gave it.
    assert fake_ffmpeg.instances[2].stdin.getvalue()[:2] == b"\xff\xd8"


def test_h264_mode_encodes_the_camera_jpegs(cfg, tmp_path, fake_ffmpeg):
    # The default mode: ffmpeg receives the JPEG bytes and encodes them to H.264 (about 4 GB per hour instead of
    # 20-30 GB for the JPEGs themselves).
    assert cfg["recording"]["mode"] == "h264"
    clock = FakeClock()
    rec, _ = make_recorder(cfg, tmp_path, clock, fake_ffmpeg, encoder="h264_qsv")
    feed(rec, clock, 1)
    cmd = fake_ffmpeg.instances[0].cmd
    assert cmd[cmd.index("-f") + 1] == "mjpeg"                      # input: JPEG frames
    assert cmd[cmd.index("-c:v") + 1] == "h264_qsv"                 # output: H.264 from Intel QuickSync
    assert rec.status()["encoder"] == "h264_qsv"


def test_encoder_choice_falls_back_to_x264():
    # Hardware encoders are tried first; a machine without them (or without their drivers) gets x264.
    from rt.recorder import pick_h264_encoder

    class Result:
        def __init__(self, rc):
            self.returncode = rc

    works = {"h264_qsv": 1, "h264_amf": 1, "libx264": 0}            # return code 0 = this encoder works here
    runner = lambda cmd, **kw: Result(works[cmd[cmd.index("-c:v") + 1]])  # noqa: E731
    assert pick_h264_encoder("ffmpeg", ["h264_qsv", "h264_amf", "libx264"], runner) == "libx264"
    works["h264_amf"] = 0
    assert pick_h264_encoder("ffmpeg", ["h264_qsv", "h264_amf", "libx264"], runner) == "h264_amf"


def test_decoded_frames_are_encoded_to_h264(cfg, tmp_path, fake_ffmpeg):
    # Without raw MJPEG the camera gives images; the recorder pipes raw pixels to an H.264 encoder.
    clock = FakeClock()
    rec, _ = make_recorder(cfg, tmp_path, clock, fake_ffmpeg, encoder="libx264")
    feed(rec, clock, 1, data_fn=lambda k: np.zeros((48, 64, 3), np.uint8))
    cmd = fake_ffmpeg.instances[0].cmd
    assert cmd[cmd.index("-c:v") + 1] == "libx264" and "veryfast" in cmd
    assert cmd[cmd.index("-s") + 1] == "64x48"
    assert len(fake_ffmpeg.instances[0].stdin.getvalue()) == 8 * 48 * 64 * 3     # 1 s = 8 images of 64x48x3


def test_disk_guard_warns_then_stops_recording_then_resumes(cfg, tmp_path, fake_ffmpeg):
    clock = FakeClock()
    rec, free = make_recorder(cfg, tmp_path, clock, fake_ffmpeg, free_gb=100.0)
    feed(rec, clock, 20)
    assert rec.status()["state"] == "recording" and rec.status()["disk_state"] == "ok"

    free[0] = 15.0                                  # under 20 GB: a warning, still recording
    feed(rec, clock, 20)
    assert rec.status()["disk_state"] == "low" and rec.status()["state"] == "recording"

    free[0] = 4.0                                   # under 5 GB: recording stops (detection is not touched)
    feed(rec, clock, 20)
    st = rec.status()
    assert st["disk_state"] == "stopped" and st["state"] == "stopped: low disk"
    assert st["skipped_low_disk"] > 0

    free[0] = 10.0                                  # some space back, but under 20 GB: stays stopped
    feed(rec, clock, 20)
    assert rec.status()["disk_state"] == "stopped"

    free[0] = 50.0                                  # above 20 GB again: recording resumes in a new file
    feed(rec, clock, 20)
    assert rec.status()["state"] == "recording"
    assert len(rec.segments) == 2


def test_offer_never_blocks_and_counts_drops(cfg, tmp_path, fake_ffmpeg):
    # The capture thread calls offer(); if the writer is stuck, frames are dropped from the recording, never waited
    # for (detection must not slow down because of the disk).
    cfg["recording"]["queue_frames"] = 3
    rec, _ = make_recorder(cfg, tmp_path, FakeClock(), fake_ffmpeg)
    results = [rec.offer(jpeg_bytes(), 0.0, 0.0, k) for k in range(5)]   # the writer thread is not running
    assert results == [True, True, True, False, False]
    assert rec.dropped == 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="ffmpeg/ffprobe not on PATH")
@pytest.mark.parametrize("mode,codec", [("mjpeg", "mjpeg"), ("h264", "h264")])
def test_real_ffmpeg_writes_a_playable_file(cfg, tmp_path, mode, codec):
    # The real thing: 3 s of 10 fps JPEG frames through the recorder thread and ffmpeg into an .mkv file, once copied
    # as MJPEG and once encoded to H.264 (x264 here, so the test does not depend on this PC's GPU).
    cfg["recording"]["mode"] = mode
    rec = Recorder(cfg, tmp_path, fps=10, version="test", log=lambda m: None,
                   encoder="libx264" if mode == "h264" else None)
    rec.start()
    for k in range(30):
        img = np.full((48, 64, 3), (k * 8) % 255, np.uint8)
        rec.offer(cv2.imencode(".jpg", img)[1].reshape(-1), 1_800_000_000.0 + k / 10, 100.0 + k / 10, k)
    rec.stop()
    path = rec.segments[0]
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,nb_read_frames,width,height", "-of", "json", str(path)],
                         capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    assert s["codec_name"] == codec and int(s["nb_read_frames"]) == 30 and (s["width"], s["height"]) == (64, 48)
    side = json.loads(path.with_suffix(".json").read_text())
    assert side["frames"] == 30 and side["ffmpeg_returncode"] == 0
