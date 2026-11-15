"""Tests of the recorder (rt/recorder.py): files every 30 minutes with their sidecars, a new file after a gap or a
camera loss, the disk-space guard, and the watchdog that handles an ffmpeg that freezes or dies.

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

from .conftest import plenty_of_free_disk


class FakeFfmpeg:
    """Stands in for subprocess.Popen(ffmpeg ...): remembers the command, collects what is written to stdin, and
    says it is still running (poll() returns None) until it is waited for or killed."""

    instances = []

    def __init__(self, cmd, stdin=None, stdout=None, stderr=None):
        self.cmd = cmd
        self.stderr_target = stderr             # where the recorder sent ffmpeg's messages (should be a log file)
        self.stdin = io.BytesIO()
        self.stdin.close = lambda: None         # keep the bytes readable after the recorder "closes" the pipe
        self.returncode = None
        self.killed = False
        type(self).instances.append(self)         # each fake class keeps its own list

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.killed = True
        if self.returncode is None:
            self.returncode = -9


@pytest.fixture
def fake_ffmpeg():
    FakeFfmpeg.instances = []
    return FakeFfmpeg


def jpeg_bytes(seed=0):
    img = np.full((48, 64, 3), seed % 255, np.uint8)
    return cv2.imencode(".jpg", img)[1].reshape(-1)


def make_recorder(cfg, tmp_path, clock, popen, free_gb=100.0, encoder="libx264", logs=None):
    """A recorder writing into a temporary folder, with a fake clock and a fake 'GB free' reading (a list, so the
    test can change it while the recorder runs). The encoder is given, so no real ffmpeg is probed. No watchdog
    thread: tests call watchdog_check() themselves when they want it to look."""
    free = [free_gb]
    rec = Recorder(cfg, tmp_path, fps=10, version="test-1", log=(logs.append if logs is not None else lambda m: None),
                   clock=clock, disk_free_gb=lambda folder: free[0], popen=popen, encoder=encoder, watchdog_poll_s=0)
    rec.set_camera_info(dict(fourcc="MJPG", width=64, height=48, settings={"exposure": {"read_back": -10.0}}))
    return rec, free


STEP = 0.125   # 8 frames per second in these tests: 0.125 is exact in binary, so 1800 s is exactly 14400 steps


def feed(rec, clock, seconds, data_fn=jpeg_bytes):
    """Hand the recorder `seconds` of frames (8 per second), advancing the fake clock between frames."""
    for frame_no in range(int(round(seconds / STEP))):
        rec.handle(data_fn(frame_no), clock.wall(), clock.mono(), frame_no)
        clock.advance(STEP)


def test_one_file_every_30_minutes_with_sidecars(cfg, tmp_path, fake_ffmpeg):
    clock = FakeClock(wall=1_800_000_000.0, mono=100.0)
    rec, _ = make_recorder(cfg, tmp_path, clock, fake_ffmpeg)
    feed(rec, clock, 65 * 60)                       # 65 minutes of frames -> 3 segments (30 + 30 + 5 min)
    rec.close_segment("end of test")

    assert len(rec.segments) == 3
    # The file name is the wall-clock start time (local time), so files sort by time.
    names = [path.name for path in rec.segments]
    assert names == sorted(names) and all(n.startswith("rec_") and n.endswith(".mkv") for n in names)

    # The JSON sidecar of the second segment: start on both clocks, the camera settings, the software version.
    side = json.loads(rec.segments[1].with_suffix(".json").read_text())
    assert side["start_wall"] == 1_800_000_000.0 + 1800
    assert side["start_mono"] == 100.0 + 1800
    assert side["camera"]["settings"]["exposure"]["read_back"] == -10.0
    assert side["software_version"] == "test-1"
    assert side["frames"] == 30 * 60 * 8            # 30 min of 8 fps, written at close
    assert side["encoder"] == "libx264"
    assert side["closed_because"] == "segment time" and side["status"] == "closed"

    # The frames CSV has one row per frame with its true times.
    rows = list(csv.DictReader(open(rec.segments[2].with_suffix(".frames.csv"))))
    assert len(rows) == 5 * 60 * 8
    assert float(rows[1]["t_mono"]) - float(rows[0]["t_mono"]) == pytest.approx(STEP)

    # ffmpeg was asked to encode the JPEG frames to H.264 in Matroska, with the start time as creation_time.
    cmd = fake_ffmpeg.instances[0].cmd
    assert cmd[cmd.index("-c:v") + 1] == "libx264" and cmd[cmd.index("-f", cmd.index("-c:v")) + 1] == "matroska"
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
        def __init__(self, return_code):
            self.returncode = return_code

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


def test_offer_never_blocks_drops_the_oldest_and_counts(cfg, tmp_path, fake_ffmpeg):
    # The capture thread calls offer(); if the writer is stuck, frames are dropped from the recording, never waited
    # for (detection must not slow down because of the disk). The queue is small on purpose: a 1080p frame is
    # ~6 MB, so 10 frames (1 s) is already 60 MB. When it is full, the OLDEST frame is thrown away.
    cfg["recording"]["queue_frames"] = 3
    rec, _ = make_recorder(cfg, tmp_path, FakeClock(), fake_ffmpeg)
    results = [rec.offer(jpeg_bytes(), 0.0, 0.0, camera_frame_no) for camera_frame_no in range(5)]
    assert results == [True] * 5                    # offer() always accepts the newest frame ...
    assert rec.dropped == 2                         # ... and dropped the 2 oldest to make room
    queued = [rec.q.get_nowait()[3] for _ in range(rec.q.qsize())]
    assert queued == [2, 3, 4]                      # the queue holds the 3 newest frames, in order


def test_default_queue_is_about_one_second(cfg):
    # 10 frames at 10 fps: at most ~60 MB of 1080p BGR frames waiting.
    assert cfg["recording"]["queue_frames"] <= cfg["capture"]["fps"]


@pytest.mark.parametrize("bad, message", [({"mode": "mjpeg"}, "not possible"), ({"mode": "avi"}, "h264"),
                                          ({"all_frames": True}, "removed")])
def test_impossible_recording_settings_are_refused_with_a_reason(cfg, tmp_path, bad, message):
    # mjpeg (copy the camera's JPEGs) cannot work: DirectShow hands rt decoded images. rt must say so at start
    # instead of silently recording something else.
    cfg["recording"].update(bad)
    with pytest.raises(SystemExit, match=message):
        Recorder(cfg, tmp_path, fps=10)


def test_encoder_is_probed_once_at_start_and_logged(cfg, tmp_path):
    # choose_encoder() runs the probe (here: a fake runner where only AMF works) and logs the encoder in use.
    from rt import recorder as recorder_module

    logs = []
    rec = Recorder(cfg, tmp_path, fps=10, log=logs.append, watchdog_poll_s=0)
    probed = []

    def fake_probe(ffmpeg, name, runner=None):
        probed.append(name)
        # (worked, ffmpeg's first error line)
        return (True, "") if name == "h264_amf" else (False, "Error creating a MFX session: -9.")

    original_probe, original_find = recorder_module.probe_encoder, recorder_module.find_ffmpeg
    recorder_module.probe_encoder = fake_probe
    recorder_module.find_ffmpeg = lambda ffmpeg: "C:/ffmpeg/bin/ffmpeg.exe"   # pretend ffmpeg is installed
    try:
        assert rec.choose_encoder() == "h264_amf"
        assert rec.choose_encoder() == "h264_amf"   # second call: no new probe
    finally:
        recorder_module.probe_encoder, recorder_module.find_ffmpeg = original_probe, original_find
    assert probed == ["h264_qsv", "h264_amf"]
    assert any("recording encoder: h264_amf" in line and "h264_qsv failed" in line for line in logs)
    # The failed encoder's reason (from ffmpeg's error output) is in the log line too.
    assert any("h264_qsv failed (Error creating a MFX session: -9.)" in line for line in logs)


# ------------------------------------------------------------------------------------------------ ffmpeg missing
# On a fresh Windows laptop ffmpeg is often not installed (or not on PATH). Every encoder probe then "failed", and rt
# used to log "no H.264 encoder passed the probe ... trying libx264 anyway", which hides the real problem. Now a
# missing ffmpeg is told apart from an encoder that does not work, recording is switched off with a clear message,
# and detection carries on (run.py does not start the recorder at all).

class ProbeResult:
    """What subprocess.run returns, reduced to what the probe reads: the exit code and the error output."""
    def __init__(self, returncode, stderr=b""):
        self.returncode, self.stderr = returncode, stderr


def test_probe_says_when_ffmpeg_itself_cannot_be_started():
    from rt.recorder import FfmpegNotFound, probe_encoder

    def runner(cmd, **kw):
        raise FileNotFoundError(2, "The system cannot find the file specified")   # what Windows says

    with pytest.raises(FfmpegNotFound):
        probe_encoder("ffmpeg", "libx264", runner)


def test_probe_gives_the_first_useful_line_of_an_encoder_failure():
    from rt.recorder import probe_encoder
    # ffmpeg runs, but this encoder does not work here: exit code 1, and the reason in its error output (with an
    # empty first line, which must be skipped).
    stderr = b"\n[h264_qsv @ 000001] Error creating a MFX session: -9.\nError initializing output stream\n"
    worked, detail = probe_encoder("ffmpeg", "h264_qsv", lambda cmd, **kw: ProbeResult(1, stderr))
    assert worked is False
    assert detail == "[h264_qsv @ 000001] Error creating a MFX session: -9."
    # An encoder that works: (True, "").
    assert probe_encoder("ffmpeg", "libx264", lambda cmd, **kw: ProbeResult(0)) == (True, "")


def test_missing_ffmpeg_turns_recording_off_with_a_clear_message(cfg, tmp_path, monkeypatch):
    from rt import recorder as recorder_module
    # find_ffmpeg() looks the program up (on PATH, or the configured full path). Here: not found.
    monkeypatch.setattr(recorder_module, "find_ffmpeg", lambda ffmpeg: None)
    cfg["capture"]["ffmpeg"] = "C:/tools/ffmpeg.exe"
    logs = []
    rec = Recorder(cfg, tmp_path, fps=10, log=logs.append, watchdog_poll_s=0)
    assert rec.choose_encoder() is None                 # no encoder: recording cannot work
    msg = rec.unavailable
    # The message names the configured path, says recording is off, and how to fix it.
    assert "ffmpeg not found at 'C:/tools/ffmpeg.exe'" in msg
    assert "recording is off" in msg and "winget install Gyan.FFmpeg" in msg and "capture.ffmpeg" in msg
    assert any(line == "WARNING recording: " + msg for line in logs)
    # Not the old, misleading message.
    assert not any("no H.264 encoder passed the probe" in line for line in logs)
    # The recorder refuses frames (so it never tries to start ffmpeg) and says why on the status page.
    assert rec.offer(b"frame", 0.0, 0.0, 1) is False
    assert rec.status()["state"].startswith("off: ffmpeg not found")


def test_ffmpeg_on_path_but_failing_to_start_is_also_reported_as_missing(cfg, tmp_path, monkeypatch):
    # PATH says ffmpeg is there, but starting it fails (a broken install): same outcome as "not found".
    from rt import recorder as recorder_module
    monkeypatch.setattr(recorder_module, "find_ffmpeg", lambda ffmpeg: "/usr/bin/ffmpeg")

    def probe(ffmpeg, name, runner=None):
        raise recorder_module.FfmpegNotFound("cannot start")

    monkeypatch.setattr(recorder_module, "probe_encoder", probe)
    rec = Recorder(cfg, tmp_path, fps=10, log=lambda m: None, watchdog_poll_s=0)
    assert rec.choose_encoder() is None and "ffmpeg not found" in rec.unavailable


def test_ffmpeg_present_but_every_encoder_failing_keeps_trying_x264(cfg, tmp_path, monkeypatch):
    # ffmpeg is installed but no encoder passed the probe: the old behaviour (try libx264 anyway), now with reasons.
    from rt import recorder as recorder_module
    monkeypatch.setattr(recorder_module, "find_ffmpeg", lambda ffmpeg: "/usr/bin/ffmpeg")
    monkeypatch.setattr(recorder_module, "probe_encoder", lambda ffmpeg, name, runner=None: (False, "no " + name))
    logs = []
    rec = Recorder(cfg, tmp_path, fps=10, log=logs.append, watchdog_poll_s=0)
    assert rec.choose_encoder() == "libx264" and rec.unavailable is None
    line = next(x for x in logs if "no H.264 encoder passed the probe" in x)
    assert "h264_qsv failed (no h264_qsv)" in line and "libx264 failed (no libx264)" in line


def test_status_page_shows_the_missing_ffmpeg_notice_in_yellow(cfg):
    from rt.recorder import ffmpeg_missing_message
    from rt.status import StatusBoard, render_html
    board = StatusBoard(cfg)
    board.notices.append(ffmpeg_missing_message("ffmpeg"))     # what run.py does when recording is off
    page = render_html(board.payload())
    assert '<div class="notice">ffmpeg not found at &#x27;ffmpeg&#x27;: recording is off' in page


# ------------------------------------------------------------------------------------------------ gaps and losses
def test_a_gap_in_the_frames_starts_a_new_file(cfg, tmp_path, fake_ffmpeg):
    # The video file plays at a fixed 10 fps. If 1 s of frames is missing and the same file went on, everything
    # after the gap would play (and replay) 1 s too early. So a gap of more than 2 frame periods (0.2 s) closes the
    # file, and the next frame starts a new one with its own start time.
    clock = FakeClock(wall=1_800_000_000.0, mono=100.0)
    rec, _ = make_recorder(cfg, tmp_path, clock, fake_ffmpeg)
    feed(rec, clock, 10)
    clock.advance(1.0)                              # 1 s without frames
    feed(rec, clock, 10)
    rec.close_segment("end of test")
    assert len(rec.segments) == 2
    first = json.loads(rec.segments[0].with_suffix(".json").read_text())
    second = json.loads(rec.segments[1].with_suffix(".json").read_text())
    assert first["closed_because"].startswith("frame gap of 1.1")    # 1 s + the normal 0.125 s step
    assert second["start_mono"] == pytest.approx(100.0 + 10 + 1.0)
    assert second["start_wall"] == pytest.approx(1_800_000_000.0 + 10 + 1.0)


def test_normal_frame_jitter_does_not_split_files(cfg, tmp_path, fake_ffmpeg):
    # Frames 0.15 s apart (a missed camera frame now and then) are under the 0.2 s limit: one file.
    clock = FakeClock()
    rec, _ = make_recorder(cfg, tmp_path, clock, fake_ffmpeg)
    for frame_no in range(50):
        rec.handle(jpeg_bytes(), clock.wall(), clock.mono(), frame_no)
        clock.advance(0.15)
    assert len(rec.segments) == 1


def test_camera_loss_closes_the_file_and_the_next_one_has_the_new_settings(cfg, tmp_path, fake_ffmpeg):
    # The capture calls camera_lost() when the camera disappears. The file is closed; when the camera is back it
    # was reopened (its settings applied and read back again), and the new file's sidecar must hold those.
    clock = FakeClock()
    rec, _ = make_recorder(cfg, tmp_path, clock, fake_ffmpeg)
    feed(rec, clock, 5)
    rec.camera_lost("camera lost")
    rec.set_camera_info(dict(fourcc="MJPG", settings={"exposure": {"read_back": -9.0}}))   # read back at reopen
    feed(rec, clock, 5)
    rec.close_segment("end of test")
    assert len(rec.segments) == 2
    first = json.loads(rec.segments[0].with_suffix(".json").read_text())
    second = json.loads(rec.segments[1].with_suffix(".json").read_text())
    assert first["closed_because"] == "camera lost"
    assert second["camera"]["settings"]["exposure"]["read_back"] == -9.0


def test_camera_loss_splits_the_file_at_the_loss_even_when_the_writer_is_behind(cfg, tmp_path, fake_ffmpeg):
    # Two threads are involved here. The CAPTURE thread puts frames in a queue (offer) and, when the camera
    # disappears, says so (camera_lost). The WRITER thread (the Recorder itself, started with start()) takes frames
    # out of the queue and writes them to ffmpeg. The writer can fall behind: ffmpeg is busy, the CPU is loaded.
    #
    # The bug this test guards: camera_lost() used to make the writer close the file at the NEXT frame it wrote,
    # whatever that frame was. If frames from before the loss were still waiting in the queue, the file was cut too
    # early and those frames went into the next file. Whether that happened depended on how far behind the writer
    # was at that instant, so a test could pass many times and then fail once (seen: 1 run in ~20).
    #
    # Here we make the "writer is behind" case happen every time, with no timing involved: we put ALL the frames
    # and the camera loss in the queue BEFORE the writer thread even starts. The writer must still split the
    # recording exactly where the camera was lost: 5 frames in the first file, 3 in the second.
    cfg["recording"]["queue_frames"] = 100           # room for every frame of the test (none dropped)
    clock = FakeClock()
    rec, _ = make_recorder(cfg, tmp_path, clock, fake_ffmpeg)
    for frame_no in range(5):                        # 5 frames before the loss
        rec.offer(jpeg_bytes(), clock.wall(), clock.mono(), frame_no)
        clock.advance(0.1)
    rec.camera_lost("camera lost")
    for frame_no in range(3):                        # 3 frames after the camera came back. No time gap, on purpose:
        rec.offer(jpeg_bytes(), clock.wall(), clock.mono(), frame_no)   # only camera_lost may split the file here
        clock.advance(0.1)
    rec.start()                                      # the writer starts only now: it is as far behind as it can be
    rec.stop()                                       # writes everything queued, then closes the last file

    assert len(rec.segments) == 2
    # Both files started in the same second, and this fake ffmpeg never creates the .mkv (a real one may also be
    # late to): the second file must still get its own name ("_1"), or it would overwrite the first one.
    assert rec.segments[1].name == rec.segments[0].stem + "_1.mkv"
    first = json.loads(rec.segments[0].with_suffix(".json").read_text())
    second = json.loads(rec.segments[1].with_suffix(".json").read_text())
    assert (first["frames"], first["closed_because"]) == (5, "camera lost")
    assert (second["frames"], second["closed_because"]) == (3, "stopped")


def test_an_empty_queue_check_does_not_close_the_file_before_the_frames_queued_just_before_a_loss(
        cfg, tmp_path, fake_ffmpeg):
    # The writer thread also checks for a camera loss when it finds the queue empty (so a file is closed even if
    # the camera never comes back). But "the queue was empty" can be out of date by the time it acts: in between,
    # the capture thread may have queued more frames and then reported the loss. Those frames still belong in
    # the current file.
    #
    # This test plays the writer thread's part by hand, step by step, so the order of events is fixed:
    #   1. three frames are offered, then the camera is lost;
    #   2. the writer writes frame 1 (the file opens);
    #   3. the writer does its "queue empty" check, as if it had looked just before frames 2 and 3 arrived:
    #      the file must stay open, because frames 2 and 3 (from before the loss) are not written yet;
    #   4. the writer writes frames 2 and 3, then does the check again: now the file closes, with 3 frames.
    cfg["recording"]["queue_frames"] = 100
    clock = FakeClock()
    rec, _ = make_recorder(cfg, tmp_path, clock, fake_ffmpeg)
    for frame_no in range(3):
        rec.offer(jpeg_bytes(), clock.wall(), clock.mono(), frame_no)
        clock.advance(0.1)
    rec.camera_lost("camera lost")

    rec.write_queued(rec.q.get_nowait())             # step 2: frame 1
    rec._apply_close_request()                       # step 3: the "queue empty" check
    assert rec.seg is not None                       # still open: frames 2 and 3 are not written yet

    rec.write_queued(rec.q.get_nowait())             # step 4: frames 2 and 3, then the check again
    rec.write_queued(rec.q.get_nowait())
    rec._apply_close_request()
    assert rec.seg is None                           # closed now
    side = json.loads(rec.segments[0].with_suffix(".json").read_text())
    assert (side["frames"], side["closed_because"]) == (3, "camera lost")


def test_sidecar_is_written_at_open_and_finalised_at_close(cfg, tmp_path, fake_ffmpeg):
    # If the laptop dies mid-file, the sidecar must already exist (start times, camera settings), and the frames
    # CSV must be on disk up to the last flush (every 30 rows), not only in memory.
    clock = FakeClock()
    rec, _ = make_recorder(cfg, tmp_path, clock, fake_ffmpeg)
    feed(rec, clock, 30 * STEP)                     # 30 frames
    path = rec.segments[0]
    side = json.loads(path.with_suffix(".json").read_text())
    assert side["status"].startswith("open") and "end_wall" not in side
    lines_on_disk = path.with_suffix(".frames.csv").read_text().splitlines()
    assert len(lines_on_disk) == 1 + 30             # header + 30 rows, flushed while the file is still open
    rec.close_segment("end of test")
    side = json.loads(path.with_suffix(".json").read_text())
    assert side["status"] == "closed" and side["frames"] == 30 and "end_wall" in side


def test_ffmpeg_messages_go_to_a_log_file_not_a_pipe(cfg, tmp_path, fake_ffmpeg):
    # A pipe nobody reads fills up (~64 KB) and then ffmpeg, and our writer with it, freeze. A file never does.
    clock = FakeClock()
    rec, _ = make_recorder(cfg, tmp_path, clock, fake_ffmpeg)
    feed(rec, clock, 1)
    target = fake_ffmpeg.instances[0].stderr_target
    assert hasattr(target, "write") and target.name.endswith(".ffmpeg.log")
    rec.close_segment("end of test")
    assert json.loads(rec.segments[0].with_suffix(".json").read_text())["ffmpeg_log"].endswith(".ffmpeg.log")


# ------------------------------------------------------------------------------------------------ watchdog
class StallingFfmpeg(FakeFfmpeg):
    """An ffmpeg that stops reading its input at the 5th frame. In real life the write would block forever; here
    the write instead moves the fake clock 11 s forward and lets the watchdog look (what the watchdog thread would
    do while the writer is stuck). When the watchdog kills it, the stuck write fails with BrokenPipeError, which is
    what really happens to a write into a killed process."""

    recorder = None
    clock = None

    def __init__(self, cmd, **kw):
        super().__init__(cmd, **kw)
        self.writes = 0
        stall = self

        class StallingPipe(io.BytesIO):
            def write(self, data):
                stall.writes += 1
                if stall.writes >= 5:
                    StallingFfmpeg.clock.advance(11.0)
                    StallingFfmpeg.recorder.watchdog_check()
                    if stall.killed:
                        raise BrokenPipeError("ffmpeg was killed")
                return super().write(data)

            def close(self):
                pass

        self.stdin = StallingPipe()


def test_watchdog_kills_a_stuck_ffmpeg_and_starts_a_new_file(cfg, tmp_path):
    StallingFfmpeg.instances = []
    clock = FakeClock()
    logs = []
    rec, _ = make_recorder(cfg, tmp_path, clock, StallingFfmpeg, logs=logs)
    StallingFfmpeg.recorder, StallingFfmpeg.clock = rec, clock
    feed(rec, clock, 5 * STEP)                       # the 5th frame gets stuck; the watchdog kills ffmpeg
    first = StallingFfmpeg.instances[0]
    assert first.killed
    status = rec.status()
    assert status["state"].startswith("error") and "did not accept data for 10 s" in status["state"]
    assert status["errors"] == 1
    assert any("did not accept data" in line for line in logs)
    side = json.loads(rec.segments[0].with_suffix(".json").read_text())
    assert side["closed_because"].startswith("failed: ffmpeg did not accept data")
    # 10 s later (retry_wait_s) the next frame opens a new file with a new ffmpeg.
    clock.advance(10.0)
    rec.handle(jpeg_bytes(), clock.wall(), clock.mono(), 99)
    assert len(StallingFfmpeg.instances) == 2 and rec.status()["state"] == "recording"


class ExitingFfmpeg(FakeFfmpeg):
    """An ffmpeg that dies (exit code 1) after receiving 3 frames; writes after that fail like a broken pipe."""

    def __init__(self, cmd, **kw):
        super().__init__(cmd, **kw)
        proc = self

        class DyingPipe(io.BytesIO):
            def write(self, data):
                if proc.returncode is not None:
                    raise BrokenPipeError("ffmpeg has exited")
                result = super().write(data)
                if len(self.getvalue()) >= 3 * len(data):
                    proc.returncode = 1
                return result

            def close(self):
                pass

        self.stdin = DyingPipe()


def test_watchdog_notices_ffmpeg_exiting(cfg, tmp_path):
    clock = FakeClock()
    rec, _ = make_recorder(cfg, tmp_path, clock, ExitingFfmpeg)
    feed(rec, clock, 3 * STEP)                       # ffmpeg takes 3 frames and exits
    rec.watchdog_check()                             # the watchdog thread's once-a-second look
    assert "ffmpeg exited with code 1" in rec.status()["state"]
    rec.handle(jpeg_bytes(), clock.wall(), clock.mono(), 3)    # the writer acts on it: file closed, retry later
    assert rec.seg is None and rec.errors == 1


def test_recording_gives_up_after_repeated_failures_but_never_blocks(cfg, tmp_path):
    # An ffmpeg that always dies: after max_failures (5) failures in a row, recording stops for this run. The
    # status page says "failed", and offer() keeps returning at once (detection never notices).
    clock = FakeClock()
    rec, _ = make_recorder(cfg, tmp_path, clock, ExitingFfmpeg)
    feed(rec, clock, 120)                            # 2 minutes: enough for 5 failures 10 s apart
    status = rec.status()
    assert rec.gave_up and status["state"].startswith("failed: gave up after 5")
    assert status["errors"] == 5
    assert rec.offer(jpeg_bytes(), 0.0, 0.0, 0) is False      # nothing is queued any more
    files_before = len(rec.segments)
    feed(rec, clock, 30)
    assert len(rec.segments) == files_before                  # no new attempts


def test_status_page_shows_recording_failures(cfg, tmp_path):
    from rt.status import StatusBoard
    clock = FakeClock()
    rec, _ = make_recorder(cfg, tmp_path, clock, ExitingFfmpeg)
    feed(rec, clock, 120)
    board = StatusBoard(cfg, clock=clock)
    board.recorder = rec
    assert any(w.startswith("recording failed") for w in board.payload()["warnings"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="ffmpeg/ffprobe not on PATH")
@pytest.mark.parametrize("kind", ["jpeg", "bgr"])
def test_real_ffmpeg_writes_a_playable_file(cfg, tmp_path, kind):
    # The real thing: 3 s of 10 fps frames through the recorder thread and ffmpeg into an H.264 .mkv file (x264
    # here, so the test does not depend on this PC's GPU). "bgr" = decoded images, what DirectShow gives; "jpeg" =
    # camera JPEG bytes, what a passthrough backend (V4L2) gives.
    cfg["recording"]["queue_frames"] = 100         # the test offers 30 frames at once, faster than real time:
                                                    # a big queue so none is dropped before ffmpeg has started
    rec = Recorder(cfg, tmp_path, fps=10, version="test", log=lambda m: None, encoder="libx264",
                   disk_free_gb=plenty_of_free_disk)   # not the real disk: see plenty_of_free_disk in conftest.py
    rec.start()
    for frame_no in range(30):
        img = np.full((48, 64, 3), (frame_no * 8) % 255, np.uint8)
        data = cv2.imencode(".jpg", img)[1].reshape(-1) if kind == "jpeg" else img
        rec.offer(data, 1_800_000_000.0 + frame_no / 10, 100.0 + frame_no / 10, frame_no)
    rec.stop()
    path = rec.segments[0]
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,nb_read_frames,width,height", "-of", "json", str(path)],
                         capture_output=True, text=True).stdout
    stream = json.loads(out)["streams"][0]
    assert stream["codec_name"] == "h264" and int(stream["nb_read_frames"]) == 30
    assert (stream["width"], stream["height"]) == (64, 48)
    side = json.loads(path.with_suffix(".json").read_text())
    assert side["frames"] == 30 and side["ffmpeg_returncode"] == 0
    assert path.with_suffix(".ffmpeg.log").exists()


# ------------------------------------------------------------------------------------------------ replay
def test_frame_times_from_the_csv(tmp_path):
    # load_frame_times reads the recorded time of each frame. If the laptop lost power while a row was being
    # written, the last line is cut short: it is ignored instead of crashing the replay.
    from rt.capture import load_frame_times, recorded_frame_time

    video = tmp_path / "rec_x.mkv"
    assert load_frame_times(video) is None                       # no CSV next to the file
    video.with_suffix(".frames.csv").write_text("n,t_wall,t_mono,camera_frame_no\n0,100.0,5.0,0\n1,100.1,5.1,3\n"
                                                "2,100.35,5.35,10\n3,100.4")
    times = load_frame_times(video)
    assert times == [100.0, 100.1, 100.35]
    # Past the end of the CSV, times continue from the last row at the nominal rate (10 fps here).
    assert recorded_frame_time(times, 2, 10.0) == 100.35
    assert recorded_frame_time(times, 4, 10.0) == pytest.approx(100.55)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="ffmpeg/ffprobe not on PATH")
def test_replay_of_a_recording_uses_the_frames_csv_times(cfg, tmp_path):
    # Record 20 frames whose real spacing is uneven (0.1 s and 0.15 s alternately: missed camera frames). The video
    # file itself plays at a nominal 10 fps, so frame 19 would be at +1.9 s from it, while it really was at
    # +2.35 s. Replaying the file through rt (FileCapture, as `run.py --source rec_....mkv` does) must give each
    # frame its recorded time from the .frames.csv.
    from rt.capture import FileCapture

    cfg["recording"]["queue_frames"] = 100
    rec = Recorder(cfg, tmp_path, fps=10, version="test", log=lambda m: None, encoder="libx264",
                   disk_free_gb=plenty_of_free_disk)   # not the real disk: see plenty_of_free_disk in conftest.py
    rec.start()
    t_wall, recorded = 1_800_000_000.0, []
    for frame_no in range(20):
        recorded.append(t_wall)
        rec.offer(np.full((48, 64, 3), frame_no * 10, np.uint8), t_wall, 100.0 + (t_wall - 1_800_000_000.0),
                  frame_no)
        t_wall += 0.1 if frame_no % 2 == 0 else 0.15
    rec.stop()
    assert len(rec.segments) == 1                    # 0.15 s < 0.2 s: no split

    cfg["capture"]["fps"] = 10.0                     # process every frame of the 10 fps file
    replay = FileCapture(cfg, rec.segments[0], log=lambda m: None)
    assert replay.clock_how.startswith("frames.csv")
    replay.start()
    times = [frame.t for frame in replay.frames()]
    assert times == pytest.approx(recorded, abs=1e-6)
