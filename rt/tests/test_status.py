"""Tests of the status page (rt/status.py): the numbers it shows, the pass list with thumbnails, and the web server.

The page is what someone looks at on a phone during the race: is it running at 10 fps, is the camera OK, is the
clock right, is it recording, which cars passed. The tests build the same data the page shows (`payload()`).
"""
import json
import urllib.request

import numpy as np
import pytest

from rt.backoff import FakeClock
from rt.status import StatusBoard, StatusServer, StatusSink, TimeChecker, render_html

from .conftest import make_frame


class FakeCapture:
    """What the board reads from the camera capture: a status() dict and a dropped-frame count."""
    dropped = 3

    def status(self):
        return dict(state="streaming", detail="", device="0", info=dict(fourcc="MJPG", width=1920, height=1080, fps=30,
                    settings={"exposure": dict(requested=-10, accepted=True, read_back=-10.0)}),
                    read_mode="retrieve_every_frame", raw_mjpeg=False, camera_frames=900, failed_reads=0, corrupt=0,
                    reopens=0, last_frame_age_s=0.03)


class FakeRecorder:
    def __init__(self, disk_state="ok", free_gb=120.0):
        self.disk_state, self.free_gb = disk_state, free_gb

    def status(self):
        return dict(state="recording", file="rec_2026-11-21_09-00-00.mkv", frames=100, dropped=0,
                    free_gb=self.free_gb, disk_state=self.disk_state, warn_free_gb=20, stop_free_gb=5,
                    encoder="libx264")


def board_running(cfg, seconds=12.0, fps=10):
    """A board that has seen `seconds` of frames at `fps` (fake clock, small frames)."""
    clock = FakeClock(wall=1_800_000_000.0, mono=0.0)
    b = StatusBoard(cfg, clock)
    b.capture, b.recorder = FakeCapture(), FakeRecorder()
    img = np.zeros((90, 160, 3), np.uint8)
    for frame_no in range(int(seconds * fps)):
        clock.advance(1.0 / fps)
        b.on_frame(make_frame(clock.wall(), frame_no, img))
    return b, clock


def test_processing_fps_and_camera_state(cfg):
    b, _ = board_running(cfg)
    p = b.payload()
    assert p["processing"]["fps"] == pytest.approx(10.0, abs=0.2)   # frames in the last 10 s / 10 s
    assert p["processing"]["dropped"] == 3
    assert p["camera"]["state"] == "streaming"
    assert p["camera"]["info"]["settings"]["exposure"]["read_back"] == -10.0
    assert p["recording"]["state"] == "recording"
    assert p["warnings"] == []                                      # all good: nothing in red


def test_fps_drops_to_zero_when_frames_stop(cfg):
    # No frame for more than the 10 s window: fps 0, and a warning. (The launcher restarts rt on this.)
    b, clock = board_running(cfg)
    clock.advance(11)
    p = b.payload()
    assert p["processing"]["fps"] == 0.0
    assert any("processing" in w for w in p["warnings"])


def test_clock_offset_above_half_a_second_is_red(cfg):
    b, _ = board_running(cfg)
    b.time = dict(offset_s=0.8, source="SNTP 192.168.88.1", warnings=[], checked=None)
    p = b.payload()
    assert p["clock"]["bad"] is True
    assert any("clock offset" in w for w in p["warnings"])
    assert 'class="bad">+0.800 s' in render_html(p)                # red on the page

    b.time = dict(offset_s=0.02, source="SNTP 192.168.88.1", warnings=[], checked=None)
    assert b.payload()["clock"]["bad"] is False


def test_time_checker_fills_the_offset(cfg):
    # The checker asks rt/timesync.py for the offset; here a fake measurement says +0.123 s.
    b, _ = board_running(cfg, seconds=1)
    TimeChecker(cfg, b, measure=lambda c: (0.123, "fake"), check=lambda c, log: []).once()
    assert b.time["offset_s"] == pytest.approx(0.123) and b.time["source"] == "fake"


def test_disk_warnings(cfg):
    b, _ = board_running(cfg, seconds=1)
    b.recorder = FakeRecorder("low", 15.0)
    assert any("disk low" in w for w in b.payload()["warnings"])
    b.recorder = FakeRecorder("stopped", 4.0)
    assert any("recording STOPPED" in w for w in b.payload()["warnings"])


def test_last_passes_with_number_thumbnail_and_t_cross(cfg):
    b, clock = board_running(cfg, seconds=15)
    sink = StatusSink(b)
    now = clock.wall()
    # Events as rt emits them for one pass (times in epoch seconds).
    sink.event(dict(type="pass_start", pass_id=7, t_enter=now - 5, t_video=now - 4.5))
    sink.event(dict(type="pass_provisional", pass_id=7, number=14))
    sink.event(dict(type="pass_end", pass_id=7, t_enter=now - 5, t_cross=now - 3.0, t_exit=now - 1))
    sink.event(dict(type="pass_identity", pass_id=7, t_enter=now - 5, t_cross=now - 3.0, number=14, team="Team 14",
                    confidence=0.97, auto_confirm=True))
    p = b.payload()["passes"][0]
    assert (p["pass_id"], p["number"], p["confidence"], p["auto_confirm"]) == (7, 14, 0.97, True)
    assert p["provisional"] == 14
    # The thumbnail is the kept frame nearest to t_cross (frames are kept 5 per second, so within 0.1 s).
    assert p["thumb_t"] == pytest.approx(now - 3.0, abs=0.1)
    assert p["thumb_url"].startswith("/thumb/7.jpg")
    assert b.thumb(7)[:2] == b"\xff\xd8"                            # a JPEG image


def test_only_the_last_20_passes_are_kept(cfg):
    b, clock = board_running(cfg, seconds=1)
    for pid in range(25):
        b.on_event(dict(type="pass_start", pass_id=pid, t_enter=clock.wall(), t_video=clock.wall()))
    ids = [p["pass_id"] for p in b.payload()["passes"]]
    assert ids == list(range(24, 4, -1))                           # newest first, 20 of them


def test_http_server_serves_page_json_and_thumbnails(cfg):
    # Start the real web server on a free port of this machine and read from it like a phone would.
    b, clock = board_running(cfg, seconds=15)
    b.on_event(dict(type="pass_start", pass_id=1, t_enter=clock.wall() - 2, t_video=clock.wall() - 2))
    srv = StatusServer(b, "127.0.0.1", 0).start()                  # port 0 = any free port
    try:
        base = "http://127.0.0.1:%d" % srv.port
        data = json.loads(urllib.request.urlopen(base + "/status.json", timeout=5).read())
        assert data["processing"]["fps"] == pytest.approx(10.0, abs=0.2)
        page = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "<title>rt status</title>" in page and 'http-equiv="refresh"' in page
        jpg = urllib.request.urlopen(base + "/thumb/1.jpg", timeout=5).read()
        assert jpg[:2] == b"\xff\xd8"
        with pytest.raises(urllib.error.HTTPError):                # unknown pages: 404
            urllib.request.urlopen(base + "/thumb/999.jpg", timeout=5)
    finally:
        srv.stop()


# ------------------------------------------------------------------------------------------------ busy port
@pytest.fixture
def busy_port():
    """A port on this machine that some other program is already listening on (here: a plain socket)."""
    import socket
    other_program = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    other_program.bind(("127.0.0.1", 0))           # port 0 = the system picks a free port
    other_program.listen(1)
    yield other_program.getsockname()[1]
    other_program.close()


def test_busy_port_when_required_exits_with_code_3_and_says_why(cfg, busy_port):
    # Started by the launcher (required = True): rt must stop with a clear message and exit code 3. Running on
    # without the page would make the launcher kill it as "stalled" every ~90 s, forever, without saying why.
    from rt.status import start_status_server
    warnings = []
    with pytest.raises(SystemExit) as stopped:
        start_status_server(StatusBoard(cfg), "127.0.0.1", busy_port, required=True, log=lambda m: None,
                            warn=warnings.append)
    assert stopped.value.code == 3
    assert "cannot listen on port %d" % busy_port in warnings[0] and "status.port" in warnings[0]


def test_busy_port_in_a_manual_run_only_warns(cfg, busy_port):
    # Run by hand (not required): rt goes on without the page, with a warning.
    from rt.status import start_status_server
    warnings = []
    server = start_status_server(StatusBoard(cfg), "127.0.0.1", busy_port, required=False, log=lambda m: None,
                                 warn=warnings.append)
    assert server is None
    assert "Running without the status page" in warnings[0]


def test_free_port_starts_the_page(cfg):
    from rt.status import start_status_server
    lines = []
    server = start_status_server(StatusBoard(cfg), "127.0.0.1", 0, required=True, log=lines.append)
    try:
        assert server is not None and lines[0].startswith("status page: http://127.0.0.1:")
    finally:
        server.stop()
