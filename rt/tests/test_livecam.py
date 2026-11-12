"""Tests of the live camera capture (rt/livecam.py) with a fake camera and a fake clock: no webcam, no waiting.

What is checked:
  * decimation: the camera sends 30 fps, rt processes 10. Only the processed frames may be retrieved (= decoded),
    because decoding 1080p MJPEG is the biggest CPU cost of the capture;
  * timestamps: each frame carries the wall-clock time and the monotonic time of its grab();
  * the clock_map event, once a minute, relating the two clocks;
  * the camera disappearing and coming back: the capture keeps the process alive and reopens it;
  * raw MJPEG: JPEG bytes go to the recorder as they are, and only processed frames are decoded.
"""
import numpy as np
import pytest

from rt.backoff import FakeClock
from rt.camsource import FakeCamera
from rt.livecam import CameraCapture, ClockMapper, Decimator


# ------------------------------------------------------------------------------------------------ helpers
class FakeRecorder:
    """Collects what the capture offers to the recorder."""

    def __init__(self, all_frames=False):
        self.all_frames = all_frames
        self.offers = []
        self.camera_info = None

    def wants_all_frames(self):
        return self.all_frames

    def set_camera_info(self, info):
        self.camera_info = info

    def offer(self, data, t_wall, t_mono, grab_no):
        self.offers.append((data, t_wall, t_mono, grab_no))
        return True


class StopAfterReopens:
    """The capture's "sleep" function in tests: records each wait instead of sleeping, and stops the capture after
    it has waited to reopen the camera `n` times (waits of 1 s or more; the 0.05 s waits are failed grabs)."""

    def __init__(self, n=1):
        self.n, self.waits, self.cap = n, [], None

    def __call__(self, seconds):
        self.waits.append(seconds)
        if len([w for w in self.waits if w >= 1]) >= self.n:
            self.cap.stop_ev.set()

    def reopen_waits(self):
        return [w for w in self.waits if w >= 1]


def run_capture(cfg, cam, clock, reopens=1, recorder=None, events=None):
    """Run the capture loop in this thread until the camera has been lost `reopens` times; return the frames."""
    cfg["capture"]["queue_size"] = 100000          # the test reads the frames after the run: never drop any
    sleeper = StopAfterReopens(reopens)
    cap = CameraCapture(cfg, camera_factory=lambda: cam, clock=clock, recorder=recorder,
                        emit=(events.append if events is not None else None), sleep=sleeper, log=lambda m: None)
    sleeper.cap = cap
    cap.run()                                      # runs the loop here (not as a thread) until it stops
    frames = []
    while True:
        fr = cap.q.get_nowait()
        if fr is None:
            break
        frames.append(fr)
    return cap, frames, sleeper


# ------------------------------------------------------------------------------------------------ decimation
def processed(fps_in, n, fps_out=10.0, jitter=None):
    """Which of n frames arriving at fps_in the Decimator picks (optional per-frame time jitter in seconds)."""
    d = Decimator(fps_out, fps_in)
    return [i for i in range(n) if d.due(i / fps_in + (jitter[i] if jitter is not None else 0.0))]


def test_decimator_keeps_every_third_frame_of_a_30fps_camera():
    assert processed(30, 30) == list(range(0, 30, 3))


def test_decimator_other_camera_rates():
    assert processed(60, 60) == list(range(0, 60, 6))   # 60 fps camera: every 6th
    assert processed(10, 10) == list(range(10))         # already 10 fps: every frame


def test_decimator_ignores_usb_timing_jitter():
    # Real frames don't arrive exactly every 33.3 ms: +-5 ms of random jitter must not change the pattern.
    rng = np.random.default_rng(1)
    assert processed(30, 300, jitter=rng.uniform(-0.005, 0.005, 300)) == list(range(0, 300, 3))


def test_decimator_does_not_burst_after_a_stall():
    # The camera stops for 2 s (frames 10..69 never arrive). After it, processing continues at 10 fps from the next
    # frame; it does not try to "catch up" by processing the next frames back to back.
    d = Decimator(10.0)
    times = [i / 30 for i in range(10)] + [i / 30 for i in range(70, 100)]
    picked = [t for t in times if d.due(t)]
    after = [t for t in picked if t >= 70 / 30]
    assert after[0] == pytest.approx(70 / 30)
    assert np.diff(after) == pytest.approx(np.full(len(after) - 1, 0.1))


def test_only_processed_frames_are_retrieved(cfg):
    # 3 seconds of a 30 fps camera = 90 frames grabbed. Only every 3rd one may be retrieved (decoded).
    clock = FakeClock()
    cam = FakeCamera(fps=30, clock=clock, n_frames=90)
    cap, frames, _ = run_capture(cfg, cam, clock)
    assert cam.grabs == 90
    assert cam.retrieved == list(range(0, 90, 3))
    assert len(frames) == 30 and cap.retrieved == 30
    assert [f.idx for f in frames] == list(range(30))     # processed-frame numbers count up without gaps


# ------------------------------------------------------------------------------------------------ timestamps
def test_frames_carry_wall_and_monotonic_time_of_their_grab(cfg):
    # The fake camera advances the fake clock by 1/30 s at each grab, so grab n happens at start + (n + 1) / 30.
    # Frame time = wall clock at grab - configured camera latency (+ clock.offset_s, 0 here).
    cfg["capture"]["live_latency_s"] = 0.05
    clock = FakeClock(wall=1_800_000_000.0, mono=500.0)
    cam = FakeCamera(fps=30, clock=clock, n_frames=30)
    _, frames, _ = run_capture(cfg, cam, clock)
    for fr, grab_no in zip(frames, range(0, 30, 3)):
        assert fr.t == pytest.approx(1_800_000_000.0 + (grab_no + 1) / 30 - 0.05)
        assert fr.t_cap == pytest.approx(500.0 + (grab_no + 1) / 30)   # monotonic time of the same grab


def test_clock_map_is_logged_once_a_minute(cfg):
    # 3 minutes of camera frames: a clock_map event at the first grab, then every 60 s (3 in total). Each relates
    # the wall clock to the monotonic clock at one grab; the beam input (WP-L2) uses it to put both on one clock.
    clock = FakeClock(wall=1_800_000_000.0, mono=500.0)
    cam = FakeCamera(fps=30, clock=clock, n_frames=30 * 180)
    events = []
    run_capture(cfg, cam, clock, events=events)
    maps = [e for e in events if e["type"] == "clock_map"]
    assert len(maps) == 3
    # 60 s = 1800 grabs at 30 fps; adding 1/30 s 1800 times can land a hair under 60.0, so allow one grab later.
    assert maps[0]["grab_no"] == 0
    assert abs(maps[1]["grab_no"] - 1800) <= 1 and abs(maps[2]["grab_no"] - 3600) <= 2
    for m in maps:
        assert m["wall_minus_mono"] == pytest.approx(1_800_000_000.0 - 500.0)


def test_clock_mapper_period():
    cm = ClockMapper(60)
    assert cm.maybe(100.0, 10.0, 0) is not None      # first grab: logged
    assert cm.maybe(130.0, 40.0, 1) is None          # 30 s later: not yet
    assert cm.maybe(170.0, 70.0, 2) is not None      # 60 s later: logged


# ------------------------------------------------------------------------------------------------ reconnect
def test_camera_lost_and_back_keeps_running(cfg):
    # The camera gives 30 frames, then stops (unplugged); when reopened it gives 30 more, and stops again.
    clock = FakeClock()
    cam = FakeCamera(fps=30, clock=clock, lose_after=30)
    events = []
    cap, frames, sleeper = run_capture(cfg, cam, clock, reopens=2, events=events)
    assert cam.opens == 2                     # opened, lost, reopened
    assert cap.reopens == 2                   # lost twice (the test stops the loop at the second loss)
    assert len(frames) == 20                  # 10 processed frames from each 30-frame period
    assert [f.idx for f in frames] == list(range(20))
    # It waited 1 s before reopening each time (the backoff restarts at 1 s once frames flow again).
    assert sleeper.reopen_waits() == [1.0, 1.0]
    states = [e["state"] for e in events if e["type"] == "camera"]
    assert states == ["opening", "streaming", "lost", "opening", "streaming", "lost", "stopped"]


def test_missing_camera_at_start_is_retried_with_growing_waits(cfg):
    # The camera is not there for the first 3 attempts (e.g. still being plugged in): waits of 1, 2, 4 s, then it
    # opens and frames flow. The process never exits.
    clock = FakeClock()
    cam = FakeCamera(fps=30, clock=clock, fail_opens=3, n_frames=30)
    cap, frames, sleeper = run_capture(cfg, cam, clock, reopens=4)
    assert sleeper.reopen_waits() == [1.0, 2.0, 4.0, 1.0]   # the last one: the camera ran out of frames ("lost")
    assert len(frames) == 10


# ------------------------------------------------------------------------------------------------ raw MJPEG
def test_raw_jpeg_is_recorded_as_is_and_only_processed_frames_are_decoded(cfg):
    clock = FakeClock()
    cam = FakeCamera(fps=30, clock=clock, n_frames=90, jpeg=True, size=(64, 48))
    rec = FakeRecorder(all_frames=False)
    cap, frames, _ = run_capture(cfg, cam, clock, recorder=rec)
    assert cap.raw is True
    # The recorder got the 30 processed frames as JPEG bytes (1-D uint8 starting with the JPEG marker FF D8).
    assert len(rec.offers) == 30
    data = rec.offers[0][0]
    assert data.ndim == 1 and data[:2].tobytes() == b"\xff\xd8"
    assert [o[3] for o in rec.offers] == list(range(0, 90, 3))
    # Detection got decoded images of the camera's size.
    assert frames[0].img.shape == (48, 64, 3)
    assert rec.camera_info["fourcc"] == "MJPG"


def test_record_all_frames_without_decoding_them(cfg):
    # all_frames = true: every camera frame goes to the recording (retrieving raw JPEG is only a copy), but still
    # only every 3rd one is decoded and processed.
    clock = FakeClock()
    cam = FakeCamera(fps=30, clock=clock, n_frames=90, jpeg=True)
    rec = FakeRecorder(all_frames=True)
    _, frames, _ = run_capture(cfg, cam, clock, recorder=rec)
    assert len(rec.offers) == 90
    assert len(frames) == 30


def test_backend_without_raw_passthrough_falls_back_to_decoded_frames(cfg):
    # The driver said "yes" to raw mode but still returns decoded images. The capture notices on the first frame
    # and continues with images (the recorder then encodes them to H.264).
    clock = FakeClock()
    cam = FakeCamera(fps=30, clock=clock, n_frames=30, jpeg=False, info=dict(raw_accepted=True))
    rec = FakeRecorder()
    cap, frames, _ = run_capture(cfg, cam, clock, recorder=rec)
    assert cap.raw is False
    assert len(frames) == 10
    assert rec.offers[0][0].ndim == 3          # images, not JPEG bytes
