"""Tests of the live camera capture (rt/livecam.py) with fake cameras and a fake clock: no webcam, no waiting.

What is checked:
  * decimation: the camera sends 30 fps, rt processes 10;
  * timestamps: each frame carries the wall-clock time and the monotonic time at which it was received;
  * the clock_map event, once a minute, relating the two clocks;
  * the camera disappearing and coming back: the capture keeps the process alive and reopens it;
  * raw MJPEG: JPEG bytes go to the recorder as they are, and only processed frames are decoded;
  * the DirectShow behaviour (FakeDshowCamera): grab() does not wait, retrieve() does. The capture must not spin
    in a busy loop, and it must notice an unplugged camera although DirectShow reports it only once.

Two fake cameras are used:
  * FakeCamera: its grab() waits for the next frame (like Linux V4L2). rt grabs every frame and retrieves (decodes)
    only the ones it processes.
  * FakeDshowCamera: behaves like Windows DirectShow (the race laptop). rt retrieves every frame and drops 2 of 3.
"""
import numpy as np
import pytest

from rt.backoff import FakeClock
from rt.camsource import FakeCamera, FakeDshowCamera
from rt.livecam import CameraCapture, ClockMapper, Decimator


# ------------------------------------------------------------------------------------------------ helpers
class FakeRecorder:
    """Collects what the capture hands to the recorder."""

    def __init__(self):
        self.offers = []          # (data, t_wall, t_mono, camera_frame_no) of every offered frame
        self.camera_info = None
        self.lost_calls = []      # reasons given each time the capture said "the camera is lost"

    def set_camera_info(self, info):
        self.camera_info = info

    def offer(self, data, t_wall, t_mono, camera_frame_no):
        self.offers.append((data, t_wall, t_mono, camera_frame_no))
        return True

    def camera_lost(self, reason):
        self.lost_calls.append(reason)


class FakeSleep:
    """The capture's "sleep" function in tests. Instead of really waiting, it moves the fake clock forward by the
    same amount (so time passes as it would on the laptop), and it stops the capture once it has waited to reopen
    the camera `n` times. Waits of 1 s or more are reopen waits; the short 0.05 s waits follow a failed read.
    `on_reopen_wait` (optional) is called at each reopen wait: tests use it to look at the status meanwhile."""

    def __init__(self, clock, n=1, on_reopen_wait=None):
        self.clock, self.n, self.on_reopen_wait = clock, n, on_reopen_wait
        self.waits, self.cap = [], None

    def __call__(self, seconds):
        self.waits.append(seconds)
        if seconds >= 1 and self.on_reopen_wait is not None:
            self.on_reopen_wait(self.cap)
        self.clock.advance(seconds)
        if len(self.reopen_waits()) >= self.n:
            self.cap.stop_ev.set()

    def reopen_waits(self):
        return [w for w in self.waits if w >= 1]


def run_capture(cfg, cam, clock, reopens=1, recorder=None, events=None, on_reopen_wait=None):
    """Run the capture loop in this thread until it has waited to reopen the camera `reopens` times; return the
    capture object, the frames it produced for detection, and the fake sleep (with its list of waits)."""
    cfg["capture"]["queue_size"] = 100000          # the test reads the frames after the run: never drop any
    sleeper = FakeSleep(clock, reopens, on_reopen_wait)
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
    # A camera whose grab() waits for each frame (Linux V4L2, FakeCamera): 3 seconds of a 30 fps camera = 90 frames
    # grabbed. Only every 3rd one may be retrieved (decoded): decoding is the expensive part.
    clock = FakeClock()
    cam = FakeCamera(fps=30, clock=clock, n_frames=90)
    cap, frames, _ = run_capture(cfg, cam, clock)
    assert cam.grabs == 90
    assert cam.retrieved == list(range(0, 90, 3))
    assert cap.read_mode == "grab_then_retrieve"
    assert len(frames) == 30 and cap.retrieve_calls == 30 and cap.camera_frames == 90
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
    # 60 s = 1800 frames at 30 fps; adding 1/30 s 1800 times can land a hair under 60.0, so allow one frame later.
    assert maps[0]["camera_frame_no"] == 0
    assert abs(maps[1]["camera_frame_no"] - 1800) <= 1 and abs(maps[2]["camera_frame_no"] - 3600) <= 2
    assert maps[0]["stamped_at"] == "grab"           # FakeCamera's grab() waits, so frames are stamped there
    for clock_map in maps:
        assert clock_map["wall_minus_mono"] == pytest.approx(1_800_000_000.0 - 500.0)


def test_clock_mapper_period():
    cm = ClockMapper(60)
    assert cm.maybe(100.0, 10.0, 0) is not None      # first grab: logged
    assert cm.maybe(130.0, 40.0, 1) is None          # 30 s later: not yet
    assert cm.maybe(170.0, 70.0, 2) is not None      # 60 s later: logged


# ------------------------------------------------------------------------------------------------ reconnect
def test_camera_lost_and_back_keeps_running(cfg):
    # The camera gives 30 frames, then stops (every grab fails); when reopened it gives 30 more, and stops again.
    # 10 failed reads in a row (camera.lost_after_failures) = lost.
    clock = FakeClock()
    cam = FakeCamera(fps=30, clock=clock, lose_after=30)
    events = []
    recorder = FakeRecorder()
    cap, frames, sleeper = run_capture(cfg, cam, clock, reopens=2, events=events, recorder=recorder)
    assert cam.opens == 2                     # opened, lost, reopened
    assert cap.reopens == 2                   # lost twice (the test stops the loop at the second loss)
    assert len(frames) == 20                  # 10 processed frames from each 30-frame period
    assert [f.idx for f in frames] == list(range(20))
    # It waited 1 s before reopening each time (the backoff restarts at 1 s once frames flow again).
    assert sleeper.reopen_waits() == [1.0, 1.0]
    states = [e["state"] for e in events if e["type"] == "camera"]
    assert states == ["opening", "streaming", "lost", "opening", "streaming", "lost", "stopped"]
    lost = [e for e in events if e["type"] == "camera" and e["state"] == "lost"]
    assert "10 failed reads in a row" in lost[0]["detail"]
    # Each loss told the recorder, so it closes the file (a new one starts with the next frame).
    assert recorder.lost_calls == ["camera lost", "camera lost"]


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
    rec = FakeRecorder()
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


# ------------------------------------------------------------------------------------------------ DirectShow
# On Windows, OpenCV's DirectShow backend does not behave like the FakeCamera above: grab() returns True at once
# (it does not wait for a frame), and retrieve() is what waits for the next frame and decodes it. These tests use
# FakeDshowCamera, which copies that behaviour, to check the capture loop the race laptop really runs.

def test_directshow_uses_the_retrieve_every_frame_loop(cfg):
    # rt picks the loop from the camera's `grab_blocks` flag: DirectShow's grab() does not block.
    clock = FakeClock()
    cam = FakeDshowCamera(fps=30, clock=clock, n_frames=30)
    cap, _, _ = run_capture(cfg, cam, clock)
    assert cap.read_mode == "retrieve_every_frame"


def test_directshow_no_busy_loop_one_grab_per_frame(cfg):
    # 3 s of a 30 fps camera = 90 frames. A busy loop would call grab() thousands of times per frame. The capture
    # must call grab() and retrieve() once per camera frame (plus a few calls while it notices the stream ended).
    clock = FakeClock(wall=1_800_000_000.0, mono=500.0)
    cam = FakeDshowCamera(fps=30, clock=clock, n_frames=90)
    cap, frames, _ = run_capture(cfg, cam, clock)
    assert cam.delivered == 90
    # Every grab() that did not lead to a frame is a failed read (the stream had ended): grab calls minus failed
    # reads = frames delivered, i.e. exactly one grab per frame.
    assert cam.grab_calls - cap.failed_reads == 90
    assert cam.retrieve_calls == cam.grab_calls          # each grab is followed by one (waiting) retrieve
    assert cap.failed_reads <= 4                         # the end of the stream: ~3 one-second timeouts
    assert len(frames) == 30                             # and 1 in 3 frames is processed


def test_directshow_keeps_every_third_frame(cfg):
    # The fake camera's frame number n is a grey image of brightness n. The frames that reach detection must be
    # camera frames 0, 3, 6, ... (every 3rd, evenly spaced), numbered 0, 1, 2, ... for detection.
    clock = FakeClock()
    cam = FakeDshowCamera(fps=30, clock=clock, n_frames=90)
    _, frames, _ = run_capture(cfg, cam, clock)
    assert [int(fr.img[0, 0, 0]) for fr in frames] == list(range(0, 90, 3))
    assert [fr.idx for fr in frames] == list(range(30))


def test_directshow_frames_are_stamped_when_retrieve_returns(cfg):
    # On DirectShow the frame is only in hand when retrieve() returns, so that is when both clocks are read. The
    # camera opens at mono = 500 and frame n arrives at 500 + (n + 1) / 30 s. A processed frame's monotonic time
    # (t_cap) and wall time (t) must be its own arrival time, not the time of some earlier grab().
    clock = FakeClock(wall=1_800_000_000.0, mono=500.0)
    cam = FakeDshowCamera(fps=30, clock=clock, n_frames=90)
    events = []
    _, frames, _ = run_capture(cfg, cam, clock, events=events)
    for fr in frames:
        camera_frame_no = int(fr.img[0, 0, 0])
        assert fr.t_cap == pytest.approx(500.0 + (camera_frame_no + 1) / 30)
        assert fr.t == pytest.approx(1_800_000_000.0 + (camera_frame_no + 1) / 30)
    clock_map = [e for e in events if e["type"] == "clock_map"][0]
    assert clock_map["stamped_at"] == "after retrieve" and clock_map["camera_frame_no"] == 0


def test_directshow_recorder_gets_only_processed_frames_with_their_stamps(cfg):
    # The recorder is offered exactly the processed frames, with the same times detection got.
    clock = FakeClock()
    cam = FakeDshowCamera(fps=30, clock=clock, n_frames=90)
    recorder = FakeRecorder()
    _, frames, _ = run_capture(cfg, cam, clock, recorder=recorder)
    assert [offer[3] for offer in recorder.offers] == list(range(0, 90, 3))     # camera frame numbers
    assert [offer[2] for offer in recorder.offers] == pytest.approx([fr.t_cap for fr in frames])


def test_directshow_unplug_and_replug_recovers_in_process(cfg):
    # The race-day scenario: the USB cable is pulled after 60 frames (2 s) and plugged back 10 s later.
    # DirectShow says so only once (one grab() returns False); after that retrieve() just times out every second.
    # rt must: notice it by time (no frame for 3 s), show "lost" on the status page while the camera is out, try
    # to reopen with growing waits, and stream again once the camera is back, all without restarting the process.
    from rt.status import StatusBoard

    clock = FakeClock(wall=1_800_000_000.0, mono=500.0)
    cam = FakeDshowCamera(fps=30, clock=clock, unplug_after=60, unplugged_for_s=10.0, n_frames=120)
    board = StatusBoard(cfg, clock=clock)
    seen_while_out = []          # (camera state, status page warnings) at each reopen wait

    def look_at_status(cap):
        board.capture = cap
        payload = board.payload()
        seen_while_out.append((payload["camera"]["state"], payload["warnings"]))

    events, recorder = [], FakeRecorder()
    # n_frames=120: after 60 more frames the stream ends, which is the 4th loss/wait and stops the test.
    cap, frames, sleeper = run_capture(cfg, cam, clock, reopens=4, events=events, recorder=recorder,
                                       on_reopen_wait=look_at_status)
    # Waits: 1 s after the loss, then 2 s and 4 s while the camera is still unplugged (open fails), then it opens;
    # the last 1 s is the stream ending (the backoff went back to 1 s once frames flowed again).
    assert sleeper.reopen_waits() == [1.0, 2.0, 4.0, 1.0]
    assert cam.opens == 4                                 # first open, 2 failed reopens, 1 successful reopen
    # The status page said "lost" (with a warning) during each wait while the camera was out.
    for state, warnings in seen_while_out[:3]:
        assert state == "lost"
        assert any(w.startswith("camera lost") for w in warnings)
    lost = [e for e in events if e["type"] == "camera" and e["state"] == "lost"]
    assert "no frame for 3" in lost[0]["detail"]
    # 20 processed frames before the unplug, 20 after the replug, numbered without a gap for detection.
    assert len(frames) == 40 and [fr.idx for fr in frames] == list(range(40))
    # The frames after the replug really are the camera's frames 60.. (brightness = camera frame number).
    assert int(frames[20].img[0, 0, 0]) == 60
    assert recorder.lost_calls[0] == "camera lost"     # the recording file was closed at the loss


def test_old_grab_then_retrieve_loop_would_spin_on_directshow(cfg):
    # Why the DirectShow loop exists: forcing the "grab every frame, retrieve some" loop on a DirectShow-like
    # camera makes grab() return at once, forever, without time passing: a busy loop. Here grab() is stopped after
    # 5,000 calls; in that time hardly any frame was delivered. (On the laptop this burned a whole core.)
    cfg["camera"]["read_mode"] = "grab_then_retrieve"
    clock = FakeClock()
    cam = FakeDshowCamera(fps=30, clock=clock, n_frames=90)
    real_grab = cam.grab

    def counting_grab():
        if cam.grab_calls >= 5000:
            cap_holder[0].stop_ev.set()
        return real_grab()

    cam.grab = counting_grab
    cap_holder = []
    cfg["capture"]["queue_size"] = 100000
    cap = CameraCapture(cfg, camera_factory=lambda: cam, clock=clock, sleep=lambda s: None, log=lambda m: None)
    cap_holder.append(cap)
    cap.run()
    assert cam.grab_calls >= 5000 and cam.delivered <= 2


def test_unknown_read_mode_is_refused(cfg):
    cfg["camera"]["read_mode"] = "fastest"
    with pytest.raises(SystemExit, match="camera.read_mode"):
        CameraCapture(cfg, camera_factory=lambda: None, clock=FakeClock())
