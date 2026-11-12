"""End-to-end tests on a real 7.5 s clip of the 26SE race (rt/tests/fixtures/2026-10-17 26SE car14 clip.mp4).

The clip: car 14 crosses the start line alone at 11:45:17 (pass 22 of session 1's whole-race table; no other car within
100 s before or 31 s after). It was cut from "2026-10-17 11-29-49.mkv" (927.5 s to 935.0 s), at 10 fps, H.264 CRF 28,
3.2 MB. The OBS clock overlay (top left) is kept, so rt reads the real race time from it.

1. Camera path, no models: the clip played through the fake camera -> live capture -> recorder (real ffmpeg, x264).
   The recording must contain every processed frame, and its frame times must be exactly the times detection saw.
2. Whole pipeline, with the ONNX models (deploy/models/*.onnx, not in git): rt/run.py on the clip must find exactly one
   pass, at the right time, and read number 14. Skipped when the models or ffmpeg are missing.
"""
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rt import config as C
from rt.backoff import FakeClock
from rt.camsource import FakeCamera
from rt.livecam import CameraCapture
from rt.recorder import Recorder

ROOT = Path(__file__).resolve().parents[2]
CLIP = Path(__file__).parent / "fixtures" / "2026-10-17 26SE car14 clip.mp4"
UTC_3 = dt.timezone(dt.timedelta(hours=-3))      # the 26SE overlay clock and the human counts are in UTC-3


def tod(epoch):
    """Epoch seconds -> 'HH:MM:SS.s' in the race's time zone."""
    return dt.datetime.fromtimestamp(epoch, UTC_3).strftime("%H:%M:%S.%f")[:10]


def test_clip_is_small_and_present():
    # The fixture is committed to git; keep it small.
    assert CLIP.exists() and CLIP.stat().st_size < 4 * 2**20


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_recording_lines_up_with_the_frames_detection_saw(cfg, tmp_path):
    # The clip is 10 fps, so we pretend it is a 10 fps camera: every frame is processed and recorded.
    # The fake camera hands out decoded images, like DirectShow does on the race laptop.
    clock = FakeClock(wall=1_800_000_000.0, mono=50.0)
    cam = FakeCamera.from_video(CLIP, fps=10, clock=clock)
    cfg["capture"]["queue_size"] = 1000
    cfg["camera"]["fps"] = 10
    cfg["recording"]["queue_frames"] = 1000         # the fake camera runs faster than real time: keep every frame
    rec = Recorder(cfg, tmp_path, fps=10, version="test", log=lambda m: None, encoder="libx264")
    rec.start()
    stop_at_first_loss = lambda s: cap.stop_ev.set() if s >= 1 else None  # noqa: E731
    cap = CameraCapture(cfg, camera_factory=lambda: cam, clock=clock, recorder=rec, sleep=stop_at_first_loss,
                        log=lambda m: None)
    cap.run()
    rec.stop()
    frames = []
    while (fr := cap.q.get_nowait()) is not None:
        frames.append(fr)

    assert len(frames) == 75                        # 7.5 s at 10 fps, all processed
    assert frames[0].img.shape == (1080, 1920, 3)   # decoded full-resolution frames for detection

    # The recording: one segment, 75 frames, encoded to H.264 (x264 here).
    assert len(rec.segments) == 1
    seg = rec.segments[0]
    side = json.loads(seg.with_suffix(".json").read_text())
    assert side["frames"] == 75 and side["encoder"] == "libx264"
    # Frame k of the recording has exactly the wall time detection used for frame k (latency 0 here): this is how a
    # pass's thumbnail time is found in the recording afterwards.
    rows = list(csv.DictReader(open(seg.with_suffix(".frames.csv"))))
    assert [round(float(row["t_wall"]), 6) for row in rows] == [round(f.t, 6) for f in frames]
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(seg)], capture_output=True, text=True).stdout
    assert int(out.strip()) == 75


def models_present():
    cfg = C.load()
    m = cfg["models"]
    return all(Path(m[k]).exists() for k in ("car_detector", "panel_detector", "recognizer"))


@pytest.mark.skipif(not models_present(), reason="ONNX models not present (deploy/models/*.onnx are not in git)")
@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not on PATH")
def test_rt_finds_car_14_at_11_45_17(tmp_path):
    # Run rt exactly as on the command line, on the clip, as fast as possible. The status page is switched off so
    # the test never fights another rt over port 8765.
    out = tmp_path / "clip.jsonl"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    result = subprocess.run([sys.executable, str(ROOT / "rt" / "run.py"), "--source", str(CLIP), "--event", "26SE",
                             "--out", str(out), "--quiet", "--set", "status.enabled=false", "--set",
                             "capture.max_width=0"], capture_output=True, text=True, timeout=600, env=env,
                            cwd=str(ROOT))
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    events = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    ends = [e for e in events if e["type"] == "pass_end"]
    ids = [e for e in events if e["type"] == "pass_identity"]

    # One car, one pass.
    assert len(ends) == 1 and len(ids) == 1
    # The time comes from the burnt-in race clock: the car enters at about 11:45:17.4 (session 1 found 11:45:17.42
    # with YOLO-World at 30 fps; at 10 fps the first frame can be up to 0.1 s later; allow 0.5 s).
    t_enter = dt.datetime.fromtimestamp(ends[0]["t_enter"], UTC_3)
    expected = dt.datetime(2026, 10, 17, 11, 45, 17, 420000, tzinfo=UTC_3)
    assert abs((t_enter - expected).total_seconds()) < 0.5, tod(ends[0]["t_enter"])
    # And the number is read: 14.
    assert ids[0]["number"] == 14, ids[0]
