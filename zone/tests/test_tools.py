"""Tests for the offline tools on a small synthetic recording: the replay harness, the detection check and the
spotter recorder. They need ffmpeg on PATH (skipped otherwise)."""
import csv
import json
import shutil

import numpy as np
import pytest

from zone import zonefile
from zone.testing import Car, Scene, write_video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="needs ffmpeg and ffprobe")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    d = tmp_path_factory.mktemp("rec")
    sc = Scene()
    write_video(d / "spotter.mp4", sc, [Car(sc, 1.0), Car(sc, 14.0, factor=1.2)], 26.0)
    zonefile.save(sc.make_zone(), d / "zone.json")
    return d


def test_replay_scores_each_followed_pass(recording, tmp_path):
    from zone.tools import replay
    rc = replay.main(["--video", str(recording / "spotter.mp4"), "--zone", str(recording / "zone.json"),
                      "--detector", "blobs", "--out", str(tmp_path)])
    assert rc == 0
    passes = read_csv(tmp_path / "passes.csv")
    assert len(passes) == 2
    for p in passes:
        assert p["drop_reason"] == "exit"
        # errors are in degrees (the report's units per degree): small for a perfect detector
        assert float(p["err_mean"]) < 3.0
    assert (tmp_path / "pass_01.png").exists() and (tmp_path / "pass_02.png").exists()
    assert (tmp_path / "cycles.csv").exists() and json.loads((tmp_path / "summary.json").read_text())["passes"]
    # the detections were saved: re-running with --tracks skips detection and gives the same passes
    rc = replay.main(["--tracks", str(tmp_path / "tracks.json"), "--zone", str(recording / "zone.json"),
                      "--out", str(tmp_path / "again")])
    again = read_csv(tmp_path / "again" / "passes.csv")
    assert [p["err_mean"] for p in again] == [p["err_mean"] for p in passes]


def test_detect_check_reports_one_continuous_track_per_car(recording, tmp_path):
    from zone.tools import detect_check
    detect_check.main(["--video", str(recording / "spotter.mp4"), "--zone", str(recording / "zone.json"),
                       "--detector", "blobs", "--passes", "2", "--out", str(tmp_path)])
    rep = json.loads((tmp_path / "detect_check.json").read_text())
    modes = {m["mode"]: m for m in rep["modes"]}
    assert set(modes) == {"full", "downscale", "tile"}
    for m in modes.values():
        assert m["continuous_tracks"] == 2 and m["tracks_per_pass"] == 1.0
        assert m["ms_per_frame_median"] is not None
    assert modes["tile"]["inferences_per_frame"] > 1 and modes["downscale"]["inferences_per_frame"] == 1
    assert (tmp_path / "tracks_tile.csv").exists()


def test_spotter_recorder_writes_video_and_frame_times(tmp_path):
    from zone.recorder import SpotterRecorder
    from zone.source import ZoneFrame
    rec = SpotterRecorder(tmp_path, 10, (320, 240), 1_800_000_000.0, codec_args=["-c:v", "libx264", "-preset",
                                                                                   "ultrafast"])
    for k in range(20):
        rec.write(ZoneFrame(np.full((240, 320, 3), k * 10, np.uint8), 100.0 + k / 10, 1_800_000_000.0 + k / 10, k))
    rec.write(ZoneFrame(np.zeros((10, 10, 3), np.uint8), 0, 0, 99))     # wrong size: dropped, never crashes
    rec.close()
    assert rec.path.exists() and rec.path.stat().st_size > 0
    rows = read_csv(rec.frames_path)
    assert len(rows) == 20 and rows[3]["t_mono"] == "100.3000"
    assert rec.dropped == 1
    assert json.loads(rec.meta_path.read_text())["fps"] == 10


# ===================================================================================== replay: real frame times
def write_sidecar(path, n_frames, t0=5000.0, late_after=None, late_by=0.0):
    """A `.frames.csv` like the SpotterRecorder writes: one row per frame of the video, with the real grab times.
    From frame `late_after` on, every frame was grabbed `late_by` seconds later than the nominal 10 fps says (the
    webcam stalled once): the video's pts can't know that, only the sidecar can."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_wall", "t_mono", "src_idx"])
        for k in range(n_frames):
            t = t0 + k * 0.1 + (late_by if late_after is not None and k >= late_after else 0.0)
            w.writerow([k, "%.3f" % (1_800_000_000.0 + t), "%.4f" % t, k])


def test_video_frames_take_their_times_from_the_sidecar(recording, tmp_path):
    from zone.spotter import rt_config
    from zone import config as C
    from zone.tools.common import load_frame_times, video_frames
    side = tmp_path / "spotter.frames.csv"
    write_sidecar(side, 260, late_after=100, late_by=0.5)
    rtcfg = rt_config(C.load(None, []))
    stats = {}
    frames = list(video_frames(rtcfg, recording / "spotter.mp4", frame_times=load_frame_times(side), stats=stats))
    assert len(frames) == 260 and stats == dict(matched=260, missing=0)
    # frame 10: nominal time, from the sidecar's monotonic clock (5000 + 1.0)
    assert frames[10].t == pytest.approx(5001.0)
    # frame 150: the pts says 15.0 s into the file, but the sidecar knows it was grabbed 0.5 s later than that
    assert frames[150].t == pytest.approx(5015.5)
    assert frames[150].t_wall == pytest.approx(1_800_000_000.0 + 5015.5)
    # without the sidecar, the same frame's time is just its pts
    plain = list(video_frames(rtcfg, recording / "spotter.mp4"))
    assert plain[150].t == pytest.approx(15.0)


def test_replay_finds_the_sidecar_next_to_the_video_by_itself(recording, tmp_path):
    from zone.tools import replay
    video = tmp_path / "spotter_20261115-101500.mp4"
    shutil.copy(recording / "spotter.mp4", video)
    write_sidecar(tmp_path / "spotter_20261115-101500.frames.csv", 260)
    out = tmp_path / "out"
    assert replay.main(["--video", str(video), "--zone", str(recording / "zone.json"), "--detector", "blobs",
                        "--out", str(out)]) == 0
    saved = json.loads((out / "tracks.json").read_text())
    assert saved["frame_times"].endswith("spotter_20261115-101500.frames.csv")
    # each saved frame is (t, index, t_wall, tracks): t is the sidecar's monotonic time, not the pts
    assert saved["frames"][0][0] == pytest.approx(5000.0) and saved["frames"][30][0] == pytest.approx(5003.0)
    # the passes are still found and followed with the real times
    assert len(read_csv(out / "passes.csv")) == 2
    # --no-frames: back to the pts
    out2 = tmp_path / "out2"
    replay.main(["--video", str(video), "--zone", str(recording / "zone.json"), "--detector", "blobs",
                 "--out", str(out2), "--no-frames"])
    assert json.loads((out2 / "tracks.json").read_text())["frames"][30][0] == pytest.approx(3.0)

