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
