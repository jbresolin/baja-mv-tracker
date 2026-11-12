"""Tests for the small helpers that line a recording up with its `.frames.csv` sidecar (zone/tools/common.py).
No video and no ffmpeg needed: these only do file names and arithmetic."""
import csv

from zone.tools.common import file_frame_number, load_frame_times, sidecar_for


def test_frame_number_from_pts():
    # a 10 fps file: the frame shown at 0.0 s is frame 0, at 1.5 s frame 15. Rounding absorbs the tiny float
    # errors of pts computed as k * step / fps (1.4999999 is still frame 15).
    assert file_frame_number(0.0, 10) == 0
    assert file_frame_number(1.5, 10) == 15
    assert file_frame_number(1.4999999, 10) == 15
    assert file_frame_number(2.0, 30) == 60


def test_sidecar_is_found_next_to_the_video(tmp_path):
    video = tmp_path / "spotter_20261115-101500.mkv"
    video.write_bytes(b"")
    assert sidecar_for(video) is None                       # no sidecar yet: None (use the pts)
    side = tmp_path / "spotter_20261115-101500.frames.csv"
    side.write_text("frame,t_wall,t_mono,src_idx\n", encoding="utf-8")
    assert sidecar_for(video) == side


def test_load_frame_times(tmp_path):
    side = tmp_path / "x.frames.csv"
    with open(side, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_wall", "t_mono", "src_idx"])
        w.writerow([0, "1800000000.000", "100.0000", 7])
        w.writerow([1, "1800000000.130", "100.1300", 8])     # this frame came 30 ms late
    # frame number in the file -> (wall time, monotonic time)
    assert load_frame_times(side) == {0: (1800000000.0, 100.0), 1: (1800000000.13, 100.13)}
