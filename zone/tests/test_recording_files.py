"""Tests for the spotter recording's files: the container it is written in (zone/recorder.py), and the small helpers
that line a recording up with its `.frames.csv` sidecar (zone/tools/common.py). No video and no ffmpeg needed: these
only look at file names, command lines and arithmetic."""
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


def test_spotter_recording_is_matroska_so_a_crash_leaves_a_readable_file(tmp_path):
    # An .mp4 is unreadable until ffmpeg writes its index at the very end, so a crash or a power cut loses the whole
    # recording. Matroska (.mkv) stays readable up to the last written piece. Here ffmpeg isn't really started: a
    # fake `popen` just records the command line the recorder would run.
    from zone.recorder import SpotterRecorder
    started = []

    class FakeProc:
        stdin = None

        def poll(self):
            return None

    def fake_popen(cmd, **kw):
        started.append(cmd)
        return FakeProc()
    rec = SpotterRecorder(tmp_path, 10, (320, 240), 1_800_000_000.0, codec_args=["-c:v", "libx264"],
                          popen=fake_popen)
    assert rec.path.suffix == ".mkv"
    assert rec.frames_path.name == rec.path.stem + ".frames.csv"      # the replay finds the sidecar by this name
    assert started[0][-3:] == ["-f", "matroska", str(rec.path)]          # the container is asked for explicitly
    rec.q.put(None)                                                    # let the writer thread end
    rec.thread.join(5)


def test_replay_error_converts_each_axis_with_its_own_units_per_degree():
    # A camera with 20 raw units per degree in pan but 10 in tilt: an error of 60 pan units and 40 tilt units is
    # 3 degrees and 4 degrees, so 5 degrees in total (a 3-4-5 triangle). Dividing both by the pan factor would
    # say sqrt(3^2 + 2^2) = 3.6 degrees: the tilt error would be halved.
    from zone.tools.replay import angular_error
    assert angular_error(60, 40, 0, 0, (20.0, 10.0)) == 5.0
    # without units per degree, the error stays in raw units
    assert angular_error(3, 4, 0, 0, None) == 5.0
