"""Tests for zone/supervisor.py (restart with backoff) and zone/tools/record_ptz.py (the PTZ recording command)."""
from zone.clock import FakeClock
from zone.supervisor import supervise
from zone.tools import record_ptz


def runner(codes, clock, run_s=1.0):
    """A fake process: each run lasts run_s (fake) seconds and exits with the next code."""
    codes = list(codes)
    starts = []

    def run(cmd):
        starts.append(clock.monotonic())
        clock.advance(run_s)
        return codes.pop(0)
    return run, starts


def test_exit_zero_is_not_restarted():
    clock = FakeClock(0.0)
    run, starts = runner([0], clock)
    assert supervise(["x"], clock, run, log=lambda m: None) == 0
    assert len(starts) == 1


def test_crashes_restart_with_doubling_backoff():
    clock = FakeClock(0.0)
    run, starts = runner([1, 3, 1, 1, 0], clock)
    supervise(["x"], clock, run, backoff_initial_s=2, backoff_max_s=5, healthy_after_s=60, log=lambda m: None)
    waits = [b - a - 1.0 for a, b in zip(starts, starts[1:])]    # minus the 1 s each run lasted
    assert waits == [2, 4, 5, 5]                                # doubles, capped at 5


def test_a_long_healthy_run_resets_the_backoff():
    clock = FakeClock(0.0)
    run, starts = runner([1, 1, 1, 0], clock, run_s=100.0)       # every run lasted 100 s (> healthy_after 60)
    supervise(["x"], clock, run, backoff_initial_s=2, backoff_max_s=60, healthy_after_s=60, log=lambda m: None)
    waits = [b - a - 100.0 for a, b in zip(starts, starts[1:])]
    assert waits == [2, 2, 2]


def test_record_command_copies_the_stream_and_names_the_file_by_wall_time(tmp_path):
    from zone import config as C
    rc = C.load()["recording"]
    cmd, out = record_ptz.build_command(rc, tmp_path, 1_800_000_000.0)
    assert ["-c", "copy"] == cmd[cmd.index("-c"):cmd.index("-c") + 2]
    assert out.name.startswith("ptz_") and out.suffix == ".mkv"
    rc = dict(rc, ptz_segment_s=600)
    cmd, out = record_ptz.build_command(rc, tmp_path, 1_800_000_000.0)
    assert "segment" in cmd and "%Y%m%d" in str(out)


def test_recorder_disabled_exits_zero():
    assert record_ptz.main(["--set", "recording.ptz_usb=false"]) == 0
