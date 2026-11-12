"""Tests of the launcher's restart rules (rt/supervise.py), with a fake rt process and a fake clock: the restart-rule
tests start no real process and never sleep. (The last test, test_kill_tree_also_kills_grandchildren, is the
exception: it starts two real, sleeping Python processes to check they are both killed; it waits only for them to
start and stop, at most a few seconds.)

The rules being checked:
  * rt exits -> start it again after 5 s, then 10, 20, 40, 60, 60 s if it keeps dying;
  * a run that lasted 10 minutes before dying resets the wait to 5 s;
  * rt alive but processing 0 fps (or not answering) for 30 s -> kill it, restart it;
  * ... EXCEPT when the status page answers and says the camera is lost / retrying / opening: rt is reopening the
    camera by itself, so the launcher leaves it alone (killing it would not bring the camera back any sooner);
  * nothing is checked during the first 60 s of a run (models load, the camera opens).
"""
import pytest

from rt.supervise import Supervisor, build_command


class FakeProc:
    """A fake rt process: poll() returns None while running, the exit code once it has "died"."""
    next_pid = 100

    def __init__(self, cmd):
        self.cmd, self.rc, self.killed = cmd, None, False
        self.pid = FakeProc.next_pid
        FakeProc.next_pid += 1

    def poll(self):
        return self.rc


class World:
    """Everything the supervisor talks to, faked: time, the processes it started, the fps the status page says."""

    def __init__(self):
        self.now = 0.0
        self.procs = []
        self.fps = 10.0
        self.logs = []

    def spawn(self, cmd):
        p = FakeProc(cmd)
        self.procs.append(p)
        return p

    def kill(self, p):
        p.killed, p.rc = True, -9

    def supervisor(self, **kw):
        return Supervisor(make_cmd=lambda: ["python", "rt/run.py"], spawn=self.spawn, probe=lambda: self.fps,
                          kill=self.kill, clock=lambda: self.now, log=self.logs.append, **kw)

    def run(self, sup, seconds, step=1.0):
        """Advance the fake time `seconds`, calling tick() every `step` like the real loop does."""
        end = self.now + seconds
        while self.now < end:
            sup.tick()
            self.now += step


def test_starts_rt_immediately():
    w = World()
    sup = w.supervisor()
    sup.tick()
    assert len(w.procs) == 1


def test_restart_waits_double_up_to_60_s_when_rt_keeps_dying():
    w = World()
    sup = w.supervisor()
    for _ in range(7):
        w.run(sup, 1)                    # rt starts ...
        w.procs[-1].rc = 1               # ... and crashes right away
        w.run(sup, 61)                   # long enough for any wait
    waits = [r[2] for r in sup.restarts]
    assert waits == [5, 10, 20, 40, 60, 60, 60]
    assert all("exited with code 1" in r[1] for r in sup.restarts)


def test_restart_happens_after_the_wait_not_before():
    w = World()
    sup = w.supervisor()
    sup.tick()                           # t = 0: start #1
    w.procs[-1].rc = 1
    w.now = 1.0
    sup.tick()                           # t = 1: sees the exit, restart planned at t = 6
    w.now = 5.9
    sup.tick()
    assert len(w.procs) == 1             # not yet
    w.now = 6.0
    sup.tick()
    assert len(w.procs) == 2             # now


def test_a_long_healthy_run_resets_the_wait():
    w = World()
    sup = w.supervisor()
    w.run(sup, 1)
    w.procs[-1].rc = 1
    w.run(sup, 10)                       # crash 1: wait 5 s
    w.procs[-1].rc = 1
    w.run(sup, 20)                       # crash 2: wait 10 s
    w.run(sup, 11 * 60)                  # then it runs fine for 11 minutes ...
    w.procs[-1].rc = 1                   # ... and crashes once: back to 5 s, not 20 s
    w.run(sup, 2)
    assert [r[2] for r in sup.restarts] == [5, 10, 5]


def test_stalled_rt_is_killed_and_restarted():
    w = World()
    sup = w.supervisor(grace_s=60, stall_s=30)
    w.run(sup, 100)                      # running fine at 10 fps
    assert len(w.procs) == 1
    w.fps = 0.0                          # detection hangs: the status page says 0 fps
    w.run(sup, 29)
    assert not w.procs[0].killed         # 29 s of 0 fps: still waiting
    w.run(sup, 2)
    assert w.procs[0].killed             # 30 s: killed
    w.fps = 10.0
    w.run(sup, 10)
    assert len(w.procs) == 2             # and started again after 5 s
    assert sup.restarts[0][1] == "rt stalled"


def test_status_page_not_answering_counts_as_stalled():
    # probe() returns None when the status page does not answer at all (rt frozen).
    w = World()
    sup = w.supervisor(grace_s=60, stall_s=30)
    w.run(sup, 70)
    w.fps = None
    w.run(sup, 31)
    assert w.procs[0].killed


def test_a_short_dip_to_zero_fps_is_not_a_stall():
    # fps 0 for 20 s (e.g. the camera was replugged and came back) then normal again: no restart.
    w = World()
    sup = w.supervisor(grace_s=60, stall_s=30)
    w.run(sup, 70)
    w.fps = 0.0
    w.run(sup, 20)
    w.fps = 10.0
    w.run(sup, 5)
    w.fps = 0.0
    w.run(sup, 20)                       # a second dip: the 30 s count started again at its beginning
    assert not w.procs[0].killed and len(w.procs) == 1


def test_no_stall_check_during_startup_grace():
    # Loading the models and opening the camera takes a while: fps is 0 at first, which must not count.
    w = World()
    w.fps = 0.0
    sup = w.supervisor(grace_s=60, stall_s=30)
    # w.run(sup, n) ticks once a second at t = now, now + 1, ..., now + n - 1. rt was started at t = 0.
    w.run(sup, 90)                       # last tick t = 89: fps 0 is only counted from t = 60 (end of grace): 29 s
    assert not w.procs[0].killed
    w.run(sup, 1)                        # tick at t = 90: 60 s grace + 30 s of 0 fps -> killed
    assert w.procs[0].killed


def test_rt_command_has_a_fresh_output_file_per_run(cfg):
    import datetime as dt
    cfg["launch"].update(event="27SE", source="camera", out_dir="out/lapline", extra_args=["--stats"])
    cmd = build_command(cfg, "rt/configs/lapline.toml", python="python", now=dt.datetime(2026, 11, 21, 9, 15, 2))
    assert cmd[:2] == ["python", cmd[1]] and cmd[1].endswith("run.py")
    assert cmd[cmd.index("--event") + 1] == "27SE" and cmd[cmd.index("--source") + 1] == "camera"
    out = cmd[cmd.index("--out") + 1].replace("\\", "/")
    assert out.endswith("out/lapline/2026-11-21/lapline_09-15-02.jsonl")
    assert cmd[-1] == "--stats"
    # The status page is on, so rt is told it is required: if its port is busy rt exits and says why, instead of
    # running without it (the launcher would then kill it as "stalled" every ~90 s without a reason).
    assert "--require-status" in cmd


def test_rt_command_without_status_page_does_not_require_it(cfg):
    cfg["launch"].update(event="27SE")
    cfg["status"]["enabled"] = False
    assert "--require-status" not in build_command(cfg, "rt/configs/lapline.toml", python="python")


def test_busy_status_port_exit_is_explained_in_the_launcher_log():
    # rt exits with code 3 when its status port is busy. The launcher still restarts it (the port may be freed),
    # but its log names the reason, so whoever reads launcher.log knows what to fix.
    w = World()
    sup = w.supervisor()
    w.run(sup, 1)
    w.procs[-1].rc = 3
    w.run(sup, 2)
    assert "status page's port is busy" in sup.restarts[0][1]


@pytest.mark.parametrize("rc", [0, 1, -9])
def test_any_exit_is_restarted(rc):
    # Even a clean exit (code 0) is restarted: on race day rt should never stop by itself.
    w = World()
    sup = w.supervisor()
    w.run(sup, 1)
    w.procs[-1].rc = rc
    w.run(sup, 10)
    assert len(w.procs) == 2


def test_kill_tree_also_kills_grandchildren():
    # On Windows, .venv\Scripts\python.exe is only a launcher: the real rt runs as its child. Killing just the launcher
    # would leave rt running and holding the camera. Here a real Python process starts a child that sleeps; after
    # kill_tree, both must be gone. (Real processes, but no waiting: they are killed at once.)
    import subprocess
    import sys

    import psutil

    from rt.supervise import kill_tree

    code = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); " \
           "time.sleep(60)"
    parent = subprocess.Popen([sys.executable, "-c", code])
    p = psutil.Process(parent.pid)
    for _ in range(100):                       # wait (at most ~5 s) until the grandchild exists
        kids = p.children(recursive=True)
        if kids:
            break
        psutil.wait_procs([], timeout=0.05)
    assert kids, "the child process did not start"
    kill_tree(parent, timeout=5)
    assert parent.poll() is not None
    assert not any(k.is_running() and k.status() != psutil.STATUS_ZOMBIE for k in kids)


# ------------------------------------------------------------------------------------------------ camera lost
# The status page also tells the camera's state. In these tests the fake status page answers with a small dict, like
# rt.supervise.probe_status() returns from the real page: {"fps": ..., "camera_state": ...}.

def supervisor_with_camera_state(w, **kw):
    """A supervisor whose fake status page reports both the fps and the camera state stored in the World."""
    w.camera_state = "streaming"
    return Supervisor(make_cmd=lambda: ["python", "rt/run.py"], spawn=w.spawn,
                      probe=lambda: None if w.fps is None else dict(fps=w.fps, camera_state=w.camera_state),
                      kill=w.kill, clock=lambda: w.now, log=w.logs.append, **kw)


@pytest.mark.parametrize("state", ["lost", "retrying", "opening"])
def test_camera_lost_is_left_to_rt_not_killed(state):
    # The webcam was unplugged: rt keeps running, its page answers, fps is 0 and the camera state is "lost" (or
    # "retrying" / "opening" while it tries again). Even after 5 minutes the launcher must NOT kill rt.
    w = World()
    sup = supervisor_with_camera_state(w, grace_s=60, stall_s=30)
    w.run(sup, 70)                       # normal running
    w.fps, w.camera_state = 0.0, state
    w.run(sup, 300)
    assert not w.procs[0].killed and len(w.procs) == 1
    # It said why, once (not every 2 s for 5 minutes).
    assert sum("reopening the camera itself" in m for m in w.logs) == 1


def test_zero_fps_while_camera_claims_streaming_is_still_a_stall():
    # The camera says "streaming" but no frame is processed: detection is hung. That is the real stall: kill it.
    w = World()
    sup = supervisor_with_camera_state(w, grace_s=60, stall_s=30)
    w.run(sup, 70)
    w.fps = 0.0                          # camera_state stays "streaming"
    w.run(sup, 31)
    assert w.procs[0].killed
    assert any("camera streaming" in m for m in w.logs)


def test_no_answer_is_still_a_stall_with_the_new_probe():
    # The page does not answer at all (probe gives None): rt is frozen, whatever the camera was doing. Kill it.
    w = World()
    sup = supervisor_with_camera_state(w, grace_s=60, stall_s=30)
    w.run(sup, 70)
    w.fps = None
    w.run(sup, 31)
    assert w.procs[0].killed


def test_camera_back_then_hang_counts_from_the_hang():
    # Camera lost for 2 minutes (no kill), it comes back and streams, then detection hangs: the 30 s count starts
    # at the hang, not at the moment the camera was lost.
    w = World()
    sup = supervisor_with_camera_state(w, grace_s=60, stall_s=30)
    w.run(sup, 70)
    w.fps, w.camera_state = 0.0, "lost"
    w.run(sup, 120)
    w.camera_state = "streaming"         # back, but fps still 0: hung
    w.run(sup, 29)
    assert not w.procs[0].killed         # only 29 s of "streaming at 0 fps"
    w.run(sup, 2)
    assert w.procs[0].killed


def test_probe_status_reads_fps_and_camera_state_from_a_real_page():
    # A tiny real HTTP server that serves a status.json like rt's: probe_status() must read both fields from it.
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from rt.supervise import probe_fps, probe_status

    body = json.dumps(dict(processing=dict(fps=0.0), camera=dict(state="lost"))).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = "http://127.0.0.1:%d/status.json" % server.server_address[1]
        assert probe_status(url) == dict(fps=0.0, camera_state="lost", camera_attempts=None)
        assert probe_fps(url) == 0.0                  # the old helper still works
    finally:
        server.shutdown()
        server.server_close()
    # Nothing listening on that port any more: "no answer" is None.
    assert probe_status(url, timeout=0.5) is None


# ------------------------------------------------------------------------------------------------ camera lost too long
# "Leave a lost camera to rt" has two limits, so a camera that never comes back (or a reopen loop that hangs) cannot
# keep rt useless for the whole race day:
#   * camera_recover_max_s: the camera has been "recovering" this long in one go -> restart rt anyway;
#   * camera_attempt_stall_s: rt's count of camera open attempts (on its status page) has not gone up this long ->
#     rt's own reopen loop is stuck -> restart rt.
# The fake status page below also reports that open-attempt counter (w.attempts).

def supervisor_with_attempts(w, **kw):
    """Like supervisor_with_camera_state, plus the open-attempt counter rt shows on its page."""
    w.camera_state = "streaming"
    w.attempts = 1
    return Supervisor(make_cmd=lambda: ["python", "rt/run.py"], spawn=w.spawn,
                      probe=lambda: None if w.fps is None else dict(fps=w.fps, camera_state=w.camera_state,
                                                                    camera_attempts=w.attempts),
                      kill=w.kill, clock=lambda: w.now, log=w.logs.append, **kw)


def run_with_attempts_every(w, sup, seconds, every_s):
    """Advance fake time `seconds` (1 s per tick); rt "tries to open the camera" once every `every_s` seconds."""
    for k in range(int(seconds)):
        if every_s and k % every_s == 0:
            w.attempts += 1
        sup.tick()
        w.now += 1.0


def test_camera_lost_with_a_live_reopen_loop_is_left_alone_up_to_the_cap():
    # rt keeps trying every 30 s (its normal maximum wait): 9 minutes of that is fine, no kill.
    w = World()
    sup = supervisor_with_attempts(w, grace_s=60, stall_s=30, camera_recover_max_s=600, camera_attempt_stall_s=60)
    w.run(sup, 70)
    w.fps, w.camera_state = 0.0, "lost"
    run_with_attempts_every(w, sup, 540, every_s=30)
    assert not w.procs[0].killed


def test_camera_lost_longer_than_the_cap_restarts_rt():
    # Same, but the camera never comes back: after 600 s of "lost" the launcher restarts rt anyway.
    w = World()
    sup = supervisor_with_attempts(w, grace_s=60, stall_s=30, camera_recover_max_s=600, camera_attempt_stall_s=60)
    w.run(sup, 70)
    w.fps, w.camera_state = 0.0, "lost"
    run_with_attempts_every(w, sup, 599, every_s=30)
    assert not w.procs[0].killed          # 599 s: not yet
    run_with_attempts_every(w, sup, 2, every_s=30)
    assert w.procs[0].killed              # 600 s: restart
    assert any("camera not back after 600 s" in m for m in w.logs)
    assert sup.restarts[-1][1] == "camera recovery stuck"


def test_reopen_loop_that_stops_trying_restarts_rt_after_60_s():
    # The camera is "opening" and rt's attempt counter does not move (an open() that never returns): after 60 s
    # without a new attempt, restart rt. Long before the 600 s cap.
    w = World()
    sup = supervisor_with_attempts(w, grace_s=60, stall_s=30, camera_recover_max_s=600, camera_attempt_stall_s=60)
    w.run(sup, 70)
    w.fps, w.camera_state = 0.0, "opening"
    run_with_attempts_every(w, sup, 59, every_s=0)     # every_s=0: the counter never changes
    assert not w.procs[0].killed
    run_with_attempts_every(w, sup, 2, every_s=0)
    assert w.procs[0].killed
    assert any("reopen loop is stuck" in m for m in w.logs)


def test_camera_coming_back_resets_both_limits():
    # 500 s lost, then the camera streams again for a while, then it is lost again: the 600 s count starts over.
    w = World()
    sup = supervisor_with_attempts(w, grace_s=60, stall_s=30, camera_recover_max_s=600, camera_attempt_stall_s=60)
    w.run(sup, 70)
    w.fps, w.camera_state = 0.0, "lost"
    run_with_attempts_every(w, sup, 500, every_s=30)
    w.fps, w.camera_state = 10.0, "streaming"
    w.run(sup, 20)
    w.fps, w.camera_state = 0.0, "lost"
    run_with_attempts_every(w, sup, 500, every_s=30)
    assert not w.procs[0].killed


def test_old_rt_without_the_attempt_counter_still_gets_the_cap():
    # A status page without camera_attempts (None): only the 600 s cap applies; the 60 s rule cannot be used.
    w = World()
    sup = supervisor_with_camera_state(w, grace_s=60, stall_s=30, camera_recover_max_s=600,
                                       camera_attempt_stall_s=60)
    w.run(sup, 70)
    w.fps, w.camera_state = 0.0, "retrying"
    w.run(sup, 300)
    assert not w.procs[0].killed
    w.run(sup, 301)
    assert w.procs[0].killed
