"""Tests for `python -m launcher --shutdown` (launcher/__main__.py: request_shutdown and stop_leftover_roles).

The installer runs --shutdown before an upgrade and before an uninstall, because a running Python holds the program's
files. Two situations:
  * a launcher (the tray app) is running: --shutdown asks it, over HTTP, to stop both roles and exit;
  * NO launcher is running, but the roles are (the operator closed the tray with "Fechar o painel", which on purpose
    leaves the race running). Nobody answers the HTTP request, so --shutdown must find the roles itself, through
    their pid files, and stop them. For the zone that means the SAFE stop (PARAR on the zone page, kill, then a VISCA
    stop to the PTZ): otherwise Windows would kill it hard and a moving camera would keep moving.

No real process, network or camera is used: processes are fakes from conftest.py, and the "network calls" are small
functions that only write down that they were called.
"""
from launcher import settings as S
from launcher.__main__ import request_shutdown, stop_leftover_roles
from launcher.procs import ManagedProcess
from launcher.roles import RoleManager

START_TIME = 123.0          # the fake "process start time" written into the pid files


def make_roles(paths, fake_system, fake_clock, calls, find=None):
    """A RoleManager whose processes are fakes. `calls` collects the zone's PARAR and VISCA stop requests.
    find(pid, start time) is how a new manager recognises a process from a pid file (None = "not running")."""
    def make_process(name, make_cmd):
        return ManagedProcess(name, make_cmd, paths.logs_dir, spawn=fake_system.spawn, kill=fake_system.kill,
                              clock=fake_clock, create_time=lambda pid: START_TIME,
                              find=find or (lambda pid, created: None))
    return RoleManager(paths, make_process=make_process, zone_stop=lambda port: calls.append(("PARAR", port)),
                       ptz_stop=lambda host, port: calls.append(("VISCA stop", host)), log=lambda m: None)


def save_both_roles(paths):
    S.write_lapline(paths, S.validate_lapline(dict(event="27SE", camera_device="0"))[0])
    S.write_zone(paths, S.validate_zone(dict(camera_device="1", ptz_host="192.168.0.100", web_port=8091))[0])


def start_roles_then_close_the_launcher(paths, fake_system, fake_clock):
    """Launcher 1 starts both roles and is then closed: the roles keep running, their pid files stay behind.
    Returns a find() that recognises those still-running fake processes, as psutil would."""
    save_both_roles(paths)
    first = make_roles(paths, fake_system, fake_clock, [])
    assert first.start("lapline")[0] and first.start("zone")[0]
    running = {process.pid: process for process in fake_system.started}

    def find(pid, created):
        process = running.get(pid)
        return process if process is not None and process.poll() is None and created == START_TIME else None
    return find


def test_with_no_launcher_answering_the_left_over_roles_are_stopped_safely(paths, fake_system, fake_clock):
    find = start_roles_then_close_the_launcher(paths, fake_system, fake_clock)
    calls = []
    code = request_shutdown(8780, paths=paths, ping_fn=lambda port, timeout=1.0: None,
                            make_roles=lambda: make_roles(paths, fake_system, fake_clock, calls, find=find),
                            log=lambda m: None)
    assert code == 0
    # both processes were killed ...
    assert sorted(fake_system.killed) == sorted(p.pid for p in fake_system.started)
    # ... and the zone got its safe stop: PARAR on its page (port 8091) and a VISCA stop to the PTZ
    assert ("PARAR", 8091) in calls
    assert ("VISCA stop", "192.168.0.100") in calls
    # the pid files are gone, so the next launcher does not try to adopt dead processes
    assert not list(paths.logs_dir.glob("*.pid"))


def test_nothing_running_and_no_launcher_is_a_clean_success(paths, fake_system, fake_clock):
    calls = []
    stopped = stop_leftover_roles(paths, make_roles=lambda: make_roles(paths, fake_system, fake_clock, calls),
                                  log=lambda m: None)
    assert stopped == []
    assert fake_system.killed == [] and calls == []


def test_a_running_launcher_is_asked_first_and_left_overs_are_checked_after(paths, fake_system, fake_clock):
    # The launcher answers the first ping, takes the shutdown request, and is gone at the next ping.
    answers = iter([{"app": "bajamv-launcher"}, None, None])
    posted = []
    code = request_shutdown(8780, paths=paths, ping_fn=lambda port, timeout=1.0: next(answers),
                            post=posted.append, sleep=lambda s: None,
                            make_roles=lambda: make_roles(paths, fake_system, fake_clock, []), log=lambda m: None)
    assert code == 0
    assert posted == ["http://127.0.0.1:8780/api/shutdown"]


def test_a_failed_shutdown_request_is_reported_with_exit_code_1(paths, fake_system, fake_clock):
    def broken_post(url):
        raise OSError("connection reset")
    answers = iter([{"app": "bajamv-launcher"}, None, None])
    code = request_shutdown(8780, paths=paths, ping_fn=lambda port, timeout=1.0: next(answers), post=broken_post,
                            sleep=lambda s: None, make_roles=lambda: make_roles(paths, fake_system, fake_clock, []),
                            log=lambda m: None)
    assert code == 1
