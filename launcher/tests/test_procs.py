"""Tests for launcher/procs.py: one managed child process (start, stop, state, log file, taking over a process an
earlier launcher left running).

No real process is started: FakeSystem (conftest.py) hands out FakeProcess objects and records kills, and FakeClock
is a clock the test moves by hand. So "the process ran for 5 minutes" takes no time at all.
"""
import json

from launcher.procs import ManagedProcess, tail_lines


def make(fake_system, fake_clock, tmp_path, name="lapline", find=None, create_time=lambda pid: 123.0):
    return ManagedProcess(name, lambda: ["python", "rt/supervise.py", "--config", "x.toml"], tmp_path / "logs",
                          spawn=fake_system.spawn, kill=fake_system.kill, clock=fake_clock, wall=lambda: 1.7e9,
                          find=find or (lambda pid, created: None), create_time=create_time)


def test_start_runs_the_command_and_reports_running(fake_system, fake_clock, tmp_path):
    process = make(fake_system, fake_clock, tmp_path)
    ok, message = process.start()
    assert ok, message
    assert fake_system.started[0].cmd == ["python", "rt/supervise.py", "--config", "x.toml"]
    fake_clock.advance(300)                                   # five minutes later
    status = process.status()
    assert status["running"] is True
    assert status["uptime_s"] == 300
    # the start is written into the log the page shows
    assert any("iniciando" in line for line in status["log"])


def test_a_second_start_is_refused_while_running(fake_system, fake_clock, tmp_path):
    process = make(fake_system, fake_clock, tmp_path)
    process.start()
    ok, message = process.start()
    assert not ok and "já está rodando" in message
    assert len(fake_system.started) == 1


def test_stop_kills_the_process_and_runs_the_before_kill_step_first(fake_system, fake_clock, tmp_path):
    process = make(fake_system, fake_clock, tmp_path)
    process.start()
    order = []
    ok, _ = process.stop(before_kill=lambda: order.append("before kill (still alive: %s)" % process.running()))
    assert ok
    assert order == ["before kill (still alive: True)"]
    assert fake_system.killed == [fake_system.started[0].pid]
    status = process.status()
    assert status["running"] is False and status["stopped_by_user"] is True


def test_a_failing_before_kill_step_does_not_prevent_the_kill(fake_system, fake_clock, tmp_path):
    def broken():
        raise ConnectionError("zone page did not answer")
    process = make(fake_system, fake_clock, tmp_path)
    process.start()
    process.stop(before_kill=broken)
    assert fake_system.killed                                # killed anyway
    assert any("zone page did not answer" in line for line in process.status()["log"])


def test_a_process_that_exits_by_itself_shows_its_exit_code(fake_system, fake_clock, tmp_path):
    # e.g. rt/supervise.py stops at once because launch.event is empty: the page must say "parou (código 1)".
    process = make(fake_system, fake_clock, tmp_path)
    process.start()
    fake_system.started[0].finish(1)
    status = process.status()
    assert status["running"] is False
    assert status["exit_code"] == 1
    assert status["stopped_by_user"] is False
    assert any("terminou sozinho" in line for line in status["log"])


def test_stopping_a_stopped_process_is_harmless(fake_system, fake_clock, tmp_path):
    process = make(fake_system, fake_clock, tmp_path)
    ok, message = process.stop()
    assert ok and "já estava parado" in message
    assert fake_system.killed == []


def test_pid_file_lets_a_new_launcher_take_over_a_running_process(fake_system, fake_clock, tmp_path):
    # Launcher 1 starts the lap line, then is closed (the lap line keeps running, on purpose).
    first = make(fake_system, fake_clock, tmp_path)
    first.start()
    pid_file = json.loads(first.pid_path.read_text())
    assert pid_file["pid"] == fake_system.started[0].pid and pid_file["create_time"] == 123.0
    # Launcher 2 finds it again through the pid file (find() confirms the pid AND its start time).
    still_running = fake_system.started[0]
    second = make(fake_system, fake_clock, tmp_path,
                  find=lambda pid, created: still_running if (pid, created) == (still_running.pid, 123.0) else None)
    assert second.adopt() is True
    assert second.status()["running"] is True and second.status()["adopted"] is True
    second.stop()
    assert fake_system.killed == [still_running.pid]
    assert not second.pid_path.exists()


def test_a_stale_pid_file_is_ignored_and_removed(fake_system, fake_clock, tmp_path):
    # The pid now belongs to some other program (or nothing): find() says no, and nothing is adopted or killed.
    first = make(fake_system, fake_clock, tmp_path)
    first.start()
    second = make(fake_system, fake_clock, tmp_path, find=lambda pid, created: None)
    assert second.adopt() is False
    assert not second.pid_path.exists()
    assert second.status()["running"] is False


def test_no_pid_file_when_the_start_time_cannot_be_read(fake_system, fake_clock, tmp_path):
    # Without the start time, a recycled pid could be mistaken for ours later: better no pid file at all.
    process = make(fake_system, fake_clock, tmp_path, create_time=lambda pid: None)
    process.start()
    assert not process.pid_path.exists()


def test_tail_lines_returns_the_end_of_a_big_file(tmp_path):
    log = tmp_path / "big.log"
    log.write_text("".join("line %d\n" % i for i in range(100000)), encoding="utf-8")
    assert tail_lines(log, 3) == ["line 99997", "line 99998", "line 99999"]
    assert tail_lines(tmp_path / "missing.log") == []
