"""Shared helpers for the launcher tests. pytest reads this file before the tests in this folder.

Run the tests from the repo folder:
    python -m pytest launcher -v

Nothing here needs Windows, a camera, a PTZ, the network or the real models. Processes, clocks, web pages and
cameras are replaced by small fake objects defined below, so every test is fast and gives the same answer every time.

A "fixture" is a function marked @pytest.fixture: a test that names it as an argument gets its result. pytest's own
`tmp_path` fixture gives each test a fresh empty folder, removed afterwards.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]       # the repo folder (the one that contains launcher/, rt/, zone/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from launcher.paths import AppPaths  # noqa: E402


@pytest.fixture
def paths(tmp_path):
    """AppPaths that use the real code (this repo) but a throw-away data folder, like the self-test does."""
    app_paths = AppPaths(code_root=ROOT, data_root=tmp_path / "data", installed=False, python=sys.executable)
    app_paths.make_folders()
    return app_paths


class FakeClock:
    """A clock that only moves when the test says so: clock.advance(5) = "5 seconds later"."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeProcess:
    """Stands in for subprocess.Popen: poll() is None while "running"; finish(code) makes it exit."""

    next_pid = 4000

    def __init__(self, cmd=None):
        FakeProcess.next_pid += 1
        self.pid = FakeProcess.next_pid
        self.cmd = cmd
        self.code = None
        self.killed = False

    def poll(self):
        return self.code

    def finish(self, code):
        self.code = code

    def wait(self, timeout=None):
        return self.code


class FakeSystem:
    """Records what the launcher asks the operating system to do: which processes it started and killed."""

    def __init__(self):
        self.started = []         # FakeProcess objects, in start order
        self.killed = []          # pids, in kill order
        self.events = []          # ("spawn" | "kill", name) in order, for tests about the order of steps

    def spawn(self, cmd, log_file, cwd, env):
        process = FakeProcess(cmd)
        self.started.append(process)
        self.events.append(("spawn", cmd))
        return process

    def kill(self, process):
        process.killed = True
        process.code = -9
        self.killed.append(process.pid)
        self.events.append(("kill", process.pid))


@pytest.fixture
def fake_clock():
    return FakeClock()


@pytest.fixture
def fake_system():
    return FakeSystem()
