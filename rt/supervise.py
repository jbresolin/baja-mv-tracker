"""Launcher for an unattended rt (WP-L1): starts rt/run.py with a named config and restarts it when it dies or stalls.

usage:
  python rt/supervise.py --config rt/configs/lapline.toml        (rt\\run_lapline.bat does this)
  python rt/supervise.py --config rt/configs/lapline.toml --dry-run   (print the rt command and exit)

Rules:
  * rt exits (crash, or any exit code): start it again after a wait that doubles each time: 5 s, 10 s, 20 s, 40 s,
    60 s, 60 s, ... If the run that just ended had been up for `healthy_reset_min` minutes, the wait starts again at
    5 s (one crash after hours of running is not a crash loop).
  * rt is alive but its status page (http://127.0.0.1:<status.port>/status.json) reports processing fps = 0, or
    does not answer, for `stall_s` seconds in a row: kill it and restart it the same way. Nothing is checked during
    the first `startup_grace_s` seconds of a run (models load, the camera opens).
  * Every start, exit, stall and kill is written to the launcher log (`<out_dir>/launcher.log`) and the console.

The kill takes the whole process tree: on Windows `.venv\\Scripts\\python.exe` is a launcher that starts the real
interpreter as a child, and killing only the launcher would leave rt running (holding the camera).

Settings: the [launch] section of the config (event, source, out_dir, extra_args, and the numbers above).
Each run writes its own `<out_dir>/<date>/lapline_<HH-MM-SS>.jsonl` (+ .csv, .stats.json, and rec/ for the video).
"""
import argparse
import datetime as dt
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rt import config as C  # noqa: E402
from rt.backoff import Backoff  # noqa: E402


def kill_tree(proc, timeout=10.0):
    """Stop a process and all its children. psutil if installed; else Windows taskkill /T; else proc.kill()."""
    try:
        import psutil
        try:
            parent = psutil.Process(proc.pid)
            procs = parent.children(recursive=True) + [parent]
        except psutil.NoSuchProcess:
            procs = []
        for process in procs:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(procs, timeout=timeout)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
    except ImportError:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        else:
            proc.kill()
    try:
        proc.wait(timeout=timeout)
    except Exception:  # noqa: BLE001
        pass


def probe_fps(url, timeout=2.0):
    """processing fps from rt's status page, or None if it does not answer."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return float(json.loads(response.read().decode("utf-8"))["processing"]["fps"])
    except Exception:  # noqa: BLE001
        return None


# Exit codes of rt/run.py that mean something specific, named in the launcher log.
EXIT_REASONS = {C.EXIT_STATUS_PORT_BUSY: "the status page's port is busy: another program or another rt uses it; "
                                         "see rt's console. Close it or change status.port"}


class Supervisor:
    """The restart rules as a small state machine. tick() is called every `poll_s` seconds by run(); tests call it
    directly with a fake clock, a fake process and a fake fps probe, so no test ever sleeps.

    make_cmd() -> the command list for one run; spawn(cmd) -> a process object (poll(), pid); probe() -> fps or
    None; kill(proc) stops it; clock() -> seconds (monotonic); log(msg) writes a line."""

    def __init__(self, make_cmd, spawn, probe, kill, clock, log, first_wait_s=5.0, max_wait_s=60.0, stall_s=30.0,
                 grace_s=60.0, healthy_reset_s=600.0, check_stall=True):
        self.make_cmd, self.spawn, self.probe, self.kill, self.clock, self.log = make_cmd, spawn, probe, kill, clock, log
        self.backoff = Backoff(first_wait_s, 2.0, max_wait_s)
        self.stall_s, self.grace_s, self.healthy_reset_s = float(stall_s), float(grace_s), float(healthy_reset_s)
        self.check_stall = check_stall
        self.proc = None
        self.started_at = None
        self.next_start_at = None     # None = start now
        self.zero_since = None
        self.starts = 0
        self.restarts = []            # (time, reason, wait) of every scheduled restart

    def _start(self, now):
        cmd = self.make_cmd()
        self.log("start #%d: %s" % (self.starts + 1, " ".join('"%s"' % c if " " in str(c) else str(c) for c in cmd)))
        self.proc = self.spawn(cmd)
        self.starts += 1
        self.started_at = now
        self.zero_since = None
        self.next_start_at = None

    def _schedule_restart(self, now, reason):
        ran = now - self.started_at if self.started_at is not None else 0.0
        if ran >= self.healthy_reset_s:
            self.backoff.reset()
        wait = self.backoff.next()
        self.proc = None
        self.next_start_at = now + wait
        self.restarts.append((now, reason, wait))
        self.log("%s after %.0f s of running; restarting in %.0f s" % (reason, ran, wait))

    def tick(self):
        now = self.clock()
        if self.proc is None:
            if self.next_start_at is None or now >= self.next_start_at:
                self._start(now)
            return
        return_code = self.proc.poll()
        if return_code is not None:
            why = EXIT_REASONS.get(return_code)
            self._schedule_restart(now, "rt exited with code %s%s" % (return_code, " (%s)" % why if why else ""))
            return
        if not self.check_stall or now - self.started_at < self.grace_s:
            return
        fps = self.probe()
        if fps is not None and fps > 0:
            self.zero_since = None
            return
        if self.zero_since is None:
            self.zero_since = now
            self.log("status: %s; restart if it lasts %.0f s" % ("no answer" if fps is None else "fps = 0",
                                                                 self.stall_s))
        if now - self.zero_since >= self.stall_s:
            self.log("rt stalled (%s for %.0f s): killing pid %s" % (
                "no status answer" if fps is None else "fps = 0", now - self.zero_since, self.proc.pid))
            self.kill(self.proc)
            self._schedule_restart(now, "rt stalled")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.log("stopping rt (pid %s)" % self.proc.pid)
            self.kill(self.proc)
        self.proc = None


class FileLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = "%s  %s" % (dt.datetime.now().astimezone().isoformat(timespec="seconds"), msg)
        print("[launcher] " + line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def build_command(cfg, config_path, python=None, now=None):
    """The rt/run.py command for one run, with a fresh output file named after the start time. With the status page
    on, rt is told it is required (--require-status): the launcher watches rt through it, so rt must exit with a
    clear message, not run on blind, if its port is busy."""
    launch_cfg = cfg["launch"]
    now = now or dt.datetime.now()
    out_dir = C.resolve(launch_cfg["out_dir"]) / now.strftime("%Y-%m-%d")
    out = out_dir / ("%s_%s.jsonl" % (launch_cfg.get("name", "lapline"), now.strftime("%H-%M-%S")))
    cmd = [python or sys.executable, str(ROOT / "rt" / "run.py"), "--source", str(launch_cfg["source"]), "--event",
           str(launch_cfg["event"]), "--config", str(config_path), "--out", str(out)]
    if cfg["status"].get("enabled", True):
        cmd.append("--require-status")
    return cmd + [str(arg) for arg in launch_cfg.get("extra_args", [])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="rt config with a [launch] section")
    ap.add_argument("--dry-run", action="store_true", help="print the rt command and exit")
    a = ap.parse_args()
    cfg = C.load(a.config)
    launch_cfg, status_cfg = cfg["launch"], cfg["status"]
    if not launch_cfg.get("event"):
        raise SystemExit("launch.event is empty in %s" % a.config)
    if a.dry_run:
        print(build_command(cfg, a.config))
        return
    log = FileLog(C.resolve(launch_cfg["out_dir"]) / "launcher.log")
    url = "http://127.0.0.1:%d/status.json" % int(status_cfg["port"])
    check = bool(status_cfg.get("enabled", True))
    if not check:
        log("WARNING status page disabled: stalls cannot be detected, only exits")
    sup = Supervisor(make_cmd=lambda: build_command(cfg, a.config),
                     spawn=lambda cmd: subprocess.Popen(cmd, cwd=str(ROOT)),
                     probe=lambda: probe_fps(url), kill=kill_tree, clock=time.monotonic, log=log,
                     first_wait_s=float(launch_cfg["restart_first_s"]), max_wait_s=float(launch_cfg["restart_max_s"]),
                     stall_s=float(launch_cfg["stall_s"]), grace_s=float(launch_cfg["startup_grace_s"]),
                     healthy_reset_s=float(launch_cfg["healthy_reset_min"]) * 60, check_stall=check)
    log("launcher start: config %s, status %s" % (a.config, url))
    try:
        while True:
            sup.tick()
            time.sleep(float(launch_cfg.get("poll_s", 2)))
    except KeyboardInterrupt:
        log("launcher interrupted (Ctrl+C)")
    finally:
        sup.stop()
        log("launcher end")


if __name__ == "__main__":
    main()
