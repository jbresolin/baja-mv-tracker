"""One managed child process: start it, stop it (with its whole process tree), tell whether it runs, and keep its
console output in a log file the page can show.

The child is a supervisor (rt/supervise.py or zone/supervisor.py), which itself restarts the real program with
backoff. So this class never restarts anything: if the supervisor exits by itself (e.g. a config error), the page
shows "parou" with the exit code and the last log lines, and the operator decides.

A launcher that is closed or crashes leaves its children running (they keep the race going). The next launcher
finds them again through `<data>/logs/<name>.pid` (pid + process start time, so a recycled pid is not mistaken for
ours) and "adopts" them: it shows them as running and can stop them.

Everything that touches the operating system is injectable (spawn, kill, clock, process lookup), so the tests run
with fake processes and a fake clock.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

LOG_ROTATE_BYTES = 20 * 2**20     # a log bigger than this is renamed to .1 at the next start


def tail_lines(path, max_lines=40, max_bytes=64 * 1024):
    """The last lines of a text file, read from its end (logs can be large)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        data = f.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]               # the first line was cut in the middle
    return lines[-max_lines:]


def windows_flags(visible=False):
    """Process creation flags on Windows: no console window for background children (CREATE_NO_WINDOW), a new
    one for a tool the operator talks to (CREATE_NEW_CONSOLE). 0 elsewhere."""
    if sys.platform != "win32":
        return 0
    return subprocess.CREATE_NEW_CONSOLE if visible else subprocess.CREATE_NO_WINDOW


def default_spawn(cmd, log_file, cwd, env):
    """Start a background child with its stdout + stderr going to the open log file and no stdin."""
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=cwd,
                            env=env, creationflags=windows_flags(visible=False))


def default_kill(proc):
    """Stop a process and all of its children (the supervisor's own kill_tree: psutil, else taskkill /T)."""
    from rt.supervise import kill_tree
    kill_tree(proc, timeout=10.0)


class AdoptedProcess:
    """A process started by an earlier launcher, found again by pid. Looks like a Popen for the code below."""

    def __init__(self, ps_process):
        self.ps = ps_process
        self.pid = ps_process.pid

    def poll(self):
        try:
            import psutil
            if self.ps.is_running() and self.ps.status() != psutil.STATUS_ZOMBIE:
                return None
        except Exception:  # noqa: BLE001
            pass
        return -1           # gone; its real exit code is unknown

    def wait(self, timeout=None):
        try:
            self.ps.wait(timeout)
        except Exception:  # noqa: BLE001
            pass


def find_process(pid, create_time):
    """psutil.Process for a pid that still runs and started at `create_time`, else None."""
    try:
        import psutil
        process = psutil.Process(int(pid))
        if abs(process.create_time() - float(create_time)) < 1.0:
            return process
    except Exception:  # noqa: BLE001
        pass
    return None


def process_create_time(pid):
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:  # noqa: BLE001
        return None


class ManagedProcess:
    """make_cmd() -> the command list (built at each start, so a changed config is picked up)."""

    def __init__(self, name, make_cmd, logs_dir, cwd=None, env=None, spawn=default_spawn, kill=default_kill,
                 clock=time.monotonic, wall=time.time, find=find_process, create_time=process_create_time):
        self.name = name
        self.make_cmd = make_cmd
        self.logs_dir = Path(logs_dir)
        self.cwd, self.env = cwd, env
        self.spawn, self.kill, self.clock, self.wall = spawn, kill, clock, wall
        self.find, self.create_time = find, create_time
        self.lock = threading.RLock()
        self.proc = None
        self.started_at = None          # launcher clock (monotonic) at start
        self.started_wall = None        # epoch seconds, for the page
        self.exit_code = None           # of the last run that ended by itself
        self.stopped_by_user = False
        self.adopted = False

    @property
    def log_path(self):
        return self.logs_dir / ("%s.log" % self.name)

    @property
    def pid_path(self):
        return self.logs_dir / ("%s.pid" % self.name)

    # ---- state
    def running(self):
        with self.lock:
            if self.proc is None:
                return False
            code = self.proc.poll()
            if code is None:
                return True
            self.exit_code = code         # ended by itself
            self.proc = None
            self._forget_pid()
            self._log_line("processo terminou sozinho (código %s)" % code)
            return False

    def status(self, log_lines=30):
        running = self.running()
        with self.lock:
            uptime = (self.clock() - self.started_at) if running and self.started_at is not None else None
            return dict(name=self.name, running=running, pid=self.proc.pid if running else None,
                        adopted=self.adopted if running else False,
                        started=dt.datetime.fromtimestamp(self.started_wall).isoformat(timespec="seconds")
                        if running and self.started_wall else None,
                        uptime_s=round(uptime, 1) if uptime is not None else None,
                        exit_code=None if running else self.exit_code,
                        stopped_by_user=self.stopped_by_user,
                        log=tail_lines(self.log_path, log_lines), log_file=str(self.log_path))

    # ---- actions
    def start(self):
        """Start the process. Returns (ok, message)."""
        with self.lock:
            if self.running():
                return False, "%s já está rodando (pid %s)" % (self.name, self.proc.pid)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self._rotate_log()
            cmd = [str(part) for part in self.make_cmd()]
            self._log_line("iniciando: " + " ".join('"%s"' % c if " " in c else c for c in cmd))
            log_file = open(self.log_path, "ab")       # the child gets its own handle; ours is closed below
            try:
                self.proc = self.spawn(cmd, log_file, self.cwd, self.env)
            finally:
                log_file.close()
            self.started_at, self.started_wall = self.clock(), self.wall()
            self.exit_code, self.stopped_by_user, self.adopted = None, False, False
            self._remember_pid()
            return True, "%s iniciado (pid %s)" % (self.name, self.proc.pid)

    def stop(self, before_kill=None):
        """Stop the process tree. before_kill() runs first while it is still alive (the zone's PARAR request).
        Returns (ok, message)."""
        with self.lock:
            if not self.running():
                self.stopped_by_user = True
                return True, "%s já estava parado" % self.name
            pid = self.proc.pid
            if before_kill is not None:
                try:
                    before_kill()
                except Exception as e:  # noqa: BLE001 (stopping must go on)
                    self._log_line("antes de parar: %s: %s" % (type(e).__name__, e))
            self._log_line("parando (pid %s) a pedido do painel" % pid)
            self.kill(self.proc)
            self.proc = None
            self.stopped_by_user = True
            self._forget_pid()
            return True, "%s parado" % self.name

    def adopt(self):
        """Take over a process an earlier launcher started (see the module's docstring). Returns True if found."""
        with self.lock:
            if self.proc is not None or not self.pid_path.exists():
                return False
            try:
                data = json.loads(self.pid_path.read_text(encoding="utf-8"))
                found = self.find(data["pid"], data["create_time"])
            except (OSError, ValueError, KeyError, TypeError):
                found = None
            if found is None:
                self._forget_pid()
                return False
            self.proc = AdoptedProcess(found) if not hasattr(found, "poll") else found
            self.started_at, self.started_wall = self.clock(), float(data.get("started_wall") or self.wall())
            self.adopted = True
            self._log_line("launcher reiniciado: processo pid %s retomado" % self.proc.pid)
            return True

    # ---- files
    def _remember_pid(self):
        data = dict(pid=self.proc.pid, create_time=self.create_time(self.proc.pid), started_wall=self.started_wall)
        if data["create_time"] is None:
            return                         # cannot identify it safely later: don't write a pid file
        self.pid_path.write_text(json.dumps(data), encoding="utf-8")

    def _forget_pid(self):
        try:
            self.pid_path.unlink()
        except OSError:
            pass

    def _rotate_log(self):
        try:
            if self.log_path.exists() and self.log_path.stat().st_size > LOG_ROTATE_BYTES:
                self.log_path.replace(self.log_path.with_suffix(".log.1"))
        except OSError:
            pass

    def _log_line(self, message):
        """A launcher line inside the child's log, so the page shows starts and stops in order."""
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            stamp = dt.datetime.fromtimestamp(self.wall()).isoformat(timespec="seconds")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("[launcher %s] %s\n" % (stamp, message))
        except OSError:
            pass
