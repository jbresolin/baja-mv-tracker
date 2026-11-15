"""The two roles of a laptop and how they are started and stopped. Only the existing entry points are used:

* Lap line  ("lapline"):  python rt/supervise.py --config <data>/config/lapline.toml
  (the same thing rt\\run_lapline.bat runs: it starts rt/run.py and restarts it if it dies or stalls)
* Zone PTZ  ("zone"):     python -m zone.supervisor --name zone --config <cfg> -- python -m zone.app --config <cfg>
  (the same thing zone\\run_zone.bat runs), plus, only when the config has recording.ptz_usb = true, the PTZ USB
  recorder under its own supervisor ("ptz_recorder"), like the .bat's second window.

Stopping the zone safely. A VISCA PTZ has no dead-man of its own: the zone process sends every move with a short
TTL and stops the camera itself. A killed process runs no code, so a camera that was moving keeps moving
(zone/README.md, "Stopping"). The launcher therefore stops the zone in three steps:
  1. ask the zone's own page for the big PARAR (POST /api/stop): STOPPED mode, the camera is stopped;
  2. kill the process tree (as the supervisors themselves do);
  3. send the PTZ a VISCA stop + zoom stop itself, in case step 1 could not reach the zone.
"""
import json
import os
import urllib.request
from dataclasses import asdict

from launcher import settings as S
from launcher.procs import ManagedProcess

ROLE_NAMES = {"lapline": "Linha de chegada (lap line)", "zone": "Zona PTZ"}


def child_env(paths, base=None):
    """Environment of every child: the data root, the build's version (rt writes it into its outputs), unbuffered
    UTF-8 output (it goes to a log file), and the NVIDIA GPU hidden (CPU / Intel iGPU only, as the .bat files do;
    an empty value would be dropped by Windows, hence -1)."""
    env = dict(os.environ if base is None else base)
    env.update({
        "BAJAMV_DATA": str(paths.data_root),
        "CUDA_VISIBLE_DEVICES": "-1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    if paths.installed:
        env["RT_VERSION"] = paths.version
        # Like `python -P`: the current folder / script folder is not put first on the import path, so a child
        # always imports the installed code (INSTALL\\app, through the .pth file), never a stray copy of rt/ or zone/
        # in whatever folder it happens to start from. Children also start in the code folder (cwd), as a second
        # guard. Not in repo mode: there the repo folder on the path is how `python -m zone.app` finds the code.
        env["PYTHONSAFEPATH"] = "1"
    return env


def lapline_command(paths):
    return [paths.python, str(paths.code_root / "rt" / "supervise.py"), "--config", str(paths.lapline_config)]


def zone_command(paths):
    cfg = str(paths.zone_config)
    return [paths.python, "-m", "zone.supervisor", "--name", "zone", "--config", cfg, "--",
            paths.python, "-m", "zone.app", "--config", cfg]


def ptz_recorder_command(paths):
    cfg = str(paths.zone_config)
    return [paths.python, "-m", "zone.supervisor", "--name", "ptz_recorder", "--config", cfg, "--",
            paths.python, "-m", "zone.tools.record_ptz", "--config", cfg]


def zone_wants_ptz_recorder(paths):
    return bool(S.get_key(S.read_toml(paths.zone_config), "recording.ptz_usb", False))


def request_zone_stop(port, timeout=2.0):
    """The zone page's big PARAR (STOPPED mode + camera stop). Raises if the zone does not answer."""
    request = urllib.request.Request("http://127.0.0.1:%d/api/stop" % int(port), data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def visca_stop(host, port=52381):
    """Send the PTZ a stop + zoom stop directly (step 3 of the safe stop). Raises if it cannot."""
    from zone.clock import SystemClock
    from zone.ptz.visca import ViscaPtz
    ptz = ViscaPtz(host, int(port), 0.2, clock=SystemClock())
    try:
        ptz.stop()
        ptz.zoom_stop()
    finally:
        ptz.close()


class RoleManager:
    """The managed processes of both roles. `make_process(name, make_cmd)` builds a ManagedProcess (tests pass one
    with fake processes); `zone_stop` and `ptz_stop` replace the two network calls of the zone's safe stop."""

    def __init__(self, paths, make_process=None, zone_stop=request_zone_stop, ptz_stop=visca_stop, log=print):
        self.paths = paths
        self.log = log
        env = child_env(paths)
        make_process = make_process or (lambda name, make_cmd: ManagedProcess(
            name, make_cmd, paths.logs_dir, cwd=str(paths.code_root), env=env))
        self.processes = {
            "lapline": make_process("lapline", lambda: lapline_command(paths)),
            "zone": make_process("zone", lambda: zone_command(paths)),
            "ptz_recorder": make_process("ptz_recorder", lambda: ptz_recorder_command(paths)),
        }
        self.zone_stop, self.ptz_stop = zone_stop, ptz_stop

    def adopt_running(self):
        """After a launcher restart: take over the role processes the last launcher left running."""
        for process in self.processes.values():
            if process.adopt():
                self.log("retomado: %s (pid %s)" % (process.name, process.proc.pid))

    # ---- checks before a start
    def config_problem(self, role):
        """Why a role cannot start yet (pt-BR, for the page), or None."""
        if role == "lapline":
            if not self.paths.lapline_config.exists():
                return "Salve as configurações da linha de chegada primeiro."
            settings, errors = S.validate_lapline(asdict(S.read_lapline(self.paths)))
            if errors:
                return "Configuração com erro: " + "; ".join("%s: %s" % kv for kv in errors.items())
        elif role == "zone":
            if not self.paths.zone_config.exists():
                return "Salve as configurações da zona PTZ primeiro."
        else:
            return "Papel desconhecido: %s" % role
        return None

    # ---- actions (each returns (ok, message))
    def start(self, role):
        problem = self.config_problem(role)
        if problem:
            return False, problem
        ok, message = self.processes[role].start()
        if ok and role == "zone" and zone_wants_ptz_recorder(self.paths):
            self.processes["ptz_recorder"].start()
        return ok, message

    def stop(self, role):
        if role == "zone":
            self.processes["ptz_recorder"].stop()
            if not self.processes["zone"].running():
                return self.processes["zone"].stop()
            return self._stop_zone_safely()
        if role not in self.processes:
            return False, "Papel desconhecido: %s" % role
        return self.processes[role].stop()

    def restart(self, role):
        self.stop(role)
        return self.start(role)

    def stop_all(self):
        for role in ("zone", "lapline"):
            self.stop(role)

    def _stop_zone_safely(self):
        zone_cfg = S.read_toml(self.paths.zone_config)
        web_port = int(S.get_key(zone_cfg, "web.port", 8090))

        def ask_zone_to_stop():
            self.zone_stop(web_port)            # step 1: the zone's own PARAR

        ok, message = self.processes["zone"].stop(before_kill=ask_zone_to_stop)     # step 2: kill the tree
        host = S.get_key(zone_cfg, "ptz.host", "")
        if S.get_key(zone_cfg, "ptz.driver", "visca") == "visca" and host:
            try:                                                                        # step 3: VISCA stop
                self.ptz_stop(host, int(S.get_key(zone_cfg, "ptz.port", 52381)))
            except Exception as e:  # noqa: BLE001
                message += " (aviso: parada direta do PTZ falhou: %s)" % e
        return ok, message

    def running_roles(self):
        return [role for role in ("lapline", "zone") if self.processes[role].running()]

    def status(self, log_lines=30):
        return {name: process.status(log_lines) for name, process in self.processes.items()}
