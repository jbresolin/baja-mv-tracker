"""The PTZ characterization tool (zone/tools/characterize.py), started from the page.

It moves the camera, and asks before the first move (and, later, for FOV screenshots and two angle marks). Those
questions are answered in a **visible console window** (Windows: a new console), never from the page: a phone tap
should not be able to move a camera nobody is watching. The page shows the window's output live, from the log file
launcher/teerun.py writes.

Only one VISCA client should talk to the PTZ at a time: the tool refuses to start while the Zone PTZ role runs.
"""
import datetime as dt
import re
import subprocess
import time
from pathlib import Path

from launcher.procs import tail_lines, windows_flags

SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class CharacterizationRunner:
    def __init__(self, paths, env=None, spawn=None, clock=time.time):
        self.paths, self.env, self.clock = paths, env, clock
        self.spawn = spawn or (lambda cmd: subprocess.Popen(cmd, cwd=str(paths.code_root), env=env,
                                                            creationflags=windows_flags(visible=True)))
        self.proc = None
        self.log_path = None
        self.out_dir = None

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, ip, folder_name, quick=False, zone_running=False):
        """Returns (ok, message)."""
        from launcher.settings import validate_zone
        if self.running():
            return False, "A caracterização já está rodando (veja a janela dela)."
        if zone_running:
            return False, "Pare a Zona PTZ primeiro: só um programa pode comandar o PTZ por vez."
        _, errors = validate_zone(dict(ptz_host=ip))
        if "ptz_host" in errors:
            return False, "IP do PTZ: " + errors["ptz_host"]
        folder_name = (folder_name or "").strip() or dt.datetime.fromtimestamp(self.clock()).strftime(
            "ptz_%Y-%m-%d_%H-%M")
        if not SAFE_NAME.match(folder_name):
            return False, "Nome da pasta: use só letras, números, _ . - (até 64)."
        self.out_dir = self.paths.reports_dir / folder_name
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.paths.logs_dir / ("characterize_%s.log" % folder_name)
        cmd = [self.paths.python, "-m", "launcher.teerun", str(self.log_path), "zone.tools.characterize",
               "--ip", ip.strip(), "--out", str(self.out_dir)]
        if quick:
            cmd.append("--quick")
        self.proc = self.spawn(cmd)
        return True, ("Caracterização iniciada numa janela própria: confirme o movimento NELA. Resultado em %s"
                      % self.out_dir)

    def status(self, lines=60):
        return dict(running=self.running(), log=tail_lines(self.log_path, lines) if self.log_path else [],
                    out_dir=str(self.out_dir) if self.out_dir else None,
                    exit_code=None if self.proc is None or self.running() else self.proc.poll())


def list_reports(reports_dir):
    """Characterization reports in the data root: [dict(path, name, modified)] newest first."""
    found = []
    for report in Path(reports_dir).glob("*/report.json"):
        found.append(dict(path=str(report), name=report.parent.name,
                          modified=dt.datetime.fromtimestamp(report.stat().st_mtime).isoformat(timespec="minutes")))
    return sorted(found, key=lambda r: r["modified"], reverse=True)
