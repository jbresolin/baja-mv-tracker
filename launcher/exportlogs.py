"""The "Exportar logs" button: one zip with what is needed to look at a problem from afar.

Contents: every log in <data>/logs (launcher, lapline, zone, characterization), the config files (with tokens and
PINs blanked), the launcher log of rt (out/lapline/launcher.log), the newest lap-line run (its .jsonl, .csv and
.stats.json; not the video) and the newest zone control log (zone_*.jsonl). Recordings are never included (GBs).
"""
import datetime as dt
import re
import socket
import zipfile
from pathlib import Path

SECRET_LINE = re.compile(r'^(\s*(token|setup_token|pin)\s*=\s*)".*"', re.M)
SECRET_JSON = re.compile(r'("(pin|token)"\s*:\s*)"[^"]*"')


def redact(text):
    """Blank the values of token / setup_token / pin keys (TOML and JSON)."""
    return SECRET_JSON.sub(r'\1"***"', SECRET_LINE.sub(r'\1"***"', text))


def newest(files):
    files = [f for f in files if f.is_file()]
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def files_to_export(paths):
    """-> [(path on disk, name in the zip, redact?)]"""
    chosen = []
    for log in sorted(paths.logs_dir.glob("*.log*")):
        chosen.append((log, "logs/" + log.name, False))
    for cfg in sorted(paths.config_dir.glob("*.toml")) + sorted(paths.config_dir.glob("*.json")):
        chosen.append((cfg, "config/" + cfg.name, True))
    lapline = paths.out_dir / "lapline"
    if (lapline / "launcher.log").exists():
        chosen.append((lapline / "launcher.log", "out/lapline/launcher.log", False))
    latest_run = newest(lapline.glob("*/*.jsonl")) if lapline.exists() else None
    if latest_run is not None:
        stem = latest_run.with_suffix("")
        for sibling in (latest_run, stem.with_suffix(".csv"), Path(str(stem) + ".stats.json")):
            if sibling.exists():
                chosen.append((sibling, "out/lapline/%s/%s" % (latest_run.parent.name, sibling.name), False))
    zone_out = paths.out_dir / "zone"
    latest_zone = newest(zone_out.glob("zone_*.jsonl")) if zone_out.exists() else None
    if latest_zone is not None:
        chosen.append((latest_zone, "out/zone/" + latest_zone.name, False))
    return chosen


def export_logs(paths, now=None, hostname=None):
    """Write the zip into <data>/exports and return its path."""
    now = now or dt.datetime.now()
    hostname = re.sub(r"[^A-Za-z0-9_-]", "_", hostname or socket.gethostname())
    paths.exports_dir.mkdir(parents=True, exist_ok=True)
    target = paths.exports_dir / ("BajaMV-logs-%s-%s.zip" % (hostname, now.strftime("%Y-%m-%d_%H-%M-%S")))
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, name, secret in files_to_export(paths):
            try:
                if secret:
                    zf.writestr(name, redact(path.read_text(encoding="utf-8", errors="replace")))
                else:
                    zf.write(path, name)
            except OSError as e:          # a file being written can still be read on Windows; if not, note it
                zf.writestr(name + ".ERRO.txt", "não foi possível ler: %s" % e)
        zf.writestr("versao.txt", "versão %s\nmodelos: %s\ndados: %s\n" % (
            paths.version, paths.build_info.get("models_note", "?"), paths.data_root))
    return target
