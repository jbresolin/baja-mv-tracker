"""Where things are: the code, the bundled tools (Python, ffmpeg) and the writable data folder.

Two modes, decided by one file:

* **repo mode** (a git checkout, run with `python -m launcher`): the code root is the repo. Data goes to
  `<repo>/out/bajamv/` unless told otherwise, and ffmpeg is whatever is on PATH. The repo's own workflow
  (rt\\run_lapline.bat, zone\\run_zone.bat, configs under rt/configs and zone/configs) is not touched at all.
* **installed mode** (the Windows installer): `build_info.json` sits next to the code (the build writes it).
  The install folder (Program Files\\BajaMV) is read-only for normal users, so everything that is written goes to
  the data root (default C:\\BajaMV, chosen in the installer and written to `install.ini` next to the code).

Installed layout (Program Files\\BajaMV = INSTALL):
    INSTALL\\python\\            private Python 3.12 with every runtime package
    INSTALL\\ffmpeg\\bin\\       ffmpeg.exe, ffprobe.exe
    INSTALL\\app\\               the code root: rt\\ zone\\ launcher\\ rosters\\ deploy\\models\\ build_info.json install.ini

Data root layout (both modes):
    config\\     lapline.toml, zone.toml (+ lapline.beam.toml from the status page), launcher.json
    out\\        lapline\\<date>\\*.jsonl + rec\\, zone\\ (control logs, recordings)
    logs\\       one log per managed process (launcher.log, lapline.log, zone.log, characterize_*.log)
    rosters\\    rosters fetched with the "Baixar lista de equipes" button (win over the bundled ones)
    zones\\      zone.json (the taught zone), zone.agent.json, zone_backups\\
    reports\\    PTZ characterization reports (one folder per run)
    exports\\    zips made by "Exportar logs"
"""
import configparser
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The folder that holds rt/, zone/ and launcher/: the repo, or INSTALL\app.
CODE_ROOT = Path(__file__).resolve().parent.parent

# Environment variable that points every process at the data root (the launcher sets it for its children).
DATA_ROOT_ENV = "BAJAMV_DATA"

# Default data root of an installed app (the installer asks, this is its suggestion).
DEFAULT_INSTALLED_DATA_ROOT = Path("C:/BajaMV")

# The launcher's own web page.
DEFAULT_LAUNCHER_PORT = 8780


def read_build_info(code_root):
    """build_info.json written by tools/installer/stage_app.py, or None in a repo checkout."""
    path = Path(code_root) / "build_info.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": "unknown", "error": "build_info.json unreadable"}


def decode_ini_bytes(raw):
    """Text of install.ini. Inno Setup writes INI files with the Windows API, which uses the PC's ANSI code page
    (cp1252 on a Brazilian Windows), not UTF-8: a data folder like C:\\Gravações arrives as cp1252 bytes. Try
    UTF-8 first (a file we or a test wrote), then the ANSI code page ("mbcs", Windows only), then cp1252, which can
    decode any byte, so this never raises."""
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    for encoding in ("utf-8", "mbcs", "cp1252"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):       # LookupError: "mbcs" exists only on Windows
            continue
    return raw.decode("latin-1")


def read_install_ini(code_root):
    """The data root the installer wrote to install.ini ([paths] data_root = ...), or None."""
    path = Path(code_root) / "install.ini"
    if not path.exists():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read_string(decode_ini_bytes(path.read_bytes()))
    except (OSError, configparser.Error):
        return None
    value = parser.get("paths", "data_root", fallback="").strip()
    return value or None


def installed_app_of_python(executable=None):
    """INSTALL\\app when this Python is an installed one (INSTALL\\python\\python.exe with INSTALL\\app\\install.ini or
    build_info.json next to it), else None. The self-test uses it to catch an installed Python that imported some
    other copy of the code (e.g. a repo checkout in the current folder, which `python -m` puts first on sys.path)."""
    exe = Path(executable or sys.executable).resolve()
    app = exe.parent.parent / "app"
    if (app / "install.ini").exists() or (app / "build_info.json").exists():
        return app
    return None


def same_folder(a, b):
    """Do two paths name the same folder? (resolved; case-insensitive on Windows)"""
    a, b = os.path.normcase(str(Path(a).resolve())), os.path.normcase(str(Path(b).resolve()))
    return a == b


def console_python(executable=None):
    """The Python to start child processes with. The tray app runs under pythonw.exe (no console window), but the
    children must be python.exe: they print to stdout, which the launcher writes to their log files."""
    exe = Path(executable or sys.executable)
    if exe.name.lower() == "pythonw.exe":
        sibling = exe.with_name("python.exe")
        if sibling.exists():
            return str(sibling)
    return str(exe)


@dataclass
class AppPaths:
    code_root: Path
    data_root: Path
    installed: bool
    build_info: dict = field(default_factory=dict)
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    python: str = ""

    # ---- folders inside the data root
    @property
    def config_dir(self):
        return self.data_root / "config"

    @property
    def out_dir(self):
        return self.data_root / "out"

    @property
    def logs_dir(self):
        return self.data_root / "logs"

    @property
    def rosters_dir(self):
        return self.data_root / "rosters"

    @property
    def zones_dir(self):
        return self.data_root / "zones"

    @property
    def reports_dir(self):
        return self.data_root / "reports"

    @property
    def exports_dir(self):
        return self.data_root / "exports"

    # ---- files
    @property
    def lapline_config(self):
        return self.config_dir / "lapline.toml"

    @property
    def zone_config(self):
        return self.config_dir / "zone.toml"

    @property
    def launcher_settings(self):
        return self.config_dir / "launcher.json"

    @property
    def manifest(self):
        return self.code_root / "deploy" / "models" / "manifest.json"

    @property
    def bundled_rosters_dir(self):
        return self.code_root / "rosters"

    @property
    def version(self):
        return str(self.build_info.get("version") or "repo (sem instalador)")

    def make_folders(self):
        """Create every data folder (the installer creates the root; this fills it in on first run)."""
        for folder in (self.config_dir, self.out_dir, self.logs_dir, self.rosters_dir, self.zones_dir,
                       self.reports_dir, self.exports_dir):
            folder.mkdir(parents=True, exist_ok=True)


def resolve_paths(data_root=None, env=None, code_root=None, executable=None):
    """Work out every path. Data root precedence: the `data_root` argument (--data-root) > the BAJAMV_DATA
    environment variable > install.ini (installed mode) > the default of the mode."""
    env = os.environ if env is None else env
    code_root = Path(code_root or CODE_ROOT)
    build_info = read_build_info(code_root)
    installed = build_info is not None
    chosen = data_root or env.get(DATA_ROOT_ENV) or (read_install_ini(code_root) if installed else None)
    if chosen:
        root = Path(chosen)
    elif installed:
        root = DEFAULT_INSTALLED_DATA_ROOT
    else:
        root = code_root / "out" / "bajamv"
    ffmpeg, ffprobe = "ffmpeg", "ffprobe"
    if installed:
        # INSTALL\app is the code root, so the bundled ffmpeg is INSTALL\ffmpeg\bin.
        bundled = code_root.parent / "ffmpeg" / "bin"
        exe = ".exe" if sys.platform == "win32" else ""
        if (bundled / ("ffmpeg" + exe)).exists():
            ffmpeg, ffprobe = str(bundled / ("ffmpeg" + exe)), str(bundled / ("ffprobe" + exe))
    return AppPaths(code_root=code_root, data_root=root, installed=installed, build_info=build_info or {},
                    ffmpeg=ffmpeg, ffprobe=ffprobe, python=console_python(executable))
