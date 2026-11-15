"""The launcher's web page and its HTTP API (FastAPI). The page itself is plain HTML + JS in launcher/static/
(pt-BR, no build step) and works from a phone on the LAN: http://<laptop IP>:8780/

Reading is open (GET). Every change (POST) needs the PIN when one is set in the page (header X-Launcher-Pin), like
the rt status page's setup token. /api/shutdown (used by the installer before an upgrade) only answers this PC.
"""
import hmac
import json
import os
import re
import socket
import subprocess
import sys
import threading
from dataclasses import asdict
from pathlib import Path

from launcher import settings as S
from launcher.cameras import list_cameras, snapshot_jpeg
from launcher.exportlogs import export_logs
from launcher.netprofile import CachedNetworkWarning
from launcher.roles import ROLE_NAMES, RoleManager, child_env
from launcher.rosterfetch import fetch_roster, roster_status
from launcher.tools import SAFE_NAME, CharacterizationRunner, list_reports

STATIC = Path(__file__).resolve().parent / "static"
EXPORT_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,200}\.zip$")


def models_note(paths):
    try:
        return json.loads(paths.manifest.read_text(encoding="utf-8")).get("note", "")
    except (OSError, ValueError):
        return "manifest.json não encontrado"


def lan_addresses():
    """This PC's IPv4 addresses (for the page's links), loopback excluded."""
    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    return sorted(a for a in found if not a.startswith("127."))


class Launcher:
    """Everything the page talks to. Built once by __main__ (and by the self-test and the tests)."""

    def __init__(self, paths, port, roles=None, characterization=None, snapshot=snapshot_jpeg, cameras=list_cameras,
                 roster_fetch=fetch_roster, open_folder=None, network_warning=None, log=print):
        self.paths, self.port, self.log = paths, int(port), log
        self.roles = roles or RoleManager(paths, log=log)
        self.characterization = characterization or CharacterizationRunner(paths, env=child_env(paths))
        self.snapshot, self.cameras, self.roster_fetch = snapshot, cameras, roster_fetch
        self.open_folder = open_folder or open_in_file_manager
        # "this network is Public" warning for the page (Windows only; asked at most once a minute)
        self.network_warning = network_warning or CachedNetworkWarning().get
        self.exit_requested = threading.Event()
        self.lock = threading.Lock()          # one settings write / role action at a time

    @property
    def launcher_settings(self):
        return S.read_launcher(self.paths)

    def state(self):
        lapline, zone = S.read_lapline(self.paths), S.read_zone(self.paths)
        launcher = self.launcher_settings
        _, lapline_errors = S.validate_lapline(asdict(lapline))
        # The studio token is a secret, and reading the state needs no PIN: the page only learns whether one is set
        # (like the launcher's own PIN). Saving the form with the field empty keeps it (save_role_settings).
        zone_settings = asdict(zone)
        token_set = bool(zone_settings.pop("agent_token", ""))
        return dict(
            version=self.paths.version, build=self.paths.build_info, models_note=models_note(self.paths),
            installed=self.paths.installed, data_root=str(self.paths.data_root), hostname=socket.gethostname(),
            addresses=lan_addresses(), launcher_port=self.port, network_warning=self.network_warning(),
            launcher=dict(roles=launcher.roles, autostart=launcher.autostart, pin_set=bool(launcher.pin)),
            role_names=ROLE_NAMES,
            lapline=dict(settings=asdict(lapline), config_exists=self.paths.lapline_config.exists(),
                         config_path=str(self.paths.lapline_config),
                         errors=lapline_errors if self.paths.lapline_config.exists() else {},
                         roster=roster_status(lapline.event, self.paths)),
            zone=dict(settings=zone_settings, token_set=token_set, config_exists=self.paths.zone_config.exists(),
                      config_path=str(self.paths.zone_config), bench=zone.profile in S.BENCH_PROFILES),
            processes=self.roles.status(log_lines=25),
            characterization=self.characterization.status(),
            reports=list_reports(self.paths.reports_dir),
            choices=dict(resolutions=list(S.RESOLUTIONS), profiles=list(S.PROFILES),
                         bench_profiles=list(S.BENCH_PROFILES), car_detectors=list(S.CAR_DETECTORS)),
        )

    # ---- settings
    def save_role_settings(self, role, data):
        """Validate and write one role's config. -> (ok, errors or {"path": ...})."""
        with self.lock:
            if role == "lapline":
                settings, errors = S.validate_lapline(data)
                other_zone = S.read_zone(self.paths) if "zone" in self.launcher_settings.roles else None
                errors.update({k.split(".", 1)[1]: v for k, v in S.port_conflicts(settings, other_zone, self.port, saving="lapline")
                               .items() if k.startswith("lapline.")})
                if errors:
                    return False, errors
                return True, dict(path=str(S.write_lapline(self.paths, settings)))
            if role == "zone":
                data = dict(data)
                # empty or missing token field = keep the saved token (the page never receives it); to remove the
                # token, type a single space (validate_zone strips it to "")
                if data.get("agent_token") in (None, ""):
                    data["agent_token"] = S.read_zone(self.paths).agent_token
                settings, errors = S.validate_zone(data)
                other = S.read_lapline(self.paths) if "lapline" in self.launcher_settings.roles else None
                errors.update({k.split(".", 1)[1]: v for k, v in S.port_conflicts(other, settings, self.port, saving="zone")
                               .items() if k.startswith("zone.")})
                if errors:
                    return False, errors
                return True, dict(path=str(S.write_zone(self.paths, settings)))
            return False, {"role": "papel desconhecido"}

    def save_launcher_settings(self, data):
        with self.lock:
            data = dict(data)
            if "pin" not in data:                 # the page leaves the PIN out to keep it
                data["pin"] = self.launcher_settings.pin
            settings, errors = S.validate_launcher(data)
            if errors:
                return False, errors
            S.write_launcher(self.paths, settings)
            return True, {}

    # ---- roles
    def role_action(self, role, action):
        if role not in ("lapline", "zone") or action not in ("start", "stop", "restart"):
            return False, "ação desconhecida"
        with self.lock:
            return getattr(self.roles, action)(role)

    def start_autostart_roles(self):
        launcher = self.launcher_settings
        if not launcher.autostart:
            return
        for role in launcher.roles:
            if not self.roles.processes[role].running():
                ok, message = self.roles.start(role)
                self.log("início automático %s: %s" % (role, message))

    def camera_in_use(self, device):
        """The running role that holds this camera, or None. A config may name the camera by its index ("0") or by
        its device path, and the page may ask with the other form: when the plain comparison does not match, both
        sides are translated through the camera list (index -> device path) and compared again."""
        running = self.roles.running_roles()
        if not running:
            return None
        uses = {"lapline": S.read_lapline(self.paths).camera_device, "zone": S.read_zone(self.paths).camera_device}
        for role in running:
            if same_text(uses.get(role), device):
                return role
        try:
            cameras, _ = self.cameras(self.paths.ffmpeg)
        except Exception:  # noqa: BLE001 (no list: the plain comparison above is all we can do)
            return None
        wanted = canonical_camera(device, cameras)
        for role in running:
            if same_text(canonical_camera(uses.get(role), cameras), wanted):
                return role
        return None

    def request_exit(self, stop_roles):
        if stop_roles:
            self.roles.stop_all()
        self.exit_requested.set()


def same_text(a, b):
    """Camera names compared without case or surrounding spaces (Windows device paths differ in case)."""
    return str(a or "").strip().lower() == str(b or "").strip().lower()


def canonical_camera(device, cameras):
    """The device path of a camera given by index ("0") or by path, using the list from list_cameras()."""
    text = str(device or "").strip()
    if text.isdigit():
        for camera in cameras or []:
            if str(camera.get("index")) == text:
                return str(camera.get("device") or text)
    return text


def open_in_file_manager(folder):
    folder = str(folder)
    if sys.platform == "win32":
        os.startfile(folder)                                     # noqa: S606 (Explorer)
    else:
        subprocess.Popen(["xdg-open", folder], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def create_app(launcher):
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, Response

    app = FastAPI(title="Baja MV", docs_url=None, redoc_url=None, openapi_url=None)

    def require_pin(request: Request):
        pin = launcher.launcher_settings.pin
        if pin and not hmac.compare_digest(request.headers.get("X-Launcher-Pin", ""), pin):
            raise HTTPException(401, "PIN necessário")

    def answer(ok, payload, status_on_error=400):
        if ok:
            return dict(ok=True, **(payload if isinstance(payload, dict) else {"message": payload}))
        body = dict(ok=False, errors=payload) if isinstance(payload, dict) else dict(ok=False, message=payload)
        return JSONResponse(body, status_code=status_on_error)

    # ---- page
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/static/{name}")
    def static(name: str):
        path = STATIC / name
        if not SAFE_NAME.match(name) or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.get("/api/ping")
    def ping():
        return dict(ok=True, app="bajamv-launcher", version=launcher.paths.version)

    @app.get("/api/state")
    def state():
        return launcher.state()

    # ---- settings
    @app.post("/api/settings/{role}")
    def save_settings(role: str, request: Request, body: dict = Body(...)):
        require_pin(request)
        if role == "launcher":
            return answer(*launcher.save_launcher_settings(body))
        return answer(*launcher.save_role_settings(role, body))

    # ---- roles
    @app.post("/api/roles/{role}/{action}")
    def role_action(role: str, action: str, request: Request):
        require_pin(request)
        ok, message = launcher.role_action(role, action)
        return answer(ok, message, status_on_error=409)

    @app.get("/api/roles/{role}/log")
    def role_log(role: str, lines: int = 200):
        if role not in launcher.roles.processes:
            raise HTTPException(404)
        return launcher.roles.processes[role].status(log_lines=max(1, min(int(lines), 2000)))

    # ---- roster
    @app.post("/api/roster/fetch")
    def roster_fetch(request: Request, body: dict = Body(...)):
        require_pin(request)
        result = launcher.roster_fetch(str(body.get("event", "")), launcher.paths)
        result["roster"] = roster_status(str(body.get("event", "")), launcher.paths)
        return result

    # ---- cameras
    @app.get("/api/cameras")
    def cameras():
        found, error = launcher.cameras(launcher.paths.ffmpeg)
        return dict(cameras=found, error=error)

    @app.get("/api/cameras/snapshot")
    def snapshot(device: str, resolution: str = "1920x1080"):
        holder = launcher.camera_in_use(device)
        if holder:
            return JSONResponse(dict(ok=False, message="Câmera em uso por %s: pare o papel para ver a prévia, ou use "
                                     "a página do papel." % ROLE_NAMES[holder]), status_code=409)
        width, height = S.RESOLUTIONS.get(resolution, (1920, 1080))
        try:
            jpeg, _ = launcher.snapshot(device, launcher.paths.ffmpeg, width, height)
        except Exception as e:  # noqa: BLE001
            return JSONResponse(dict(ok=False, message=str(e)), status_code=409)
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # ---- characterization
    @app.post("/api/characterize")
    def characterize(request: Request, body: dict = Body(...)):
        require_pin(request)
        zone_running = "zone" in launcher.roles.running_roles()
        ok, message = launcher.characterization.start(str(body.get("ip", "")), str(body.get("folder", "")),
                                                      quick=bool(body.get("quick")), zone_running=zone_running)
        return answer(ok, message, status_on_error=409)

    @app.get("/api/characterize")
    def characterize_status():
        return launcher.characterization.status(lines=200)

    # ---- data folder, logs
    @app.post("/api/open-data-folder")
    def open_data_folder(request: Request):
        require_pin(request)
        try:
            launcher.open_folder(launcher.paths.data_root)
        except Exception as e:  # noqa: BLE001
            return answer(False, "Não foi possível abrir a pasta: %s" % e)
        return answer(True, "Pasta aberta neste computador: %s" % launcher.paths.data_root)

    @app.post("/api/export")
    def export(request: Request):
        require_pin(request)
        path = export_logs(launcher.paths)
        return answer(True, dict(name=path.name, path=str(path)))

    @app.get("/api/export/{name}")
    def download_export(name: str):
        path = launcher.paths.exports_dir / name
        if not EXPORT_NAME.match(name) or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="application/zip", filename=name)

    # ---- installer: stop everything before an upgrade / uninstall
    @app.post("/api/shutdown")
    def shutdown(request: Request):
        if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
            raise HTTPException(403, "only from this PC")
        threading.Thread(target=launcher.request_exit, args=(True,), daemon=True).start()
        return dict(ok=True)

    return app
