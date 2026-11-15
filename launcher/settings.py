"""The essential settings of each role: defaults, validation (messages in pt-BR, shown on the page) and the TOML
config files they become.

The launcher owns a few keys of each config ("managed keys", listed in apply_lapline / apply_zone). Every other key
of an existing file is kept: the file is read, the managed keys are changed, and it is written back. Hand edits of
other keys therefore survive; comments do not (the file's header says so).

* Lap line: <data>/config/lapline.toml, an rt config (rt/config.toml + only the keys it changes). A new one starts
  from rt/configs/lapline.example.toml, the race-ready settings (camera exposure, recording, status page...).
* Zone PTZ: <data>/config/zone.toml, a zone config (zone/configs/zone.example.toml + only the keys it changes).
  A new one holds only the managed keys.

Both files also get "plumbing" keys that point every written path at the data root, and the programs at the bundled
ffmpeg: that is what lets the installed app run from a read-only Program Files folder.
"""
import ipaddress
import json
import re
import tomllib
from dataclasses import asdict, dataclass, field, fields

from launcher import tomlwrite

HEADER = ("Escrito pelo Baja MV (launcher). Chaves que o painel não mostra são mantidas quando ele grava;\n"
          "comentários não. Written by the Baja MV launcher: other keys are kept on save, comments are not.")

RESOLUTIONS = {"1920x1080": (1920, 1080), "1280x720": (1280, 720)}

# Car detector of the lap line: (rt runtime.car_backend, runtime.car_device). "" = leave what the file has.
CAR_DETECTORS = {
    "openvino_gpu": ("openvino", "GPU"),        # the T14's Intel iGPU (rt falls back to ONNX Runtime CPU if missing)
    "openvino_cpu": ("openvino", "CPU"),
    "onnxruntime_cpu": ("onnxruntime", "CPU"),
}

# Zone profiles (zone/configs/zone.example.toml [profiles.*]); "normal" = no profile.
PROFILES = ("normal", "strict", "bench", "bench_strict")
BENCH_PROFILES = ("bench", "bench_strict")     # COCO street-car detector: NEVER at the race

LAUNCHER_ROLES = ("lapline", "zone")

EVENT_CODE = re.compile(r"^[0-9]{2}[A-Z]{2,3}$")        # 26SE, 27SU, 03NE ...
HOSTNAME = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$")


# ============================================================================================ the settings themselves
@dataclass
class LaplineSettings:
    event: str = ""                      # event code: picks rosters/<EV>.csv
    camera_device: str = "0"             # camera index or DirectShow device path (or a unique part of it)
    resolution: str = "1920x1080"
    ntp_server: str = ""                 # the event router's IP: rt checks this PC's clock against it
    status_port: int = 8765
    car_detector: str = "openvino_gpu"   # see CAR_DETECTORS; "" = keep the file's value


@dataclass
class ZoneSettings:
    camera_device: str = "0"             # the spotter webcam
    resolution: str = "1920x1080"
    ntp_server: str = ""
    ptz_host: str = ""                   # the PTZ camera's IP (VISCA over IP)
    ptz_report: str = ""                 # characterization report.json ("" = none: Branch B with defaults)
    profile: str = "normal"
    web_port: int = 8090
    agent_port: int = 8890
    agent_token: str = ""                # the studio's X-Agent-Token; set one at the event


@dataclass
class LauncherSettings:
    roles: list = field(default_factory=list)   # which roles this laptop runs: "lapline", "zone" (both on a bench)
    autostart: bool = False                     # start those roles when the launcher starts (e.g. at login)
    pin: str = ""                               # "" = anyone on the LAN can use the page; else it asks for this PIN


# ============================================================================================ validation
def _text(value):
    return "" if value is None else str(value).strip()


def _port(value, name, errors):
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        errors[name] = "Porta inválida: use um número de 1024 a 65535."
        return value
    if not 1024 <= port <= 65535:
        errors[name] = "Porta fora da faixa 1024-65535."
    return port


def _host(value, name, errors, required):
    text = _text(value)
    if not text:
        if required:
            errors[name] = "Obrigatório (ex.: 192.168.0.100)."
        return text
    try:
        ipaddress.ip_address(text)
    except ValueError:
        if not HOSTNAME.match(text):
            errors[name] = "Endereço inválido: use um IP (ex.: 192.168.0.100) ou um nome sem espaços."
    return text


def _camera(value, errors):
    text = _text(value)
    if not text:
        errors["camera_device"] = "Escolha uma câmera (índice ou caminho do dispositivo)."
    elif len(text) > 300 or any(c in text for c in "\r\n\t\""):
        errors["camera_device"] = "Câmera inválida."
    return text


def _choice(value, name, choices, errors, message):
    text = _text(value)
    if text not in choices:
        errors[name] = message
    return text


def validate_lapline(data):
    """dict (from the page) -> (LaplineSettings, {field: error message}). Fields not given keep their defaults."""
    values, errors = asdict(LaplineSettings()), {}
    values.update({k: v for k, v in (data or {}).items() if k in values})
    values["event"] = _text(values["event"]).upper()
    if not EVENT_CODE.match(values["event"]):
        errors["event"] = "Código do evento inválido: dois dígitos + letras, ex.: 27SE."
    values["camera_device"] = _camera(values["camera_device"], errors)
    values["resolution"] = _choice(values["resolution"], "resolution", RESOLUTIONS, errors,
                                   "Resolução inválida (use 1920x1080 ou 1280x720).")
    values["ntp_server"] = _host(values["ntp_server"], "ntp_server", errors, required=False)
    values["status_port"] = _port(values["status_port"], "status_port", errors)
    values["car_detector"] = _choice(values["car_detector"], "car_detector", list(CAR_DETECTORS) + [""], errors,
                                     "Detector inválido.")
    return LaplineSettings(**values), errors


def validate_zone(data, reports_exist=None):
    """dict -> (ZoneSettings, errors). reports_exist(path) -> bool checks the report file (injectable for tests)."""
    from pathlib import Path
    reports_exist = reports_exist or (lambda p: Path(p).is_file())
    values, errors = asdict(ZoneSettings()), {}
    values.update({k: v for k, v in (data or {}).items() if k in values})
    values["camera_device"] = _camera(values["camera_device"], errors)
    values["resolution"] = _choice(values["resolution"], "resolution", RESOLUTIONS, errors,
                                   "Resolução inválida (use 1920x1080 ou 1280x720).")
    values["ntp_server"] = _host(values["ntp_server"], "ntp_server", errors, required=False)
    values["ptz_host"] = _host(values["ptz_host"], "ptz_host", errors, required=True)
    values["ptz_report"] = _text(values["ptz_report"])
    if values["ptz_report"] and not reports_exist(values["ptz_report"]):
        errors["ptz_report"] = "Arquivo de caracterização não encontrado."
    values["profile"] = _choice(values["profile"] or "normal", "profile", PROFILES, errors, "Perfil inválido.")
    values["web_port"] = _port(values["web_port"], "web_port", errors)
    values["agent_port"] = _port(values["agent_port"], "agent_port", errors)
    if "web_port" not in errors and "agent_port" not in errors and values["web_port"] == values["agent_port"]:
        errors["agent_port"] = "A porta do estúdio precisa ser diferente da porta da página."
    values["agent_token"] = _text(values["agent_token"])
    if len(values["agent_token"]) > 128 or any(c in values["agent_token"] for c in "\r\n\""):
        errors["agent_token"] = "Token inválido (até 128 caracteres, sem aspas)."
    return ZoneSettings(**values), errors


def validate_launcher(data):
    values, errors = asdict(LauncherSettings()), {}
    values.update({k: v for k, v in (data or {}).items() if k in values})
    roles = values["roles"] if isinstance(values["roles"], list) else []
    values["roles"] = [r for r in LAUNCHER_ROLES if r in roles]       # known roles only, in a fixed order
    values["autostart"] = bool(values["autostart"])
    values["pin"] = _text(values["pin"])
    if len(values["pin"]) > 32:
        errors["pin"] = "PIN longo demais (até 32 caracteres)."
    return LauncherSettings(**values), errors


def port_conflicts(lapline, zone, launcher_port, saving="zone"):
    """Ports two programs on this laptop would both want (both roles on one bench laptop, or the launcher's own).
    -> {field: message} keyed like the settings fields ("lapline.status_port", "zone.web_port", ...). The role
    being saved (`saving`) claims its ports last, so the message lands on the field the operator just changed."""
    owners = {}
    lapline_claims = [("lapline.status_port", lapline.status_port)] if lapline is not None else []
    zone_claims = [("zone.web_port", zone.web_port), ("zone.agent_port", zone.agent_port)] if zone is not None else []
    claims = [("launcher", int(launcher_port))]
    claims += (zone_claims + lapline_claims) if saving == "lapline" else (lapline_claims + zone_claims)
    errors = {}
    for name, port in claims:
        if port in owners:
            errors[name] = "A porta %s já é usada por %s." % (port, owners[port])
        else:
            owners[port] = name
    return errors


# ============================================================================================ TOML files
def read_toml(path):
    """A TOML file as a dict ({} when it does not exist)."""
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def get_key(cfg, dotted, default=None):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_key(cfg, dotted, value):
    *parents, key = dotted.split(".")
    node = cfg
    for part in parents:
        node = node.setdefault(part, {})
    node[key] = value


def _device_value(text):
    """A camera index is written as a number (rt treats both the same; a number reads better)."""
    return int(text) if re.fullmatch(r"\d+", text) else text


def _resolution_of(width, height, default="1920x1080"):
    for name, size in RESOLUTIONS.items():
        if (width, height) == size:
            return name
    return default


def read_lapline(paths):
    """The lap line's settings as they are in its config file (defaults for what the file lacks)."""
    cfg = read_toml(paths.lapline_config)
    if not cfg:
        return LaplineSettings()
    detector = ""
    for name, (backend, device) in CAR_DETECTORS.items():
        if (get_key(cfg, "runtime.car_backend"), str(get_key(cfg, "runtime.car_device", "")).upper()) == (backend, device):
            detector = name
    return LaplineSettings(
        event=str(get_key(cfg, "launch.event", "")), camera_device=str(get_key(cfg, "camera.device", 0)),
        resolution=_resolution_of(get_key(cfg, "camera.width"), get_key(cfg, "camera.height")),
        ntp_server=str(get_key(cfg, "clock.ntp_server", "")), status_port=int(get_key(cfg, "status.port", 8765)),
        car_detector=detector)


def apply_lapline(cfg, settings, paths):
    """Write the managed keys (and the data-root plumbing) into an rt config dict."""
    width, height = RESOLUTIONS[settings.resolution]
    set_key(cfg, "launch.event", settings.event)
    set_key(cfg, "launch.source", "camera")
    set_key(cfg, "camera.device", _device_value(settings.camera_device))
    set_key(cfg, "camera.width", width)
    set_key(cfg, "camera.height", height)
    set_key(cfg, "clock.ntp_server", settings.ntp_server)
    set_key(cfg, "status.port", int(settings.status_port))
    if settings.car_detector:
        backend, device = CAR_DETECTORS[settings.car_detector]
        set_key(cfg, "runtime.car_backend", backend)
        set_key(cfg, "runtime.car_device", device)
    # plumbing: everything written goes under the data root; the bundled programs and models are used
    set_key(cfg, "launch.out_dir", str(paths.out_dir / "lapline"))
    set_key(cfg, "identity.rosters_dir", str(paths.rosters_dir))
    set_key(cfg, "capture.ffmpeg", paths.ffmpeg)
    set_key(cfg, "capture.ffprobe", paths.ffprobe)
    set_key(cfg, "models.manifest", str(paths.manifest))
    return cfg


def write_lapline(paths, settings):
    """Create or update <data>/config/lapline.toml. A new file starts from rt/configs/lapline.example.toml."""
    cfg = read_toml(paths.lapline_config) or read_toml(paths.code_root / "rt" / "configs" / "lapline.example.toml")
    apply_lapline(cfg, settings, paths)
    write_toml_file(paths.lapline_config, cfg)
    return paths.lapline_config


def read_zone(paths):
    cfg = read_toml(paths.zone_config)
    if not cfg:
        return ZoneSettings()
    source = str(get_key(cfg, "spotter.source", "camera"))
    # "camera" -> rt's camera.device; a bare index -> that index; a file / URL set by hand is shown as it is
    device = str(get_key(cfg, "rt.camera.device", 0)) if source == "camera" else source
    return ZoneSettings(
        camera_device=device,
        resolution=_resolution_of(get_key(cfg, "rt.camera.width"), get_key(cfg, "rt.camera.height")),
        ntp_server=str(get_key(cfg, "rt.clock.ntp_server", "")), ptz_host=str(get_key(cfg, "ptz.host", "")),
        ptz_report=str(get_key(cfg, "ptz.report", "")), profile=str(cfg.get("profile") or "normal"),
        web_port=int(get_key(cfg, "web.port", 8090)), agent_port=int(get_key(cfg, "agent.port", 8890)),
        agent_token=str(get_key(cfg, "agent.token", "")))


def is_camera_source(text):
    """A spotter.source the launcher manages: a camera index or "camera". Files and stream URLs set by hand are
    left alone."""
    return text == "camera" or bool(re.fullmatch(r"\d+", text))


def apply_zone(cfg, settings, paths):
    width, height = RESOLUTIONS[settings.resolution]
    cfg["profile"] = "" if settings.profile == "normal" else settings.profile
    current_source = str(get_key(cfg, "spotter.source", "camera"))
    hand_set_source = not is_camera_source(current_source) and settings.camera_device == current_source
    if not hand_set_source:
        # the camera goes through rt's [camera] section, which also takes a DirectShow device path
        set_key(cfg, "spotter.source", "camera")
        set_key(cfg, "rt.camera.device", _device_value(settings.camera_device))
    set_key(cfg, "rt.camera.width", width)
    set_key(cfg, "rt.camera.height", height)
    set_key(cfg, "rt.clock.ntp_server", settings.ntp_server)
    set_key(cfg, "ptz.driver", "visca")
    set_key(cfg, "ptz.host", settings.ptz_host)
    set_key(cfg, "ptz.report", settings.ptz_report)
    set_key(cfg, "web.port", int(settings.web_port))
    set_key(cfg, "agent.port", int(settings.agent_port))
    set_key(cfg, "agent.token", settings.agent_token)
    # plumbing
    set_key(cfg, "zone.file", str(paths.zones_dir / "zone.json"))
    set_key(cfg, "zone.out_dir", str(paths.out_dir / "zone"))
    set_key(cfg, "recording.ffmpeg", paths.ffmpeg)
    set_key(cfg, "rt.capture.ffmpeg", paths.ffmpeg)
    set_key(cfg, "rt.capture.ffprobe", paths.ffprobe)
    set_key(cfg, "rt.models.manifest", str(paths.manifest))
    return cfg


def write_zone(paths, settings):
    """Create or update <data>/config/zone.toml (only the keys it changes; zone.example.toml has the rest)."""
    cfg = read_toml(paths.zone_config)
    apply_zone(cfg, settings, paths)
    write_toml_file(paths.zone_config, cfg)
    return paths.zone_config


def write_toml_file(path, cfg):
    """Write atomically: a crash mid-write never leaves half a config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(tomlwrite.dumps(cfg, HEADER), encoding="utf-8")
    tmp.replace(path)


# ============================================================================================ launcher.json
def read_launcher(paths):
    path = paths.launcher_settings
    if not path.exists():
        return LauncherSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return LauncherSettings()
    settings, _ = validate_launcher(data)
    return settings


def write_launcher(paths, settings):
    paths.launcher_settings.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.launcher_settings.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(settings), indent=1), encoding="utf-8")
    tmp.replace(paths.launcher_settings)


def field_names(settings_class):
    return [f.name for f in fields(settings_class)]
