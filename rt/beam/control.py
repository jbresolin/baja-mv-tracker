"""Beam setup from the status page (on-site, from a phone): settings validation, the override file, the list of
serial ports, and BeamController, which owns the running beam subsystem and can restart it with new settings.

Why: the sensor and its USB adapter only reach the owner at the event, so everything about the beam must be found
out and set on site, from a phone, without editing files or restarting rt. The status page's "Beam setup" panel
talks to the controller through a few POST endpoints (rt/status.py); everything else on the page stays read-only.

The override file: "Apply" writes the beam settings to `<config>.beam.toml` next to the config file
(rt/configs/lapline.toml -> rt/configs/lapline.beam.toml). rt/config.py loads it at start ON TOP of the config's
[beam] section (and --set options on top of that), so the settings survive a restart by the launcher. Delete the
file to go back to the config's [beam]. Only the keys in SETTABLE_KEYS are ever written.

Safety rules:
  * every value is validated here, server-side (validate_settings); a bad request changes nothing;
  * "Apply" restarts ONLY the beam (BeamService): never detection, capture or recording;
  * if the new settings still fail to start (a mistake validation could not catch), the beam stays off and the
    error is shown on the page; rt goes on;
  * setup mode stops the beam driver (the monitor needs the COM port). If the page is left open-ended (phone locked,
    tab closed) setup mode ends by itself after `beam.setup_idle_s` (300 s) without a page refresh, and the beam
    restarts with the saved settings: an unattended rt is never left without its beam because of a forgotten setup.
"""
import copy
import datetime as dt
import os
import re
import threading
import tomllib
from pathlib import Path

from ..backoff import SystemClock
from ..sinks import Sink
from .sources import parse_hex

CP210X_VID_PID = (0x10C4, 0xEA60)          # Silicon Labs CP210x USB-UART (the adapter the event uses)

# The [beam] keys the page can set (and the override file can hold). Nothing else is ever written or loaded from it.
SETTABLE_KEYS = ("driver", "port", "line", "active", "baud", "byte_pattern", "byte_release_pattern", "group_gap_s",
                 "stuck_s", "set_dtr", "set_rts")
DRIVERS = ("off", "modem-line", "byte-stream")
LINES = ("cts", "dsr", "cd", "ri")
BAUD_RATES = (300, 600, 1200, 2400, 4800, 9600, 14400, 19200, 38400, 57600, 115200, 230400, 460800, 921600)
PORT_RE = re.compile(r"^(?:\\\\\.\\)?COM[0-9]{1,3}$|^/dev/[A-Za-z0-9_.\-/]{1,60}$", re.IGNORECASE)
MAX_PATTERN_BYTES = 32


# ------------------------------------------------------------------------------------------------ validation
def _as_bool(value, name, errors):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false", "1", "0", "on", "off", "yes", "no"):
        return value.lower() in ("true", "1", "on", "yes")
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    errors.append("%s must be true or false" % name)
    return None


def _as_number(value, name, low, high, errors):
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append("%s must be a number" % name)
        return None
    if not (low <= number <= high):         # also rejects NaN
        errors.append("%s must be between %g and %g" % (name, low, high))
        return None
    return number


def _as_pattern(value, name, errors):
    text = "" if value is None else str(value)
    if len(text) > 4 * MAX_PATTERN_BYTES:
        errors.append("%s is too long" % name)
        return None
    try:
        raw = parse_hex(text)
    except ValueError:
        errors.append("%s must be hex bytes, e.g. \"0D 0A\"" % name)
        return None
    if len(raw) > MAX_PATTERN_BYTES:
        errors.append("%s: at most %d bytes" % (name, MAX_PATTERN_BYTES))
        return None
    return raw.hex(" ").upper()


def validate_settings(data, current=None):
    """Check beam settings coming from the page (or the override file). `data` = dict of SETTABLE_KEYS (missing keys
    keep the `current` value). Returns (clean settings dict with every SETTABLE_KEY, list of error messages). Unknown
    keys are an error (nothing unexpected is ever stored). Settings are only usable when the error list is empty."""
    errors = []
    if not isinstance(data, dict):
        return {}, ["settings must be a JSON object"]
    unknown = sorted(set(data) - set(SETTABLE_KEYS))
    if unknown:
        errors.append("unknown setting(s): %s" % ", ".join(str(k)[:40] for k in unknown))
    merged = {k: (current or {}).get(k) for k in SETTABLE_KEYS}
    merged.update({k: v for k, v in data.items() if k in SETTABLE_KEYS})
    out = {}
    driver = str(merged.get("driver") or "off").lower()
    if driver not in DRIVERS:
        errors.append("driver must be one of %s" % ", ".join(DRIVERS))
    out["driver"] = driver
    port = str(merged.get("port") or "").strip()
    if port and not PORT_RE.match(port):
        errors.append("port must look like COM5 (Windows) or /dev/ttyUSB0")
    elif not port and driver != "off":
        errors.append("choose a port")
    out["port"] = port
    line = str(merged.get("line") or "cts").lower()
    line = "cd" if line == "dcd" else line
    if line not in LINES:
        errors.append("line must be CTS, DSR, DCD or RI")
    out["line"] = line
    active = str(merged.get("active") or "high").lower()
    if active not in ("high", "low"):
        errors.append("active must be high or low")
    out["active"] = active
    baud = merged.get("baud") if merged.get("baud") is not None else 9600
    try:
        baud = int(str(baud).strip())
    except ValueError:
        baud = None
    if baud not in BAUD_RATES:
        errors.append("baud must be one of %s" % ", ".join(str(b) for b in BAUD_RATES))
    out["baud"] = baud
    out["byte_pattern"] = _as_pattern(merged.get("byte_pattern"), "byte pattern", errors)
    out["byte_release_pattern"] = _as_pattern(merged.get("byte_release_pattern"), "release pattern", errors)
    if out["byte_release_pattern"] and not out["byte_pattern"]:
        errors.append("a release pattern needs a byte pattern too")
    if out["byte_pattern"] and out["byte_pattern"] == out["byte_release_pattern"]:
        errors.append("the byte pattern and the release pattern must differ")
    gap = merged.get("group_gap_s") if merged.get("group_gap_s") is not None else 0.25
    out["group_gap_s"] = _as_number(gap, "group gap (s)", 0.02, 2.0, errors)
    stuck = merged.get("stuck_s") if merged.get("stuck_s") is not None else 5.0
    out["stuck_s"] = _as_number(stuck, "stuck after (s)", 0.5, 600.0, errors)
    if out["group_gap_s"] is not None and out["stuck_s"] is not None and out["stuck_s"] <= out["group_gap_s"]:
        errors.append("stuck after (s) must be longer than the group gap")
    for key in ("set_dtr", "set_rts"):
        value = merged.get(key)
        out[key] = True if value is None else _as_bool(value, key, errors)
    return out, errors


# ------------------------------------------------------------------------------------------------ override file
def override_path(config_path):
    """rt/configs/lapline.toml -> rt/configs/lapline.beam.toml (next to the config it overrides)."""
    config_path = Path(config_path)
    return config_path.with_name(config_path.stem + ".beam.toml")


def load_override(path):
    """Read an override file. Returns (settings dict or None when there is no file, error text or None). A broken or
    invalid file never raises: the caller turns the beam off and shows the error."""
    path = Path(path)
    if not path.exists():
        return None, None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        return None, "beam override %s cannot be read (%s): beam off until it is fixed or deleted" % (path, e)
    beam = data.get("beam")
    if not isinstance(beam, dict):
        return None, "beam override %s has no [beam] section: beam off until it is fixed or deleted" % path
    settings, errors = validate_settings(beam)
    if errors:
        return None, "beam override %s is invalid (%s): beam off until it is fixed or deleted" % (
            path, "; ".join(errors))
    return settings, None


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return '"%s"' % str(value).replace("\\", "\\\\").replace('"', '\\"')


def write_override(path, settings, config_path, now=None):
    """Write the override file atomically (a temporary file renamed over the old one: a crash mid-write cannot leave
    half a file). Only validated SETTABLE_KEYS are written."""
    path = Path(path)
    when = (now or dt.datetime.now().astimezone()).isoformat(timespec="seconds")
    lines = ["# Written by rt's status page (Beam setup) on %s." % when,
             "# Loaded at start on top of the [beam] section of %s." % Path(config_path).name,
             "# Delete this file to go back to that [beam] section.",
             "[beam]"]
    lines += ["%s = %s" % (key, _toml_value(settings[key])) for key in SETTABLE_KEYS]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ serial ports
def list_serial_ports(comports=None):
    """The serial ports this PC has, for the page: [dict(device, description, usb_id, cp210x)]. The CP2102 (USB id
    10C4:EA60) is flagged. `comports` is for the tests (pyserial's serial.tools.list_ports.comports by default)."""
    if comports is None:
        try:
            from serial.tools.list_ports import comports
        except ImportError:
            return []
    out = []
    try:
        found = list(comports())
    except Exception:  # noqa: BLE001 (listing ports must never break the page)
        return []
    for p in found:
        vid, pid = getattr(p, "vid", None), getattr(p, "pid", None)
        out.append(dict(device=str(p.device), description=str(getattr(p, "description", "") or ""),
                        usb_id="%04X:%04X" % (vid, pid) if vid is not None and pid is not None else "",
                        cp210x=(vid, pid) == CP210X_VID_PID))
    out.sort(key=lambda p: (not p["cp210x"], p["device"]))
    return out


# ------------------------------------------------------------------------------------------------ controller
class BeamController:
    """Owns the beam subsystem of a running rt: the BeamService (or none when the beam is off), the setup monitor,
    and the override file. run.py creates one; the status page talks to it; the detection loop and rt's event
    emitter only call on_frame() / on_event(), which just forward to the current service.

    `make_service(cfg)` builds a BeamService (tests pass fakes); `make_monitor(port, baud, set_dtr, set_rts)` builds
    a BeamMonitor; `comports` lists serial ports."""

    def __init__(self, cfg, emit, make_service=None, make_monitor=None, comports=None, clock=None,
                 config_path=None, use_watchdog=True):
        self.base_cfg = copy.deepcopy(cfg)
        self.emit = emit
        self.clock = clock or SystemClock()
        self.make_service = make_service or self._default_service
        self.make_monitor = make_monitor or self._default_monitor
        self.comports = comports
        override = cfg.get("beam_override") or {}
        self.override_file = Path(override["path"]) if override.get("path") else (
            override_path(config_path) if config_path else None)
        self.config_path = config_path or (override.get("config") if override else None) or "config"
        self.lock = threading.RLock()        # one page action at a time (apply / setup / off)
        self.service = None
        self.monitor = None
        self.error = override.get("error")   # shown on the page (bad override file, failed start, ...)
        self.setup_idle_s = float(cfg["beam"].get("setup_idle_s", 300))
        self.setup_touched = None            # monotonic time of the last setup-page request while in setup mode
        self.use_watchdog = use_watchdog     # tests call check_setup_idle() themselves, with a fake clock
        self.watchdog = None
        self.halt = threading.Event()
        self.restarts = 0
        # The settings the page starts from. If the config's own [beam] has something the page would not accept
        # (e.g. an unusual baud), that value shows as empty; the running beam still uses the config as it is.
        self.settings, _ = validate_settings({k: cfg["beam"].get(k) for k in SETTABLE_KEYS})

    # ------------------------------------------------------------------ the running beam
    @staticmethod
    def _default_service(cfg, emit):
        from .service import BeamService
        return BeamService(cfg, emit)

    @staticmethod
    def _default_monitor(port, baud, set_dtr, set_rts):
        from .monitor import BeamMonitor
        return BeamMonitor(port, baud, set_dtr, set_rts)

    def _beam_cfg(self, settings=None):
        cfg = copy.deepcopy(self.base_cfg)
        if settings is not None:
            cfg["beam"].update(settings)
        return cfg

    def start_from_config(self):
        """Start the beam as the config (plus the override file) says. Never raises."""
        with self.lock:
            if self.error and (self.base_cfg.get("beam_override") or {}).get("error"):
                self.emit(dict(type="warning", msg=self.error))
            self._start_service(self._beam_cfg())

    def _start_service(self, cfg):
        """Start a BeamService for `cfg` (none when driver = off). A failure leaves the beam off with the error
        shown; it never raises."""
        if str(cfg["beam"].get("driver", "off")).lower() == "off":
            self.service = None
            return
        try:
            service = self.make_service(cfg, self.emit)
            service.start()
        except Exception as e:  # noqa: BLE001 (a bad setting must never crash rt)
            self.service = None
            self.error = "beam input NOT started: %s: %s" % (type(e).__name__, e)
            self.emit(dict(type="warning", msg=self.error))
            return
        self.service = service
        self.emit(dict(type="info", msg="beam input: driver %s, port %s" % (service.source.driver,
                                                                           service.source.port)))

    def _stop_service(self):
        service, self.service = self.service, None
        if service is not None:
            try:
                service.stop()
            except Exception as e:  # noqa: BLE001
                self.emit(dict(type="warning", msg="beam stop: %s" % e))

    # ------------------------------------------------------------------ what rt calls
    def on_frame(self, fr):
        service = self.service           # one read: Apply may swap it at any moment
        if service is not None:
            service.on_frame(fr)

    def on_event(self, ev):
        service = self.service
        if service is not None:
            service.on_event(ev)

    def status(self):
        """The Beam section of the status page: the running service's numbers, or "off"."""
        service = self.service
        if service is not None:
            out = service.status()
        else:
            out = dict(enabled=False, driver="off")
        out.update(setup_mode=self.monitor is not None, error=self.error)
        return out

    def stop(self):
        """rt is ending: stop the monitor and the beam."""
        self.halt.set()
        with self.lock:
            self._stop_monitor()
            self._stop_service()

    # ------------------------------------------------------------------ setup page actions
    def setup_state(self):
        """Everything the setup panel shows (GET /beam/setup.json). Counts as "the page is still open"."""
        with self.lock:
            if self.monitor is not None:
                self.setup_touched = self.clock.mono()
            monitor = self.monitor.snapshot() if self.monitor is not None else None
            return dict(settings=dict(self.settings), setup_mode=monitor is not None, monitor=monitor,
                        ports=list_serial_ports(self.comports), error=self.error, beam=self.status(),
                        override_file=str(self.override_file) if self.override_file else None,
                        setup_idle_s=self.setup_idle_s, restarts=self.restarts)

    def start_setup(self, data):
        """Enter setup mode: stop the beam driver (it holds the COM port) and open the port in monitor mode."""
        port = str((data or {}).get("port") or self.settings.get("port") or "")
        check = dict(self.settings, driver="off", port=port)
        for key in ("baud", "set_dtr", "set_rts"):
            if key in (data or {}):
                check[key] = data[key]
        settings, errors = validate_settings(check)
        if not port:
            errors.append("choose a port")
        if errors:
            return False, errors
        with self.lock:
            self._stop_monitor()
            self._stop_service()
            self.monitor = self.make_monitor(port, settings["baud"], settings["set_dtr"], settings["set_rts"])
            self.monitor.start()
            self.setup_touched = self.clock.mono()
            self._start_watchdog()
        self.emit(dict(type="info", msg="beam setup: monitoring %s at %d baud (beam driver paused)" % (
            port, settings["baud"])))
        return True, []

    def leave_setup(self, why="setup closed"):
        """Leave setup mode without changes: close the monitor, restart the beam with the saved settings."""
        with self.lock:
            if self.monitor is None:
                return False
            self._stop_monitor()
            self._start_service(self._beam_cfg(self.settings))
        self.emit(dict(type="info", msg="beam setup ended (%s): beam restarted with the saved settings" % why))
        return True

    def clear_monitor(self):
        with self.lock:
            if self.monitor is None:
                return False
            self.monitor.clear()
            return True

    def suggest(self):
        with self.lock:
            if self.monitor is None:
                return None
            self.setup_touched = self.clock.mono()
            return self.monitor.suggest()

    def apply(self, data):
        """Validate, persist to the override file, restart ONLY the beam with the new settings. Returns (ok, errors).
        Bad values change nothing."""
        settings, errors = validate_settings(data, current=self.settings)
        if errors:
            return False, errors
        with self.lock:
            try:
                if self.override_file is not None:
                    write_override(self.override_file, settings, self.config_path)
            except OSError as e:
                return False, ["cannot save the settings to %s: %s" % (self.override_file, e)]
            self.settings = settings
            self.error = None
            self._stop_monitor()
            self._stop_service()
            self.restarts += 1
            self._start_service(self._beam_cfg(settings))
        self.emit(dict(type="info", msg="beam settings applied from the status page: %s" % ", ".join(
            "%s=%s" % (k, settings[k]) for k in SETTABLE_KEYS)))
        return True, []

    def turn_off(self):
        return self.apply(dict(driver="off"))

    # ------------------------------------------------------------------ setup-mode watchdog
    def _stop_monitor(self):
        monitor, self.monitor = self.monitor, None
        self.setup_touched = None
        if monitor is not None:
            try:
                monitor.stop()
            except Exception:  # noqa: BLE001
                pass

    def check_setup_idle(self):
        """Leave setup mode when the page has not asked for anything for setup_idle_s. Returns True if it did."""
        with self.lock:
            idle = (self.monitor is not None and self.setup_touched is not None and
                    self.clock.mono() - self.setup_touched >= self.setup_idle_s)
        if idle:
            return self.leave_setup("no page refresh for %.0f s" % self.setup_idle_s)
        return False

    def _start_watchdog(self):
        if not self.use_watchdog or (self.watchdog is not None and self.watchdog.is_alive()):
            return

        def loop():
            while not self.halt.wait(5.0):
                try:
                    self.check_setup_idle()
                except Exception as e:  # noqa: BLE001
                    self.emit(dict(type="warning", msg="beam setup watchdog: %s" % e))

        self.watchdog = threading.Thread(target=loop, name="beam-setup-watchdog", daemon=True)
        self.watchdog.start()


class ControllerSink(Sink):
    """The rt/sinks.py sink that hands rt's events to the controller (which forwards them to the running beam). It
    is called with rt's emitter lock held: it only reads one attribute and queues, never emits."""

    def __init__(self, controller):
        self.controller = controller

    def event(self, ev):
        self.controller.on_event(ev)
