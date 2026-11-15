"""Zone tracker configuration: zone/configs/zone.example.toml (all defaults, commented) + an optional user TOML (only
the keys it changes) + `--set section.key=value` overrides. Then every "auto" value is resolved from the PTZ
characterization report, and the settings objects of each module are built from the result.

The control branch (A or B) is chosen here, from config, never by editing code:
    controller.branch = "auto"  ->  the report's verdict ("A" | "B"), or "B" without a report.
"""
import ast
import copy
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from zone import direction
from zone.controller import ControllerSettings
from zone.director import DirectorSettings
from zone.progress import ProgressSettings
from zone.ptz.sim import DEFAULT_UNITS_PER_DEGREE, PtzModel
from zone.speedmap import AxisSpeedMap

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = Path(__file__).resolve().parent / "configs" / "zone.example.toml"


def _parse_value(v):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        return v


def _merge(base, user):
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v


def load(path=None, overrides=()):
    """-> nested dict. `overrides` like ["controller.branch=B", "rt.capture.fps=5"].

    Order: defaults (zone.example.toml) < user TOML < profile < overrides. A profile is a named set of changes in
    the [profiles.<name>] tables, picked with the top-level key `profile` (in the user TOML, or
    `--set profile=strict`). Profiles are for side-by-side comparison runs; `--set` still wins over them."""
    cfg = tomllib.loads(DEFAULT.read_text(encoding="utf-8"))
    if path and Path(path).resolve() != DEFAULT.resolve():
        _merge(cfg, tomllib.loads(Path(path).read_text(encoding="utf-8")))
    cfg = copy.deepcopy(cfg)
    sets = {}
    for o in overrides:
        k, v = o.split("=", 1)
        parts = k.strip().split(".")
        d = sets
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = _parse_value(v.strip())
    name = sets.get("profile", cfg.get("profile", "")) or ""
    if name:
        profiles = cfg.get("profiles", {})
        if name not in profiles:
            raise ValueError("unknown config profile %r (known: %s)" % (name, ", ".join(sorted(profiles)) or "none"))
        _merge(cfg, copy.deepcopy(profiles[name]))
    _merge(cfg, sets)
    return cfg


def resolve_path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# ======================================================================================================== report
@dataclass
class Report:
    """The characterization report, as far as the zone tracker cares: the PtzModel (speed tables, latency, inquiry
    behaviour...) and the verdict. `path` is "" when there is no report (defaults are used, Branch B)."""
    model: PtzModel
    branch: str
    reason: str
    path: str = ""
    raw: dict = None

    def summary(self):
        return dict(path=self.path, branch=self.branch, reason=self.reason)


def load_report(path) -> Report:
    if not path:
        return Report(PtzModel(), "B", "no characterization report: Branch B with default speed tables", "", {})
    p = resolve_path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    verdict = raw.get("verdict") or {}
    branch = str(verdict.get("branch", "B")).upper()
    if branch not in ("A", "B"):
        branch = "B"
    return Report(PtzModel.from_report(raw), branch, str(verdict.get("reason", "")), str(p), raw)


# ======================================================================================================== settings
def _auto(v, fallback):
    return fallback if v == "auto" or v is None else v


@dataclass
class Settings:
    """Everything the runtime needs, resolved."""
    cfg: dict
    report: Report
    controller: ControllerSettings
    director: DirectorSettings
    progress: ProgressSettings
    pan_map: AxisSpeedMap
    tilt_map: AxisSpeedMap
    pose_source: str
    # pan/tilt sign (zone/direction.py): from the config and the report (sign_base, sign_base_how), then the zone's
    # own measurement on top (apply_zone writes controller.pan_sign / tilt_sign and sign_how)
    sign_base: dict = None
    sign_base_how: dict = None
    sign_how: dict = None


def build(cfg, report: Report = None) -> Settings:
    report = report or load_report(cfg["ptz"].get("report", ""))
    m = report.model
    c = cfg["controller"]
    branch = c["branch"].upper() if isinstance(c["branch"], str) else "auto"
    branch = report.branch if branch == "AUTO" else branch
    if branch == "A" and m.inquiry_mode != "always":
        # a forced Branch A on a camera that can't answer while moving would feed back stale positions
        branch = "B"
    # which way the reported pan / tilt count (zone/direction.py): config, else the report's measurement, else +1
    signs, signs_how = direction.resolve(cfg.get("ptz", {}), report)
    latency = _auto(c["latency_s"], None)
    if latency is None:
        latency = (m.command_latency_s + m.ramp_s / 2.0) if report.path else 0.25
    inquiry_hz = _auto(c["inquiry_hz"], None)
    if inquiry_hz is None:
        inquiry_hz = min(20.0, float(m.inquiry_poll_hz)) if getattr(m, "inquiry_poll_hz", None) else 10.0
    zoom_wide = _auto(c["zoom_wide"], None)
    if zoom_wide is None:
        zoom_wide = float(m.zoom_limits[0]) if report.path else 0.0
    ctrl = ControllerSettings(
        branch=branch, latency_s=float(latency), include_processing_delay=bool(c["include_processing_delay"]),
        k_pan=float(c["k_pan"]), k_tilt=float(c["k_tilt"]), max_feedback_pan=float(c["max_feedback_pan"]),
        max_feedback_tilt=float(c["max_feedback_tilt"]), inquiry_hz=float(inquiry_hz),
        inquiry_stale_s=float(c["inquiry_stale_s"]), dead_reckoning_trust_s=float(c["dead_reckoning_trust_s"]),
        k_quant=float(c["k_quant"]), ttl_s=float(c["ttl_s"]), zoom_enabled=bool(c["zoom_enabled"]),
        zoom_direct_supported=bool(getattr(m, "supports_zoom_direct", True)),
        zoom_min_interval_s=float(c["zoom_min_interval_s"]), zoom_threshold=float(c["zoom_threshold"]),
        zoom_loosen=float(c["zoom_loosen"]), zoom_wide=float(zoom_wide), zoom_limits=tuple(m.zoom_limits),
        model_latency_s=float(m.command_latency_s),
        model_stop_latency_s=float(getattr(m, "stop_latency_s", m.command_latency_s)),
        model_ramp_s=float(m.ramp_s),
        pan_sign=signs["pan"], tilt_sign=signs["tilt"],
        divergence_pan_units=float(c.get("divergence_deg", 0.0)) * units_per_degree(m, "pan"),
        divergence_tilt_units=float(c.get("divergence_deg", 0.0)) * units_per_degree(m, "tilt"),
        divergence_cycles=int(c.get("divergence_cycles", 5)))

    d = cfg["director"]
    settle = _auto(d["settle_s"], None)
    if settle is None:
        settle = float(getattr(m, "preset_settle_s", 1.5)) if report.path else 1.5
    director = DirectorSettings(
        s_lock=float(d["s_lock"]), lock_window=float(d["lock_window"]), v_min=float(d["v_min"]),
        min_updates=int(d["min_updates"]),
        lost_timeout_s=float(d["lost_timeout_s"]), stopped_v=float(d["stopped_v"]),
        stopped_timeout_s=float(d["stopped_timeout_s"]), off_corridor_frames=int(d["off_corridor_frames"]),
        s_exit=float(d["s_exit"]), settle_s=float(settle))

    p = cfg["progress"]
    progress = ProgressSettings(
        meas_sigma_px=float(p["meas_sigma_px"]), accel_sigma=float(p["accel_sigma"]), v_sigma0=float(p["v_sigma0"]),
        window_back_px=float(p["window_back_px"]), window_ahead_px=float(p["window_ahead_px"]),
        window_sigmas=float(p.get("window_sigmas", 3.0)),
        corridor_factor=float(p["corridor_factor"]), finish_after_s=float(p["finish_after_s"]),
        profile_min_span=float(p["profile_min_span"]), max_prediction_s=float(p["max_prediction_s"]))

    s = cfg["speedmap"]
    pan_min = _auto(s["pan_min_smooth_step"], getattr(m, "pan_min_smooth_step", 1) or 1)
    tilt_min = _auto(s["tilt_min_smooth_step"], getattr(m, "tilt_min_smooth_step", 1) or 1)
    pan_map = AxisSpeedMap(m.pan_speed_table, getattr(m, "pan_speed_table_negative", None), int(pan_min),
                           float(s["hysteresis"]), int(s["max_step_change"]), float(s["start_margin"]))
    tilt_map = AxisSpeedMap(m.tilt_speed_table, getattr(m, "tilt_speed_table_negative", None), int(tilt_min),
                            float(s["hysteresis"]), int(s["max_step_change"]), float(s["start_margin"]))

    ease = float(cfg["zone"].get("ease_in", 0.0))
    if ease > 0 and director.s_lock > -ease + 1e-9:
        # the ease-in starts moving the camera at s = -ease_in: a car locked later would find it already gone
        director.s_lock = -ease
    return Settings(cfg, report, ctrl, director, progress, pan_map, tilt_map,
                    pose_source_for(cfg["teach"]["pose_source"], report), dict(signs), dict(signs_how),
                    dict(signs_how))


def units_per_degree(model, axis):
    """Raw units per degree from the report, or the usual VISCA value when the bench didn't measure it."""
    return float(getattr(model, axis + "_units_per_degree", None) or DEFAULT_UNITS_PER_DEGREE)


def pose_source_for(choice, report: Report):
    """Teach pose source, in the brief's order of preference: inquiry > commanded absolute > dead reckoning."""
    if choice and choice != "auto":
        return choice
    m = report.model
    if not report.path:
        return "inquiry"          # no report: try inquiry (the teach page shows when it fails)
    if m.inquiry_mode in ("always", "at_rest", "frozen_while_moving"):
        return "inquiry"          # the teach pose is read at rest, so "at rest only" is enough
    if getattr(m, "supports_absolute", False):
        return "commanded"
    return "dead_reckoning"
