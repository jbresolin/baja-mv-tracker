"""Config: rt/config.toml (+ --set overrides) and an optional deploy manifest (one-line model switch)."""
import ast
import copy
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = Path(__file__).resolve().parent / "config.toml"
ROLES = ("car_detector", "car_detector_alt", "panel_detector", "recognizer", "vote")
# rt/run.py exits with this code when the status page's port cannot be opened and the page is required (started by
# the launcher, which watches rt through that page). rt/supervise.py names the reason in its log.
EXIT_STATUS_PORT_BUSY = 3


def _parse_value(v):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        return v


def resolve(p, base=ROOT):
    p = Path(p)
    return p if p.is_absolute() else (base / p)


def load(path=None, overrides=()):
    path = Path(path) if path else DEFAULT
    base = tomllib.loads(DEFAULT.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT.resolve():  # a user config only needs the keys it changes
        user = tomllib.loads(path.read_text(encoding="utf-8"))
        for sec, kv in user.items():
            base.setdefault(sec, {}).update(kv)
    cfg = copy.deepcopy(base)
    for o in overrides:
        k, v = o.split("=", 1)
        *path, key = k.split(".")       # section.key, or section.table.key (camera.settings.exposure=-10)
        d = cfg
        for part in path:
            d = d.setdefault(part, {})
        d[key] = _parse_value(v)
    apply_manifest(cfg)
    m = cfg["models"]
    for r in ROLES:
        if isinstance(m.get(r), str) and m[r]:
            m[r] = str(resolve(m[r]))
    return cfg


ALIASES = {"car_detector": ("car_detector", "car"), "car_detector_alt": ("car_detector_alt",),
           "panel_detector": ("panel_detector", "panel"), "recognizer": ("recognizer", "recog"), "vote": ("vote",)}


def apply_manifest(cfg):
    """deploy/models/manifest.json (published by the final-model agent). Accepted layouts: {"models": {"car_n": {"file":
    "car_n.onnx", "input": {"shape": [1, 3, 384, 640]}}, "car_s": ..., "panel": ..., "recog": ...}, "pipeline": {"vote":
    {"tau", "min_support", "score"}}} (the published one), or roles keyed car_detector / panel_detector / recognizer /
    vote, each a path or {"path"|"file", "imgsz"}. models.manifest_car picks "car_n" (default) or "car_s" as the car
    detector; the other one becomes car_detector_alt. Paths are relative to the manifest's folder."""
    mf = cfg["models"].get("manifest", "")
    if not mf:
        return
    mp = resolve(mf)
    man = json.loads(mp.read_text(encoding="utf-8"))
    roles = dict(man.get("roles", man.get("models", man)))
    m = cfg["models"]
    pick = m.get("manifest_car", "car_n")
    if pick in roles:
        roles.setdefault("car_detector", roles[pick])
        other = {"car_n": "car_s", "car_s": "car_n"}.get(pick)
        if other in roles:
            roles.setdefault("car_detector_alt", roles[other])
    vote = (man.get("pipeline") or {}).get("vote")
    if isinstance(vote, dict) and "tau" in vote:
        roles.setdefault("vote", vote)
    for r, names in ALIASES.items():
        v = next((roles[n] for n in names if n in roles), None)
        if v is None:
            continue
        if r == "vote" and isinstance(v, dict) and "tau" in v:
            m["vote"] = v  # inline rule
            continue
        if isinstance(v, dict):
            f = v.get("path", v.get("file"))
            shp = (v.get("input") or {}).get("shape")
            sz = v.get("imgsz") or (shp[2:4] if shp and all(isinstance(x, int) for x in shp[2:4]) else None)
            if sz and r == "car_detector":
                m["car_imgsz"] = sz
            if sz and r == "panel_detector":
                m["panel_imgsz"] = sz if isinstance(sz, int) else int(max(sz))
        else:
            f = v
        m[r] = str(resolve(f, mp.parent))
    m["manifest_note"] = man.get("note", "")


def vote_rule(cfg):
    v = cfg["models"]["vote"]
    if isinstance(v, dict):
        d = v
    else:
        d = json.loads(Path(v).read_text(encoding="utf-8"))
        d = d.get("chosen", d)
    return dict(tau=float(d["tau"]), min_support=int(d.get("min_support", 1)), score=d.get("score", "share"))


def read_roster(event):
    """rosters/<EV>.csv (numero,equipe) -> (sorted numbers, {numero: equipe})"""
    import csv
    p = ROOT / "rosters" / ("%s.csv" % event)
    if not p.exists():
        raise SystemExit("no roster %s (rosters/<EV>.csv)" % p)
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    teams = {int(r["numero"]): r.get("equipe", "") for r in rows}
    return sorted(teams), teams
