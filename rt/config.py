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
    apply_beam_override(base, path)
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


def apply_beam_override(cfg, config_path):
    """The beam settings saved from the status page (Beam setup -> Apply) live in `<config>.beam.toml` next to the
    config file. Load order: rt/config.toml, then the user config, then this file's [beam], then --set options.
    cfg["beam_override"] says what happened (path, loaded, error). A broken or invalid file never stops rt: the
    beam is turned off and the error is shown on the status page (rt/beam/control.py)."""
    from rt.beam.control import load_override, override_path
    path = override_path(config_path)
    settings, error = load_override(path)
    cfg["beam_override"] = dict(path=str(path), config=str(config_path), loaded=settings is not None, error=error)
    if error:
        cfg.setdefault("beam", {})["driver"] = "off"
    elif settings is not None:
        cfg.setdefault("beam", {}).update(settings)


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


# Open-set mode (no roster for the event yet): the numbers the reader may answer. Every roster scraped so far
# (03SU-26SE, 60 events) used numbers 1-99; 999 leaves room for 3-digit numbers. The recognizer reads any digit
# string, so this is only a limit of the snapping step. Snapping to 999 numbers costs ~6x more CPU per crop than a
# ~100-number roster (numpy CTC, ~25 ms per crop on a desktop), which the identity thread absorbs (it never blocks
# detection). Set identity.open_set_max = 99 to make it cheaper.
OPEN_SET_MAX = 999


def roster_path(event, rosters_dir=None):
    return Path(rosters_dir or (ROOT / "rosters")) / ("%s.csv" % event)


def read_roster(event, rosters_dir=None):
    """rosters/<EV>.csv (numero,equipe) -> (sorted numbers, {numero: equipe}). Raises FileNotFoundError when the
    file is missing; an existing file with no rows gives ([], {})."""
    import csv
    p = roster_path(event, rosters_dir)
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    teams = {int(r["numero"]): r.get("equipe", "") for r in rows if str(r.get("numero") or "").strip()}
    return sorted(teams), teams


def pick_rosters_dir(event, rosters_dir=None, override_dir=None):
    """The folder to read <event>.csv from. `override_dir` (identity.rosters_dir: the installed app's data folder,
    where the launcher's "Baixar lista de equipes" button saves fetched rosters) wins when it has a file for this
    event; otherwise `rosters_dir` (default: rosters/ in the repo, i.e. the rosters bundled with the code)."""
    if override_dir and roster_path(event, override_dir).exists():
        return override_dir
    return rosters_dir


def load_roster(event, open_set_max=OPEN_SET_MAX, rosters_dir=None, override_dir=None):
    """The event roster for rt, or an "open set" when it cannot be used. Never stops rt.

    `override_dir` (optional, identity.rosters_dir): a folder searched first (see pick_rosters_dir).

    Returns dict(numbers, teams, open_set, warning):
      * roster found: numbers/teams from the file, open_set False, warning None.
      * file missing, empty or unreadable: numbers 1..open_set_max, no team names, open_set True, and a warning to
        log and to show on the status page. In open-set mode auto-confirm is off: its ~99% precision was measured
        with the real roster, which also throws away misreads that are not entry numbers."""
    rosters_dir = pick_rosters_dir(event, rosters_dir, override_dir)
    p = roster_path(event, rosters_dir)
    problem = None
    try:
        numbers, teams = read_roster(event, rosters_dir)
        if not numbers:
            problem = "%s is empty" % p
    except FileNotFoundError:
        problem = "%s not found" % p
    except (OSError, ValueError, KeyError) as e:      # unreadable file, a bad number, no "numero" column
        problem = "%s unreadable (%s: %s)" % (p, type(e).__name__, e)
    if problem is None:
        return dict(numbers=numbers, teams=teams, open_set=False, warning=None, path=str(p))
    fix = ("press \"Baixar lista de equipes\" in the Baja MV page" if override_dir else
           "run `python teacher/rosters.py %s`" % event)      # override_dir is set by the installed app's launcher
    warning = ("no roster for %s (%s): numbers are not checked against the entry list (any number 1-%d can be "
               "read, auto-confirm is off); %s once the event's results page lists the teams, then restart rt" % (
                   event, problem, int(open_set_max), fix))
    return dict(numbers=list(range(1, int(open_set_max) + 1)), teams={}, open_set=True, warning=warning,
                path=str(p))
