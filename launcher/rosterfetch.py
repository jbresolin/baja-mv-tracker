"""The "Baixar lista de equipes" button: fetch an event's roster (car number -> team) from the official results
site with teacher/rosters.py, and save it in the data root's rosters/ folder, where rt looks first
(identity.rosters_dir). The rosters bundled with the code stay as they were.

The results site lists the teams only shortly before the event, so "no roster yet" is a normal answer; rt then
runs in open-set mode (rt/README.md, Troubleshooting).
"""
import csv
import importlib.util
from pathlib import Path

from launcher.settings import EVENT_CODE


def load_rosters_module(code_root):
    """teacher/rosters.py as a module (teacher/ is not a package, and only this one file of it is installed)."""
    path = Path(code_root) / "teacher" / "rosters.py"
    spec = importlib.util.spec_from_file_location("bajamv_teacher_rosters", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_rows(path):
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for row in csv.DictReader(f) if str(row.get("numero") or "").strip())
    except (OSError, ValueError, csv.Error):
        return 0


def valid_event(event):
    """The event code in capitals when it looks like one (26SE, 27SU, 03NE: EVENT_CODE), else None. The code ends
    up in a file name and in the results site's URL, so anything else ("../x", "26SE/..", a URL) is refused."""
    event = (event or "").strip().upper()
    return event if EVENT_CODE.match(event) else None


def roster_status(event, paths):
    """Which roster rt would use for this event: dict(event, where = "baixada" | "embutida" | None, path, cars)."""
    raw = (event or "").strip().upper()
    event = valid_event(raw)
    if event is None:
        return dict(event=raw, where=None, path=None, cars=0)
    for where, folder in (("baixada", paths.rosters_dir), ("embutida", paths.bundled_rosters_dir)):
        path = Path(folder) / ("%s.csv" % event)
        if event and path.exists():
            return dict(event=event, where=where, path=str(path), cars=count_rows(path))
    return dict(event=event, where=None, path=None, cars=0)


def fetch_roster(event, paths, module=None, get_page=None, sleep=None):
    """Fetch and save one event's roster. Returns dict(ok, message, cars, path) for the page; never raises.
    module / get_page / sleep: injectable (tests pass a fake page and no sleeping)."""
    raw = (event or "").strip().upper()
    if not raw:
        return dict(ok=False, message="Informe o código do evento primeiro.", cars=0, path=None)
    event = valid_event(raw)
    if event is None:
        return dict(ok=False, cars=0, path=None,
                    message="Código de evento inválido: %r (exemplo: 27SE, 27SU, 03NE)." % raw[:40])
    try:
        module = module or load_rosters_module(paths.code_root)
        kwargs = {"get_page": get_page} if get_page else {}
        if sleep is not None:
            kwargs["sleep"] = sleep
        rows, source = module.roster(event, **kwargs)
    except Exception as e:  # noqa: BLE001 (no network, site down, page changed...)
        return dict(ok=False, message="Não foi possível acessar o site de resultados: %s: %s" % (
            type(e).__name__, e), cars=0, path=None)
    if not rows:
        return dict(ok=False, cars=0, path=None,
                    message="Ainda não há lista de equipes para %s no site de resultados. O site publica a lista "
                            "pouco antes do evento; até lá a linha de chegada lê qualquer número (sem confirmação "
                            "automática)." % event)
    path = paths.rosters_dir / ("%s.csv" % event)
    cars = module.write_roster(rows, source, path)
    return dict(ok=True, cars=cars, path=str(path),
                message="Lista de %s salva: %d carros (página %s). Reinicie a linha de chegada para usá-la." % (
                    event, cars, source))
