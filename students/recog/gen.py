"""WP10b: generalisation helpers shared by train.py options, train_parseq.py and heldout_eval.py.

* held-out-number folds (students/recog/folds/numbers_v1.json, built by folds.py): `fold_numbers("v1:0")`
* livery subsampling for the data-scaling curve: keep a fraction of the training (event, number) cars
* team identity of a crop (roster name, normalised) for the "seen number, unseen style" proxy
* uniform synthetic number text (1-99 plus "0N" side-plate forms, near-uniform digit frequency)
"""
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FOLDS = HERE / "folds"


def fold_numbers(spec):
    """'v1:0' -> set of held-out numbers (ints); None / '' -> empty set."""
    if not spec:
        return set()
    ver, k = spec.split(":")
    doc = json.loads((FOLDS / f"numbers_{ver}.json").read_text())
    return set(doc["folds"][int(k)]["heldout"])


def event_of(item):
    return item["video"][:4]


def exclude_numbers(items, held):
    """Drop every crop of a held-out number and the "no number" windows cut from the same car crops."""
    if not held:
        return items
    bad_crops = {i["crop"] for i in items if not i["neg"] and int(i["number"]) in held}
    return [i for i in items if not (i["crop"] in bad_crops or (not i["neg"] and int(i["number"]) in held))]


def keep_cars(items, frac, seed=0):
    """Keep a fraction of the training cars (event, number), chosen by a stable hash so the same cars are kept in every
    item list (digits cache, panel proposals); negatives follow their car crops."""
    if frac >= 1.0:
        return items
    import hashlib
    keep = lambda ev, n: int(hashlib.md5(f"{ev}:{n}:{seed}".encode()).hexdigest()[:8], 16) / 2 ** 32 < frac  # noqa: E731
    keep_crops = {i["crop"] for i in items if not i["neg"] and keep(event_of(i), i["number"])}
    return [i for i in items if i["crop"] in keep_crops]


def norm_team(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


@lru_cache(None)
def roster_teams(ev):
    import pandas as pd
    return {int(r.numero): norm_team(r.equipe) for r in pd.read_csv(ROOT / "rosters" / f"{ev}.csv").itertuples()}


def team_of(item):
    if item["neg"]:
        return None
    return roster_teams(event_of(item)).get(int(item["number"]))


def rand_text_uniform(rng):
    """1-99 with near-uniform digits: 12% '0N', 23% 'N' (1-9), 65% two digits (first 1-9, second 0-9 uniform)."""
    r = rng.random()
    if r < 0.12:
        return "0" + str(rng.randint(1, 9))
    if r < 0.35:
        return str(rng.randint(1, 9))
    return str(rng.randint(1, 9)) + str(rng.randint(0, 9))
