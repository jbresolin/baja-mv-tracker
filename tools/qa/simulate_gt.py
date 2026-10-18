"""Test harness: answer a batch from its hidden GT (meta.hidden.gt_car) by POSTing to a running serve.py.

usage: python tools/qa/simulate_gt.py <batch.jsonl> [--port 8765] [--limit N] [--error-rate 0.03] [--seed 0]
Skips items the server already has a verdict for (so it also exercises resume). Rules:
  no GT lap -> not_car;  proposed == GT -> correct;  proposed != GT -> wrong + GT number;
  no proposal: GT car 1 (mud, unreadable) -> correct ("no readable number"), else wrong + GT number.
--error-rate flips that fraction of answers to a deliberate mistake, to check that report.py's GT comparison sees them.
"""
import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qalib  # noqa: E402


def call(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--error-rate", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    rng = random.Random(a.seed)
    items = qalib.load_batch(a.batch)           # read locally: the server never sends meta.hidden
    done = set(call(base + "/api/batch")["verdicts"])
    n = flips = 0
    for it in items:
        if it["item_id"] in done:
            continue
        if a.limit and n >= a.limit:
            break
        gt = it["meta"]["hidden"]["gt_car"]
        pn = it["proposed_number"]
        if gt is None:
            v, num = "not_car", None
        elif pn is not None:
            v, num = ("correct", None) if pn == gt else ("wrong", gt)
        else:
            v, num = ("correct", None) if gt == 1 else ("wrong", gt)
        if rng.random() < a.error_rate:
            flips += 1
            v, num = ("wrong", None) if v == "correct" else ("correct", None)
        r = call(base + "/api/verdict", dict(item_id=it["item_id"], verdict=v, number=num, boxes_wrong=False,
                                             dt_ms=rng.randint(2000, 7000)))
        assert r["ok"], r
        n += 1
    print(f"posted {n} verdicts ({flips} deliberate mistakes); server now has {len(call(base + '/api/batch')['verdicts'])}")


if __name__ == "__main__":
    main()
