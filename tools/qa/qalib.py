"""Shared helpers for the QA review tool: batch / verdict / roster I/O (stdlib only; parquet needs pyarrow)."""
import csv
import json
from pathlib import Path

VERDICTS = ("correct", "wrong", "not_car", "unsure")


def load_batch(path):
    """Items of a review batch (.jsonl, or .parquet with the same columns)."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        items = pq.read_table(path).to_pylist()
    else:
        with open(path, encoding="utf-8") as f:
            items = [json.loads(l) for l in f if l.strip()]
    seen = set()
    for it in items:
        if it["item_id"] in seen:
            raise ValueError(f"duplicate item_id {it['item_id']!r} in {path}")
        seen.add(it["item_id"])
    return items


def verdicts_path(batch):
    return Path(str(batch) + ".verdicts.jsonl")


def load_verdicts(batch):
    """Current verdict per item_id (last record wins; an 'undo' record clears the item)."""
    p = verdicts_path(batch)
    cur = {}
    if not p.exists():
        return cur
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:   # a torn last line from a crash: ignore it
                continue
            if r.get("verdict") == "undo":
                cur.pop(r["item_id"], None)
            else:
                cur[r["item_id"]] = r
    return cur


def load_roster(root, event):
    p = Path(root) / "rosters" / f"{event}.csv"
    if not p.exists():
        return {}
    with open(p, newline="", encoding="utf-8") as f:
        return {str(int(r["numero"])): r["equipe"] for r in csv.DictReader(f) if r.get("numero", "").strip()}
