"""Build a QA review batch from the session-1 start-line data (out/world_*), so the review tool can be tested.

usage: python tools/qa/make_batch.py [--n 150] [--seed 0] [--out out/qa/26SE_session1.jsonl]
Run from the repo root. One item per pass (passes_all.csv, i.e. passes after 09:18:30 rebuilt with the current
pass rules), crops = the ~10 crops the VLM read for that pass, proposed number = roster-snapped majority vote of
the per-crop VLM reads (same rule as lapvision/analyze.py vote_vlm). The GT car goes to meta.hidden (never shown).

Sampling is stratified by (source file, read / no read), proportional allocation with a floor per stratum, so passes
with no read are always present. Each item carries weight = stratum size / stratum sample size.
"""
import argparse
import ast
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVENT = "26SE"
MIN_VOTES = 2          # as lapvision/analyze.py


def num(x):
    """'20.0' -> 20, '' / 'nan' -> None."""
    if x is None or str(x).strip() in ("", "nan", "None"):
        return None
    return int(float(x))


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def roster(ev):
    return {int(r["numero"]): r["equipe"] for r in read_csv(ROOT / "rosters" / f"{ev}.csv")}


def vote(reads, ros):
    nums = [n for n in (num(r["vlm"]) for r in reads) if n is not None and n in ros]
    c = Counter(nums)
    if not c:
        return None, 0.0, 0
    n, k = c.most_common(1)[0]
    return (n if k >= MIN_VOTES else None), k / len(reads), k


def allocate(sizes, n, floor):
    """Proportional allocation (largest remainder) with a per-stratum floor, total == n."""
    tot = sum(sizes.values())
    if n >= tot:
        return dict(sizes)
    alloc = {k: min(s, floor) for k, s in sizes.items()}
    left = n - sum(alloc.values())
    want = {k: s * n / tot for k, s in sizes.items()}
    while left > 0:
        k = max((k for k in sizes if alloc[k] < sizes[k]), key=lambda k: want[k] - alloc[k])
        alloc[k] += 1
        left -= 1
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--floor", type=int, default=3, help="minimum items per stratum")
    ap.add_argument("--out", default="out/qa/26SE_session1.jsonl")
    a = ap.parse_args()

    ros = roster(EVENT)
    pa = ROOT / "out" / "passes_all.session1.csv"
    if not pa.exists():
        pa = ROOT / "out" / "passes_all.csv"
    passes = read_csv(pa)

    # reads per (src, segment:track); paths are relative to lapvision/ ("..\out\world_1129\...")
    reads = defaultdict(list)
    for d in sorted((ROOT / "out").glob("world_*")):
        if not (d / "reads.csv").exists():
            continue
        for r in read_csv(d / "reads.csv"):
            reads[(d.name, f"{r['event']}:{r['track']}")].append(r)

    items, mism = [], 0
    for p in passes:
        src = p["src"]
        keys = ast.literal_eval(p["tracks"])
        rs = sorted((r for k in keys for r in reads.get((src, k), [])), key=lambda r: (r["event"], int(r["frame"])))
        if not rs:
            continue
        car, share, k = vote(rs, ros)
        if car != num(p["vlm_car"]):
            mism += 1
        crops = []
        for r in rs:
            rel = Path(r["path"].replace("\\", "/"))
            crops.append((ROOT / "lapvision" / rel).resolve().relative_to(ROOT).as_posix())
        items.append(dict(
            item_id=f"{EVENT}:{src}:{p['pass_id']}",
            event=EVENT,
            video_id=src,
            shot_tag="start_line",
            track_key=f"{src}:{p['pass_id']}",
            proposed_number=car,
            vote_share=round(share, 3),
            n_votes=k,
            n_crops=len(rs),
            overlay_number=None,
            crops=crops,
            crop_reads=[r["vlm_raw"] for r in rs],
            number_boxes=None,
            car_box_context=None,
            meta=dict(src=src, pass_id=int(p["pass_id"]), time=p["time"], tracks=keys,
                      hidden=dict(gt_car=num(p["gt_car"]),
                                  gt_dt=None if p["gt_dt"] in ("", "nan") else round(float(p["gt_dt"]), 2))),
        ))

    strata = defaultdict(list)
    for it in items:
        strata[(it["video_id"], it["proposed_number"] is not None)].append(it)
    alloc = allocate({k: len(v) for k, v in strata.items()}, a.n, a.floor)
    rng = random.Random(a.seed)
    out = []
    for k in sorted(strata):
        pick = rng.sample(strata[k], alloc[k])
        for it in pick:
            it["stratum"] = f"{k[0]}|{'read' if k[1] else 'noread'}"
            it["weight"] = round(len(strata[k]) / alloc[k], 4)
        out += pick
    rng.shuffle(out)

    dst = ROOT / a.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} passes from {pa.name}, vote mismatches vs vlm_car: {mism}")
    for k in sorted(strata):
        print(f"  stratum {k[0]:<11} {'read' if k[1] else 'noread':<7} {alloc[k]:>3} / {len(strata[k])}")
    print(f"wrote {len(out)} items -> {dst.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
