"""QA report: label precision / coverage / not-a-car rate with Wilson 95% intervals, per group.

usage: python tools/qa/report.py <batch.jsonl|.parquet> [--by event,shot_tag] [--json out.json]
--by takes item fields or meta fields (meta.src). Groups are printed for each --by key and for the whole batch.
If items carry meta.hidden.gt_car (session-1 batches), the reviewer's verdicts are also compared with that GT.

Definitions (verdicts: correct / wrong / not_car / unsure, last record per item wins, 'undo' clears):
  precision  = correct / (correct + wrong), items WITH a proposed number   ("unsure" and "not_car" excluded)
  coverage   = items with a proposed number / items judged cars (all verdicts except not_car)
  not_car    = not_car / reviewed
  missed     = no-proposal items where the reviewer typed a readable number ("wrong" on a null proposal) / no-proposal cars
  boxes_bad  = items flagged "number boxes wrong" / reviewed items that have number boxes
Weighted precision (if items have a 'weight', i.e. unequal sampling per stratum) is shown for the 'all' row.
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qalib  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt(w):
    return "      -" if w is None else f"{w[0]:5.1%} [{w[1]:4.0%}-{w[2]:4.0%}]"


def field(it, key):
    if key.startswith("meta."):
        return (it.get("meta") or {}).get(key[5:])
    return it.get(key)


def has_boxes(it):
    return any(b for b in (it.get("number_boxes") or []))


def summarize(rows):
    """rows: list of (item, verdict record)."""
    c = Counter()
    wsum = defaultdict(float)
    for it, v in rows:
        vd = v["verdict"]
        prop = it.get("proposed_number") is not None
        c["reviewed"] += 1
        c[vd] += 1
        if vd != "not_car":
            c["cars"] += 1
            c["cars_prop" if prop else "cars_noprop"] += 1
        if prop and vd in ("correct", "wrong"):
            c["p_" + vd] += 1
            wsum[vd] += float(it.get("weight") or 1.0)
        if not prop and vd == "wrong" and v.get("number") is not None:
            c["missed"] += 1
        if has_boxes(it):
            c["with_boxes"] += 1
            c["boxes_bad"] += bool(v.get("boxes_wrong"))
    out = dict(
        n=c["reviewed"],
        precision=wilson(c["p_correct"], c["p_correct"] + c["p_wrong"]),
        n_judged=c["p_correct"] + c["p_wrong"],
        coverage=wilson(c["cars_prop"], c["cars"]),
        not_car=wilson(c["not_car"], c["reviewed"]),
        unsure=wilson(c["unsure"], c["reviewed"]),
        missed=wilson(c["missed"], c["cars_noprop"]),
        boxes_bad=wilson(c["boxes_bad"], c["with_boxes"]),
    )
    tw = wsum["correct"] + wsum["wrong"]
    out["precision_weighted"] = wsum["correct"] / tw if tw else None
    return out


def print_table(title, groups):
    print(f"\n== by {title}")
    hdr = f"{'group':<22}{'n':>5}  {'precision':<19}{'(n)':>5}  {'coverage':<19}{'not a car':<19}{'unsure':<19}{'missed read':<19}{'boxes wrong':<19}"
    print(hdr)
    for g, s in groups:
        print(f"{str(g):<22}{s['n']:>5}  {fmt(s['precision']):<19}{s['n_judged']:>5}  {fmt(s['coverage']):<19}"
              f"{fmt(s['not_car']):<19}{fmt(s['unsure']):<19}{fmt(s['missed']):<19}{fmt(s['boxes_bad']):<19}")


def gt_compare(rows):
    """Session-1 only: reviewer verdicts vs meta.hidden.gt_car (None = pass matched to no human lap)."""
    rows = [(it, v) for it, v in rows if "gt_car" in ((it.get("meta") or {}).get("hidden") or {})]
    if not rows:
        return None
    print(f"\n== reviewer vs hidden GT ({len(rows)} reviewed items with a GT field)")
    t = Counter()
    bad = []
    for it, v in rows:
        gt = it["meta"]["hidden"]["gt_car"]
        pn, vd, typed = it.get("proposed_number"), v["verdict"], v.get("number")
        if gt is None:
            t[f"nogt_{vd}"] += 1
            continue
        if vd == "not_car":
            t["notcar_but_gt_car"] += 1
            bad.append((it["item_id"], pn, vd, typed, gt))
        if pn is not None:
            if pn == gt:
                t["gt_correct"] += 1
            else:
                t["gt_wrong"] += 1
            if vd in ("correct", "wrong"):
                key = ("rev_ok" if vd == "correct" else "rev_bad") + ("_gt_ok" if pn == gt else "_gt_bad")
                t[key] += 1
                if (vd == "correct") != (pn == gt):
                    bad.append((it["item_id"], pn, vd, typed, gt))
        if typed is not None:
            t["typed_agree" if typed == gt else "typed_disagree"] += 1
            if typed != gt and (not bad or bad[-1][0] != it["item_id"]):
                bad.append((it["item_id"], pn, vd, typed, gt))
    rp = wilson(t["rev_ok_gt_ok"] + t["rev_ok_gt_bad"], sum(t[k] for k in ("rev_ok_gt_ok", "rev_ok_gt_bad",
                                                                             "rev_bad_gt_ok", "rev_bad_gt_bad")))
    gp = wilson(t["gt_correct"], t["gt_correct"] + t["gt_wrong"])
    print(f"precision of proposed numbers, by reviewer: {fmt(rp)}   by GT: {fmt(gp)}  (n={t['gt_correct'] + t['gt_wrong']})")
    print("verdict vs GT on proposed numbers:          GT says right   GT says wrong")
    print(f"  reviewer 'correct'                        {t['rev_ok_gt_ok']:>13}   {t['rev_ok_gt_bad']:>13}   (missed errors)")
    print(f"  reviewer 'wrong'                          {t['rev_bad_gt_ok']:>13}   {t['rev_bad_gt_bad']:>13}")
    print(f"  (false alarms = reviewer 'wrong' where GT says right: {t['rev_bad_gt_ok']})")
    nt = t["typed_agree"] + t["typed_disagree"]
    print(f"typed numbers == GT car: {t['typed_agree']}/{nt}")
    nogt = {k[5:]: n for k, n in t.items() if k.startswith("nogt_")}
    print(f"items with no GT lap (false pass or unmatched): {nogt or 0};  'not a car' where GT has a lap: "
          f"{t['notcar_but_gt_car']}")
    for d in bad[:20]:
        print(f"  mismatch {d[0]}: proposed {d[1]}, verdict {d[2]}{' #%s' % d[3] if d[3] is not None else ''}, GT {d[4]}")
    return dict(t, gt_precision=gp, reviewer_precision=rp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--by", default="event,shot_tag")
    ap.add_argument("--json", help="also write the numbers to this JSON file")
    a = ap.parse_args()

    items = qalib.load_batch(a.batch)
    ver = qalib.load_verdicts(a.batch)
    rows = [(it, ver[it["item_id"]]) for it in items if it["item_id"] in ver]
    print(f"{Path(a.batch).name}: {len(items)} items, {len(rows)} reviewed")
    dts = sorted(v["dt_ms"] for _, v in rows if isinstance(v.get("dt_ms"), (int, float)) and 0 < v["dt_ms"] < 120000)
    if dts:
        print(f"review time: median {dts[len(dts) // 2] / 1000:.1f} s/item, total {sum(dts) / 60000:.1f} min "
              f"(pauses > 2 min excluded)")
    if not rows:
        return
    res = {"all": summarize(rows)}
    print_table("all", [("all", res["all"])])
    pw = res["all"]["precision_weighted"]
    if pw is not None and len({it.get("weight") for it, _ in rows}) > 1:
        print(f"stratum-weighted precision (all): {pw:.1%}")
    for key in [k.strip() for k in a.by.split(",") if k.strip()]:
        g = defaultdict(list)
        for it, v in rows:
            g[field(it, key)].append((it, v))
        groups = [(k, summarize(r)) for k, r in sorted(g.items(), key=lambda kv: str(kv[0]))]
        res[key] = dict(groups)
        print_table(key, groups)
    gc = gt_compare(rows)
    if gc:
        res["gt_compare"] = gc
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
