"""WP10b: the unseen-number protocol. Compares a model family trained on all numbers vs with each fold's numbers held out.

A family is a set of runs in students/runs/recog/: <family>_full (all training numbers) and <family>_f0, _f1, _f2
(trained with --fold v1:K: every crop of fold K's numbers removed from training, all events). Same recipe otherwise.
Val (25SU) and test (24SU) only hold numbers 1-16 and the folds partition 1-16, so every val/test crop is "unseen"
for exactly one fold model. Hence the key number, the unseen-number gap, is perfectly paired (same crops, same box
heights): accuracy of the full model on all val/test crops vs accuracy of the fold model that never saw the crop's
number. Also reported:
* seen-number control: fold models on the val/test crops of numbers they did see (they have ~20% less training data);
* track level: soft vote over the track's crops (unseen = fold model posteriors), top-1 and precision / coverage at tau;
* "unseen number, seen event": fold models on the train-split crops of their held-out numbers (incl. 17-58);
* style proxy (full model): val/test crops by whether the crop's team (roster name) occurs in training with another
  number, never occurs in training, or with this same number (24SU #6 only), height-standardised.

usage: CUDA_VISIBLE_DEVICES=-1 python students/recog/heldout_eval.py --family gx_base [--ckpt last.pt] [--threads 4]
       (any run name can be given as --full NAME --folds A,B,C instead of a family)
Output: students/runs/recog/heldout_<family>.json and a printed summary. Per-run predictions are cached in
<run>/eval/heldout_preds_<ckpt>.pkl (delete to recompute).
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen  # noqa: E402
from common import RUNS, load_model, read_roster, strip0, vote  # noqa: E402
from evaluate import HBINS, run, wilson  # noqa: E402

EV = {"val": "25SU", "test": "24SU"}
TAUS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def load_cache(split):
    return pickle.load(open(RUNS / f"cache_{split}.pkl", "rb"))


def predictions(run_name, ckpt, items_by_split, threads):
    """dict split -> list of (pred, P, roster) aligned with items_by_split[split]"""
    d = RUNS / run_name
    cache = d / "eval" / f"heldout_preds_{Path(ckpt).stem}.pkl"
    if cache.exists():
        out = pickle.load(open(cache, "rb"))
        if all(len(out.get(sp, ())) == len(v) for sp, v in items_by_split.items()):
            return out
    torch.set_num_threads(threads)
    model, _ = load_model(d / ckpt)
    out = {}
    for sp, items in items_by_split.items():
        res = [None] * len(items)
        by_ev = defaultdict(list)
        for j, it in enumerate(items):
            by_ev[gen.event_of(it)].append(j)
        for ev, idx in by_ev.items():
            roster = read_roster(ev)
            rr = run(model, [items[j] for j in idx], roster)
            for j, r in zip(idx, rr):
                res[j] = (r["pred"], r["P"].astype(np.float32), ev)
        out[sp] = res
    cache.parent.mkdir(exist_ok=True)
    pickle.dump(out, open(cache, "wb"))
    return out


def correct(pred_tuple, it):
    pred, P, ev = pred_tuple
    roster = read_roster(ev)
    return strip0(pred) == it["number"], roster[int(np.argmax(P))] == int(it["number"])


def hbin(h):
    for lo, hi in zip(HBINS[:-1], HBINS[1:]):
        if lo <= h < hi:
            return f"{lo}-{hi if hi < 1e8 else 'inf'}"
    return "lt16"


def summarize(rows):
    """rows: list of (acc, snap) booleans -> dict with n, acc, snap, CI"""
    n = len(rows)
    if not n:
        return dict(n=0)
    a = sum(r[0] for r in rows)
    s = sum(r[1] for r in rows)
    return dict(n=n, acc=round(a / n, 4), snap=round(s / n, 4), snap_ci=[round(float(x), 3) for x in wilson(s, n)])


def std_height(rows, heights, ref_dist):
    """height-standardised snap accuracy: per-bin accuracy weighted by a reference bin distribution"""
    by = defaultdict(list)
    for r, h in zip(rows, heights):
        by[hbin(h)].append(r[1])
    tot, w = 0.0, 0.0
    for b, p in ref_dist.items():
        if by.get(b):
            tot += p * np.mean(by[b])
            w += p
    return round(tot / w, 4) if w else float("nan")


def track_stats(entries, taus=TAUS):
    """entries: list of (track_key, gt, roster, P) per crop -> top-1 and curve"""
    by = defaultdict(list)
    for k, gt, roster, P in entries:
        by[k].append((gt, roster, P))
    tr = []
    for k, v in by.items():
        j, s, _ = vote([x[2] for x in v])
        tr.append((v[0][1][j] == v[0][0], s))
    out = dict(n_tracks=len(tr), top1=round(float(np.mean([t[0] for t in tr])), 4) if tr else None, curve=[])
    for t in taus:
        ans = [x for x in tr if x[1] >= t]
        ok = sum(x[0] for x in ans)
        out["curve"].append(dict(tau=t, coverage=round(len(ans) / max(len(tr), 1), 3),
                                 precision=round(ok / max(len(ans), 1), 4), n=len(ans)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None)
    ap.add_argument("--full", default=None)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--extra-full", default="", help="comma list of extra full-data runs (seed noise reference)")
    a = ap.parse_args()
    full = a.full or f"{a.family}_full"
    nf = len(json.loads((gen.FOLDS / f"numbers_{a.version}.json").read_text())["folds"])
    folds = a.folds.split(",") if a.folds else [f"{a.family}_f{k}" for k in range(nf)]
    if not a.folds:
        missing = [f for f in folds if not (RUNS / f / a.ckpt).exists()]
        if missing:
            print("WARNING: missing fold runs (folds from the first missing one on are skipped):", missing)
            folds = folds[:folds.index(missing[0])]
    tag = a.family or full
    held = [gen.fold_numbers(f"{a.version}:{k}") for k in range(len(folds))]
    all_held = set().union(*held)

    vt = {sp: [i for i in load_cache(sp) if not i["neg"]] for sp in ("val", "test")}
    trh = [i for i in load_cache("train") if not i["neg"] and int(i["number"]) in all_held]
    items = dict(val=vt["val"], test=vt["test"], train_held=trh)
    preds = {r: predictions(r, a.ckpt, items, a.threads)
             for r in [full] + folds + [x for x in a.extra_full.split(",") if x]}

    vt_ok = [(sp, j, it) for sp in ("val", "test") for j, it in enumerate(items[sp]) if it["bucket"] == "ok"]
    ref = defaultdict(float)
    for _, _, it in vt_ok:
        ref[hbin(it["box_h"])] += 1 / len(vt_ok)
    fold_of = {n: k for k, h in enumerate(held) for n in h}

    res = dict(family=tag, full=full, folds=folds, ckpt=a.ckpt, heldout=[sorted(h) for h in held])
    # 1. paired unseen-number gap on val+test 'ok' crops
    R = defaultdict(list)
    H = defaultdict(list)
    for sp, j, it in vt_ok:
        n = int(it["number"])
        k = fold_of.get(n)
        cf = correct(preds[full][sp][j], it)
        R["full_all"].append(cf)
        H["full_all"].append(it["box_h"])
        R[f"full_{sp}"].append(cf)
        if k is None:
            continue
        cu = correct(preds[folds[k]][sp][j], it)
        R["unseen_all"].append(cu)
        H["unseen_all"].append(it["box_h"])
        R[f"unseen_{sp}"].append(cu)
        R[f"full_on_f{k}"].append(cf)
        R[f"unseen_f{k}"].append(cu)
        for kk, fr in enumerate(folds):
            if kk != k:
                cs = correct(preds[fr][sp][j], it)
                R["seen_by_fold_models"].append(cs)
                H["seen_by_fold_models"].append(it["box_h"])
                R[f"full_seen_ctrl"].append(cf)
                H[f"full_seen_ctrl"].append(it["box_h"])
        R[f"unseen_h{hbin(it['box_h'])}"].append(cu)
        R[f"full_h{hbin(it['box_h'])}"].append(cf)
    res["crop"] = {k: summarize(v) for k, v in R.items()}
    for k in ("full_all", "unseen_all", "seen_by_fold_models", "full_seen_ctrl"):
        res["crop"][k]["snap_std_height"] = std_height(R[k], H[k], ref)
    for e in a.extra_full.split(","):
        if e:
            res["crop"][f"extra_{e}"] = summarize([correct(preds[e][sp][j], it) for sp, j, it in vt_ok])
    # per number (val+test): full vs unseen
    pn = defaultdict(lambda: defaultdict(list))
    for sp, j, it in vt_ok:
        n = int(it["number"])
        pn[(sp, n)]["full"].append(correct(preds[full][sp][j], it)[1])
        if n in fold_of:
            pn[(sp, n)]["unseen"].append(correct(preds[folds[fold_of[n]]][sp][j], it)[1])
    res["per_number"] = {f"{sp}:{n}": dict(n=len(v["full"]), full=round(np.mean(v["full"]), 3),
                                           unseen=round(np.mean(v["unseen"]), 3) if v["unseen"] else None)
                         for (sp, n), v in sorted(pn.items())}
    # 2. track level (val, test separately; all number crops incl. tiny)
    res["track"] = {}
    for sp in ("val", "test"):
        roster = read_roster(EV[sp])
        ef, eu = [], []
        for j, it in enumerate(items[sp]):
            key = (it["video"], it["track"])
            gt = int(it["number"])
            ef.append((key, gt, roster, preds[full][sp][j][1]))
            k = fold_of.get(gt)
            if k is not None:
                eu.append((key, gt, roster, preds[folds[k]][sp][j][1]))
        res["track"][sp] = dict(full=track_stats(ef), unseen=track_stats(eu))
    # 3. unseen number, seen event (train-split crops of held-out numbers, fold model)
    R2, H2 = defaultdict(list), defaultdict(list)
    for j, it in enumerate(items["train_held"]):
        n = int(it["number"])
        if it["bucket"] != "ok" or n not in fold_of:
            continue
        c = correct(preds[folds[fold_of[n]]]["train_held"][j], it)
        cats = ["all", "1digit" if n < 10 else ("2digit_le16" if n <= 16 else ("2digit_17_39" if n < 40 else
                                                                            "2digit_ge40")), f"fold{fold_of[n]}"]
        for cat in cats + [f"n{n}"]:
            R2[cat].append(c)
            H2[cat].append(it["box_h"])
    ref2 = defaultdict(float)
    for h in H2["all"]:
        ref2[hbin(h)] += 1 / len(H2["all"])
    res["train_held_unseen"] = {}
    for k, v in R2.items():
        res["train_held_unseen"][k] = summarize(v)
        res["train_held_unseen"][k]["snap_std_height"] = std_height(v, H2[k], ref2)
    # 4. style proxy (full model): team seen in training with another number / never / same number
    tr_items = [i for i in load_cache("train") if not i["neg"]]
    team_nums = defaultdict(set)
    for i in tr_items:
        team_nums[gen.team_of(i)].add(int(i["number"]))
    R3, H3 = defaultdict(list), defaultdict(list)
    for sp, j, it in vt_ok:
        t, n = gen.team_of(it), int(it["number"])
        cat = ("same_team_same_number" if n in team_nums.get(t, ()) else
               "team_seen_other_number" if t in team_nums else "team_unseen")
        c = correct(preds[full][sp][j], it)
        R3[cat].append(c)
        H3[cat].append(it["box_h"])
        if n in fold_of:   # same split, unseen number
            cu = correct(preds[folds[fold_of[n]]][sp][j], it)
            R3[cat + "|unseen_number"].append(cu)
            H3[cat + "|unseen_number"].append(it["box_h"])
    res["style_proxy"] = {}
    for k, v in R3.items():
        s = summarize(v)
        s["snap_std_height"] = std_height(v, H3[k], ref)
        res["style_proxy"][k] = s
    out = RUNS / f"heldout_{tag}.json"
    json.dump(res, open(out, "w"), indent=1, default=float)

    c = res["crop"]
    print(f"== {tag} ({a.ckpt}); val+test 'ok' crops: {c['full_all']['n']}")
    print(f"  full model (number seen):      acc {c['full_all']['acc']:.3f}  snap {c['full_all']['snap']:.3f}")
    print(f"  fold model (number held out):  acc {c['unseen_all']['acc']:.3f}  snap {c['unseen_all']['snap']:.3f}"
          f"  {c['unseen_all']['snap_ci']}")
    print(f"  seen-number control: fold models {c['seen_by_fold_models']['snap']:.3f} vs full "
          f"{c['full_seen_ctrl']['snap']:.3f} (same crops)")
    for sp in ("val", "test"):
        print(f"  {sp}: full {c[f'full_{sp}']['snap']:.3f}  unseen {c[f'unseen_{sp}']['snap']:.3f}")
    for k in [k for k in range(len(folds)) if f'unseen_f{k}' in c]:
        print(f"  fold {k} {sorted(held[k])}: full {c[f'full_on_f{k}']['snap']:.3f} -> unseen {c[f'unseen_f{k}']['snap']:.3f}"
              f" (n={c[f'unseen_f{k}']['n']})")
    print("  by height (snap full/unseen):", " ".join(
        f"{b}:{c['full_h' + b]['snap']:.2f}/{c['unseen_h' + b]['snap']:.2f}(n{c['unseen_h' + b]['n']})"
        for b in [hbin(x) for x in HBINS[:-1]] if "unseen_h" + b in c))
    for sp in ("val", "test"):
        t = res["track"][sp]
        f6 = [x for x in t["full"]["curve"] if x["tau"] == 0.6][0]
        u6 = [x for x in t["unseen"]["curve"] if x["tau"] == 0.6][0]
        print(f"  tracks {sp}: top1 full {t['full']['top1']:.3f} unseen {t['unseen']['top1']:.3f}; tau 0.6 "
              f"full {f6['coverage']:.2f}/{f6['precision']:.3f} unseen {u6['coverage']:.2f}/{u6['precision']:.3f}")
    th = res["train_held_unseen"]
    print("  unseen number, seen event (train split; snap / height-std):",
          " ".join(f"{k}:{v['snap']:.3f}/{v['snap_std_height']:.3f}(n{v['n']})" for k, v in th.items()
                   if not k.startswith("n")))
    print("  style proxy (full model, snap / height-std):",
          " ".join(f"{k}:{v['snap']:.3f}/{v['snap_std_height']:.3f}(n{v['n']})" for k, v in res["style_proxy"].items()))
    for e in a.extra_full.split(","):
        if e:
            print(f"  extra full run {e}: snap {c['extra_' + e]['snap']:.3f} acc {c['extra_' + e]['acc']:.3f}")


if __name__ == "__main__":
    main()
