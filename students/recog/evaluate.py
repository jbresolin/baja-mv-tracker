"""WP10 evaluation on the digits val (25SU) / test (24SU) splits.

usage: CUDA_VISIBLE_DEVICES=-1 python students/recog/evaluate.py --name real [--splits val,test] [--sheet]

Reports (students/runs/recog/<name>/eval/{metrics.json, report.md, errors_<split>.jpg}):
* crop level ('ok' bucket = box >= 16 px): greedy exact match on text_read, accuracy after stripping leading zeros,
  roster-snapped accuracy (argmax of P(number | crop) over the event roster, CTC likelihood of "n" / "0n"),
  accuracy by box-height bin, digit confusion (same-length reads), tiny bucket, false reads on "no number" windows;
* track level: soft vote (mean over the track's crops of the roster posterior), abstain below a threshold;
  precision / coverage curve. Thresholds are picked on val and applied unchanged to test.
Labels are the teacher's consensus (VLM, ~97% vs broadcast overlays), and the crops are those whose grounded read
matched it, i.e. readable ones: these numbers are recognizer-vs-teacher on teacher-readable crops.
"""
import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment import plain  # noqa: E402
from common import RUNS, RosterScorer, batch_tensor, greedy, load_model, read_roster, strip0, vote  # noqa: E402

HBINS = [16, 24, 32, 48, 64, 96, 1e9]
EVENTS = {"val": "25SU", "test": "24SU", "train": None}
TAUS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@torch.no_grad()
def run(model, items, roster, bs=512):
    sc = RosterScorer(roster)
    res = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        X = batch_tensor([plain(it["img"], it["box"]) for it in chunk])
        lp = model(X)
        g = greedy(lp)
        P, po = sc(lp)
        for it, (s, pr), p, o in zip(chunk, g, P.numpy(), po.numpy()):
            res.append(dict(it=it, pred=s, p_path=pr, P=p, p_other=float(o)))
    return res


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def crop_metrics(res, roster):
    pos = [r for r in res if not r["it"]["neg"] and r["it"]["bucket"] == "ok"]
    tiny = [r for r in res if r["it"]["bucket"] == "tiny"]
    neg = [r for r in res if r["it"]["neg"]]
    m = {}

    def acc(rs):
        return float(np.mean([strip0(r["pred"]) == r["it"]["number"] for r in rs])) if rs else float("nan")

    def snap_acc(rs):
        return float(np.mean([roster[int(np.argmax(r["P"]))] == int(r["it"]["number"]) for r in rs])) if rs else float("nan")
    m["n"] = len(pos)
    m["exact_text_read"] = float(np.mean([r["pred"] == r["it"]["text"] for r in pos]))
    m["acc_strip0"] = acc(pos)
    m["acc_roster_snap"] = snap_acc(pos)
    m["tiny_n"], m["tiny_acc"], m["tiny_snap"] = len(tiny), acc(tiny), snap_acc(tiny)
    m["by_height"] = []
    for lo, hi in zip(HBINS[:-1], HBINS[1:]):
        rs = [r for r in pos if lo <= r["it"]["box_h"] < hi]
        m["by_height"].append(dict(bin=f"{lo}-{hi if hi < 1e8 else 'inf'}", n=len(rs), acc=acc(rs), snap=snap_acc(rs)))
    # crop-level snapped precision / coverage (crop answered if max P >= tau)
    m["crop_curve"] = []
    for t in TAUS:
        ans = [r for r in pos if r["P"].max() >= t]
        ok = sum(roster[int(np.argmax(r["P"]))] == int(r["it"]["number"]) for r in ans)
        m["crop_curve"].append(dict(tau=t, coverage=len(ans) / max(len(pos), 1), precision=ok / max(len(ans), 1)))
    # digit confusion on same-length reads (raw strings vs text_read)
    conf = np.zeros((10, 10), int)
    length_err = Counter()
    for r in pos:
        p, t = r["pred"], r["it"]["text"]
        if len(p) == len(t):
            for a, b in zip(t, p):
                conf[int(a), int(b)] += 1
        else:
            length_err["short" if len(p) < len(t) else "long"] += 1
    m["confusion"] = conf.tolist()
    m["length_errors"] = dict(length_err)
    off = [(int(conf[i, j]), f"{i}->{j}") for i in range(10) for j in range(10) if i != j and conf[i, j]]
    m["top_confusions"] = [f"{s} x{c}" for c, s in sorted(off, reverse=True)[:12]]
    m["per_digit_recall"] = {str(i): round(conf[i, i] / max(conf[i].sum(), 1), 3) for i in range(10)}
    m["per_number"] = {}
    for num in sorted({r["it"]["number"] for r in pos}, key=int):
        rs = [r for r in pos if r["it"]["number"] == num]
        m["per_number"][num] = dict(n=len(rs), acc=round(acc(rs), 3), snap=round(snap_acc(rs), 3))
    m["neg_n"] = len(neg)
    m["neg_false_read"] = float(np.mean([r["pred"] != "" for r in neg])) if neg else float("nan")
    for t in (0.5, 0.7, 0.9):
        m[f"neg_snap_ge_{t}"] = float(np.mean([r["P"].max() >= t for r in neg])) if neg else float("nan")
    return m


def track_metrics(res, roster):
    pos = [r for r in res if not r["it"]["neg"]]          # ok + tiny crops of the track (all number crops)
    by = defaultdict(list)
    for r in pos:
        by[(r["it"]["video"], r["it"]["track"])].append(r)
    tracks = []
    for k, rs in by.items():
        j, s, s2 = vote([r["P"] for r in rs])
        # session-1-like baseline: majority over greedy strings snapped to the roster
        reads = [int(strip0(r["pred"])) for r in rs if r["pred"] and strip0(r["pred"]) and int(strip0(r["pred"])) in roster]
        mj = Counter(reads).most_common(1)
        tracks.append(dict(key=k, n=len(rs), gt=int(rs[0]["it"]["number"]), pred=roster[j], score=s, margin=s - s2,
                           maj=mj[0][0] if mj else None, maj_share=(mj[0][1] / len(rs)) if mj else 0.0))
    m = dict(n_tracks=len(tracks), crops_per_track=float(np.mean([t["n"] for t in tracks])) if tracks else 0)
    m["curve"] = []
    for t in TAUS:
        ans = [x for x in tracks if x["score"] >= t]
        ok = sum(x["pred"] == x["gt"] for x in ans)
        lo, hi = wilson(ok, len(ans))
        m["curve"].append(dict(tau=t, coverage=len(ans) / max(len(tracks), 1), precision=ok / max(len(ans), 1),
                               n=len(ans), ci=[round(lo, 3), round(hi, 3)]))
    ans = [x for x in tracks if x["maj"] is not None]
    m["majority_greedy"] = dict(coverage=len(ans) / max(len(tracks), 1),
                                precision=sum(x["maj"] == x["gt"] for x in ans) / max(len(ans), 1))
    m["by_ncrops"] = {}
    for n in sorted({x["n"] for x in tracks}):
        xs = [x for x in tracks if x["n"] == n]
        m["by_ncrops"][str(n)] = dict(tracks=len(xs), acc=sum(x["pred"] == x["gt"] for x in xs) / len(xs))
    return m, tracks


def pick_tau(curve, target):
    """lowest tau whose precision >= target (max coverage)."""
    ok = [c for c in curve if c["precision"] >= target]
    return min(ok, key=lambda c: c["tau"])["tau"] if ok else None


def sheet(res, roster, path, n=40):
    errs = [r for r in res if not r["it"]["neg"] and r["it"]["bucket"] == "ok"
            and roster[int(np.argmax(r["P"]))] != int(r["it"]["number"])]
    rng = np.random.default_rng(0)
    if len(errs) > n:
        errs = [errs[i] for i in sorted(rng.choice(len(errs), n, replace=False))]
    tiles = []
    for r in errs:
        it = r["it"]
        x1, y1, x2, y2 = it["box"]
        im = it["img"]
        a = cv2.resize(im, (int(im.shape[1] * 96 / im.shape[0]), 96)) if im.shape[0] else im
        a = a[:, :200]
        t = np.full((150, 200, 3), 255, np.uint8)
        t[:a.shape[0], :a.shape[1]] = a
        s = 96 / im.shape[0]
        cv2.rectangle(t, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), (0, 255, 0), 1)
        inp = plain(it["img"], it["box"])[:, :, ::-1]
        t[98:130, 0:96] = inp
        snap = roster[int(np.argmax(r["P"]))]
        cv2.putText(t, f"gt {it['text']} rd '{r['pred']}' sn {snap}", (98, 112), 0, 0.33, (0, 0, 255), 1)
        cv2.putText(t, f"P {r['P'].max():.2f} h{it['box_h']:.0f}", (98, 126), 0, 0.33, (0, 0, 255), 1)
        cv2.putText(t, it["name"][:30], (2, 145), 0, 0.3, (80, 80, 80), 1)
        tiles.append(t)
    while len(tiles) % 8:
        tiles.append(np.full((150, 200, 3), 255, np.uint8))
    if tiles:
        cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
    return len(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = RUNS / a.name
    model, _ = load_model(d / a.ckpt)
    out = d / "eval"
    out.mkdir(exist_ok=True)
    allm = {}
    for sp in a.splits.split(","):
        items = pickle.load(open(RUNS / f"cache_{sp}.pkl", "rb"))
        roster = read_roster(EVENTS[sp])
        res = run(model, items, roster)
        cm = crop_metrics(res, roster)
        tm, tracks = track_metrics(res, roster)
        allm[sp] = dict(crop=cm, track=tm)
        if a.sheet:
            sheet(res, roster, out / f"errors_{sp}.jpg")
        with open(out / f"tracks_{sp}.json", "w") as f:
            json.dump([{k: (list(v) if k == "key" else v) for k, v in t.items()} for t in tracks], f)
    if "val" in allm:
        for tgt in (0.95, 0.97, 0.98, 0.99):
            allm.setdefault("tau_from_val", {})[str(tgt)] = pick_tau(allm["val"]["track"]["curve"], tgt)
    json.dump(allm, open(out / "metrics.json", "w"), indent=1, default=float)
    for sp, m in allm.items():
        if sp == "tau_from_val":
            print("tau picked on val:", m)
            continue
        c, t = m["crop"], m["track"]
        print(f"== {sp}: crops {c['n']}  exact {c['exact_text_read']:.3f}  acc(strip0) {c['acc_strip0']:.3f}  "
              f"snap {c['acc_roster_snap']:.3f}  tiny {c['tiny_acc']:.3f}/{c['tiny_snap']:.3f} (n={c['tiny_n']})  "
              f"neg false read {c['neg_false_read']:.3f} snapP>=.5 {c['neg_snap_ge_0.5']:.3f}")
        print("   by height:", " ".join(f"{b['bin']}:{b['acc']:.2f}/{b['snap']:.2f}(n{b['n']})" for b in c["by_height"]))
        print("   top confusions:", c["top_confusions"], "length errors", c["length_errors"])
        print("   per number (n, snap acc):", " ".join(f"{k}:{v['n']}/{v['snap']:.2f}" for k, v in c["per_number"].items()))
        print(f"   tracks {t['n_tracks']} ({t['crops_per_track']:.1f} crops/track), majority-greedy "
              f"{t['majority_greedy']['coverage']:.3f} cov / {t['majority_greedy']['precision']:.3f} prec")
        print("   track curve:", " ".join(f"t{x['tau']}:{x['coverage']:.2f}/{x['precision']:.3f}" for x in t["curve"]))


if __name__ == "__main__":
    main()
