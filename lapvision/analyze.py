"""Stage 5: identity voting per pass + evaluation against human GT.

usage: python analyze.py [crop_dirs...]   (default: all out/crops_*)
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import clock
import evaluate as E
import gt
import passes as P

OUT = Path(__file__).resolve().parent.parent / "out"
ROSTER = set(gt.equipes().keys())
VALID_FROM = 9 * 3600 + 18 * 60 + 30   # camera set-up finished (see LESSONS.md)
MIN_VOTES = 2


def vote_vlm(r):
    nums = [int(x) for x in r.vlm.dropna() if int(x) in ROSTER]
    c = Counter(nums)
    if not c:
        return None, 0.0, 0
    n, k = c.most_common(1)[0]
    return (n if k >= MIN_VOTES else None), k / len(r), k


def vote_easy(r):
    s = Counter()
    for e in r.easy.dropna():
        for item in str(e).split("|"):
            if ":" in item:
                t, c = item.rsplit(":", 1)
                if t.isdigit() and int(t) in ROSTER:
                    s[int(t)] += float(c)
    if not s:
        return None, 0.0
    n, v = s.most_common(1)[0]
    return (n if v >= 1.0 else None), v


def load(dirs):
    PS, RS = [], []
    for d in dirs:
        if not (d / "passes.csv").exists() or not (d / "reads.csv").exists():
            print("skip", d)
            continue
        p = P.build(pd.read_csv(d / "tracks.csv"))  # rebuild with current pass rules
        r = pd.read_csv(d / "reads.csv")
        # reads were taken per (event, track); re-attach them to the rebuilt passes
        owner = {k: pid for pid, ks in zip(p.pass_id, p.tracks) for k in ks}
        r["pass_id"] = (r.event + ":" + r.track.astype(str)).map(owner)
        r = r[r.pass_id.notna()]
        p["src"] = d.name
        r["src"] = d.name
        PS.append(p)
        RS.append(r)
    return pd.concat(PS, ignore_index=True), pd.concat(RS, ignore_index=True)


def main():
    dirs = [Path(x) for x in sys.argv[1:]] or sorted(OUT.glob("crops_[0-9]*"))
    ps, rd = load(dirs)
    rows = []
    for (src, pid), r in rd.groupby(["src", "pass_id"]):
        v, vs, vk = vote_vlm(r)
        e, es = vote_easy(r)
        rows.append(dict(src=src, pass_id=pid, vlm_car=v, vlm_share=vs, vlm_k=vk, easy_car=e, easy_score=es,
                         n_crops=len(r)))
    ids = pd.DataFrame(rows)
    ps = ps.merge(ids, on=["src", "pass_id"], how="left")
    ps["combo_car"] = ps.vlm_car.where(ps.vlm_car.notna(), ps.easy_car)
    ps = ps[ps.t0 >= VALID_FROM].reset_index(drop=True)

    t0, t1 = ps.t0.min() - 10, ps.t1.max() + 10
    # restrict GT to time covered by the analysed files
    g = E.gt_laps(max(t0, VALID_FROM), t1)
    print("passes %d, GT laps in window %d" % (len(ps), len(g)))
    results = {}
    for col in ["vlm_car", "easy_car", "combo_car"]:
        rep, pairs = E.report(ps, g, tcol="t0", idcol=col)
        results[col] = rep
        if col == "vlm_car":
            best_pairs = pairs
    print(pd.DataFrame(results).T.round(3).to_string())

    # per-pass table with GT match
    ps["gt_car"] = np.nan
    ps["gt_dt"] = np.nan
    for i, j in best_pairs:
        ps.loc[i, "gt_car"] = g.loc[j, "car"]
        ps.loc[i, "gt_dt"] = ps.loc[i, "t0"] - g.loc[j, "t"]
    ps["time"] = ps.tcross.map(clock.fmt)
    ps.to_csv(OUT / "passes_all.csv", index=False)

    # per-car breakdown
    m = ps[ps.gt_car.notna()]
    per = m.groupby("gt_car").agg(passes=("pass_id", "size"),
                                  vlm_ok=("vlm_car", lambda s: int((s == m.loc[s.index, "gt_car"]).sum())),
                                  vlm_wrong=("vlm_car", lambda s: int((s.notna() & (s != m.loc[s.index, "gt_car"])).sum())),
                                  easy_ok=("easy_car", lambda s: int((s == m.loc[s.index, "gt_car"]).sum())))
    gl = g.groupby("car").size().rename("gt_laps")
    per = per.join(gl, how="outer").fillna(0).astype(int)
    print(per.to_string())
    per.to_csv(OUT / "per_car.csv")

    # wrong reads (for the log)
    w = m[m.vlm_car.notna() & (m.vlm_car != m.gt_car)]
    print("\nVLM wrong ids:")
    print(w[["src", "pass_id", "time", "gt_car", "vlm_car", "vlm_share", "easy_car"]].to_string())
    fp = ps[ps.gt_car.isna()]
    print("\nunmatched passes (FP candidates): %d" % len(fp))
    print(fp[["src", "pass_id", "time", "tracks", "n", "exited", "vlm_car", "easy_car"]].head(40).to_string())


if __name__ == "__main__":
    main()
