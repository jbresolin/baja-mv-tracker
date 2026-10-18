"""Compare detected passes (with identities) to the human consolidated laps.

GT: contagem_volta (consolidated laps) and contagem_passagem (raw human entries),
shifted by CLOCK_OFFSET (video PC clock - server clock, measured ~ +80 s).
"""
import numpy as np
import pandas as pd
import gt

CLOCK_OFFSET = 80.0
TOL = 6.0


def gt_laps(t0, t1, offset=CLOCK_OFFSET):
    rows = [dict(car=x["car"], lap=x["lap"], t=gt.tod(x["ts"]) + offset, status=x["status"],
                 disp=x["disp_ms"]) for x in gt.voltas()]
    g = pd.DataFrame(rows)
    return g[(g.t >= t0) & (g.t <= t1)].sort_values("t").reset_index(drop=True)


def match(passes, g, tcol="t0", tol=TOL, idcol="car_pred"):
    """Greedy matching: first pairs with same id within tol, then any within tol."""
    used_p, used_g, pairs = set(), set(), []
    cand = []
    for i, p in passes.iterrows():
        for j, q in g.iterrows():
            dt = p[tcol] - q.t
            if abs(dt) <= tol:
                same = idcol in p and p[idcol] == q.car
                cand.append((0 if same else 1, abs(dt), i, j))
    for _, adt, i, j in sorted(cand):
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); pairs.append((i, j))
    return pairs


def report(passes, g, tcol="t0", idcol="car_pred"):
    pairs = match(passes, g, tcol, idcol=idcol)
    n_p, n_g, n_m = len(passes), len(g), len(pairs)
    out = dict(gt_laps=n_g, detected=n_p, matched=n_m, recall=n_m / max(1, n_g),
               precision=n_m / max(1, n_p))
    if idcol in passes:
        ok = sum(passes.loc[i, idcol] == g.loc[j, "car"] for i, j in pairs)
        read = sum(pd.notna(passes.loc[i, idcol]) for i, j in pairs)
        out.update(id_attempted=int(read), id_correct=int(ok),
                   id_acc_when_read=ok / max(1, read), end_to_end=ok / max(1, n_g))
    dts = [passes.loc[i, tcol] - g.loc[j, "t"] for i, j in pairs]
    if dts:
        out.update(dt_median=float(np.median(dts)), dt_p10=float(np.percentile(dts, 10)),
                   dt_p90=float(np.percentile(dts, 90)))
    return out, pairs
