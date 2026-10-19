"""WP11 validation on the 26SE replay (evaluation only; nothing here is used at run time, nothing is tuned on 26SE).

usage:
  # 1) replay (one process per file, as fast as possible; the offline e2e read native-resolution crops -> max_width 0):
  python rt/run.py --source "2026-10-17 09-09-26.mkv" --event 26SE --out out/rt/replay_n/0909.jsonl --quiet \
      --set capture.max_width=0            # ... and 09-56-32, 11-09-03, 11-29-49
  # 2) metrics, same matching as students/e2e_26se.py (lapvision/evaluate.report, +80 s, +-6 s on t0, after 09:18:30):
  python rt/eval_26se.py metrics out/rt/replay_n --offline out/eval_cars_n640_v2_e2e/e2e_pq_spl_full/passes.csv
  # 3) pass logic only: the offline 10 fps tracks (students/eval_26se.py --track-fps 10) through the online logic
  python rt/eval_26se.py passlogic out/eval_cars_n640_v2_t10
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lapvision"))
sys.path.insert(0, str(ROOT / "students"))
import evaluate as E  # noqa: E402  (lapvision/evaluate.py)
import gt  # noqa: E402
import passes as PL  # noqa: E402  (lapvision/passes.py, the offline logic)

VALID_FROM = 9 * 3600 + 18 * 60 + 30
FILES = {"0909": (9 * 3600 + 9 * 60 + 26, 9 * 3600 + 56 * 60 + 8), "0956": (9 * 3600 + 56 * 60 + 32, 11 * 3600 + 9 * 60 + 2),
         "1109": (11 * 3600 + 9 * 60 + 3, 11 * 3600 + 29 * 60 + 47), "1129": (11 * 3600 + 29 * 60 + 49, 12 * 3600 + 10 * 60 + 10)}
ORDER = list(FILES)


def gt_windows(keys):
    """= students/eval_26se.gt_windows (file spans, extended to the next selected file's start)."""
    win = []
    for k in keys:
        s, e = FILES[k]
        i = ORDER.index(k)
        if i + 1 < len(ORDER) and ORDER[i + 1] in keys:
            e = FILES[ORDER[i + 1]][0]
        win.append((max(s, VALID_FROM), e))
    return win


def file_of(t):
    for k, (s, e) in FILES.items():
        if s - 1 <= t <= e + 1:
            return k
    return "gap"


def gt_laps(keys):
    g = E.gt_laps(VALID_FROM, 86400)
    win = gt_windows(keys)
    return g[np.any([(g.t >= s) & (g.t <= e) for s, e in win], axis=0)].reset_index(drop=True)


def load_runs(d):
    rows = []
    for f in sorted(Path(d).glob("*.jsonl")):
        if f.name.endswith(".lapapp.jsonl"):
            continue
        ends, ids, starts, provs = {}, {}, {}, {}
        for line in open(f, encoding="utf-8"):
            ev = json.loads(line)
            if ev["type"] == "pass_end":
                ends[ev["pass_id"]] = ev
            elif ev["type"] == "pass_identity":
                ids[ev["pass_id"]] = ev
            elif ev["type"] == "pass_start":
                starts[ev["pass_id"]] = ev
            elif ev["type"] == "pass_provisional":
                provs[ev["pass_id"]] = ev
        for pid, ev in ends.items():
            i = ids.get(pid, {})
            st, pv = starts.get(pid, {}), provs.get(pid)
            rows.append(dict(run=f.stem, pass_id=pid, t0=gt.tod(ev["t_enter"]), t1=gt.tod(ev["t_exit"]),
                             tcross=gt.tod(ev["t_cross"]), n=ev["n_frames"], dwell=ev["dwell_s"], exited=ev["exited"],
                             n_crops=i.get("n_crops"), n_boxed=i.get("n_boxed"), support=i.get("support"),
                             pred=i.get("number"), score=i.get("confidence"), auto=i.get("auto_confirm"),
                             latency=i.get("identity_latency_s"),
                             start_after_s=(st["t_video"] - st["t_enter"]) if st else None,
                             prov=pv["number"] if pv else None,
                             prov_after_enter_s=(pv["t_video"] - ev["t_enter"]) if pv and pv.get("t_video") else None,
                             prov_before_exit_s=(ev["t_exit"] - pv["t_video"]) if pv and pv.get("t_video") else None,
                             end_after_exit_s=ev["t_video"] - ev["t_exit"]))
    D = pd.DataFrame(rows)
    D["pred_rule"] = [p if a else None for p, a in zip(D.pred, D.auto)]
    D["file"] = D.t0.map(file_of)
    return D.sort_values("t0").reset_index(drop=True)


def rep(D, g, col):
    r, pairs = E.report(D, g, tcol="t0", idcol=col)
    r = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
    if "id_attempted" in r:
        r["id_coverage_of_matched"] = round(r["id_attempted"] / max(r["matched"], 1), 4)
    return r, pairs


def match_sets(A, B, tol=3.0):
    """greedy 1-1 time match of two pass lists on t0 (within tol s) -> list of (i, j)."""
    cand = sorted((abs(a - b), i, j) for i, a in enumerate(A.t0) for j, b in enumerate(B.t0) if abs(a - b) <= tol)
    ui, uj, out = set(), set(), []
    for _, i, j in cand:
        if i in ui or j in uj:
            continue
        ui.add(i); uj.add(j); out.append((i, j))
    return out


def cmd_metrics(a):
    D = load_runs(a.runs)
    keys = [k for k in ORDER if k in set(D.file)]
    D = D[D.t0 >= VALID_FROM].reset_index(drop=True)
    g = gt_laps(keys)
    if a.window:  # HH:MM:SS-HH:MM:SS (video clock): a replayed segment; passes and laps inside it only
        lo, hi = [sum(float(x) * m for x, m in zip(w.split(":"), (3600, 60, 1))) for w in a.window.split("-")]
        D = D[(D.t0 >= lo) & (D.t0 <= hi)].reset_index(drop=True)
        g = g[(g.t >= lo + 3) & (g.t <= hi - 3)].reset_index(drop=True)  # 3 s margin: no half-seen passes
    res = dict(runs=str(a.runs), passes=len(D), files=keys)
    res["detection"], pairs = rep(D, g, "__none__")
    res["no_abstain"], _ = rep(D, g, "pred")
    res["val_rule"], ipairs = rep(D, g, "pred_rule")
    D["gt_car"] = np.nan
    for i, j in ipairs:
        D.loc[i, "gt_car"] = g.loc[j, "car"]
    g["matched"] = False
    g.loc[[j for _, j in pairs], "matched"] = True
    g["file"] = g.t.map(file_of)
    res["per_file"] = {k: dict(gt_laps=int((g.file == k).sum()), passes=int((D.file == k).sum()),
                               matched=int(g[g.file == k].matched.sum())) for k in keys}
    res["false_passes"] = D[D.gt_car.isna()][["run", "pass_id", "t0", "n", "dwell", "exited"]].assign(
        t0=lambda x: x.t0.map(fmt)).to_dict("records")
    res["missed_laps"] = g[~g.matched][["car", "lap", "t"]].assign(t=lambda x: x.t.map(fmt)).to_dict("records")
    res["crops_per_pass_mean"] = round(float(D.n_crops.mean()), 1)
    res["pass_start_after_enter_s"] = dict(p50=float(D.start_after_s.median()), p95=float(D.start_after_s.quantile(.95)))
    res["pass_end_after_exit_s"] = dict(p50=float(D.end_after_exit_s.median()), p95=float(D.end_after_exit_s.quantile(.95)))
    M = D[D.gt_car.notna()]
    pv = M[M.prov.notna()]
    res["provisional"] = dict(
        matched_passes=len(M), answered=len(pv), coverage=round(len(pv) / max(1, len(M)), 4),
        precision=round(float((pv.prov == pv.gt_car).mean()), 4) if len(pv) else None,
        wrong=pv[pv.prov != pv.gt_car][["t0", "prov", "gt_car"]].assign(t0=lambda x: x.t0.map(fmt)).to_dict("records"),
        agrees_with_final=round(float((pv.prov == pv.pred).mean()), 4) if len(pv) else None,
        after_enter_s_p50=float(pv.prov_after_enter_s.median()) if len(pv) else None,
        after_enter_s_p90=float(pv.prov_after_enter_s.quantile(.9)) if len(pv) else None,
        before_exit_share=round(float((pv.prov_before_exit_s > 0).mean()), 4) if len(pv) else None)
    res["identity_latency_s"] = dict(p50=float(D.latency.median()), p95=float(D.latency.quantile(0.95)))
    if a.offline:
        O = pd.read_csv(a.offline)
        O = O[O.t0 >= VALID_FROM].reset_index(drop=True)
        O["pred"] = O["pred"].astype("Int64")
        O["pred_rule"] = O["pred_rule"].astype("Int64")
        mp = match_sets(D, O)
        same = sum(pd.notna(D.pred[i]) and pd.notna(O.pred[j]) and int(D.pred[i]) == int(O.pred[j]) for i, j in mp)
        auto_both = sum(pd.notna(D.pred_rule[i]) and pd.notna(O.pred_rule[j]) for i, j in mp)
        auto_on = sum(pd.notna(D.pred_rule[i]) and pd.isna(O.pred_rule[j]) for i, j in mp)
        auto_off = sum(pd.isna(D.pred_rule[i]) and pd.notna(O.pred_rule[j]) for i, j in mp)
        um_on = sorted(set(range(len(D))) - {i for i, _ in mp})
        um_off = sorted(set(range(len(O))) - {j for _, j in mp})
        res["vs_offline"] = dict(file=a.offline, offline_passes=len(O), common=len(mp), online_only=len(um_on),
                                 offline_only=len(um_off), same_number_on_common=same,
                                 auto_both=auto_both, auto_online_only=auto_on, auto_offline_only=auto_off,
                                 online_only_list=[dict(t0=fmt(D.t0[i]), n=int(D.n[i]), gt=D.gt_car[i]) for i in um_on],
                                 offline_only_list=[dict(t0=fmt(O.t0[j]), time=O.time[j] if "time" in O else None)
                                                    for j in um_off])
    out = Path(a.runs) / "metrics.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    D.to_csv(Path(a.runs) / "passes_eval.csv", index=False)
    for k in ("detection", "no_abstain", "val_rule"):
        r = res[k]
        print("%-10s passes %d matched %d/%d recall %.4f precision %.4f%s" % (
            k, r["detected"], r["matched"], r["gt_laps"], r["recall"], r["precision"],
            "" if k == "detection" else "  id answered %d (%.1f%% of matched) precision %.4f end-to-end %.4f" % (
                r["id_attempted"], 100 * r["id_coverage_of_matched"], r["id_acc_when_read"], r["end_to_end"])))
    print("per file", res["per_file"])
    if "vs_offline" in res:
        print("vs offline", {k: v for k, v in res["vs_offline"].items() if not k.endswith("_list")})
    print("->", out)


def fmt(t):
    return "%02d:%02d:%06.3f" % (t // 3600, t % 3600 // 60, t % 60)


# ------------------------------------------------------------------------------------------------ pass logic check
class _Fr:
    def __init__(self, t, idx, W, H):
        self.t, self.idx = t, idx
        self.img = np.zeros((H, W, 3), np.uint8)


def cmd_passlogic(a):
    """Offline tracks (students/eval_26se.py tracks.csv, ultralytics ByteTrack per event window) replayed frame by
    frame through rt.passlogic, vs lapvision/passes.build on the same tracks. Tracker death = 30 frames after the last
    row (ultralytics buffer) or the end of the event window (tracker reset)."""
    from rt import config as C
    from rt.passlogic import PassLogic
    cfg = C.load()
    cfg["identity"]["crop_min_h"] = 9.0  # no crops in this check
    res = {}
    for d in sorted(Path(a.evaldir).glob("trk_*")):
        tr = pd.read_csv(d / "tracks.csv")
        if not len(tr):
            continue
        off = PL.build(tr)
        evs = []
        pl = PassLogic(cfg, evs.append, None)
        W, H = int(tr.W.iloc[0]), int(tr.H.iloc[0])
        step = (tr.frame.diff().replace(0, np.nan).dropna().abs().min()) or 1
        for ev_id, te in tr.groupby("event", sort=False):
            pl.new_source()
            last = te.groupby("track").frame.max()
            frames = np.arange(te.frame.min(), te.frame.max() + step, step)
            byf = {f: g for f, g in te.groupby("frame")}
            dt_ = float(np.median(np.diff(sorted(te.tod.unique())))) if te.tod.nunique() > 1 else 0.1
            t_first = float(te.tod.min()) - float(te.frame.min()) / step * dt_
            for f in frames:
                g = byf.get(f)
                t = float(g.tod.iloc[0]) if g is not None else t_first + f / step * dt_
                outs = [] if g is None else [(int(r.track), (r.x1, r.y1, r.x2, r.y2), r.conf) for r in g.itertuples()]
                alive = {int(k) for k, v in last.items() if f <= v + 30 * step}
                fr = _Fr(t, int(f) + 10**7 * pl.epoch, 2, 2)
                fr.img = _Shape(H, W)
                pl.step(fr, outs, alive)
        pl.flush()
        on = pd.DataFrame([e for e in evs if e["type"] == "pass_end"])
        on = on.assign(t0=on.t_enter, t1=on.t_exit, tcross=on.t_cross)
        o2 = off.rename(columns={"tcross": "tcross"})
        mp = match_sets(on.reset_index(drop=True), o2.reset_index(drop=True), tol=0.01)
        same_t1 = sum(abs(on.t1.iloc[i] - o2.t1.iloc[j]) < 0.01 for i, j in mp)
        res[d.name] = dict(offline=len(off), online=len(on), same_t0=len(mp), same_t0_and_t1=same_t1,
                           online_only=[fmt(on.t0.iloc[i]) for i in sorted(set(range(len(on))) - {i for i, _ in mp})],
                           offline_only=[fmt(o2.t0.iloc[j]) for j in sorted(set(range(len(o2))) - {j for _, j in mp})])
        print(d.name, json.dumps(res[d.name], default=int))
    (Path(a.evaldir) / "rt_passlogic_check.json").write_text(json.dumps(res, indent=1, default=int))


class _Shape:
    """stand-in image (only .shape is used when crops are disabled)"""

    def __init__(self, H, W):
        self.shape = (H, W, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("metrics")
    m.add_argument("runs")
    m.add_argument("--offline", default=None, help="students/e2e_26se.py passes.csv to compare pass by pass")
    m.add_argument("--window", default=None, help="HH:MM:SS-HH:MM:SS: score only this replayed segment")
    p = sp.add_parser("passlogic")
    p.add_argument("evaldir")
    a = ap.parse_args()
    {"metrics": cmd_metrics, "passlogic": cmd_passlogic}[a.cmd](a)


if __name__ == "__main__":
    main()
