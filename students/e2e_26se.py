"""Fully student end-to-end on 26SE: student car detector passes -> car crops of each pass -> number-panel detector ->
recognizer -> boxed vote with the val-picked rule -> 26SE roster -> match to the human laps. Evaluation only.

usage:
  # 1) passes + crops from the student (GPU; reuse the timeline of an existing eval run, track at native resolution):
  mkdir out/eval_cars_s640_e2e && cp out/eval_cars_s640/tl_*.csv out/eval_cars_s640_e2e/
  python students/eval_26se.py --name cars_s640_e2e --weights students/runs/cars_s640/weights/best.pt \
      --stages track,metrics --crops --track-width 0
  # 2) identity (CPU):
  CUDA_VISIBLE_DEVICES=-1 python students/e2e_26se.py --eval out/eval_cars_s640_e2e \
      --panel-weights students/runs/num_n320/weights/best.pt --recog pq_spl_full

Crops per pass: every saved crop of the pass's tracks (out/eval_<name>/trk_<file>/<event>/t###_f#####.jpg), evenly
subsampled to --max-crops (as harness_26se.py --all-crops). Panel boxes: conf 0.1, +5% pad, imgsz 320 (picked on val,
LESSONS WP9). Vote: students/recog/common.vote_boxed with tau / k / score kind from <recog>/eval/panel_tau.json
(picked on 25SU val by WP10b). Matching: the student's passes vs contagem_volta (+80 s, +-6 s), id-preferring like
analyze.py. Output: <eval>/e2e_<recog>[_<tag>]/{passes.csv, metrics.json, panel_boxes.csv}.
"""
import argparse
import ast
import json
import sys
import time
from pathlib import Path

import torch  # before ultralytics

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lapvision"))
sys.path.insert(0, str(ROOT / "students"))
sys.path.insert(0, str(ROOT / "students" / "recog"))  # first: its `evaluate` / `common` win
from common import RUNS as RECOG_RUNS, RosterScorer, load_any, read_roster, vote_boxed  # noqa: E402
from evaluate import wilson  # noqa: E402  (students/recog/evaluate.py)
from harness_26se import read_panels  # noqa: E402
from panel_val import enlarge  # noqa: E402
import eval_26se as V  # noqa: E402  (FILES / windows only: its `evaluate` import resolves to students/recog's here)
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("lv_evaluate", ROOT / "lapvision" / "evaluate.py")
E = importlib.util.module_from_spec(_spec)  # lapvision/evaluate.py (GT laps, matching); name clash with recog/evaluate
_spec.loader.exec_module(E)


def pass_crops(evaldir, row, max_crops):
    paths = []
    for tk in ast.literal_eval(row.tracks):
        ev, tid = tk.rsplit(":", 1)
        paths += sorted(str(p) for p in (evaldir / row.src / ev).glob("t%03d_f*.jpg" % int(tid)))
    if len(paths) > max_crops:
        idx = np.linspace(0, len(paths) - 1, max_crops).round().astype(int)
        paths = [paths[i] for i in sorted(set(idx))]
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--panel-weights", required=True)
    ap.add_argument("--panel-imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--pad", type=float, default=0.05)
    ap.add_argument("--recog", default="pq_spl_full")
    ap.add_argument("--max-crops", type=int, default=30)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from ultralytics import YOLO
    det = YOLO(a.panel_weights, task="detect")  # .pt or a (dynamic-shape) .onnx export
    rec = load_any(RECOG_RUNS / a.recog, a.threads)
    roster = read_roster("26SE")
    sc = RosterScorer(roster)
    pt = json.load(open(RECOG_RUNS / a.recog / "eval" / "panel_tau.json"))["chosen"]
    tau, k, kind = pt["tau"], pt["min_support"], pt.get("score", "mean")
    od = a.eval / ("e2e_%s%s" % (a.recog, ("_" + a.tag) if a.tag else ""))
    od.mkdir(exist_ok=True)

    P = pd.read_csv(a.eval / "passes.csv")  # eval_26se metrics stage: passes after 09:18:30, gt_car by time-only match
    rows, brows, t0 = [], [], time.time()
    t_det = t_rec = 0.0
    n_crops = n_boxes = 0
    for i, r in enumerate(P.itertuples()):
        Ps, HB = [], []
        for p in pass_crops(a.eval, r, a.max_crops):
            im = cv2.imread(p)
            if im is None:
                continue
            n_crops += 1
            H, W = im.shape[:2]
            s = time.time()
            res = det.predict(im, conf=a.conf, imgsz=a.panel_imgsz, device="cpu", verbose=False)[0]
            t_det += time.time() - s
            bx = [enlarge(b, a.pad, W, H) for b in res.boxes.xyxy.tolist()]
            for b, c in zip(bx, res.boxes.conf.tolist()):
                brows.append(dict(path=p, x1=b[0], y1=b[1], x2=b[2], y2=b[3], conf=c))
            n_boxes += len(bx)
            s = time.time()
            Ps.append(read_panels(rec, im, bx, sc))
            t_rec += time.time() - s
            HB.append(len(bx) > 0)
        j, score, s2, nb, sup = vote_boxed(Ps, HB, score=kind) if Ps else (-1, 0.0, 0.0, 0, 0)
        pred = roster[j] if j >= 0 else None
        rows.append(dict(src=r.src, pass_id=r.pass_id, t0=r.t0, t1=r.t1, time=r.time, n_crops=len(Ps), n_boxed=nb,
                         support=sup, pred=pred, score=score,
                         pred_rule=pred if (pred is not None and score >= tau and sup >= k) else None))
        if i % 25 == 0:
            print(i, len(P), "%.0f s" % (time.time() - t0), flush=True)
    D = pd.DataFrame(rows)
    pd.DataFrame(brows).to_csv(od / "panel_boxes.csv", index=False)

    # match the student's passes to the human laps, id-preferring (as analyze.py), once per identity column
    win = V.gt_windows([k_ for k_ in V.ORDER if k_ in set(D.src.str[-4:])])
    g = E.gt_laps(V.VALID_FROM, 86400)
    g = g[np.any([(g.t >= s) & (g.t <= e) for s, e in win], axis=0)].reset_index(drop=True)
    res = dict(eval=str(a.eval), panel=a.panel_weights, recog=a.recog, rule=dict(tau=tau, min_support=k, score=kind),
               crops=n_crops, boxes=n_boxes, crops_per_pass=round(n_crops / max(len(D), 1), 1),
               ms_panel_per_crop=round(1000 * t_det / max(n_crops, 1), 1),
               ms_recog_per_crop=round(1000 * t_rec / max(n_crops, 1), 1))
    for col, name in [("pred", "no_abstain"), ("pred_rule", "val_rule")]:
        rep, pairs = E.report(D, g, tcol="t0", idcol=col)
        ok = rep["id_correct"]
        rep = {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in rep.items()}
        rep["id_coverage_of_matched"] = round(rep["id_attempted"] / max(rep["matched"], 1), 4)
        rep["id_precision_ci"] = [round(float(x), 3) for x in wilson(ok, rep["id_attempted"])]
        res[name] = rep
        if name == "val_rule":
            D["gt_car"] = np.nan
            for i_, j_ in pairs:
                D.loc[i_, "gt_car"] = g.loc[j_, "car"]
    res["teacher_pipeline"] = dict(recall=0.933, precision=0.975, end_to_end=0.679, note="YOLO-World + VLM, LESSONS WP0")
    D.to_csv(od / "passes.csv", index=False)
    (od / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    for name in ["no_abstain", "val_rule"]:
        r_ = res[name]
        print("%-10s passes %d, matched %d/%d (recall %.3f, precision %.3f), id answered %d (%.0f%% of matched), "
              "id precision %.3f %s, end-to-end %.3f" % (name, r_["detected"], r_["matched"], r_["gt_laps"], r_["recall"],
                                                       r_["precision"], r_["id_attempted"], 100 * r_["id_coverage_of_matched"],
                                                       r_["id_acc_when_read"], r_["id_precision_ci"], r_["end_to_end"]))
    print(json.dumps({kk: v for kk, v in res.items() if kk not in ("no_abstain", "val_rule")}, default=float))


if __name__ == "__main__":
    main()
