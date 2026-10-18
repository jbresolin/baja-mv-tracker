"""WP10b: comparison table across recognizer families (reads the outputs of evaluate.py, heldout_eval.py, export_onnx.py).

For each family F: F_full/eval/metrics.json (evaluate.py --name F_full --ckpt last.pt), heldout_F.json
(heldout_eval.py --family F), F_full/bench.json (export_onnx.py --name F_full; uses best.pt -> pass --ckpt to export
the evaluated checkpoint, see export_onnx.py). Track precision / coverage on test at the tau picked on val for 97%
precision (evaluate.py's tau_from_val).
usage: python students/recog/wp10b_table.py gx_base gx_spl pq_base ...
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS  # noqa: E402


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def row(fam):
    full = RUNS / f"{fam}_full"
    m = json.load(open(full / "eval" / "metrics.json")) if (full / "eval" / "metrics.json").exists() else {}
    h = json.load(open(RUNS / f"heldout_{fam}.json")) if (RUNS / f"heldout_{fam}.json").exists() else {}
    b = json.load(open(full / "bench.json")) if (full / "bench.json").exists() else {}
    ms = {(x["kind"], x["threads"], x["batch"]): x["ms_per_crop"] for x in b.get("bench", [])}
    tau = g(m, "tau_from_val", "0.97")
    tc = [c for c in g(m, "test", "track", "curve", default=[]) if c["tau"] == tau]
    vc = [c for c in g(m, "val", "track", "curve", default=[]) if c["tau"] == tau]
    th = g(h, "train_held_unseen", default={})
    f = lambda x: f"{100 * x:.1f}" if isinstance(x, (int, float)) else "-"  # noqa: E731
    return [fam, f"{b.get('params', 0) / 1e6:.2f}M" if b else "-",
            f"{ms.get(('ort', 1, 1), float('nan')):.1f} / {ms.get(('ort', 4, 1), float('nan')):.1f}" if ms else "-",
            f"{f(g(m, 'val', 'crop', 'acc_strip0'))} / {f(g(m, 'val', 'crop', 'acc_roster_snap'))}",
            f"{f(g(m, 'test', 'crop', 'acc_strip0'))} / {f(g(m, 'test', 'crop', 'acc_roster_snap'))}",
            f"{tau}: val {f(vc[0]['coverage']) if vc else '-'}/{f(vc[0]['precision']) if vc else '-'}, "
            f"test {f(tc[0]['coverage']) if tc else '-'}/{f(tc[0]['precision']) if tc else '-'}",
            f"{f(g(h, 'crop', 'full_all', 'snap'))} -> {f(g(h, 'crop', 'unseen_all', 'snap'))}",
            f"{f(g(th, '2digit_le16', 'snap'))} / {f(g(th, '2digit_17_39', 'snap'))} / {f(g(th, '2digit_ge40', 'snap'))}"
            f" (fold3 {f(g(th, 'fold3', 'snap'))})"]


def main():
    hdr = ["family", "params", "ORT ms/crop 1 thr / 4 thr (batch 1)", "val crop acc / snap", "test crop acc / snap",
           "test track cov / prec at tau(val, 97%)", "val+test snap: seen -> held-out number",
           "unseen number, seen event (snap): 10-16 / 17-39 / >=40"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for fam in sys.argv[1:]:
        print("| " + " | ".join(row(fam)) + " |")


if __name__ == "__main__":
    main()
