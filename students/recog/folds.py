"""WP10b: fixed held-out-number folds (the unseen-number protocol). Writes students/recog/folds/numbers_v1.json.

Each fold removes ~10 numbers entirely from training: every crop of those numbers (and the "no number" windows cut from
the same car crops) in every training event. Val (25SU) and test (24SU) only contain numbers 1-16, so each fold takes
5-6 of 1-16 (balanced by val+test crop count, >= 2 one-digit and >= 2 two-digit numbers) plus 5 of the train-only
numbers 17-61 (balanced by train crop count). Every digit still occurs in the remaining training numbers of every fold
(>= 30% of its crop-weighted count).
Fold 3 = the remaining training numbers >= 40 (high-number case).
Deterministic (seeded search); re-running reproduces the committed file. Never change v1 once results exist: add v2.

usage: python students/recog/folds.py [--write]
"""
import argparse
import itertools
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "students" / "runs" / "recog"
OUT = HERE / "folds" / "numbers_v1.json"


def counts(split):
    it = pickle.load(open(RUNS / f"cache_{split}.pkl", "rb"))
    return Counter(int(i["number"]) for i in it if not i["neg"] and i["bucket"] == "ok")


def digit_counts(numbers):
    """crop-weighted digit counts of the training labels (text_read, so side-plate leading zeros count)"""
    it = pickle.load(open(RUNS / "cache_train.pkl", "rb"))
    return Counter(ch for i in it if not i["neg"] and i["bucket"] == "ok" and int(i["number"]) in numbers
                   for ch in i["text"])


def digit_ok(held, tr, full, min_share=0.3):
    left = digit_counts([n for n in tr if n not in held])
    return min(left[str(d)] / full[str(d)] for d in range(10)) >= min_share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    tr, va, te = counts("train"), counts("val"), counts("test")
    vt = Counter(va) + Counter(te)
    low = sorted(n for n in tr if n <= 16)
    high = sorted(n for n in tr if n > 16)
    rng = random.Random(0)
    full = digit_counts(list(tr))
    best = None
    for _ in range(4000):
        p = low[:]
        rng.shuffle(p)
        g = [sorted(p[0:5]), sorted(p[5:10]), sorted(p[10:16])]
        if any(sum(n < 10 for n in x) < 2 or sum(n >= 10 for n in x) < 2 for x in g):
            continue
        s = [sum(vt[n] for n in x) for x in g]
        cost = max(s) - min(s)
        if best is not None and cost >= best[0]:
            continue
        if not all(digit_ok(x, tr, full, 0.35) for x in g):
            continue
        if best is None or cost < best[0]:
            best = (cost, g)
    g_low = best[1]
    best = None
    for _ in range(4000):
        p = high[:]
        rng.shuffle(p)
        g = [sorted(p[0:5]), sorted(p[5:10]), sorted(p[10:15])]
        s = [sum(tr[n] for n in x) for x in g]
        cost = max(s) - min(s)
        if best is not None and cost >= best[0]:
            continue
        if not all(digit_ok(x + g_low[k], tr, full, 0.3) for k, x in enumerate(g)):
            continue
        if best is None or cost < best[0]:
            best = (cost, g)
    g_high = best[1]
    folds = []
    for k in range(3):
        held = sorted(g_low[k] + g_high[k])
        rest = [n for n in tr if n not in held]
        digits = digit_counts(rest)
        assert all(digits[str(d)] > 0 for d in range(10)), (k, digits)
        folds.append(dict(fold=k, heldout=held,
                          val_test_crops=sum(vt[n] for n in held), train_crops_removed=sum(tr[n] for n in held),
                          train_numbers_left=len(rest),
                          train_digit_counts_left={str(d): digits[str(d)] for d in range(10)}))
        print(folds[-1])
    # fold 3 (manager request): the high-number case. All training numbers >= 40 not already in folds 0-2. Those
    # numbers have crops in the train split only (val / test hold 1-16), so fold 3 is scored on "unseen number, seen
    # event" crops and compared with the folds 0-2 numbers under the same condition (heldout_eval.py).
    used = {n for f in folds for n in f["heldout"]}
    held = sorted(n for n in tr if n >= 40 and n not in used)
    rest = [n for n in tr if n not in held]
    assert digit_ok(held, tr, full, 0.3)
    digits = digit_counts(rest)
    folds.append(dict(fold=3, heldout=held, val_test_crops=0, train_crops_removed=sum(tr[n] for n in held),
                      train_numbers_left=len(rest), train_digit_counts_left={str(d): digits[str(d)] for d in range(10)},
                      note="high numbers (>= 40); train-split evaluation only"))
    print(folds[-1])
    doc = dict(version="v1", note="WP10b held-out-number folds; see students/recog/folds.py", folds=folds,
               all_train_numbers=sorted(tr), val_numbers=sorted(va), test_numbers=sorted(te))
    if a.write:
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=1))
        print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
