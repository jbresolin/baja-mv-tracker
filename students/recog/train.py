"""WP10: train the CRNN-CTC number recognizer on datasets/digits (cache from prep.py). CPU by default.

usage:
  CUDA_VISIBLE_DEVICES=-1 python students/recog/train.py --name real
  CUDA_VISIBLE_DEVICES=-1 python students/recog/train.py --name syn_pre --synth-only --epochs 25 --patience 100
  CUDA_VISIBLE_DEVICES=-1 python students/recog/train.py --name syn_ft --init students/runs/recog/syn_pre/last.pt
  options: --p-neg 0.2 (share of "no number" windows, empty target)  --p-syn 0 (synthetic share mixed into real training)
           --epoch-size 6000 --batch 128 --lr 2e-3 --threads 6 --workers 2 --width 1.0

Early stopping on val (25SU) exact-match after stripping leading zeros (greedy CTC decode vs the consensus number),
'ok' bucket. Runs: students/runs/recog/<name>/{best.pt,last.pt,log.csv,args.json}. Never trains on 26SE.
"""
import argparse
import csv
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment import augment, plain, synth_sample  # noqa: E402
from common import CRNN, RUNS, BLANK, IMG_H, batch_tensor, encode, greedy, n_params, strip0  # noqa: E402


def load(split):
    with open(RUNS / f"cache_{split}.pkl", "rb") as f:
        return pickle.load(f)


class TrainSet(Dataset):
    def __init__(self, items, size, p_neg, p_syn, synth_only=False):
        self.pos = [i for i in items if not i["neg"] and i["bucket"] == "ok"]
        self.neg = [i for i in items if i["neg"]]
        self.bgs = [i["img"] for i in self.neg[:3000]]
        self.size, self.p_neg, self.p_syn, self.synth_only = size, p_neg, p_syn, synth_only
        self.rng = None

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.rng is None:
            self.rng = random.Random(torch.initial_seed() % (1 << 31))
        rng = self.rng
        u = rng.random()
        if u < self.p_neg:
            it = rng.choice(self.neg)
            return augment(it["img"], it["box"], rng), ""
        if self.synth_only or u < self.p_neg + self.p_syn:
            return synth_sample(rng, self.bgs)
        it = rng.choice(self.pos)
        return augment(it["img"], it["box"], rng), it["text"]


def collate(b):
    imgs, texts = zip(*b)
    return batch_tensor(list(imgs)), list(texts)


@torch.no_grad()
def evaluate(model, X, texts, numbers, negX, bs=512):
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        preds += greedy(model(X[i:i + bs]))
    exact = np.mean([p == t for (p, _), t in zip(preds, texts)])
    acc = np.mean([strip0(p) == n for (p, _), n in zip(preds, numbers)])
    nr = []
    for i in range(0, len(negX), bs):
        nr += greedy(model(negX[i:i + bs]))
    neg_empty = np.mean([p == "" for p, _ in nr]) if nr else float("nan")
    model.train()
    return acc, exact, neg_empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--epoch-size", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--p-neg", type=float, default=0.2)
    ap.add_argument("--p-syn", type=float, default=0.0)
    ap.add_argument("--synth-only", action="store_true")
    ap.add_argument("--init", default=None)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    tr = load("train")
    va = load("val")
    vpos = [i for i in va if not i["neg"] and i["bucket"] == "ok"]
    vneg = [i for i in va if i["neg"]][:600]
    X = batch_tensor([plain(i["img"], i["box"]) for i in vpos])
    negX = batch_tensor([plain(i["img"], i["box"]) for i in vneg])
    vt = [i["text"] for i in vpos]
    vn = [i["number"] for i in vpos]

    ds = TrainSet(tr, a.epoch_size, a.p_neg, a.p_syn, a.synth_only)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, collate_fn=collate, persistent_workers=a.workers > 0,
                    drop_last=True)
    model = CRNN(h=IMG_H, width=a.width)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["model"])
    print("params", n_params(model), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (a.epoch_size // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05 if a.init else 0.1)
    arch = dict(h=IMG_H, width=a.width, rnn=128)
    best, best_ep, log = -1, -1, []
    t0 = time.time()
    for ep in range(a.epochs):
        tl, n = 0.0, 0
        for x, texts in dl:
            logp = model(x)
            tg, tlen = encode(texts)
            il = torch.full((x.shape[0],), logp.shape[0], dtype=torch.long)
            loss = F.ctc_loss(logp, tg, il, tlen, blank=BLANK, zero_infinity=True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tl += loss.item()
            n += 1
        acc, exact, neg_empty = evaluate(model, X, vt, vn, negX)
        row = dict(epoch=ep, loss=round(tl / max(n, 1), 4), val_acc=round(acc, 4), val_exact=round(exact, 4),
                   val_neg_empty=round(neg_empty, 4), lr=opt.param_groups[0]["lr"], min=round((time.time() - t0) / 60, 1))
        log.append(row)
        print(row, flush=True)
        ck = dict(model=model.state_dict(), arch=arch, epoch=ep, val_acc=acc, args=vars(a))
        torch.save(ck, out / "last.pt")
        if acc > best:
            best, best_ep = acc, ep
            torch.save(ck, out / "best.pt")
        with open(out / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(log)
        if ep - best_ep >= a.patience:
            print("early stop", flush=True)
            break
    print("best val_acc", best, "epoch", best_ep, flush=True)


if __name__ == "__main__":
    main()
