"""WP10b: fine-tune PARSeq-tiny (pretrained on real scene text, baudm/parseq, Apache-2.0) on datasets/digits.

Same data pipeline and options as train.py (TrainSet: real crops with the WP10 augmentation, "no number" windows with an
empty label, optional synthetic / spliced samples, --fold / --keep-cars), same 32x96 input (to_input / plain()), which
the model resizes to its native 32x128. Loss = PARSeq's permutation language modelling (forward + mirrored perms).
Digits-only token set sliced from the pretrained 94-char head (parseq_model.load_pretrained). The saved checkpoint
loads with common.load_model (arch type 'parseq') and outputs CTC-shaped log-probs, so evaluate.py, heldout_eval.py,
harness_26se.py and export_onnx.py work unchanged.

usage: CUDA_VISIBLE_DEVICES=-1 python students/recog/train_parseq.py --name pq_full --epochs 20 --threads 4
       [--fold v1:K] [--p-syn 0.2 --syn-text uniform] [--p-splice 0.2] [--p-prop 0.3] [--no-pretrained]
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen  # noqa: E402
from common import RUNS, batch_tensor, n_params  # noqa: E402
from augment import plain  # noqa: E402
from parseq_model import ParseqCTC, PermGen, load_pretrained, plm_loss  # noqa: E402
from train import TrainSet, collate, evaluate, load, load_props  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=1000)
    ap.add_argument("--epoch-size", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--p-neg", type=float, default=0.2)
    ap.add_argument("--p-syn", type=float, default=0.0)
    ap.add_argument("--syn-text", choices=["default", "uniform"], default="default")
    ap.add_argument("--p-splice", type=float, default=0.0)
    ap.add_argument("--splice-bank", default=str(RUNS / "splice_bank.pkl"))
    ap.add_argument("--p-prop", type=float, default=0.0)
    ap.add_argument("--fold", default=None)
    ap.add_argument("--keep-cars", type=float, default=1.0)
    ap.add_argument("--keep-seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--img-w", type=int, default=128, help="encoder input width (64 halves the encoder cost)")
    ap.add_argument("--init", default=None, help="start from a train_parseq.py checkpoint instead")
    ap.add_argument("--all-splits", action="store_true",
                    help="(WP9 final models) train on train + val + test (all events); val is then logged only, "
                         "NOT held out; use last.pt")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    tr = load("train") + (load("val") + load("test") if a.all_splits else [])
    held = gen.fold_numbers(a.fold)
    tr = gen.keep_cars(gen.exclude_numbers(tr, held), a.keep_cars, a.keep_seed)
    print("held out", sorted(held), "train items", len(tr), flush=True)
    splicer = None
    if a.p_splice > 0:
        from splice import Splicer
        splicer = Splicer(a.splice_bank, exclude=held, keep_crops={i["crop"] for i in tr})
    props = load_props(held, a.keep_cars, a.keep_seed) if a.p_prop > 0 else None
    va = load("val")
    vpos = [i for i in va if not i["neg"] and i["bucket"] == "ok"]
    vneg = [i for i in va if i["neg"]][:600]
    X = batch_tensor([plain(i["img"], i["box"]) for i in vpos])
    negX = batch_tensor([plain(i["img"], i["box"]) for i in vneg])
    vt = [i["text"] for i in vpos]
    vn = [i["number"] for i in vpos]

    ds = TrainSet(tr, a.epoch_size, a.p_neg, a.p_syn, False, a.syn_text, a.p_splice, splicer, a.p_prop, props)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, collate_fn=collate, persistent_workers=a.workers > 0,
                    drop_last=True)
    model = ParseqCTC(max_label_length=4, refine_iters=1, img_w=a.img_w)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["model"])
    elif not a.no_pretrained:
        print("pretrained:", load_pretrained(model.net), flush=True)
    print("params", n_params(model), flush=True)
    permgen = PermGen(seed=a.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (a.epoch_size // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    arch = dict(type="parseq", max_label_length=4, refine_iters=1, img_w=a.img_w)
    best, best_ep, log = -1, -1, []
    t0 = time.time()
    model.train()
    for ep in range(a.epochs):
        tl, n = 0.0, 0
        for x, texts in dl:
            loss = plm_loss(model.net, model.prep(x), texts, permgen)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            opt.step()
            sched.step()
            tl += loss.item()
            n += 1
        acc, exact, neg_empty = evaluate(model, X, vt, vn, negX, bs=256)
        row = dict(epoch=ep, loss=round(tl / max(n, 1), 4), val_acc=round(float(acc), 4), val_exact=round(float(exact), 4),
                   val_neg_empty=round(float(neg_empty), 4), lr=opt.param_groups[0]["lr"],
                   min=round((time.time() - t0) / 60, 1))
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
