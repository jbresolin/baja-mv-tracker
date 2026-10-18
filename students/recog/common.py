"""WP10 shared pieces: charset, CRNN model, inference preprocessing, CTC decoding, roster scoring, track voting."""
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "students" / "runs" / "recog"
CHARS = "0123456789"
BLANK = 0                      # CTC blank = class 0, digit d = class d + 1
NCLS = len(CHARS) + 1
IMG_H, IMG_W = 32, 96


def pick_device():
    # CUDA_VISIBLE_DEVICES=-1 on Windows (an empty value does not hide the GPU, LESSONS WP9)
    return torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")


def cbr(i, o, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, p, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class CRNN(nn.Module):
    """VGG-style CNN -> column features (H collapsed) -> BiLSTM -> per-column logits over blank + 10 digits.
    width: channel multiplier. Input (B,3,H,W) float in [0,1] (RGB), output (T=W/4, B, NCLS) log-probs."""

    def __init__(self, h=IMG_H, width=1.0, rnn=128):
        super().__init__()
        c = [int(round(x * width)) for x in (32, 64, 128, 128, 192, 192, 256)]
        self.cnn = nn.Sequential(
            cbr(3, c[0]), nn.MaxPool2d(2),                       # H/2, W/2
            cbr(c[0], c[1]), nn.MaxPool2d(2),                    # H/4, W/4
            cbr(c[1], c[2]), cbr(c[2], c[3]), nn.MaxPool2d((2, 1)),   # H/8
            cbr(c[3], c[4]), cbr(c[4], c[5]), nn.MaxPool2d((2, 1)),   # H/16
        )
        hr = h // 16
        self.collapse = cbr(c[5], c[6], k=(hr, 1), p=0)          # -> 1 x W/4
        self.drop = nn.Dropout(0.2)
        self.rnn = nn.LSTM(c[6], rnn, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(2 * rnn, NCLS)
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        f = self.collapse(self.cnn(x)).squeeze(2)                # B, C, T
        f = self.drop(f.permute(2, 0, 1))                        # T, B, C
        y, _ = self.rnn(f)
        return F.log_softmax(self.fc(y), dim=-1)                 # T, B, NCLS


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def to_input(img, h=IMG_H, w=IMG_W, x_off=None):
    """BGR uint8 crop -> (h, w, 3) RGB uint8, aspect kept, height h, padded right/left with the border mean.
    Crops wider than w after scaling are squashed to w. x_off: left offset (None = centred)."""
    ih, iw = img.shape[:2]
    nw = max(1, min(w, int(round(iw * h / max(ih, 1)))))
    r = cv2.resize(img, (nw, h), interpolation=cv2.INTER_AREA if ih > h else cv2.INTER_LINEAR)
    border = np.concatenate([r[:, :1].reshape(-1, 3), r[:, -1:].reshape(-1, 3)], 0).mean(0)
    out = np.empty((h, w, 3), np.uint8)
    out[:] = border.astype(np.uint8)
    if x_off is None:
        x_off = (w - nw) // 2
    out[:, x_off:x_off + nw] = r
    return out[:, :, ::-1]


def batch_tensor(imgs):
    a = np.stack(imgs).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()


def greedy(logp):
    """logp: (T, B, C) -> list of (string, prob of the best path)."""
    lp, idx = logp.max(-1)                                     # T, B
    out = []
    for b in range(idx.shape[1]):
        s, prev = [], BLANK
        for t in idx[:, b].tolist():
            if t != BLANK and t != prev:
                s.append(CHARS[t - 1])
            prev = t
        out.append(("".join(s), float(lp[:, b].sum().exp())))
    return out


def encode(texts):
    tg = [CHARS.index(ch) + 1 for t in texts for ch in t]
    return torch.tensor(tg, dtype=torch.long), torch.tensor([len(t) for t in texts], dtype=torch.long)


def strip0(s):
    return s.lstrip("0") if s else s


class RosterScorer:
    """P(roster number | crop) from the CTC sequence likelihood. Each number n has the variants "n" and, for n < 10,
    "0n" (side plates). P(n) = sum of its variants' CTC probabilities; P(other) = 1 - sum (another string, incl. none).
    """

    def __init__(self, numbers):
        self.numbers = [int(n) for n in numbers]
        self.var, self.owner = [], []
        for j, n in enumerate(self.numbers):
            for v in ([str(n), "0" + str(n)] if n < 10 else [str(n)]):
                self.var.append(v)
                self.owner.append(j)
        self.owner = torch.tensor(self.owner)
        self.tg, self.tl = encode(self.var)

    @torch.no_grad()
    def __call__(self, logp):
        """logp (T, B, C) -> probs (B, n_numbers), p_other (B,)"""
        T, B, C = logp.shape
        V = len(self.var)
        lp = logp.unsqueeze(2).expand(T, B, V, C).reshape(T, B * V, C)
        tg = self.tg.repeat(B)
        tl = self.tl.repeat(B)
        il = torch.full((B * V,), T, dtype=torch.long)
        nll = F.ctc_loss(lp, tg, il, tl, blank=BLANK, reduction="none", zero_infinity=False)
        pv = torch.exp(-nll).view(B, V)
        P = torch.zeros(B, len(self.numbers)).index_add_(1, self.owner, pv)
        P = P.clamp(0, 1)
        return P, (1 - P.sum(1)).clamp(0, 1)


def vote(P, weights=None):
    """Track vote: mean over the track's crops of P(number | crop) (soft vote, each crop's mass sums to <= 1).
    P: (n_crops, n_numbers). Returns (best index, score of best, runner-up score)."""
    P = np.asarray(P, dtype=np.float64)
    w = np.ones(len(P)) if weights is None else np.asarray(weights, dtype=np.float64)
    s = (P * w[:, None]).sum(0) / max(w.sum(), 1e-9)
    o = np.argsort(-s)
    return int(o[0]), float(s[o[0]]), float(s[o[1]]) if len(o) > 1 else 0.0


def vote_boxed(P, has_box, p_support=0.5, score="share"):
    """Crop-count-independent track vote (WP10b fix of the panel-chain score). Crops where the panel detector found no
    box carry no evidence and are ignored (vote() averaged their zeros in, so 30-crop passes scored lower than 4-crop
    val tracks and a val-picked tau did not transfer). Score = mean over boxed crops of P(number | crop); support =
    number of boxed crops whose own argmax is the voted number with P >= p_support. Answer when score >= tau and
    support >= min_support (both picked on val).
    score="share" (default): posterior-weighted vote share = sum_crops P(winner) / sum_crops max_n P(n): weak crops
    (blur, partial panel, a box on a shirt) have little weight, so the score measures the agreement of the confident
    reads and does not fall when a domain gives many weak crops. score="mean": plain mean of P(winner) over boxed crops.
    The winner is the number with the largest summed posterior in both cases.
    Returns (best index, score, runner-up score, n_boxed, support); best = -1 when no crop has a box."""
    P = np.asarray(P, dtype=np.float64)
    hb = np.asarray(has_box, dtype=bool)
    if len(P) == 0 or not hb.any():
        return -1, 0.0, 0.0, 0, 0
    Q = P[hb]
    s = Q.mean(0)
    if score == "share":
        s = Q.sum(0) / max(float(Q.max(1).sum()), 1e-9)
    o = np.argsort(-s)
    j = int(o[0])
    sup = int(((Q.argmax(1) == j) & (Q[:, j] >= p_support)).sum())
    return j, float(s[j]), float(s[o[1]]) if len(o) > 1 else 0.0, int(hb.sum()), sup


def load_model(path, device="cpu"):
    """CRNN checkpoint, or (WP10b) arch type 'parseq': PARSeq-tiny wrapped to the same (T,B,11) log-prob output."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck.get("arch", {})
    if a.get("type") == "parseq":
        from parseq_model import ParseqCTC
        m = ParseqCTC(max_label_length=a.get("max_label_length", 4), refine_iters=a.get("refine_iters", 1),
                      img_w=a.get("img_w", 128))
        m.load_state_dict(ck["model"])
        m.eval()
        return m, a
    m = CRNN(h=a.get("h", IMG_H), width=a.get("width", 1.0), rnn=a.get("rnn", 128))
    m.load_state_dict(ck["model"])
    m.eval()
    return m, a


def read_roster(ev):
    import pandas as pd
    return sorted(int(x) for x in pd.read_csv(ROOT / "rosters" / f"{ev}.csv").numero)


class OrtModel:
    """ONNX Runtime drop-in for the torch model in the eval scripts: (B,3,32,W) tensor -> (T,B,11) log-prob tensor.
    ~2.5x faster than torch on CPU (export_onnx.py bench)."""

    def __init__(self, path, threads=6):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def __call__(self, x):
        return torch.from_numpy(self.s.run(None, {"image": x.numpy()})[0])


def load_any(run_dir, threads=6, prefer_onnx=True):
    """recog.onnx (if exported) via ONNX Runtime, else best.pt via torch."""
    run_dir = Path(run_dir)
    if prefer_onnx and (run_dir / "recog.onnx").exists():
        return OrtModel(run_dir / "recog.onnx", threads)
    return load_model(run_dir / "best.pt")[0]
