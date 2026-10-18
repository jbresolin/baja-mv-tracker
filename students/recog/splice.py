"""WP10b: digit splicing. New numbers from real glyphs, so the recognizer sees digit combinations (and 63-99) that
no real crop shows.

build: run a trained CRNN (default syn_ft/best.pt) on every train-split 'ok' number crop (box + 10%, as evaluated);
       keep the crops it reads exactly (text_read incl. a leading zero), and locate the digit boundaries: CTC forced
       alignment (Viterbi on the known label) gives each digit's frames -> centre in crop pixels; the cut between two
       digits is the least-ink column within +-20% of the centre gap (Otsu inside the box, minority class = ink).
       -> students/runs/recog/splice_bank.pkl (only crop references + cut positions; pixels come from the cache).
       Only the train split is used (never val / test / 26SE). The alignment model saw all training numbers, but the
       bank's pixels are filtered by the fold at use time (Splicer(exclude=...)), so held-out numbers' crops never
       enter a fold's training.
sheet: students/runs/recog/splice_cuts.jpg (cuts on 64 random bank crops) and splice_samples.jpg (64 composites).

Splicer.sample(rng) -> (BGR patch, box, text), then the normal augment() runs on it. Three modes:
  same crop (35%):  "ab" -> "ba", "aa", "bb", "a", "b" (and "0N" crops -> "N0", "N", "00"... only valid numbers)
  same team (25%):  digits from crops of the same team (roster name; same car or the team's car in another event),
                    composed into a number that team does not carry
  cross crop (40%): target text uniform over 1-99 plus "0N" (gen.rand_text_uniform: explicitly covers 63-99 and
                    single digits in both forms), each digit from a glyph of a crop with a similar plate colour
Glyph strips are scaled to a common box height, abutted with a short linear seam blend; outer context is the
source's own context when the glyph keeps its side, else edge replication.
usage: python students/recog/splice.py build [--model syn_ft] | sheet
"""
import argparse
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen  # noqa: E402
from augment import plain  # noqa: E402
from common import BLANK, IMG_H, IMG_W, RUNS, batch_tensor, greedy, load_model  # noqa: E402

BANK = RUNS / "splice_bank.pkl"


def viterbi(logp, text):
    """CTC forced alignment. logp (T, C) numpy; returns list of frame lists, one per character."""
    lab = [CHAR + 1 for CHAR in map(int, text)]
    ext = [BLANK]
    for c in lab:
        ext += [c, BLANK]
    T, S = logp.shape[0], len(ext)
    NEG = -1e9
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), np.int64)
    dp[0, 0] = logp[0, ext[0]]
    if S > 1:
        dp[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            cands = [(dp[t - 1, s], s)]
            if s >= 1:
                cands.append((dp[t - 1, s - 1], s - 1))
            if s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2]:
                cands.append((dp[t - 1, s - 2], s - 2))
            v, b = max(cands)
            dp[t, s] = v + logp[t, ext[s]]
            bp[t, s] = b
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = [s]
    for t in range(T - 1, 0, -1):
        s = bp[t, s]
        path.append(s)
    path = path[::-1]
    frames = [[] for _ in text]
    for t, s in enumerate(path):
        if s % 2 == 1:
            frames[s // 2].append(t)
    return frames


def input_geometry(img, box):
    """plain(): crop [a:c] of the patch (box + 10%), resized to nw columns, placed at x_off. Returns (a, cw, nw, x_off)."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    H, W = img.shape[:2]
    a, b = int(max(0, np.floor(x1 - 0.1 * bw))), int(max(0, np.floor(y1 - 0.1 * bh)))
    c, d = int(min(W, np.ceil(x2 + 0.1 * bw))), int(min(H, np.ceil(y2 + 0.1 * bh)))
    cw, ch = c - a, d - b
    nw = max(1, min(IMG_W, int(round(cw * IMG_H / max(ch, 1)))))
    return a, cw, nw, (IMG_W - nw) // 2


def ink_profile(img, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    g = cv2.cvtColor(img[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2GRAY)
    if g.size == 0:
        return None
    _, bw = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 0.5:
        bw = 1 - bw
    return bw.sum(0).astype(np.float32)


def build(model_name, threads=4):
    import pickle as pk
    torch.set_num_threads(threads)
    model, _ = load_model(RUNS / model_name / "best.pt")
    items = pk.load(open(RUNS / "cache_train.pkl", "rb"))
    pos = [(j, i) for j, i in enumerate(items) if not i["neg"] and i["bucket"] == "ok"]
    bank, nread = [], 0
    for s in range(0, len(pos), 256):
        chunk = pos[s:s + 256]
        X = batch_tensor([plain(i["img"], i["box"]) for _, i in chunk])
        with torch.no_grad():
            lp = model(X)
        for (j, it), (txt, _), b in zip(chunk, greedy(lp), range(len(chunk))):
            if txt != it["text"] or len(txt) > 2:
                continue
            nread += 1
            frames = viterbi(lp[:, b].numpy(), txt)
            if any(not f for f in frames):
                continue
            a, cw, nw, xo = input_geometry(it["img"], it["box"])
            cx = [a + ((np.mean(f) * 4 + 2) - xo) * cw / nw for f in frames]
            x1, _, x2, _ = it["box"]
            cuts = []
            if len(txt) == 2:
                m = (cx[0] + cx[1]) / 2
                r = 0.2 * abs(cx[1] - cx[0])
                prof = ink_profile(it["img"], it["box"])
                if prof is not None and len(prof) > 2:
                    lo = int(max(1, m - r - x1))
                    hi = int(min(len(prof) - 1, m + r - x1 + 1))
                    if hi > lo:
                        sm = np.convolve(prof, np.ones(3) / 3, mode="same")
                        m = x1 + lo + int(np.argmin(sm[lo:hi]))
                if not (x1 + 0.15 * (x2 - x1) < m < x2 - 0.15 * (x2 - x1)):
                    continue
                cuts = [float(m)]
            bank.append(dict(idx=j, text=txt, cuts=cuts, centres=[float(c) for c in cx]))
    pk.dump(dict(model=model_name, bank=bank), open(BANK, "wb"))
    print(f"train ok crops {len(pos)}, read exactly {nread}, bank {len(bank)} "
          f"(2-char {sum(len(b['text']) == 2 for b in bank)})")


class Splicer:
    def __init__(self, bank_path=BANK, exclude=(), keep_crops=None, p_modes=(0.35, 0.25, 0.40)):
        doc = pickle.load(open(bank_path, "rb"))
        items = pickle.load(open(RUNS / "cache_train.pkl", "rb"))
        exclude = set(exclude)
        self.p_modes = p_modes
        self.crops, self.glyphs = [], []
        for b in doc["bank"]:
            it = items[b["idx"]]
            if int(it["number"]) in exclude or (keep_crops is not None and it["crop"] not in keep_crops):
                continue
            img, box = it["img"], it["box"]
            x1, y1, x2, y2 = box
            yi1, yi2 = int(max(0, y1)), int(min(img.shape[0], y2))
            plate = np.median(img[yi1:yi2, int(max(0, x1)):int(min(img.shape[1], x2))].reshape(-1, 3), 0) \
                if yi2 > yi1 else np.zeros(3)
            ci = len(self.crops)
            edges = [x1] + b["cuts"] + [x2]
            gl = []
            for k, ch in enumerate(b["text"]):
                g = dict(crop=ci, digit=ch, xa=edges[k], xb=edges[k + 1], first=k == 0, last=k == len(b["text"]) - 1)
                gl.append(len(self.glyphs))
                self.glyphs.append(g)
            self.crops.append(dict(img=img, box=box, text=b["text"], team=gen.team_of(it), plate=plate.astype(np.float32),
                                   glyphs=gl, number=int(it["number"])))
        self.by_digit = defaultdict(list)
        self.team_glyphs = defaultdict(list)
        self.team_numbers = defaultdict(set)
        for gi, g in enumerate(self.glyphs):
            self.by_digit[g["digit"]].append(gi)
            c = self.crops[g["crop"]]
            self.team_glyphs[c["team"]].append(gi)
            self.team_numbers[c["team"]].add(c["number"])
        self.two = [ci for ci, c in enumerate(self.crops) if len(c["text"]) == 2]
        self.teams = [t for t, gl in self.team_glyphs.items() if len({self.glyphs[g]["digit"] for g in gl}) >= 2]

    def summary(self):
        return dict(crops=len(self.crops), two_digit=len(self.two), glyphs=len(self.glyphs),
                    digits={d: len(v) for d, v in sorted(self.by_digit.items())}, teams_multi_digit=len(self.teams))

    # ---------------------------------------------------------------- composition
    def strip(self, gi, h0, keep_left, keep_right):
        """glyph gi scaled so its box height is h0; returns (strip, ctx_left, ctx_right) with the box rows at
        [m, m + h0], m = round(0.3 h0); context columns only where the glyph keeps its outer side."""
        g = self.glyphs[gi]
        c = self.crops[g["crop"]]
        img, (x1, y1, x2, y2) = c["img"], c["box"]
        bh = max(y2 - y1, 2.0)
        s = h0 / bh
        mv = 0.3 * bh
        ctx = 0.3 * bh
        xa = g["xa"] - (ctx if keep_left and g["first"] else 0)
        xb = g["xb"] + (ctx if keep_right and g["last"] else 0)
        pad = int(np.ceil(max(bh, ctx) + 2))
        P = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        ya, yb = y1 - mv + pad, y2 + mv + pad
        xa_p, xb_p = xa + pad, xb + pad
        src = np.float32([[xa_p, ya], [xb_p, ya], [xb_p, yb]])
        ow = max(2, int(round((xb - xa) * s)))
        oh = max(4, int(round((yb - ya) * s)))
        dst = np.float32([[0, 0], [ow, 0], [ow, oh]])
        M = cv2.getAffineTransform(src, dst)
        st = cv2.warpAffine(P, M, (ow, oh), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        cl = int(round(ctx * s)) if keep_left and g["first"] else 0
        cr = int(round(ctx * s)) if keep_right and g["last"] else 0
        return st, cl, cr

    def compose(self, gis, text, rng):
        c0 = self.crops[self.glyphs[gis[0]]["crop"]]
        h0 = float(np.clip(c0["box"][3] - c0["box"][1], 12, 90))
        parts = [self.strip(gi, h0, k == 0, k == len(gis) - 1) for k, gi in enumerate(gis)]
        oh = min(p[0].shape[0] for p in parts)
        strips = [p[0][:oh] for p in parts]
        ov = max(1, int(round(0.06 * h0)))
        out = strips[0].astype(np.float32)
        for st in strips[1:]:
            st = st.astype(np.float32)
            o = min(ov, out.shape[1] - 1, st.shape[1] - 1)
            if o > 0:
                al = np.linspace(1, 0, o + 2)[1:-1][None, :, None]
                seam = out[:, -o:] * al + st[:, :o] * (1 - al)
                out = np.concatenate([out[:, :-o], seam, st[:, o:]], 1)
            else:
                out = np.concatenate([out, st], 1)
        out = np.clip(out, 0, 255).astype(np.uint8)
        cl, cr = parts[0][1], parts[-1][2]
        ctx = int(round(0.3 * h0))
        L, R = ctx - cl if cl < ctx else 0, ctx - cr if cr < ctx else 0
        if L or R:
            out = cv2.copyMakeBorder(out, 0, 0, L, R, cv2.BORDER_REPLICATE)
        m = int(round(0.3 * h0))
        box = (float(cl + L), float(m), float(out.shape[1] - cr - R), float(min(oh, m + h0)))
        return out, box, text

    def near_glyph(self, digit, plate, rng, k=30):
        cand = self.by_digit.get(digit)
        if not cand:
            return None
        pick = [cand[rng.randrange(len(cand))] for _ in range(min(k, len(cand)))]
        d = [float(np.abs(self.crops[self.glyphs[g]["crop"]]["plate"] - plate).sum()) for g in pick]
        o = np.argsort(d)[:3]
        return pick[int(o[rng.randrange(len(o))])]

    def sample(self, rng):
        for _ in range(20):
            u = rng.random()
            if u < self.p_modes[0] and self.two:
                c = self.crops[rng.choice(self.two)]
                ga, gb = c["glyphs"]
                op = rng.choice(["swap", "aa", "bb", "a", "b"])
                gis = {"swap": [gb, ga], "aa": [ga, ga], "bb": [gb, gb], "a": [ga], "b": [gb]}[op]
            elif u < self.p_modes[0] + self.p_modes[1] and self.teams:
                t = rng.choice(self.teams)
                gl = self.team_glyphs[t]
                g1 = rng.choice(gl)
                if rng.random() < 0.3:
                    gis = [g1]
                else:
                    g2 = rng.choice(gl)
                    gis = [g1, g2]
                txt = "".join(self.glyphs[g]["digit"] for g in gis)
                if txt.lstrip("0") and int(txt) in self.team_numbers[t]:
                    continue
            else:
                txt = gen.rand_text_uniform(rng)
                g1 = self.near_glyph(txt[0], np.array([rng.uniform(0, 255) for _ in range(3)], np.float32), rng, 1)
                if g1 is None:
                    continue
                gis = [g1]
                plate = self.crops[self.glyphs[g1]["crop"]]["plate"]
                for ch in txt[1:]:
                    g = self.near_glyph(ch, plate, rng)
                    if g is None:
                        break
                    gis.append(g)
                if len(gis) != len(txt):
                    continue
            text = "".join(self.glyphs[g]["digit"] for g in gis)
            if not text.lstrip("0") or text == "00":
                continue
            return self.compose(gis, text, rng)
        c = self.crops[rng.randrange(len(self.crops))]
        return c["img"], c["box"], c["text"]


def sheet():
    items = pickle.load(open(RUNS / "cache_train.pkl", "rb"))
    doc = pickle.load(open(BANK, "rb"))
    rng = random.Random(1)
    tiles = []
    for b in rng.sample([b for b in doc["bank"] if b["cuts"]], 64):
        it = items[b["idx"]]
        im = it["img"].copy()
        s = 96 / im.shape[0]
        im = cv2.resize(im, (max(1, int(im.shape[1] * s)), 96))
        x1, y1, x2, y2 = [v * s for v in it["box"]]
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        for c in b["cuts"]:
            cv2.line(im, (int(c * s), int(y1)), (int(c * s), int(y2)), (0, 0, 255), 1)
        t = np.full((110, 200, 3), 255, np.uint8)
        t[:96, :min(200, im.shape[1])] = im[:, :200]
        cv2.putText(t, b["text"], (2, 107), 0, 0.35, (0, 0, 0), 1)
        tiles.append(t)
    cv2.imwrite(str(RUNS / "splice_cuts.jpg"), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 64, 8)]))
    sp = Splicer()
    print(sp.summary())
    tiles = []
    for _ in range(64):
        img, box, text = sp.sample(rng)
        s = 64 / img.shape[0]
        im = cv2.resize(img, (max(1, int(img.shape[1] * s)), 64))
        x1, y1, x2, y2 = [v * s for v in box]
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        t = np.full((80, 160, 3), 255, np.uint8)
        t[:64, :min(160, im.shape[1])] = im[:, :160]
        cv2.putText(t, text, (2, 77), 0, 0.35, (0, 0, 0), 1)
        tiles.append(t)
    cv2.imwrite(str(RUNS / "splice_samples.jpg"), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 64, 8)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "sheet"])
    ap.add_argument("--model", default="syn_ft")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.model, a.threads)
    else:
        sheet()


if __name__ == "__main__":
    main()
