"""WP10 interim mode: read a car crop without a number-panel detector.

The recognizer was trained with "no number" windows (empty target), so on background it emits blanks and
P(roster number) stays low. read_car_crop() scores a grid of windows (3 sizes x 2 aspects, 60% stride, ~1,000-1,400 windows) over the car
crop rescaled to a fixed height, and keeps, per roster number, the best window's P(number | window). That is the crop's
roster posterior; the track / pass vote is common.vote() over crops as usual. This is a stand-in until the WP9
number-panel detector exists: the final chain is panel boxes -> recognizer -> vote -> roster.

The window sizes follow the number-box / car-crop height ratio in datasets/numbers_crop (5/50/95% = 0.054 / 0.11 /
0.18 of the crop height): window heights 0.09 / 0.14 / 0.19 of a 320 px crop (the window is box + padding).
"""
import cv2
import numpy as np
import torch

from common import RosterScorer, batch_tensor, to_input

CROP_H = 320
WIN_H = (30, 44, 60)
ASPECTS = (1.0, 1.6)
STRIDE = 0.6
WIN_W_IN = 64          # recognizer input width for windows (32 x 64; the model is width-agnostic)


def windows(h, w):
    out = []
    for wh in WIN_H:
        for a in ASPECTS:
            ww = int(round(wh * a))
            if ww > w or wh > h:
                continue
            sy, sx = max(1, int(wh * STRIDE)), max(1, int(ww * STRIDE))
            for y in range(0, h - wh + 1, sy):
                for x in range(0, w - ww + 1, sx):
                    out.append((x, y, x + ww, y + wh))
    return out


@torch.no_grad()
def read_car_crop(model, crop, scorer: RosterScorer, bs=1024, return_best=False):
    """crop: BGR car crop. Returns P (n_numbers,) = max over windows of P(number | window), best window."""
    s = CROP_H / crop.shape[0]
    im = cv2.resize(crop, (max(8, int(round(crop.shape[1] * s))), CROP_H), interpolation=cv2.INTER_AREA)
    wins = windows(*im.shape[:2])
    best_P = np.zeros(len(scorer.numbers))
    best_w, best_v = None, -1.0
    for i in range(0, len(wins), bs):
        chunk = wins[i:i + bs]
        X = batch_tensor([to_input(im[y1:y2, x1:x2], w=WIN_W_IN) for (x1, y1, x2, y2) in chunk])
        P, _ = scorer(model(X))
        P = P.numpy()
        best_P = np.maximum(best_P, P.max(0))
        k = int(P.max(1).argmax())
        if P[k].max() > best_v:
            best_v, best_w = float(P[k].max()), tuple(int(v / s) for v in chunk[k])
    return (best_P, best_w) if return_best else best_P
