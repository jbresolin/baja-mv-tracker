"""WP10 augmentation (cv2 / numpy only) and synthetic number-plate renderer (PIL + Windows fonts).

augment(img, box, rng) works on a context patch with the number box in patch coordinates: one perspective warp
does the box jitter (each side cut or extended by up to ~12% on top of the 10% padding), rotation (+-15 deg) and
perspective; then colour jitter, mud / dust, partial occlusion, motion / Gaussian blur, low resolution (down-up),
JPEG and noise, all at the crop's native resolution (the number boxes are 16-86 px tall).
"""
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import IMG_H, IMG_W, to_input

FONT_DIR = Path("C:/Windows/Fonts")
FONTS = ["arialbd.ttf", "ariblk.ttf", "bahnschrift.ttf", "impact.ttf", "calibrib.ttf", "verdanab.ttf", "tahomabd.ttf",
         "trebucbd.ttf", "seguibl.ttf", "segoeuib.ttf", "framd.ttf", "consolab.ttf", "corbelb.ttf", "candarab.ttf",
         "georgiab.ttf", "timesbd.ttf", "courbd.ttf", "comicbd.ttf", "arial.ttf", "arialbi.ttf", "segoeuiz.ttf",
         "seguisb.ttf", "lucon.ttf", "micross.ttf"]


def warp_crop(img, box, rng, strength=1.0):
    """Box jitter + rotation + perspective in one warp. Returns the warped crop at ~native resolution."""
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 2.0), max(y2 - y1, 2.0)
    pad = 0.10
    j = 0.12 * strength
    l = x1 - bw * (pad + rng.uniform(-j, j))
    r = x2 + bw * (pad + rng.uniform(-j, j))
    t = y1 - bh * (pad + rng.uniform(-j, j))
    b = y2 + bh * (pad + rng.uniform(-j, j))
    q = np.array([[l, t], [r, t], [r, b], [l, b]], np.float64)
    c = q.mean(0)
    w, h = r - l, b - t
    if rng.random() < 0.7 * strength:
        a = math.radians(np.clip(rng.gauss(0, 7), -15, 15))
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        q = (q - c) @ R.T + c
    if rng.random() < 0.5 * strength:
        q = q + np.array([[rng.uniform(-0.08, 0.08) * w, rng.uniform(-0.08, 0.08) * h] for _ in range(4)])
    if rng.random() < 0.3 * strength:                      # horizontal shear (italic look / oblique view)
        s = rng.uniform(-0.25, 0.25) * h
        q[0, 0] += s; q[1, 0] += s; q[2, 0] -= s; q[3, 0] -= s
    oh = int(np.clip(round(h), 10, 96))
    ow = int(np.clip(round(w * oh / h), 4, 4 * IMG_W))
    dst = np.array([[0, 0], [ow, 0], [ow, oh], [0, oh]], np.float32)
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def color_jitter(im, rng):
    im = im.astype(np.float32)
    if rng.random() < 0.3:                                  # hue / saturation
        hsv = cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-90, 90)) % 180
        hsv[..., 1] *= rng.uniform(0.3, 1.4)
        im = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if rng.random() < 0.15:
        im = im[..., list(np.random.default_rng(rng.randrange(1 << 30)).permutation(3))]
    if rng.random() < 0.1:
        im = 255 - im
    if rng.random() < 0.15:
        g = im.mean(2, keepdims=True)
        im = np.repeat(g, 3, 2)
    ctr = rng.uniform(0.5, 1.4)
    bri = rng.uniform(-50, 50)
    m = im.mean()
    im = (im - m) * ctr + m + bri
    im = np.clip(im, 0, 255)
    if rng.random() < 0.3:
        gam = rng.uniform(0.6, 1.6)
        im = 255 * (im / 255) ** gam
    return im


def mud(im, rng):
    h, w = im.shape[:2]
    if rng.random() < 0.35:                                 # blotches
        mask = np.zeros((h, w), np.float32)
        for _ in range(rng.randint(1, 4)):
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            ax, ay = rng.uniform(0.05, 0.3) * w, rng.uniform(0.05, 0.3) * h
            cv2.ellipse(mask, (int(cx), int(cy)), (max(1, int(ax)), max(1, int(ay))), rng.uniform(0, 180), 0, 360,
                        rng.uniform(0.4, 0.95), -1)
        k = max(1, int(0.05 * min(h, w))) * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        col = np.array([rng.uniform(30, 90), rng.uniform(50, 110), rng.uniform(70, 150)], np.float32)
        tex = col[None, None, :] + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 12, (h, w, 1))
        im = im * (1 - mask[..., None]) + tex * mask[..., None]
    if rng.random() < 0.2:                                  # dust haze: low contrast towards brown / grey
        col = np.array([rng.uniform(60, 160)] * 3, np.float32) + np.array([-15, 0, 15], np.float32)
        a = rng.uniform(0.15, 0.5)
        im = im * (1 - a) + col * a
    return im


def occlude(im, rng):
    if rng.random() >= 0.15:
        return im
    h, w = im.shape[:2]
    im = im.copy()
    col = np.array([rng.uniform(0, 255) for _ in range(3)], np.float32)
    if rng.random() < 0.5:                                  # vertical band at the left / right edge (cut digit edge)
        bw = int(rng.uniform(0.05, 0.2) * w)
        x0 = 0 if rng.random() < 0.5 else w - bw
        im[:, x0:x0 + bw] = col
    else:                                                    # horizontal band (cage tube, top / bottom)
        bh = int(rng.uniform(0.05, 0.22) * h)
        y0 = int(rng.uniform(0, h - bh))
        im[y0:y0 + bh] = col
    return im


def degrade(im, rng):
    im = np.clip(im, 0, 255).astype(np.uint8)
    h, w = im.shape[:2]
    if rng.random() < 0.3:                                  # motion blur
        L = rng.randint(3, 9)
        k = np.zeros((L, L), np.float32)
        k[L // 2] = 1
        M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), rng.uniform(0, 180), 1)
        k = cv2.warpAffine(k, M, (L, L))
        im = cv2.filter2D(im, -1, k / max(k.sum(), 1e-6))
    elif rng.random() < 0.2:
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.35 and h > 12:                      # low resolution
        th = rng.uniform(8, min(22, h))
        s = th / h
        small = cv2.resize(im, (max(2, int(w * s)), max(2, int(h * s))), interpolation=cv2.INTER_AREA)
        im = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR if rng.random() < 0.7 else cv2.INTER_NEAREST)
    if rng.random() < 0.2:
        im = np.clip(im.astype(np.float32) + np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, rng.uniform(2, 8), im.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(15, 75)])
        im = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return im


def augment(img, box, rng, strength=1.0):
    im = warp_crop(img, box, rng, strength)
    im = color_jitter(im, rng)
    im = mud(im, rng)
    im = occlude(im, rng)
    im = degrade(im, rng)
    ih, iw = im.shape[:2]
    nw = max(1, min(IMG_W, int(round(iw * IMG_H / ih))))
    return to_input(im, x_off=rng.randint(0, IMG_W - nw))


def plain(img, box):
    """Validation / test input: box + 10% padding (the digits-set cut), no augmentation, centred."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    H, W = img.shape[:2]
    a, b = int(max(0, np.floor(x1 - 0.1 * bw))), int(max(0, np.floor(y1 - 0.1 * bh)))
    c, d = int(min(W, np.ceil(x2 + 0.1 * bw))), int(min(H, np.ceil(y2 + 0.1 * bh)))
    return to_input(img[b:d, a:c])


# ---------------------------------------------------------------- synthetic plates

_font_cache = {}


def font(name, size):
    k = (name, size)
    if k not in _font_cache:
        _font_cache[k] = ImageFont.truetype(str(FONT_DIR / name), size)
    return _font_cache[k]


def rand_text(rng):
    r = rng.random()
    if r < 0.33:
        t = str(rng.randint(0, 9))
        if rng.random() < 0.25:
            t = "0" + t
    elif r < 0.92:
        t = str(rng.randint(10, 99))
    else:
        t = str(rng.randint(100, 199))
    return t


PLATE_COLORS = [(255, 255, 255), (240, 240, 235), (20, 20, 20), (200, 30, 30), (30, 60, 180), (30, 140, 50),
                (240, 200, 20), (240, 120, 20), (120, 30, 140), (130, 130, 130), (0, 150, 200), (220, 0, 120)]


def render(rng, bg=None):
    """Synthetic number: returns (BGR patch, box of the digits, text)."""
    text = rand_text(rng)
    fname = rng.choice(FONTS)
    size = rng.randint(36, 72)
    try:
        f = font(fname, size)
    except OSError:
        f = font("arialbd.ttf", size)
    plate = rng.choice(PLATE_COLORS) if rng.random() < 0.6 else tuple(rng.randint(0, 255) for _ in range(3))
    lum = 0.299 * plate[0] + 0.587 * plate[1] + 0.114 * plate[2]
    if rng.random() < 0.8:
        ink = (rng.randint(0, 50),) * 3 if lum > 140 else rng.choice([(255, 255, 255), (250, 230, 40), (255, 255, 255)])
    else:
        ink = tuple(rng.randint(0, 255) for _ in range(3))
    stroke = rng.randint(1, 4) if rng.random() < 0.25 else 0
    stroke_col = tuple(rng.randint(0, 255) for _ in range(3))
    if rng.random() < 0.2:                                   # hollow / outlined digits (several liveries use them)
        stroke = max(1, int(size * rng.uniform(0.03, 0.08)))
        stroke_col, ink = ink, plate
    spacing = rng.uniform(-0.08, 0.2) * size
    # glyph sizes
    widths = [f.getbbox(ch, stroke_width=stroke) for ch in text]
    gw = [b[2] - b[0] for b in widths]
    tw = int(sum(gw) + spacing * (len(text) - 1))
    asc = f.getbbox("0123456789", stroke_width=stroke)
    th = asc[3] - asc[1]
    mx, my = int(rng.uniform(0.1, 0.6) * th), int(rng.uniform(0.08, 0.4) * th)
    pw, ph = tw + 2 * mx, th + 2 * my
    W, H = int(pw * rng.uniform(1.3, 2.2)), int(ph * rng.uniform(1.3, 2.2))
    if bg is not None:
        canvas = Image.fromarray(cv2.cvtColor(cv2.resize(bg, (W, H)), cv2.COLOR_BGR2RGB))
    else:
        canvas = Image.new("RGB", (W, H), tuple(rng.randint(0, 255) for _ in range(3)))
    d = ImageDraw.Draw(canvas)
    px, py = rng.randint(0, W - pw), rng.randint(0, H - ph)
    if rng.random() < 0.85:
        rad = int(rng.uniform(0, 0.25) * ph)
        d.rounded_rectangle([px, py, px + pw, py + ph], radius=rad, fill=plate)
    x = px + mx
    y0 = py + my - asc[1]
    for ch, b, w_ in zip(text, widths, gw):
        d.text((x - b[0], y0), ch, font=f, fill=ink, stroke_width=stroke, stroke_fill=stroke_col)
        x += w_ + spacing
    if rng.random() < 0.08:                                  # stencil-like gap
        yy = py + my + int(rng.uniform(0.3, 0.7) * th)
        d.rectangle([px, yy, px + pw, yy + max(1, int(0.06 * th))], fill=plate)
    im = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    box = (px + mx, py + my, px + mx + tw, py + my + th)
    return im, box, text


def synth_sample(rng, bgs=None):
    """A synthetic plate scaled so the digits are 14-60 px tall (the real box heights), then augmented."""
    bg = rng.choice(bgs) if bgs and rng.random() < 0.7 else None
    im, box, text = render(rng, bg)
    s = rng.uniform(14, 60) / max(box[3] - box[1], 1)
    if s < 1:
        im = cv2.resize(im, (max(4, int(im.shape[1] * s)), max(4, int(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
        box = tuple(v * s for v in box)
    return augment(im, box, rng), text
