"""Shared helpers for WP2b/WP4/WP5 (identity labels): rosters, fuzzy team matching, VLM prompts + batching, PAUSE,
the per-track vote rule and small parquet utilities.

The VLM is lapvision/vlm.py's Reader (Qwen3-VL-4B-Instruct, fp16 on the 2080 Ti). Three prompts:
  PLAIN      the session-1 number-reading prompt (identity; ~3 output tokens, ~0.16 s/crop at batch 4-6)
  GROUND     the WP0 grounding prompt (JSON [{"bbox_2d": [x1,y1,x2,y2], "label": "22"}], coordinates 0..1000 relative to
             the image as fed; only used for number boxes, never for identity)
  TRANSCRIBE "transcribe the on-screen graphic" for broadcast lower thirds / heat boxes (WP2b)
"""
import csv
import difflib
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, ROSTERS  # noqa: E402

PAUSE = ROOT / "PAUSE"
LV = ROOT / "lapvision"

# WP5 consensus rule (session 1: share >= 0.5 -> ~98% precise)
MIN_SHARE = 0.5
MIN_VOTES = 3

GROUND = ('Locate every race number painted on the off-road race car(s) in this image (firewall panel, side plates, '
          'nose). Output a JSON list, one entry per number: [{"bbox_2d": [x1, y1, x2, y2], "label": "<digits>"}]. '
          'Output [] if no number is legible.')
TRANSCRIBE = ("This is a cropped TV broadcast graphic (lower third / scoreboard) from a Baja SAE competition stream. "
              "Transcribe all of its text exactly, one line per text line. Answer 'none' if there is no text.")


def paused():
    return PAUSE.exists()


# ---------------------------------------------------------------- rosters / fuzzy matching

def norm_text(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


_STOP = {"baja", "team", "equipe", "sae", "de", "da", "do", "the", "racing", "e", "campus", "universidade", "federal"}


def load_roster(ev):
    """{numero: {'equipe':..., 'escola':...}} or {} when the roster is missing."""
    p = ROSTERS / f"{ev}.csv"
    if not p.exists():
        return {}
    out = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out[int(r["numero"])] = {"equipe": r.get("equipe", ""), "escola": r.get("escola", "")}
            except ValueError:
                pass
    return out


def team_score(text, entry):
    """Fuzzy match of an on-screen team text against a roster entry (0..1).

    max of: difflib ratio of the normalised strings; share of the text's distinctive tokens found (fuzzily) in the
    roster's team + school tokens. 'Baja', 'Team', 'Equipe' alone never match (stop words)."""
    t = norm_text(text)
    e = norm_text(entry["equipe"])
    if not t or not e:
        return 0.0
    r1 = difflib.SequenceMatcher(None, t, e).ratio()
    # containment both ways ("BAJA DE GALPAO" in "baja de galpao unisc")
    r2 = 0.0
    if len(t) >= 4 and (t in e or e in t):
        r2 = 0.95
    tt = [w for w in t.split() if w not in _STOP and len(w) >= 2]
    pool = set(e.split()) | set(norm_text(entry.get("escola", "")).split())
    pool -= _STOP
    r3 = 0.0
    if tt:
        hit = 0
        for w in tt:
            if w in pool or difflib.get_close_matches(w, list(pool), n=1, cutoff=0.8):
                hit += 1
        r3 = hit / len(tt) * (0.95 if len(tt) >= 1 else 0)
        if hit == 0:
            r3 = 0.0
    return round(max(r1, r2, r3), 3)


# ---------------------------------------------------------------- voting

def parse_num(raw):
    m = re.search(r"\d{1,3}", str(raw or ""))
    return int(m.group()) if m else None


def vote(nums, roster, n_read=None):
    """nums: per-crop parsed numbers (None = no read). Majority over roster numbers.
    Returns dict(number|None, top, share, votes, n, runner, consensus: bool)."""
    n = n_read if n_read is not None else len(nums)
    c = Counter(x for x in nums if x is not None and (not roster or x in roster))
    if not c or n == 0:
        return dict(top=None, share=0.0, votes=0, n=n, runner=None, consensus=False, number=None)
    (top, k), *rest = c.most_common(2)
    share = k / n
    ok = share >= MIN_SHARE and k >= MIN_VOTES
    return dict(top=int(top), share=round(share, 3), votes=int(k), n=int(n),
                runner=int(rest[0][0]) if rest else None, consensus=bool(ok), number=int(top) if ok else None)


# ---------------------------------------------------------------- VLM

_READER = None


def reader(dtype="fp16"):
    global _READER
    if _READER is None:
        if str(LV) not in sys.path:
            sys.path.insert(0, str(LV))
        from vlm import Reader  # noqa: E402
        dev = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "" else "cpu"
        import torch
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        _READER = Reader(device=dev, dtype=dtype if dev == "cuda" else "fp32")
    return _READER


def generate(paths, prompt, max_new_tokens, max_side=640):
    """paths: list of image paths or PIL images. Returns raw texts."""
    r = reader()
    txt, _ = r.generate(paths, prompt, max_side, max_new_tokens)
    return [t.strip() for t in txt]


def parse_ground(txt):
    """Grounded-read output -> list of (label, [x1, y1, x2, y2] in 0..1000) or None if not JSON."""
    t = re.sub(r"^```(json)?|```$", "", txt.strip(), flags=re.M).strip()
    try:
        j = json.loads(t)
    except Exception:  # noqa: BLE001
        m = re.search(r"\[.*\]", t, re.S)
        try:
            j = json.loads(m.group()) if m else None
        except Exception:  # noqa: BLE001
            j = None
    if j is None:
        return None
    if isinstance(j, dict):
        j = [j]
    out = []
    for e in j if isinstance(j, list) else []:
        if isinstance(e, dict) and "bbox_2d" in e and isinstance(e["bbox_2d"], (list, tuple)) and len(e["bbox_2d"]) == 4:
            try:
                out.append((str(e.get("label", e.get("number", ""))), [float(v) for v in e["bbox_2d"]]))
            except (TypeError, ValueError):
                pass
    return out


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return time.time() - self.t0


def dhash(gray, w=32, h=8):
    """Difference hash of a grayscale image (bool vector of w*h bits)."""
    import cv2
    s = cv2.resize(gray, (w + 1, h), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (s[:, 1:] > s[:, :-1]).ravel()
