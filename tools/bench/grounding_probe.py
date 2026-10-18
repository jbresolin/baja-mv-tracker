"""WP0 grounding probe: ask Qwen3-VL for every race number with its bbox (JSON), find the coordinate convention.

usage (repo root): python tools/bench/grounding_probe.py [--n=20] [--prompt=0] [--batch=1] [--dtype=fp16]
                   python tools/bench/grounding_probe.py mixed [--n=120] [--batch=4]
  mixed: stratified sample (60% numeric / 40% 'none' stored plain reads, as in vlm_fp16.py), grounded read only;
         compares the roster-snapped grounded number(s) with the stored plain read -> out/grounding_probe/mixed.csv
Picks crops from out/world_* where the pass vote is confident (share >= 0.8), equals the GT car, and the crop's own stored
plain read equals that car (one crop per pass, spread over cars, largest crops first). For each crop it draws the
returned boxes under three conventions on the original crop:
  green  = relative 0..1000 (Qwen3-VL claim)
  red    = absolute pixels of the image as fed (<=640 px side)
  blue   = absolute pixels of the processor's smart-resized image (multiple of 32; Qwen2.5-VL style)
Writes out/grounding_probe/<i>_<car>.jpg and results_p<prompt>.csv.
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
LV = ROOT / "lapvision"
sys.path.insert(0, str(LV))
import analyze  # noqa: E402
from vlm import Reader  # noqa: E402

OUT = ROOT / "out" / "grounding_probe"
PROMPTS = [
    # 0: Qwen-style grounding request, JSON list with bbox_2d
    'Locate every race number painted on the off-road race car(s) in this image (firewall panel, side plates, nose). '
    'Output a JSON list, one entry per number: [{"bbox_2d": [x1, y1, x2, y2], "label": "<digits>"}]. '
    'Output [] if no number is legible.',
    # 1: shorter
    'Detect all race numbers in the image and report their bbox coordinates in JSON format, '
    'with the digits as the label.',
]


def arg(name, default):
    for a in sys.argv:
        if a.startswith("--%s=" % name):
            return a.split("=", 1)[1]
    return default


def pick(n):
    dirs = sorted((ROOT / "out").glob("world_[0-9]*"))
    ps, rd = analyze.load(dirs)
    pa = pd.read_csv(ROOT / "out" / "passes_all.csv")
    good = pa[(pa.vlm_share >= 0.8) & (pa.vlm_car == pa.gt_car)][["src", "pass_id", "vlm_car"]]
    rd = rd.merge(good, on=["src", "pass_id"])
    rd = rd[rd.vlm == rd.vlm_car]
    rd["abs"] = rd.path.map(lambda p: str((LV / p.replace("\\", "/")).resolve()))
    rd["area"] = rd["abs"].map(lambda p: (lambda im: im.width * im.height)(Image.open(p)))
    best = rd.sort_values("area", ascending=False).groupby(["src", "pass_id"]).head(1)
    # round-robin over cars so we get variety
    best = best.sample(frac=1, random_state=0)
    best["k"] = best.groupby("vlm_car").cumcount()
    return best.sort_values(["k", "area"], ascending=[True, False]).head(n).reset_index(drop=True)


def parse(txt):
    t = re.sub(r"^```(json)?|```$", "", txt.strip(), flags=re.M).strip()
    try:
        j = json.loads(t)
    except Exception:
        m = re.search(r"\[.*\]", t, re.S)
        try:
            j = json.loads(m.group()) if m else None
        except Exception:
            j = None
    if isinstance(j, dict):
        j = [j]
    out = []
    for e in j or []:
        if isinstance(e, dict) and "bbox_2d" in e and len(e["bbox_2d"]) == 4:
            out.append((str(e.get("label", e.get("number", ""))), [float(v) for v in e["bbox_2d"]]))
    return j is not None, out


def main():
    n = int(arg("n", 20))
    pi = int(arg("prompt", 0))
    B = int(arg("batch", 1))
    rows = pick(n)
    OUT.mkdir(parents=True, exist_ok=True)
    rd = Reader(dtype=arg("dtype", "fp16"))
    rd.generate([rows["abs"][0]], PROMPTS[pi], max_new_tokens=8)  # warm-up
    res, times = [], []
    for i in range(0, len(rows), B):
        paths = list(rows["abs"][i:i + B])
        torch.cuda.synchronize()
        t = time.time()
        txt, ims = rd.generate(paths, PROMPTS[pi], max_new_tokens=256)
        torch.cuda.synchronize()
        times.append((time.time() - t) / len(paths))
        # processor's resized size (grid_thw * patch 16)
        _, inp = rd._inputs(paths, PROMPTS[pi], 640)
        for p, tx, im, g in zip(paths, txt, ims, inp["image_grid_thw"].tolist()):
            res.append(dict(path=p, raw=tx, fed_w=im.width, fed_h=im.height, sr_w=g[2] * 16, sr_h=g[1] * 16))
    for k, (r, (_, row)) in enumerate(zip(res, rows.iterrows())):
        ok, boxes = parse(r["raw"])
        r.update(car=int(row.vlm_car), src=row.src, pass_id=row.pass_id, json_ok=ok, n_boxes=len(boxes),
                 labels="|".join(b[0] for b in boxes), boxes=json.dumps([b[1] for b in boxes]))
        im = Image.open(r["path"]).convert("RGB")
        W, H = im.size
        sc = 3 if max(W, H) < 400 else 1   # upscale small crops so the overlay is visible
        im = im.resize((W * sc, H * sc))
        d = ImageDraw.Draw(im)
        for lab, (x1, y1, x2, y2) in boxes:
            for col, fx, fy in [("lime", W / 1000, H / 1000), ("red", W / r["fed_w"], H / r["fed_h"]),
                                ("blue", W / r["sr_w"], H / r["sr_h"])]:
                bx = [x1 * fx * sc, y1 * fy * sc, x2 * fx * sc, y2 * fy * sc]
                d.rectangle(bx, outline=col, width=2)
            d.text((x1 * W / 1000 * sc + 2, y1 * H / 1000 * sc + 2), lab, fill="lime")
        im.save(OUT / ("p%d_%02d_car%d.jpg" % (pi, k, r["car"])), quality=90)
    df = pd.DataFrame(res)
    df["grounded_has_car"] = [str(c) in l.split("|") for c, l in zip(df.car, df.labels)]
    df.to_csv(OUT / ("results_p%d.csv" % pi), index=False)
    print(df[["car", "fed_w", "fed_h", "sr_w", "sr_h", "json_ok", "n_boxes", "labels", "boxes"]].to_string())
    print("\nprompt %d batch %d: %.3f s/crop (plain read is ~0.16-0.29)" % (pi, B, sum(times) / len(times)))
    print("JSON ok %d/%d, grounded label set contains the plain/GT car %d/%d" % (
        df.json_ok.sum(), len(df), df.grounded_has_car.sum(), len(df)))
    print("raw example:", repr(df.raw[0][:300]))


def snap(labels):
    return sorted({int(x) for x in str(labels).split("|") if x.isdigit() and int(x) in analyze.ROSTER})


def mixed():
    from vlm_fp16 import load_reads, sample
    n, B = int(arg("n", 120)), int(arg("batch", 4))
    s = sample(load_reads(), n, seed=2)
    rd = Reader(dtype=arg("dtype", "fp16"))
    paths = list(s["abs"])
    rd.generate(paths[:1], PROMPTS[0], max_new_tokens=8)
    raws = []
    torch.cuda.synchronize()
    t = time.time()
    for i in range(0, len(paths), B):
        raws += rd.generate(paths[i:i + B], PROMPTS[0], max_new_tokens=128)[0]
    torch.cuda.synchronize()
    dt = (time.time() - t) / len(paths)
    s["g_raw"] = raws
    s["g_ok"] = [parse(r)[0] for r in raws]
    s["g_labels"] = ["|".join(b[0] for b in parse(r)[1]) for r in raws]
    s["g_snap"] = s.g_labels.map(snap)
    s["p_snap"] = s.stored.map(lambda v: [] if pd.isna(v) or int(v) not in analyze.ROSTER else [int(v)])
    s["g_first"] = s.g_snap.map(lambda l: l[0] if len(l) == 1 else (None if not l else -1))
    same = [g == p for g, p in zip(s.g_snap, s.p_snap)]
    s["same"] = same
    OUT.mkdir(parents=True, exist_ok=True)
    s.drop(columns=["abs"]).to_csv(OUT / "mixed.csv", index=False)
    print("grounded batch %d: %.3f s/crop, JSON ok %d/%d" % (B, dt, s.g_ok.sum(), len(s)))
    print("roster-snapped grounded set == plain read: %.3f" % s.same.mean())
    for name, m in [("plain numeric", s.p_snap.map(len) > 0), ("plain none/off-roster", s.p_snap.map(len) == 0)]:
        print("  %-22s n=%3d same %.3f" % (name, m.sum(), s.same[m].mean()))
    print("grounded gives >1 distinct roster number: %d" % (s.g_snap.map(len) > 1).sum())
    print(s[~s.same][["src", "pass_id", "vlm_raw", "g_labels"]].to_string())


if __name__ == "__main__":
    mixed() if "mixed" in sys.argv[1:] else main()
