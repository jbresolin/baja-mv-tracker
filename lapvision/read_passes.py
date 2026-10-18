"""Stage 4: build passes from tracks.csv, pick the best crops per pass, read numbers.

usage: python read_passes.py <crops_dir> [k] [--no-easyocr] [--dtype=fp16|bf16|fp32|auto]
(--dtype overrides env LAPVISION_VLM_DTYPE; default auto = bf16 on Ampere+, fp16 on Turing)
writes <crops_dir>/passes.csv and <crops_dir>/reads.csv
"""
import sys
from pathlib import Path

import pandas as pd

import passes as P

ROSTER = None  # filled from gt.equipes() when snapping


def main():
    d = Path(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 10
    use_easy = "--no-easyocr" not in sys.argv
    tracks = pd.read_csv(d / "tracks.csv")
    ps = P.build(tracks)
    ps.to_csv(d / "passes.csv", index=False)
    print(len(ps), "passes", flush=True)

    jobs = []
    for _, p in ps.iterrows():
        for c in P.pick_crops(tracks, p, k):
            jobs.append(dict(pass_id=p.pass_id, event=c.event, track=c.track, frame=c.frame,
                             path=str(d / c.event / c.crop)))
    print(len(jobs), "crops to read", flush=True)

    done = set()
    rf = d / "reads.csv"
    if rf.exists():
        old = pd.read_csv(rf)
        done = set(old.path)
    jobs = [j for j in jobs if j["path"] not in done]

    from vlm import Reader
    dt = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dtype=")), None)
    vr = Reader(dtype=dt)
    print("VLM dtype", vr.dtype, flush=True)
    easy = None
    if use_easy:
        import easyocr
        easy = easyocr.Reader(["en"], gpu=True, verbose=False)
    B = 8
    for i in range(0, len(jobs), B):
        chunk = jobs[i:i + B]
        res = vr.read([j["path"] for j in chunk])
        for j, (raw, num) in zip(chunk, res):
            j["vlm_raw"], j["vlm"] = raw, num
            if easy is not None:
                r = [(t, c) for _, t, c in easy.readtext(j["path"], allowlist="0123456789", mag_ratio=1.5)
                     if c > 0.3]
                j["easy"] = "|".join("%s:%.2f" % x for x in r)
        pd.DataFrame(chunk).to_csv(rf, mode="a", header=not rf.exists(), index=False)
        if i % 80 == 0:
            print(i, "/", len(jobs), flush=True)


if __name__ == "__main__":
    main()
