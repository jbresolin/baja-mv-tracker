"""Approach C: appearance re-identification.

Embeds the crops already selected per pass (reads.csv) with DINOv2, averages per pass,
then classifies passes after SPLIT by kNN against a gallery of passes before SPLIT whose
label comes from the GT time match (stand-in for "seed photos labelled by a human").

usage: python reid.py embed      -> out/emb.npz
       python reid.py eval [SPLIT_HH:MM]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "out"


def embed():
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval().half()
    rd = pd.concat([pd.read_csv(d / "reads.csv").assign(src=d.name)
                    for d in sorted(OUT.glob("crops_[0-9]*")) if (d / "reads.csv").exists()])
    paths = list(rd.path)
    embs = []
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + 32]]
            x = proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
            o = model(pixel_values=x)
            e = torch.cat([o.last_hidden_state[:, 0], o.last_hidden_state[:, 1:].mean(1)], 1)
            embs.append(torch.nn.functional.normalize(e.float(), dim=1).cpu().numpy())
    np.savez(OUT / "emb.npz", emb=np.concatenate(embs), src=rd.src.values, pass_id=rd.pass_id.values,
             path=np.array(paths))
    print("embedded", len(paths))


def evaluate(split="11:09", k=5):
    import clock  # noqa
    z = np.load(OUT / "emb.npz", allow_pickle=True)
    ps = pd.read_csv(OUT / "passes_all.csv")
    key = pd.Series(range(len(ps)), index=pd.MultiIndex.from_arrays([ps.src, ps.pass_id]))
    E = pd.DataFrame(z["emb"])
    E["src"], E["pass_id"] = z["src"], z["pass_id"]
    pe = E.groupby(["src", "pass_id"]).mean()
    pe = pe.div(np.linalg.norm(pe.values, axis=1), axis=0)
    ps = ps.set_index(["src", "pass_id"]).join(pe, how="inner").reset_index()
    feat = [c for c in ps.columns if isinstance(c, int)]
    hh, mm = map(int, split.split(":"))
    S = hh * 3600 + mm * 60
    lab = ps[ps.gt_car.notna()]
    gal, test = lab[lab.t0 < S], lab[lab.t0 >= S]
    G, T = gal[feat].values, test[feat].values
    sim = T @ G.T
    preds, confs = [], []
    for row in sim:
        idx = np.argsort(-row)[:k]
        votes = {}
        for j in idx:
            c = gal.gt_car.iloc[j]
            votes[c] = votes.get(c, 0) + row[j]
        c = max(votes, key=votes.get)
        preds.append(c)
        confs.append(votes[c] / sum(votes.values()))
    test = test.assign(reid_car=preds, reid_conf=confs)
    acc = (test.reid_car == test.gt_car).mean()
    print("gallery %d passes (%d cars), test %d passes: reid top1 acc %.3f"
          % (len(gal), gal.gt_car.nunique(), len(test), acc))
    for th in [0.5, 0.6, 0.7, 0.8]:
        s = test[test.reid_conf >= th]
        print("  conf>=%.1f: coverage %.2f acc %.3f" % (th, len(s) / len(test), (s.reid_car == s.gt_car).mean()))
    # fusion: VLM when it has a vote, else re-id
    fused = test.vlm_car.where(test.vlm_car.notna(), test.reid_car)
    print("fusion VLM>reid: acc %.3f (VLM alone %.3f, vlm coverage %.2f)" % (
        (fused == test.gt_car).mean(), (test.vlm_car == test.gt_car).mean(), test.vlm_car.notna().mean()))
    test = test.assign(fused=fused)
    rows = []
    for c, d in test.groupby("gt_car"):
        rows.append(dict(car=int(c), n=len(d), reid_ok=int((d.reid_car == c).sum()),
                         vlm_ok=int((d.vlm_car == c).sum()), fused_ok=int((d.fused == c).sum())))
    print(pd.DataFrame(rows).to_string(index=False))
    test.drop(columns=feat).to_csv(OUT / "reid_test.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        embed()
    else:
        evaluate(*(sys.argv[2:3] or []))
