"""WP10: export the recognizer to ONNX and measure CPU latency (ONNX Runtime, 1 and 4 threads; torch for reference).

usage: CUDA_VISIBLE_DEVICES=-1 python students/recog/export_onnx.py --name real [--bench-only]
-> students/runs/recog/<name>/recog.onnx (input "image": (N,3,32,W) float RGB in [0,1], dynamic N and W;
   output "logp": (W/4, N, 11) CTC log-probs, class 0 = blank, d+1 = digit d), bench.json.
Latency = per crop at batch 1 (one number box per call, the worst case) and per crop at batch 8 (several boxes / frames
per call), median of 300 runs after warm-up, preprocessing (resize + pad of a real crop) timed separately.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def export(name):
    import torch
    from common import IMG_H, IMG_W, RUNS, load_model, n_params
    m, _ = load_model(RUNS / name / "best.pt")
    x = torch.rand(1, 3, IMG_H, IMG_W)
    out = RUNS / name / "recog.onnx"
    torch.onnx.export(m, x, str(out), input_names=["image"], output_names=["logp"], opset_version=17,
                      dynamic_axes={"image": {0: "n", 3: "w"}, "logp": {0: "t", 1: "n"}}, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    xb = torch.rand(4, 3, IMG_H, IMG_W)
    d = np.abs(s.run(None, {"image": xb.numpy()})[0] - m(xb).detach().numpy()).max()
    print("exported", out, "params", n_params(m), "max |onnx - torch| =", d)
    return out, n_params(m), float(d)


def bench_worker(kind, path, threads, batch):
    """runs in a fresh process so the thread setting is clean."""
    import cv2
    from common import RUNS, batch_tensor
    from augment import plain
    items = [i for i in pickle.load(open(RUNS / "cache_test.pkl", "rb")) if not i["neg"]][:64]
    t = time.perf_counter()
    for it in items:
        plain(it["img"], it["box"])
    pre_ms = (time.perf_counter() - t) / len(items) * 1000
    X = batch_tensor([plain(it["img"], it["box"]) for it in items[:batch]]).numpy()
    if kind == "ort":
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        f = lambda: s.run(None, {"image": X})  # noqa: E731
    else:
        import torch
        torch.set_num_threads(threads)
        from common import load_model
        m, _ = load_model(path)
        xt = torch.from_numpy(X)

        def f():
            with torch.no_grad():
                return m(xt)
    for _ in range(30):
        f()
    ts = []
    for _ in range(300):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return dict(kind=kind, threads=threads, batch=batch, ms_per_call=float(np.median(ts) * 1000),
                ms_per_crop=float(np.median(ts) * 1000 / batch), pre_ms_per_crop=pre_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--worker", nargs=4)
    a = ap.parse_args()
    if a.worker:
        kind, path, th, b = a.worker
        print("RESULT", json.dumps(bench_worker(kind, path, int(th), int(b))))
        return
    from common import RUNS
    d = RUNS / a.name
    info = {}
    if not a.bench_only:
        out, n, diff = export(a.name)
        info.update(params=n, max_abs_diff=diff)
    res = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    for kind, path in (("ort", d / "recog.onnx"), ("torch", d / "best.pt")):
        for th in (1, 4):
            for b in (1, 8):
                o = subprocess.run([sys.executable, __file__, "--worker", kind, str(path), str(th), str(b)],
                                   capture_output=True, text=True, env=env)
                line = [x for x in o.stdout.splitlines() if x.startswith("RESULT")]
                if not line:
                    print(o.stderr[-2000:])
                    continue
                r = json.loads(line[0][7:])
                res.append(r)
                print(r, flush=True)
    info["bench"] = res
    json.dump(info, open(d / "bench.json", "w"), indent=1)


if __name__ == "__main__":
    main()
