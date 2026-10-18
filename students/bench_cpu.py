"""WP9/WP11 proxy: export a YOLO detector to ONNX / OpenVINO and measure CPU latency + RAM at 640 px.

usage:
  python students/bench_cpu.py --weights weights/yolo11n.pt
  python students/bench_cpu.py --weights students/runs/cars_n640/weights/best.pt --threads 1,4 --formats torch,onnx,openvino

Exports go to students/runs/bench/<stem>/ (a copy of the weights is exported there, so nothing is written next to
the source weights). Each (format, threads) is measured in a fresh subprocess, so RAM = that process alone:
  * pre  = letterbox a real 1920x1080 frame to imgsz (640 = 640x640, or HxW like 384x640 for 16:9) + normalise
  * inf  = the network forward pass (ONNX Runtime session.run / OpenVINO infer / torch forward)
  * post = confidence filter + NMS (cv2.dnn.NMSBoxes)
  * RAM  = peak working set of the process (psutil; includes the Python interpreter, ~30 MB), and the delta
           over the interpreter + imports baseline measured before loading the model.
The ONNX / OpenVINO workers import only numpy, cv2 and the runtime (no torch / ultralytics), as a laptop app would.
Target (TRAINING_PLAN WP11): <= 50 ms/frame for the detector, < 1 GB RAM.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "students" / "runs" / "bench"
FRAME_VIDEO = ROOT / "2026-10-17 11-29-49.mkv"
FRAME_T = 645.0  # a car in front of the camera (11:40:34)


def get_frame(path):
    import cv2
    import numpy as np
    if not path.exists():
        out = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(FRAME_T), "-i", str(FRAME_VIDEO), "-frames:v", "1",
                              "-vf", "scale=1920:1080", "-f", "image2pipe", "-vcodec", "png", "-"],
                             capture_output=True).stdout
        img = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR) if out else None
        if img is None:  # no ffmpeg / video: synthetic frame
            img = (np.random.default_rng(0).random((1080, 1920, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(path), img)
    return cv2.imread(str(path))


def hw(imgsz):
    """'640' -> (640, 640); '384x640' -> (384, 640) (height x width, multiples of 32)."""
    p = [int(x) for x in str(imgsz).lower().split("x")]
    return (p[0], p[0]) if len(p) == 1 else (p[0], p[1])


def letterbox(img, size):
    import cv2
    import numpy as np
    sh, sw = size
    h, w = img.shape[:2]
    r = min(sh / h, sw / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((sh, sw, 3), 114, np.uint8)
    top, left = (sh - nh) // 2, (sw - nw) // 2
    out[top:top + nh, left:left + nw] = im
    x = out[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), r, (left, top)


def postprocess(y, conf=0.25, iou=0.7):
    """y: (1, 4 + nc, N) raw YOLO11 head output -> boxes, scores, classes after NMS."""
    import cv2
    import numpy as np
    p = y[0].T  # (N, 4 + nc)
    sc = p[:, 4:]
    cls = sc.argmax(1)
    s = sc[np.arange(len(p)), cls]
    k = s > conf
    b, s, cls = p[k, :4], s[k], cls[k]
    xywh = np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 2], b[:, 3]]
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), s.tolist(), conf, iou) if len(s) else []
    idx = np.array(idx).reshape(-1)
    return xywh[idx], s[idx], cls[idx]


def worker(fmt, path, threads, imgsz, n, warm, frame_path, priority="normal"):
    import psutil
    proc = psutil.Process()
    if priority == "high" and hasattr(psutil, "HIGH_PRIORITY_CLASS"):
        proc.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows: approximates an idle machine when other jobs load the CPU
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import numpy as np  # noqa: F401
    import cv2
    cv2.setNumThreads(1)
    frame = get_frame(Path(frame_path))
    if fmt == "onnx":
        import onnxruntime as ort
        base = proc.memory_info().rss
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name

        def infer(x):
            return sess.run(None, {name: x})[0]
        version = ort.__version__
    elif fmt == "openvino":
        import openvino as ov
        base = proc.memory_info().rss
        core = ov.Core()
        xml = next(Path(path).glob("*.xml"))
        cm = core.compile_model(str(xml), "CPU", {"INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY",
                                                  "NUM_STREAMS": 1})
        req = cm.create_infer_request()

        def infer(x):
            return req.infer({0: x})[cm.output(0)]
        version = ov.__version__
    else:  # torch (the .pt, ultralytics model's nn.Module, fused)
        import torch
        torch.set_num_threads(threads)
        from ultralytics import YOLO
        base = proc.memory_info().rss
        m = YOLO(str(path)).model.float().eval().fuse()

        def infer(x):
            with torch.inference_mode():
                y = m(torch.from_numpy(x))
            return (y[0] if isinstance(y, (list, tuple)) else y).numpy()
        version = torch.__version__
    t_pre, t_inf, t_post, ndet = [], [], [], 0
    for i in range(n + warm):
        a = time.perf_counter()
        x, r, pad = letterbox(frame, hw(imgsz))
        b = time.perf_counter()
        y = infer(x)
        c = time.perf_counter()
        boxes, scores, cls = postprocess(y)
        d = time.perf_counter()
        if i >= warm:
            t_pre.append(b - a); t_inf.append(c - b); t_post.append(d - c)
            ndet = len(scores)
    mi = proc.memory_info()
    peak = getattr(mi, "peak_wset", mi.rss)
    med = lambda v: round(1000 * sorted(v)[len(v) // 2], 2)  # noqa: E731
    p90 = lambda v: round(1000 * sorted(v)[int(len(v) * 0.9)], 2)  # noqa: E731
    tot = [p + q + s for p, q, s in zip(t_pre, t_inf, t_post)]
    return dict(format=fmt, runtime=version, threads=threads, priority=priority, imgsz=imgsz, n=n, pre_ms=med(t_pre), inf_ms=med(t_inf),
                post_ms=med(t_post), total_ms=med(tot), total_p90_ms=p90(tot), detections=int(ndet),
                rss_mb=round(mi.rss / 2**20), peak_mb=round(peak / 2**20),
                model_delta_mb=round((peak - base) / 2**20), base_mb=round(base / 2**20))


def export(weights, imgsz, formats):
    from ultralytics import YOLO
    d = BENCH / Path(weights).stem / str(imgsz)
    imgsz = list(hw(imgsz))
    d.mkdir(parents=True, exist_ok=True)
    w = d / Path(weights).name
    if not w.exists():
        shutil.copy2(weights, w)
    paths = {"torch": w}
    if "onnx" in formats:
        p = d / (w.stem + ".onnx")
        if not p.exists():
            YOLO(str(w)).export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False, device="cpu")
        paths["onnx"] = p
    if "openvino" in formats:
        p = d / (w.stem + "_openvino_model")
        if not p.exists():
            YOLO(str(w)).export(format="openvino", imgsz=imgsz, device="cpu")
        paths["openvino"] = p
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "weights" / "yolo11n.pt"))
    ap.add_argument("--imgsz", default="640", help="640 (square) or HxW, e.g. 384x640 (16:9 frames, less padding)")
    ap.add_argument("--formats", default="torch,onnx,openvino")
    ap.add_argument("--threads", default="1,4")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=1, help="interleaved repeats; the fastest per config is reported")
    ap.add_argument("--priority", default="normal", choices=["normal", "high"],
                    help="process priority of the measuring worker (high: less disturbed by other CPU jobs)")
    ap.add_argument("--worker", nargs=3, metavar=("FMT", "PATH", "THREADS"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    frame_path = BENCH / "frame_1080p.png"
    if a.worker:
        fmt, path, th = a.worker
        print("RESULT " + json.dumps(worker(fmt, path, int(th), a.imgsz, a.n, a.warmup, str(frame_path), a.priority)),
              flush=True)
        return
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    BENCH.mkdir(parents=True, exist_ok=True)
    get_frame(frame_path)
    formats = a.formats.split(",")
    paths = export(a.weights, a.imgsz, formats)
    rows = []
    try:
        import psutil
        load_before = psutil.cpu_percent(interval=1.0)
    except ImportError:
        load_before = None
    configs = [(fmt, int(th)) for fmt in formats for th in a.threads.split(",")]
    allruns = {c: [] for c in configs}
    for rnd in range(a.rounds):  # interleaved rounds; keep the fastest round per config (least disturbed)
        for fmt, th in configs:
            out = subprocess.run([sys.executable, __file__, "--worker", fmt, str(paths[fmt]), str(th), "--imgsz",
                                  str(a.imgsz), "--n", str(a.n), "--warmup", str(a.warmup), "--priority", a.priority],
                                 capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
            line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
            if not line:
                print(fmt, th, "FAILED:", out.stderr[-2000:], flush=True)
                continue
            r = json.loads(line[0][7:])
            allruns[(fmt, th)].append(r)
            print("round %d" % rnd, json.dumps(r), flush=True)
    for c in configs:
        if allruns[c]:
            best = dict(min(allruns[c], key=lambda r: r["total_ms"]))
            best["rounds_total_ms"] = [r["total_ms"] for r in allruns[c]]
            rows.append(best)
    try:
        import psutil
        cpu = dict(cpu_load_pct_before=load_before, cpu_load_pct_after=psutil.cpu_percent(interval=1.0),
                   logical_cpus=psutil.cpu_count(), priority=a.priority)
    except ImportError:
        cpu = {}
    res = dict(weights=str(a.weights), imgsz=a.imgsz, when=time.strftime("%Y-%m-%d %H:%M:%S"), **cpu, results=rows)
    out = BENCH / ("%s_%s_%s.json" % (Path(a.weights).stem, a.imgsz, a.priority))
    out.write_text(json.dumps(res, indent=1))
    import pandas as pd
    print(pd.DataFrame(rows)[["format", "threads", "pre_ms", "inf_ms", "post_ms", "total_ms", "total_p90_ms",
                              "detections", "peak_mb", "model_delta_mb", "rounds_total_ms"]].to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    main()
