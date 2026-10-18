"""WP0: detector throughput on 720p stream frames (YOLO-World v2-x prompts as in lapvision/det.py, or COCO YOLO11x).

usage (repo root): python tools/bench/yolo.py <video> [--model=yolov8x-worldv2.pt] [--imgsz=960,1280] [--batch=1,8,16]
                   [--n=160] [--start=900]
Frames: n frames at 4 fps from <video> starting at <start> s, decoded once into RAM (decode is benchmarked separately).
Reports img/s (predict incl. pre/post-processing, fp16) and peak VRAM (torch.cuda.max_memory_allocated, nvidia-smi).

end-to-end: python tools/bench/yolo.py <video> e2e [--model=..] [--imgsz=1280] [--batch=8] [--fps=4] [--dur=300]
            [--start=900] [--hw=0|1]
  ffmpeg (CPU or -hwaccel cuda) pipes frames at --fps while a reader thread feeds batched predict: x real time overall.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LV = ROOT / "lapvision"
sys.path.insert(0, str(LV))
sys.path.insert(0, str(Path(__file__).resolve().parent))
WINGET = r"C:\Users\baja\AppData\Local\Microsoft\WinGet\Links"
if not shutil.which("ffmpeg") and os.path.isdir(WINGET):
    os.environ["PATH"] = WINGET + os.pathsep + os.environ["PATH"]

from vlm_fp16 import SmiPeak, arg  # noqa: E402


def frames(video, start, n, fps=4, w=1280, h=720):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", str(start), "-i", video, "-an", "-vf", "fps=%g" % fps,
           "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    b = subprocess.run(cmd, capture_output=True, check=True).stdout
    return list(np.frombuffer(b, np.uint8).reshape(-1, h, w, 3))


def main():
    video = str(Path(sys.argv[1]).resolve())
    model = arg("model", "yolov8x-worldv2.pt")
    sizes = [int(x) for x in arg("imgsz", "960,1280").split(",")]
    batches = [int(x) for x in arg("batch", "1,8,16").split(",")]
    n = int(arg("n", 160))
    fr = frames(video, float(arg("start", 900)), n)
    os.chdir(LV)  # det.py / YOLO-World look for weights relative to lapvision/
    from det import Detector
    det = Detector(model)
    print("model %s, %d frames %dx%d from %s" % (model, len(fr), fr[0].shape[1], fr[0].shape[0], Path(video).name))
    print("%-6s %-5s %8s %8s %11s %10s %8s" % ("imgsz", "batch", "img/s", "ms/img", "torch_peak", "smi_peak", "dets/img"))
    for s in sizes:
        for B in batches:
            det.predict(fr[:B], imgsz=s, conf=0.15, half=True)  # warm-up / build
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            nd = 0
            with SmiPeak() as sp:
                t = time.time()
                for i in range(0, len(fr), B):
                    for r in det.predict(fr[i:i + B], imgsz=s, conf=0.15, half=True):
                        nd += len(r.boxes)
                torch.cuda.synchronize()
                dt = time.time() - t
            print("%-6d %-5d %8.1f %8.1f %8.2f GB %7.2f GB %8.2f" % (
                s, B, len(fr) / dt, 1000 * dt / len(fr), torch.cuda.max_memory_allocated() / 1e9, sp.peak / 1024,
                nd / len(fr)), flush=True)


def e2e():
    import queue
    import threading
    video = str(Path(sys.argv[1]).resolve())
    model = arg("model", "yolov8x-worldv2.pt")
    s, B, fps = int(arg("imgsz", 1280)), int(arg("batch", 8)), float(arg("fps", 4))
    dur, start, hw = float(arg("dur", 300)), float(arg("start", 900)), arg("hw", "0") == "1"
    os.chdir(LV)
    from det import Detector
    det = Detector(model)
    det.predict([np.zeros((720, 1280, 3), np.uint8)] * B, imgsz=s, conf=0.15, half=True)
    w, h = 1280, 720
    cmd = ["ffmpeg", "-v", "error", "-nostdin"] + (["-hwaccel", "cuda"] if hw else []) + [
        "-ss", str(start), "-t", str(dur), "-i", video, "-an", "-vf", "fps=%g" % fps,
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    q = queue.Queue(maxsize=4 * B)

    def reader(p):
        fs = w * h * 3
        while True:
            b = p.stdout.read(fs)
            if len(b) < fs:
                break
            q.put(np.frombuffer(b, np.uint8).reshape(h, w, 3))
        q.put(None)

    t = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    threading.Thread(target=reader, args=(p,), daemon=True).start()
    n, done = 0, False
    while not done:
        batch = []
        while len(batch) < B:
            f = q.get()
            if f is None:
                done = True
                break
            batch.append(f)
        if batch:
            det.predict(batch, imgsz=s, conf=0.15, half=True)
            n += len(batch)
    dt = time.time() - t
    print("e2e %s imgsz %d batch %d fps %g decode=%s: %d frames in %.1f s -> %.1f img/s, %.1fx real time" % (
        model, s, B, fps, "cuda" if hw else "cpu", n, dt, n / dt, dur / dt), flush=True)


if __name__ == "__main__":
    e2e() if "e2e" in sys.argv[2:] else main()
