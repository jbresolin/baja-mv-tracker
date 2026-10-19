"""Laptop benchmark: run this first on the real track-side laptop. One command, ~2-4 minutes, no camera needed:

  python rt/bench_laptop.py                       # synthetic frame
  python rt/bench_laptop.py --video some.mkv      # a real 1080p frame (and ffmpeg decode cost) from a recording
  python rt/bench_laptop.py --sustain 120         # + 2 min at 10 fps to expose thermal throttling

Measures, per model x backend/device x threads: wall ms per call (median, p90) and CPU ms per call (all threads of the
process, i.e. what it costs the machine), for
  * the car detector(s) on a 1080p frame (letterbox + inference + NMS: one 10 fps frame),
  * the number-panel detector per car crop and the recognizer per number box (the identity chain),
on ONNX Runtime CPU (1/2/4 threads) and OpenVINO CPU (1/2/4) / GPU (Intel iGPU) when available.
Then a verdict for 10 fps: CPU share of the whole machine = detector + identity (busy finish: one pass every 6 s,
30 crops, ~0.8 boxes per crop) + decode (measured with --video, else assumed 15% of one core), against the budget
GO <= 50% of all cores and p90 <= 60 ms; MARGINAL <= 75% and p90 <= 90 ms; else NO-GO. Writes bench_laptop.json.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from rt import config as C  # noqa: E402
from rt.models import Recognizer, RosterScorer, YoloOnnx, plain  # noqa: E402

BUSY_PASS_EVERY_S = 6.0      # 26SE finish 12:04: packs of cars, ~10 passes per minute
CROPS_PER_PASS, BOXES_PER_CROP = 30, 0.8


def sysinfo():
    info = dict(platform=platform.platform(), python=sys.version.split()[0], machine=platform.machine(),
                processor=platform.processor(), logical_cpus=os.cpu_count())
    try:
        import psutil
        info.update(physical_cores=psutil.cpu_count(logical=False), ram_gb=round(psutil.virtual_memory().total / 2**30, 1))
        f = psutil.cpu_freq()
        if f:
            info.update(cpu_mhz_now=f.current, cpu_mhz_max=f.max)
    except ImportError:
        pass
    if sys.platform == "win32":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name; (Get-CimInstance Win32_VideoController).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
            info["cpu_name"], info["gpus"] = out[0].strip() if out else "", [x.strip() for x in out[1:]]
        except Exception:  # noqa: BLE001
            pass
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
    except ImportError:
        info["onnxruntime"] = None
    try:
        import openvino as ov
        info["openvino"] = ov.__version__
        info["openvino_devices"] = ov.Core().available_devices
    except Exception:  # noqa: BLE001
        info["openvino"], info["openvino_devices"] = None, []
    return info


def get_frame(a, cfg):
    if a.video:
        t = a.video_t
        out = subprocess.run([cfg["capture"]["ffmpeg"], "-v", "error", "-ss", str(t), "-i", a.video, "-frames:v", "1",
                              "-vf", "scale=1920:-2", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
        img = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR) if out else None
        if img is not None:
            return img, "video frame at %s s" % t
    if a.image:
        return cv2.imread(a.image), a.image
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur((rng.random((1080, 1920, 3)) * 255).astype(np.uint8), (0, 0), 3)
    return img, "synthetic 1920x1080"


def timeit(fn, n, warm):
    for _ in range(warm):
        fn()
    w = []
    p0 = time.process_time()  # all threads of the process; Windows ticks are 15.6 ms, so total / n, not per call
    for _ in range(n):
        t = time.perf_counter()
        fn()
        w.append(time.perf_counter() - t)
    cpu = (time.process_time() - p0) / n * 1000
    w = np.array(w) * 1000
    return dict(ms_p50=round(float(np.median(w)), 2), ms_p90=round(float(np.percentile(w, 90)), 2),
                cpu_ms=round(float(cpu), 2))


def configs(info, threads):
    out = [("onnxruntime", "CPU", t) for t in threads]
    if info.get("openvino"):
        out += [("openvino", "CPU", t) for t in threads]
        for d in info.get("openvino_devices", []):
            if d.startswith("GPU"):
                out.append(("openvino", d, 0))
    return out


def decode_cost(a, cfg):
    """ffmpeg decode of 20 s of the video, sampled to 10 fps as rt/capture.py does: CPU % of one core."""
    if not a.video:
        return None
    try:
        import psutil
    except ImportError:
        return None
    ff = cfg["capture"]["ffmpeg"]
    pr = json.loads(subprocess.run([cfg["capture"]["ffprobe"], "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=r_frame_rate,width", "-of", "json", a.video], capture_output=True,
                                   text=True).stdout)["streams"][0]
    n, d = pr["r_frame_rate"].split("/")
    step = max(1, round(float(n) / float(d) / 10))
    vf = "framestep=%d" % step + (",scale=1920:-2" if int(pr["width"]) > 1920 else "")
    hw = ["-hwaccel", a.hwaccel] if a.hwaccel and a.hwaccel != "none" else []
    t = time.perf_counter()
    p = subprocess.Popen([ff, "-v", "error"] + hw + ["-ss", str(a.video_t), "-i", a.video, "-t", "20", "-vf", vf,
                          "-fps_mode", "passthrough", "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE)
    pp = psutil.Process(p.pid)
    cpu = 0.0
    while p.poll() is None:
        try:
            ct = pp.cpu_times()
            cpu = ct.user + ct.system
        except Exception:  # noqa: BLE001
            pass
        if not p.stdout.read(1 << 20):
            break
    p.wait()
    wall = time.perf_counter() - t
    return dict(video_s=20, hwaccel=a.hwaccel, wall_s=round(wall, 2), cpu_pct_one_core_realtime=round(100 * cpu / 20, 1),
                speed_x=round(20 / wall, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--video", default=None)
    ap.add_argument("--video-t", type=float, default=645.0)
    ap.add_argument("--hwaccel", default="none", help="decode test with ffmpeg -hwaccel (e.g. d3d11va, qsv)")
    ap.add_argument("--image", default=None)
    ap.add_argument("--car", action="append", default=None, help="car detector .onnx (repeatable); default: config")
    ap.add_argument("--threads", default="1,2,4")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--sustain", type=float, default=0, help="seconds at 10 fps with the best config (throttling)")
    ap.add_argument("--out", default="bench_laptop.json")
    a = ap.parse_args()
    cfg = C.load(a.config, a.set)
    cv2.setNumThreads(1)
    info = sysinfo()
    ncpu = info.get("physical_cores") or info.get("logical_cpus") or 4  # hyper-threads are not full cores
    print(json.dumps(info, indent=1))
    frame, fdesc = get_frame(a, cfg)
    print("frame:", fdesc)
    threads = [int(x) for x in a.threads.split(",")]
    cars = a.car or [cfg["models"]["car_detector"]] + ([C.resolve(cfg["models"]["car_detector_alt"]).as_posix()]
                                                       if cfg["models"].get("car_detector_alt") else [])
    cars = [c for c in cars if Path(c).exists()]
    res = dict(system=info, frame=fdesc, car=[], panel=[], recog=[])
    log = lambda m: print("  " + m)  # noqa: E731
    for car in cars:
        for be, dev, th in configs(info, threads):
            det = YoloOnnx(car, cfg["models"]["car_imgsz"], max(th, 1), backend=be, device=dev, log=log)
            if be == "openvino" and det.runner.backend != "openvino":
                continue
            r = timeit(lambda: det(frame, conf=0.15), a.n, 5)
            r.update(model=Path(car).parent.parent.name or Path(car).name, backend=be, device=dev, threads=th,
                     desc=det.desc)
            res["car"].append(r)
            print("car   %-22s %-28s p50 %6.1f ms  p90 %6.1f ms  cpu %6.1f ms/frame -> %5.1f%% of one core at 10 fps"
                  % (r["model"][:22], det.desc[:28], r["ms_p50"], r["ms_p90"], r["cpu_ms"], r["cpu_ms"]))
    # identity chain on car-sized crops of the frame
    H, W = frame.shape[:2]
    crop = frame[int(0.45 * H):int(0.85 * H), int(0.3 * W):int(0.6 * W)]
    roster = list(range(1, 25))
    sc = RosterScorer(roster)
    m = cfg["models"]
    box = [crop.shape[1] * 0.4, crop.shape[0] * 0.3, crop.shape[1] * 0.55, crop.shape[0] * 0.5]
    for be, dev, th in configs(info, threads):
        pdet = YoloOnnx(m["panel_detector"], int(m["panel_imgsz"]), max(th, 1), rect=cfg["identity"]["panel_rect"],
                        backend=be, device=dev, log=log)
        if be == "openvino" and pdet.runner.backend != "openvino":
            continue
        r = timeit(lambda: pdet(crop, conf=0.1), a.n, 5)
        r.update(backend=be, device=dev, threads=th, desc=pdet.desc)
        res["panel"].append(r)
        rec = Recognizer(m["recognizer"], max(th, 1), backend=be, device=dev, log=log)
        x1 = [plain(crop, box)]
        r2 = timeit(lambda: sc(rec(x1)), a.n, 5)
        r2.update(backend=be, device=dev, threads=th, desc=rec.desc)
        res["recog"].append(r2)
        print("panel %-28s p50 %6.1f ms/crop  cpu %6.1f | recog p50 %6.1f ms/box  cpu %6.1f"
              % (pdet.desc[:28], r["ms_p50"], r["cpu_ms"], r2["ms_p50"], r2["cpu_ms"]))
    dec = decode_cost(a, cfg)
    res["decode"] = dec
    dec_pct = dec["cpu_pct_one_core_realtime"] if dec else 15.0
    if dec:
        print("decode (ffmpeg, 10 fps sampling): %.1f%% of one core in real time, %.1fx real time" %
              (dec_pct, dec["speed_x"]))
    # identity cost per pass with the cheapest CPU config
    best_id = min(zip(res["panel"], res["recog"]), key=lambda pr: pr[0]["cpu_ms"] + BOXES_PER_CROP * pr[1]["cpu_ms"])
    per_pass_cpu_s = CROPS_PER_PASS * (best_id[0]["cpu_ms"] + BOXES_PER_CROP * best_id[1]["cpu_ms"]) / 1000
    per_pass_wall_s = CROPS_PER_PASS * (best_id[0]["ms_p50"] + BOXES_PER_CROP * best_id[1]["ms_p50"]) / 1000
    id_pct = 100 * per_pass_cpu_s / BUSY_PASS_EVERY_S
    print("identity per pass (%d crops): %.2f s CPU, %.2f s wall (%s); busy finish (1 pass / %.0f s): %.1f%% of one "
          "core" % (CROPS_PER_PASS, per_pass_cpu_s, per_pass_wall_s, best_id[0]["desc"], BUSY_PASS_EVERY_S, id_pct))
    res["identity"] = dict(per_pass_cpu_s=round(per_pass_cpu_s, 2), per_pass_wall_s=round(per_pass_wall_s, 2),
                           busy_pct_one_core=round(id_pct, 1), config=best_id[0]["desc"])
    print("\nverdict for 10 fps (machine = %d physical cores; budget = share of ALL cores; the rest is for the OS/OBS):" % ncpu)
    for r in res["car"]:
        tot = r["cpu_ms"] + id_pct + dec_pct + 2.0  # + tracker / pass logic ~2% of a core
        share = tot / ncpu
        if r["ms_p90"] <= 60 and share <= 50:
            v = "GO"
        elif r["ms_p90"] <= 90 and share <= 75:
            v = "MARGINAL"
        else:
            v = "NO-GO"
        r.update(total_pct_one_core=round(tot, 1), machine_share_pct=round(share, 1), verdict=v)
        print("  %-8s %-22s %-30s frame p90 %6.1f ms, total %6.1f%% of one core = %5.1f%% of the machine"
              % (v, r["model"][:22], r["desc"][:30], r["ms_p90"], tot, share))
    if a.sustain and res["car"]:
        go = [r for r in res["car"] if r["verdict"] != "NO-GO"] or res["car"]
        b = min(go, key=lambda r: r["cpu_ms"])
        det = YoloOnnx([c for c in cars if Path(c).parent.parent.name == b["model"]][0], cfg["models"]["car_imgsz"],
                       max(b["threads"], 1), backend=b["backend"], device=b["device"])
        print("\nsustained %.0f s at 10 fps with %s %s ..." % (a.sustain, b["model"], b["desc"]))
        t0, k, lat = time.monotonic(), 0, []
        while time.monotonic() - t0 < a.sustain:
            due = t0 + k * 0.1
            if due > time.monotonic():
                time.sleep(due - time.monotonic())
            s = time.perf_counter()
            det(frame, conf=0.15)
            lat.append((time.monotonic() - t0, 1000 * (time.perf_counter() - s)))
            k += 1
        L = np.array(lat)
        q = max(1, len(L) // 4)
        first, last = np.percentile(L[:q, 1], 90), np.percentile(L[-q:, 1], 90)
        late = int((L[1:, 0] - L[:-1, 0] > 0.15).sum())
        res["sustain"] = dict(seconds=a.sustain, frames=len(L), p90_first_quarter=round(float(first), 1),
                              p90_last_quarter=round(float(last), 1), late_frames=late)
        print("  p90 first quarter %.1f ms, last quarter %.1f ms (%s), %d frames later than 150 ms" % (
            first, last, "THROTTLING?" if last > 1.3 * first else "stable", late))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("\n->", a.out, "(send this file back)")


if __name__ == "__main__":
    main()
