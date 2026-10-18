"""WP9: train a student car detector (YOLO11n / YOLO11s) on a WP8 YOLO dataset.

usage:
  python students/train_det.py --data datasets/cars/data.yaml --model yolo11n.pt --name cars_n640
  python students/train_det.py --data ... --model yolo11s.pt --imgsz 960 --name cars_s960
  python students/train_det.py --name cars_n640 --resume          # continue an interrupted run
  python students/train_det.py --name cars_n640 --data ... --test-only   # mAP on the data.yaml 'test' split (24SU)
  python students/train_det.py --data ... --name smoke --device cpu --epochs 1 --fraction 32 --batch 8

Runs go to students/runs/<name>/ (weights/best.pt, weights/last.pt, results.csv, student_meta.json, and
eval_test.json = mAP on the data.yaml 'test' split when it has one, run automatically after training).
Never train or tune on the 26SE start-line footage (the benchmark, see TRAINING_PLAN.md).

Augmentation for this domain (broadcast PTZ / fixed trackside cameras, 720p low-bitrate streams -> 1080p start line):
mosaic, HSV (mud, overexposure: the 26SE camera was washed out for ~30 min), scale 0.5 (far cars to frame-filling),
translate, horizontal flip only (no vertical flip, no rotation: the horizon is level), plus a cv2-only
"degrade" transform (Gaussian / horizontal motion blur, JPEG re-compression, down-up scaling) because
albumentations would pull opencv-python-headless next to opencv-python in the shared venv.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "students" / "runs"

# batch sizes that fit the 11 GB 2080 Ti (with ~1.8 GB taken by the desktop) with AMP fp16.
# Estimates from ultralytics' memory scaling; check nvidia-smi in the first epoch, or pass --batch 0.7
# (AutoBatch: use 70% of free VRAM).
DEFAULT_BATCH = {("n", 640): 64, ("s", 640): 32, ("n", 960): 24, ("s", 960): 16}

AUG = dict(
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.45,
    degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, close_mosaic=10, mixup=0.0, cutmix=0.0, copy_paste=0.0,
)
DEGRADE_P = 0.3


class Degrade:
    """Random image degradation (one of: Gaussian blur, horizontal motion blur, JPEG, down-up scale).
    Works on labels['img'] (HWC uint8 BGR) before Format; boxes are unchanged."""

    def __init__(self, p=DEGRADE_P):
        self.p = p

    def __call__(self, labels):
        if random.random() >= self.p:
            return labels
        img = labels["img"]
        k = random.random()
        if k < 0.25:
            s = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (s, s), 0)
        elif k < 0.55:
            n = random.randint(5, 15)
            ker = np.zeros((n, n), np.float32)
            ker[n // 2, :] = 1.0 / n
            img = cv2.filter2D(img, -1, ker)
        elif k < 0.85:
            q = random.randint(25, 70)
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        else:
            h, w = img.shape[:2]
            f = random.uniform(0.4, 0.7)
            small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        labels["img"] = np.ascontiguousarray(img)
        return labels


class Washout:
    """Over-exposure: optional brightening gamma, then a blend towards white (contrast loss), then partial desaturation.
    Targets the washed-out fixed-camera look (26SE 10:24-10:51: mean luminance ~235, 5th percentile ~160), which never
    occurs in the broadcast training frames (3,000 train frames: 5th-percentile luminance <= 91 in 99%, never > 120)."""

    def __init__(self, p):
        self.p = p

    def __call__(self, labels):
        if random.random() >= self.p:
            return labels
        img = labels["img"].astype(np.float32) / 255.0
        img = img ** random.uniform(0.5, 1.0)
        a = random.uniform(0.4, 0.85)
        img = img * (1 - a) + a
        g = img.mean(2, keepdims=True)
        sat = random.uniform(0.3, 1.0)
        img = g + (img - g) * sat
        labels["img"] = np.ascontiguousarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
        return labels


class DegradeDataset(YOLODataset):
    """YOLODataset whose transforms always end with Degrade before Format. Overriding build_transforms (instead of
    inserting once) keeps it after close_mosaic rebuilds the pipeline; module-level so Windows spawn workers can
    unpickle it."""
    degrade_p = 0.0
    washout_p = 0.0

    def build_transforms(self, hyp=None):
        t = super().build_transforms(hyp)
        if self.augment and self.washout_p > 0:
            t.insert(-1, Washout(self.washout_p))  # before Format (numpy -> tensor)
        if self.augment and self.degrade_p > 0:
            t.insert(-1, Degrade(self.degrade_p))
        return t


def make_trainer(degrade_p, washout_p=0.0):
    class StudentTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            ds = super().build_dataset(img_path, mode, batch)
            if mode == "train" and (degrade_p > 0 or washout_p > 0) and type(ds) is YOLODataset:
                ds.__class__ = DegradeDataset
                ds.degrade_p = degrade_p
                ds.washout_p = washout_p
                ds.transforms = ds.build_transforms(self.args)
            return ds

    return StudentTrainer


def git_rev():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def test_eval(weights, data, imgsz, device, out_dir, split="test"):
    """mAP of `weights` on the data.yaml `split` (WP8: the held-out event, e.g. 24SU, labelled by the teacher)."""
    import yaml
    from ultralytics import YOLO
    cfg = yaml.safe_load(Path(data).read_text())
    if not cfg.get(split):
        print("no '%s' split in %s: skipping the held-out evaluation" % (split, data), flush=True)
        return None
    m = YOLO(str(weights)).val(data=str(data), split=split, imgsz=imgsz, batch=16, device=device, plots=True,
                               project=str(out_dir), name="eval_" + split, exist_ok=True)
    res = dict(split=split, imgsz=imgsz, map50=round(float(m.box.map50), 4), map50_95=round(float(m.box.map), 4),
               precision=round(float(m.box.mp), 4), recall=round(float(m.box.mr), 4))
    (Path(out_dir) / ("eval_%s.json" % split)).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="data.yaml (YOLO format, e.g. datasets/cars/data.yaml)")
    ap.add_argument("--model", default="yolo11n.pt", help="yolo11n.pt | yolo11s.pt | a .yaml (from scratch) | a .pt")
    ap.add_argument("--name", required=True, help="run name -> students/runs/<name>/")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--hours", type=float, default=None, help="max training hours (ultralytics `time`, overrides epochs)")
    ap.add_argument("--patience", type=int, default=25, help="early stop after N epochs without val fitness gain")
    ap.add_argument("--batch", type=float, default=None,
                    help="int batch, or a fraction <1 for AutoBatch; default from DEFAULT_BATCH (GPU) or 8 (CPU)")
    ap.add_argument("--device", default=None, help="0 (GPU) or cpu; default: 0 if CUDA is visible")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="false", choices=["false", "ram", "disk"])
    ap.add_argument("--fraction", type=float, default=1.0, help="train subset (fraction, or count if > 1)")
    ap.add_argument("--degrade", type=float, default=DEGRADE_P, help="probability of the blur/JPEG transform")
    ap.add_argument("--washout", type=float, default=0.0,
                    help="probability of the over-exposure transform (0 = off, as cars_n640 / cars_s640)")
    ap.add_argument("--cos-lr", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="resume students/runs/<name>/weights/last.pt")
    ap.add_argument("--exist-ok", action="store_true", help="reuse an existing run dir (overwrites it)")
    ap.add_argument("--test-only", action="store_true",
                    help="only evaluate students/runs/<name>/weights/best.pt on the data.yaml 'test' split")
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO

    run_dir = RUNS / a.name
    trainer = make_trainer(a.degrade, a.washout)
    if a.test_only:
        dev = a.device if a.device is not None else ("0" if torch.cuda.device_count() > 0 else "cpu")
        test_eval(run_dir / "weights" / "best.pt", Path(a.data).resolve(), a.imgsz, dev, run_dir)
        return
    if a.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            sys.exit("nothing to resume: %s" % last)
        mf = run_dir / "student_meta.json"
        if mf.exists():  # the custom transforms are not in ultralytics' args: take them from the original run
            m = json.loads(mf.read_text())
            trainer = make_trainer(m.get("degrade", a.degrade), m.get("washout", 0.0))
        print("resuming", last, flush=True)
        YOLO(str(last)).train(resume=True, trainer=trainer)
        return
    if not a.data:
        sys.exit("--data is required (unless --resume)")
    if run_dir.exists() and not a.exist_ok:
        sys.exit("%s exists: use --resume, --exist-ok, or another --name" % run_dir)

    device = a.device if a.device is not None else ("0" if torch.cuda.device_count() > 0 else "cpu")
    size = "s" if "11s" in Path(a.model).stem else "n"
    if a.batch is None:
        batch = DEFAULT_BATCH.get((size, a.imgsz), 16) if device != "cpu" else 8
    else:
        batch = a.batch if a.batch < 1 else int(a.batch)

    args = dict(data=str(Path(a.data).resolve()), imgsz=a.imgsz, epochs=a.epochs, patience=a.patience, batch=batch,
                device=device, workers=a.workers, cache=False if a.cache == "false" else a.cache,
                fraction=a.fraction if a.fraction <= 1 else int(a.fraction), project=str(RUNS), name=a.name,
                exist_ok=True, amp=True, seed=a.seed, deterministic=False, cos_lr=a.cos_lr,
                optimizer="auto", plots=True, val=True, **AUG)
    if a.hours:
        args["time"] = a.hours
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), git=git_rev(), model=a.model, degrade=a.degrade,
                washout=a.washout,
                argv=sys.argv, train_args=args, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if device != "cpu" else None)
    (run_dir / "student_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(args), flush=True)
    t = time.time()
    YOLO(a.model).train(trainer=trainer, **args)
    meta["train_s"] = round(time.time() - t, 1)
    (run_dir / "student_meta.json").write_text(json.dumps(meta, indent=1))
    print("done in %.0f s -> %s" % (time.time() - t, run_dir / "weights" / "best.pt"), flush=True)
    meta["test"] = test_eval(run_dir / "weights" / "best.pt", args["data"], a.imgsz, device, run_dir)
    (run_dir / "student_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
