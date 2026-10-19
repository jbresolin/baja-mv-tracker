"""All-events training sets for the final deployable models (no held-out split left).

usage: python students/build_all_events.py
  datasets/cars_all              <- datasets/cars           train + val (25SU) + test (24SU)
  datasets/numbers_crop_r2_all   <- datasets/numbers_crop_r2 train + val + test

Images and labels are hard links (nothing is copied, the sources are not modified, and ultralytics' label caches land
in the new folders). images/train holds every split; images/val is 25SU again, ONLY to log curves: it is part of the
training data, so its metrics are not held-out numbers and early stopping is off (fixed epoch count).
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def link(a, b):
    if b.exists():
        return
    try:
        os.link(a, b)
    except OSError:
        shutil.copy2(a, b)


def build(src, dst, name):
    src, dst = ROOT / "datasets" / src, ROOT / "datasets" / dst
    n = 0
    for sub in ("images", "labels"):
        for d in ("train", "val"):
            (dst / sub / d).mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            for f in (src / sub / split).iterdir():
                if f.suffix == ".cache":
                    continue
                # prefix keeps names unique across splits (they already are, but cheap insurance)
                link(f, dst / sub / "train" / ("%s_%s" % (split, f.name)))
                if split == "val":
                    link(f, dst / sub / "val" / f.name)
                n += sub == "images"
    (dst / "data.yaml").write_text(
        "# all events (train + val + test of %s); val = 25SU for curves only, NOT held out\n"
        "path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: %s\n" % (src.name, dst.as_posix(), name))
    print(dst, n, "training images")


if __name__ == "__main__":
    build("cars", "cars_all", "car")
    build("numbers_crop_r2", "numbers_crop_r2_all", "number")
