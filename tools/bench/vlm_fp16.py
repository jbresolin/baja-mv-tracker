"""WP0: re-read stored crops with Qwen3-VL in another dtype and compare with the stored (bf16, RTX 5080) reads.

usage (from the repo root):
  python tools/bench/vlm_fp16.py compare [--dtype=fp16] [--attn=sdpa] [--n=500] [--batch=8] [--tag=fp16]
  python tools/bench/vlm_fp16.py speed   [--dtype=fp16] [--attn=sdpa] [--n=48] [--batches=1,4,8]
compare: stratified sample of out/world_*/reads.csv (60% numeric stored reads, 40% 'none'), writes
         out/fp16_check/<tag>.csv and prints agreement + garbage counts.
speed:   s/crop and peak VRAM (torch.cuda.max_memory_allocated + nvidia-smi, sampled) per batch size.
"""
import glob
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
LV = ROOT / "lapvision"
sys.path.insert(0, str(LV))
from vlm import Reader  # noqa: E402

OUT = ROOT / "out" / "fp16_check"
OK_RAW = re.compile(r"^(\d{1,3}|none)$", re.I)


def arg(name, default):
    for a in sys.argv:
        if a.startswith("--%s=" % name):
            return a.split("=", 1)[1]
    return default


def load_reads():
    rs = []
    for f in sorted(glob.glob(str(ROOT / "out" / "world_*" / "reads.csv"))):
        r = pd.read_csv(f)
        r["src"] = Path(f).parent.name
        rs.append(r)
    r = pd.concat(rs, ignore_index=True)
    r["abs"] = r.path.map(lambda p: str((LV / p.replace("\\", "/")).resolve()))
    r["stored"] = r.vlm.map(lambda v: None if pd.isna(v) else int(v))
    return r


def sample(r, n, seed=0):
    num = r[r.stored.notna()]
    non = r[r.stored.isna()]
    k_num = int(round(n * 0.6))
    return pd.concat([num.sample(min(k_num, len(num)), random_state=seed),
                      non.sample(min(n - k_num, len(non)), random_state=seed)]).sample(frac=1, random_state=seed)


class SmiPeak:
    """Samples nvidia-smi memory.used (MiB) in a thread; .peak after stop()."""

    def __init__(self, every=0.2):
        self.every, self.peak, self._stop = every, 0, threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout
                self.peak = max(self.peak, int(o.split()[0]))
            except Exception:
                pass
            time.sleep(self.every)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()


def make_reader():
    dt = arg("dtype", "fp16")
    attn = arg("attn", None)
    t = time.time()
    rd = Reader(dtype=dt, attn=attn)
    print("loaded %s dtype=%s attn=%s in %.0f s, weights %.2f GB" % (
        "Qwen3-VL-4B", rd.dtype, rd.model.config._attn_implementation, time.time() - t,
        torch.cuda.memory_allocated() / 1e9), flush=True)
    return rd


def compare():
    n = int(arg("n", 500))
    B = int(arg("batch", 8))
    tag = arg("tag", arg("dtype", "fp16"))
    s = sample(load_reads(), n)
    rd = make_reader()
    rows = []
    t0 = time.time()
    paths = list(s["abs"])
    for i in range(0, len(paths), B):
        res = rd.read(paths[i:i + B])
        rows += res
        if i % 80 == 0:
            print(i, "/", len(paths), flush=True)
    dt = time.time() - t0
    s["new_raw"] = [x[0] for x in rows]
    s["new"] = [x[1] for x in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    s.drop(columns=["abs"]).to_csv(OUT / ("%s.csv" % tag), index=False)
    same = (s.new.fillna(-1).astype(int) == s.stored.fillna(-1).astype(int))
    same_raw = s.new_raw.str.strip().str.lower() == s.vlm_raw.astype(str).str.strip().str.lower()
    bad = ~s.new_raw.str.strip().str.match(OK_RAW)
    print("\n== %s: %d crops, batch %d, %.3f s/crop" % (tag, len(s), B, dt / len(s)))
    print("identical parsed number (None==None): %.4f  (%d/%d)" % (same.mean(), same.sum(), len(s)))
    print("identical raw text:                   %.4f" % same_raw.mean())
    for name, m in [("stored numeric", s.stored.notna()), ("stored none", s.stored.isna())]:
        print("  %-15s n=%3d identical %.4f" % (name, m.sum(), same[m].mean()))
    print("empty outputs: %d, not digits/none: %d" % ((s.new_raw.str.strip() == "").sum(), bad.sum()))
    if bad.any():
        print(s[bad][["new_raw", "vlm_raw"]].head(20).to_string())
    d = s[~same]
    print("\ndisagreements:")
    print(d[["src", "pass_id", "vlm_raw", "new_raw"]].to_string())


def speed():
    n = int(arg("n", 48))
    batches = [int(x) for x in arg("batches", "1,4,8").split(",")]
    s = sample(load_reads(), n, seed=1)
    paths = list(s["abs"])
    rd = make_reader()
    rd.read(paths[:2])  # warm-up
    print("%-6s %8s %10s %12s" % ("batch", "s/crop", "torch_peak", "smi_peak"))
    for B in batches:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with SmiPeak() as sp:
            t = time.time()
            for i in range(0, len(paths), B):
                rd.read(paths[i:i + B])
            torch.cuda.synchronize()
            dt = time.time() - t
        print("%-6d %8.3f %8.2f GB %8.2f GB" % (B, dt / len(paths), torch.cuda.max_memory_allocated() / 1e9,
                                                sp.peak / 1024), flush=True)


if __name__ == "__main__":
    {"compare": compare, "speed": speed}[sys.argv[1]]()
