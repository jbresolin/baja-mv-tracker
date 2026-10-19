"""Real-time load matrix: replays file segments with rt/run.py --realtime --stats under several thread / affinity /
model configs (one at a time, so they do not disturb each other) and tabulates CPU %, RAM, frame lag, dropped frames
and identity latency.

usage (this PC, 26SE):
  python rt/loadtest.py --video "2026-10-17 11-29-49.mkv" --segment quiet=11:35:00 --segment busy=11:59:00 \
      --duration 600 --config n_1t:n:1:1: --config n_4t:n:4:4: --config n_cap2:n:1:1:2,4 --config n_cap3:n:2:1:2,4,6
config = name:model(n|s|path):threads_detector:threads_identity:affinity(comma list or empty)
Outputs out/rt/load/<segment>_<name>.{jsonl,csv,stats.json} and out/rt/load/table.md.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"n": "students/runs/bench/cars_n640_v2_best/384x640/best.onnx",
          "s": "students/runs/bench/cars_s640_best/384x640/best.onnx"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--event", default="26SE")
    ap.add_argument("--segment", action="append", required=True, help="name=start (HH:MM:SS or seconds)")
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--outdir", default=str(ROOT / "out" / "rt" / "load"))
    ap.add_argument("--decoder-unpinned", action="store_true", help="pin only the pipeline, not ffmpeg")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    od = Path(a.outdir)
    od.mkdir(parents=True, exist_ok=True)
    rows = []
    for seg in a.segment:
        sname, start = seg.split("=", 1)
        for c in a.config:
            name, model, td, ti, aff = (c.split(":") + [""] * 5)[:5]
            out = od / ("%s_%s.jsonl" % (sname, name))
            stf = od / ("%s_%s.stats.json" % (sname, name))
            if not (a.skip_existing and stf.exists()):
                for f in (out, out.with_suffix(".csv"), stf):
                    if f.exists():
                        f.unlink()
                cmd = [sys.executable, str(ROOT / "rt" / "run.py"), "--source", a.video, "--event", a.event,
                       "--start", start, "--duration", str(a.duration), "--realtime", "--stats", "--quiet",
                       "--out", str(out), "--set", "models.car_detector=%s" % MODELS.get(model, model),
                       "--set", "runtime.threads_detector=%s" % td, "--set", "runtime.threads_identity=%s" % ti]
                if aff:
                    cmd += ["--affinity", aff]
                    if a.decoder_unpinned:
                        cmd += ["--set", "runtime.affinity_decoder=false"]
                for s in a.set:
                    cmd += ["--set", s]
                env = {k: v for k, v in __import__("os").environ.items()}
                env["CUDA_VISIBLE_DEVICES"] = "-1"
                env["OMP_NUM_THREADS"] = str(max(int(td), int(ti)))
                print(time.strftime("%H:%M:%S"), sname, name, flush=True)
                subprocess.run(cmd, env=env, stdout=open(str(out) + ".log", "w"), stderr=subprocess.STDOUT)
            st = json.loads(stf.read_text())
            i, cpu, ram = st.get("identity") or {}, st.get("cpu") or {}, st.get("ram_mb") or {}
            rows.append(dict(segment=sname, config=name, model=model, threads="%s+%s" % (td, ti), affinity=aff or "-",
                             frames=st["frames"], dropped=st["dropped"], det_ms_p50=st["det_ms_p50"],
                             det_ms_p90=st["det_ms_p90"], lag_ms_p95=st["lag_ms_p95"], lag_ms_max=st["lag_ms_max"],
                             cpu_proc_mean=cpu.get("proc_mean"), cpu_proc_p95=cpu.get("proc_p95"),
                             cpu_ffmpeg_mean=cpu.get("ffmpeg_mean"), cpu_system_mean=cpu.get("system_mean"),
                             ram_peak_mb=ram.get("peak"), passes=st["passes"],
                             id_latency_p50=i.get("latency_s_p50"), id_latency_max=i.get("latency_s_max"),
                             id_max_queue=i.get("max_queue"), panel_ms=i.get("panel_ms_per_crop"),
                             recog_ms_box=i.get("recog_ms_per_box")))
    keys = list(rows[0])
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    (od / "table.md").write_text("\n".join(md) + "\n")
    (od / "table.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
