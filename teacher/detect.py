"""WP3 step 1 (GPU): raw per-frame car detections for every active video -> work/<id>/dets_raw.parquet.

YOLO-World v2-x, fp16, imgsz 1280, batch 8, 4 fps, CPU decode through an ffmpeg `fps=4` rawvideo pipe (WP0 settings).
Columns: frame_idx (int32, frame i of the 4 fps stream), pts (s = frame_idx / fps), cls (name: buggy, person, and the
absorbing classes if the prompt set has them), conf (>= 0.05), x1, y1, x2, y2 (source pixels).

Resumable: detections are written in parts of `--part_s` seconds of video (default 600 s) to
work/<id>/dets_raw_parts/part_NNNNN.parquet; a restart resumes after the last complete part. A `PAUSE` file at the repo
root makes the run finish the current part and exit cleanly (exit code 3). When a video is complete its parts are merged
into dets_raw.parquet, `detect_settings.json` is written and `detect.done` is created. A failure on one video goes to
work/<id>/errors.log and the queue continues. Progress + ETA: work/detect.log.

usage (repo root):
  python teacher/detect.py [id ...] [--prompts=A|B] [--imgsz=1280] [--fps=4] [--batch=8] [--force]
  python teacher/detect.py calib <id> --start=S --dur=D --out=<file.parquet> [--prompts=A|B] [--imgsz=..]
"""
import datetime as dt
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORK, log_error, select_videos, tool, wdir  # noqa: E402

MODEL = ROOT / "lapvision" / "yolov8x-worldv2.pt"
PROMPT_SETS = {
    # (prompt, stored name); only "off-road buggy" is a car label, the rest absorb false positives
    "A": [("off-road buggy", "buggy"), ("person", "person")],
    "B": [("off-road buggy", "buggy"), ("person", "person"), ("pickup truck", "pickup"), ("car", "car"),
          ("ATV", "atv"), ("tractor", "tractor"), ("motorcycle", "motorcycle")],
}
DEFAULTS = dict(prompts="A", imgsz=1280, fps=4.0, batch=8, conf=0.05, iou=0.7, max_det=300, part_s=600.0)
LOG = WORK / "detect.log"
PAUSE = ROOT / "PAUSE"


class Paused(Exception):
    pass


def log(msg):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class Model:
    def __init__(self, prompts="A"):
        warnings.filterwarnings("ignore")
        import logging
        from ultralytics.utils import LOGGER
        LOGGER.setLevel(logging.ERROR)  # 'half is deprecated' would be printed on every predict call
        cwd = os.getcwd()
        os.chdir(ROOT / "lapvision")  # YOLO-World resolves the CLIP weights relative to here / weights_dir
        try:
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(str(MODEL))
            self.model.set_classes([p for p, _ in PROMPT_SETS[prompts]])
        finally:
            os.chdir(cwd)
        self.names = [n for _, n in PROMPT_SETS[prompts]]
        self.prompts = prompts

    def predict(self, frames, imgsz, conf, iou, max_det):
        res = self.model.predict(frames, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, half=True, verbose=False)
        out = []
        for r in res:
            b = r.boxes
            out.append((b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
                        b.cls.cpu().numpy().astype(np.int16)))
        return out


def frame_pipe(path, fps, w, h, start=0.0, dur=None, maxq=32):
    """Yield (k, frame) from an ffmpeg fps-resampled bgr24 pipe; k counts from the seek point. Reader thread."""
    cmd = [tool("ffmpeg"), "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", str(path), "-an", "-sn", "-vf", f"fps={fps:g}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    fs = w * h * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=fs * 4)
    q = queue.Queue(maxsize=maxq)
    err = []

    def reader():
        try:
            while True:
                b = p.stdout.read(fs)
                if len(b) < fs:
                    break
                q.put(np.frombuffer(b, np.uint8).reshape(h, w, 3))
        finally:
            q.put(None)

    def errs():
        err.append(p.stderr.read().decode("utf-8", "replace"))

    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=errs, daemon=True).start()
    k = 0
    try:
        while True:
            f = q.get()
            if f is None:
                break
            yield k, f
            k += 1
    finally:
        p.kill() if p.poll() is None else None
        p.wait()
        if p.returncode not in (0, None) and k == 0:
            raise RuntimeError(f"ffmpeg failed ({p.returncode}): {''.join(err)[-500:]}")


def to_frame(rows, names):
    if not rows:
        return pd.DataFrame({"frame_idx": pd.Series([], "int32"), "pts": pd.Series([], "float64"),
                             "cls": pd.Series([], "str"), "conf": pd.Series([], "float32"),
                             **{c: pd.Series([], "float32") for c in ("x1", "y1", "x2", "y2")}})
    fi = np.concatenate([np.full(len(c), i, np.int32) for i, _, c, _ in rows])
    xy = np.concatenate([x for _, x, _, _ in rows]).reshape(-1, 4)
    cf = np.concatenate([c for _, _, c, _ in rows])
    cl = np.concatenate([k for _, _, _, k in rows])
    return fi, xy, cf, cl


def make_df(rows, names, fps):
    if not rows or sum(len(c) for _, _, c, _ in rows) == 0:
        return to_frame([], names)
    fi, xy, cf, cl = to_frame(rows, names)
    return pd.DataFrame({"frame_idx": fi, "pts": fi / fps, "cls": np.array(names, dtype=object)[cl], "conf": cf,
                         "x1": xy[:, 0], "y1": xy[:, 1], "x2": xy[:, 2], "y2": xy[:, 3]})


def run_frames(model, path, w, h, S, start_frame=0, dur=None, part_frames=None, on_part=None, check_pause=False):
    """Detect frames from start_frame on. Calls on_part(part_no, rows) every part_frames frames (and at the end)."""
    fps, B = S["fps"], S["batch"]
    rows, batch, idx = [], [], []
    n = pending = 0

    def flush_batch():
        for i, r in zip(idx, model.predict(batch, S["imgsz"], S["conf"], S["iou"], S["max_det"])):
            rows.append((i, r[0], r[1], r[2]))
        batch.clear()
        idx.clear()

    for k, f in frame_pipe(path, fps, w, h, start=start_frame / fps, dur=dur):
        gi = start_frame + k
        batch.append(f)
        idx.append(gi)
        n += 1
        pending += 1
        end_of_part = part_frames and (gi + 1) % part_frames == 0
        if len(batch) == B or end_of_part:
            flush_batch()
        if end_of_part:
            on_part(gi // part_frames, rows)
            rows, pending = [], 0
            if check_pause and PAUSE.exists():
                raise Paused()
    if batch:
        flush_batch()
    if part_frames and pending and on_part:
        on_part((start_frame + n - 1) // part_frames, rows)
        rows = []
    return rows, n


def settings_dict(S):
    return {"model": MODEL.name, "prompts": [p for p, _ in PROMPT_SETS[S["prompts"]]], "prompt_set": S["prompts"],
            "names": [n for _, n in PROMPT_SETS[S["prompts"]]], "car_class": "buggy", "imgsz": S["imgsz"],
            "fps": S["fps"], "batch": S["batch"], "conf": S["conf"], "iou": S["iou"], "max_det": S["max_det"],
            "half": True, "decode": "ffmpeg cpu fps filter, bgr24 pipe", "part_s": S["part_s"]}


def detect_video(model, row, S, state):
    vid = row["id"]
    d = wdir(vid)
    parts = d / "dets_raw_parts"
    parts.mkdir(exist_ok=True)
    sj = settings_dict(S)
    sfile = parts / "settings.json"
    if sfile.exists():
        old = json.loads(sfile.read_text())
        if {k: v for k, v in old.items() if k != "batch"} != {k: v for k, v in sj.items() if k != "batch"}:
            raise RuntimeError(f"existing parts were made with other settings ({old}); use --force")
    else:
        sfile.write_text(json.dumps(sj, indent=1))
    fps = S["fps"]
    pf = int(round(S["part_s"] * fps))
    total_frames = int(row["duration_s"] * fps)
    done_parts = sorted(int(p.stem.split("_")[1]) for p in parts.glob("part_*.parquet"))
    # parts are written in order; resume after the last contiguous complete part
    k = 0
    while k in done_parts:
        k += 1
    complete = (parts / "complete").exists()
    start_frame = k * pf
    t_video = time.time()
    if not complete:
        log(f"{vid}: start at part {k} (frame {start_frame}, {start_frame / fps / 60:.0f} min of "
            f"{row['duration_s'] / 60:.0f})")

        def on_part(pno, rows):
            df = make_df(rows, model.names, fps)
            tmp = parts / f"part_{pno:05d}.tmp"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, parts / f"part_{pno:05d}.parquet")
            now = time.time()
            nfr = min(pf, total_frames - pno * pf)
            state["frames"] += nfr
            rate = state["frames"] / max(now - state["t0"], 1e-6)
            left_v = max(total_frames - (pno + 1) * pf, 0)
            left_all = left_v + state["frames_after"]
            log(f"{vid}: part {pno} done ({(pno + 1) * pf / fps / 60:.0f}/{row['duration_s'] / 60:.0f} min, "
                f"{len(df)} dets, {int((df.cls == 'buggy').sum()) if len(df) else 0} buggy) "
                f"{rate:.1f} img/s; ETA video {left_v / rate / 60:.0f} min, all {left_all / rate / 3600:.2f} h")

        _, n = run_frames(model, row["path"] if Path(row["path"]).is_absolute() else ROOT / row["path"],
                          row["width"], row["height"], S, start_frame=start_frame, part_frames=pf,
                          on_part=on_part, check_pause=True)
        nframes = start_frame + n
        if nframes < total_frames - 2 * fps * 10:
            log_error(vid, "detect", f"only {nframes} frames decoded, expected ~{total_frames}")
        (parts / "complete").write_text(str(nframes))
    nframes = int((parts / "complete").read_text())
    files = sorted(parts.glob("part_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    df = df.sort_values(["frame_idx", "conf"], ascending=[True, False], kind="stable").reset_index(drop=True)
    df.to_parquet(d / "dets_raw.parquet", index=False)
    sj.update(n_frames=nframes, n_dets=len(df), n_buggy=int((df.cls == "buggy").sum()),
              n_buggy_025=int(((df.cls == "buggy") & (df.conf >= 0.25)).sum()),
              finished=f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}", wall_s_last_session=round(time.time() - t_video))
    (d / "detect_settings.json").write_text(json.dumps(sj, indent=1))
    (d / "detect.done").write_text(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {nframes} frames {len(df)} dets\n")
    shutil.rmtree(parts)
    log(f"{vid}: DONE {nframes} frames, {len(df)} dets ({sj['n_buggy_025']} buggy >= 0.25)")


def parse(argv):
    S = dict(DEFAULTS)
    ids, extra, force = [], {}, False
    for a in argv:
        if a == "--force":
            force = True
        elif a.startswith("--"):
            k, _, v = a[2:].partition("=")
            if k in S:
                S[k] = type(S[k])(v)
            else:
                extra[k] = v
        else:
            ids.append(a)
    return S, ids, extra, force


def calib(argv):
    S, ids, extra, _ = parse(argv)
    row = select_videos([ids[0]])[0]
    start, dur = float(extra["start"]), float(extra["dur"])
    model = Model(S["prompts"])
    t = time.time()
    rows, n = run_frames(model, ROOT / row["path"], row["width"], row["height"], S, dur=dur,
                         start_frame=int(round(start * S["fps"])))
    df = make_df(rows, model.names, S["fps"])
    out = Path(extra["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    out.with_suffix(".json").write_text(json.dumps({**settings_dict(S), "video": row["id"], "start": start, "dur": dur,
                                                    "n_frames": n}, indent=1))
    print(f"{row['id']} {start}+{dur}s prompts {S['prompts']} imgsz {S['imgsz']}: {n} frames, {len(df)} dets, "
          f"{n / (time.time() - t):.1f} img/s -> {out}", flush=True)


def main(argv):
    S, ids, _, force = parse(argv)
    rows = select_videos(ids or None)
    if ids:
        rows.sort(key=lambda r: ids.index(r["id"]))  # run in the order given
    todo = []
    for r in rows:
        d = wdir(r["id"])
        if force:
            for p in (d / "detect.done", d / "dets_raw.parquet"):
                p.unlink(missing_ok=True)
            shutil.rmtree(d / "dets_raw_parts", ignore_errors=True)
        if (d / "detect.done").exists():
            print(f"{r['id']}: detect already done, skip")
            continue
        todo.append(r)
    if not todo:
        return 0
    if PAUSE.exists():
        log("PAUSE file present, not starting")
        return 3
    log(f"start: {len(todo)} videos {[r['id'] for r in todo]}, settings {settings_dict(S)}")
    model = Model(S["prompts"])
    state = {"frames": 0, "t0": time.time(), "frames_after": 0}
    for i, r in enumerate(todo):
        state["frames_after"] = sum(int(x["duration_s"] * S["fps"]) for x in todo[i + 1:])
        try:
            detect_video(model, r, S, state)
        except Paused:
            log(f"{r['id']}: PAUSE file found, stopping cleanly (resume: rerun teacher/detect.py)")
            return 3
        except Exception as e:  # noqa: BLE001 - log and continue with the next video
            log_error(r["id"], "detect", traceback.format_exc())
            log(f"{r['id']}: FAILED {e!r} (see work/{r['id']}/errors.log)")
        if PAUSE.exists():
            log("PAUSE file found between videos, stopping")
            return 3
    log("all videos finished")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "calib":
        calib(sys.argv[2:])
    else:
        sys.exit(main(sys.argv[1:]))
