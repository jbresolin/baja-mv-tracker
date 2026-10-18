"""WP9 benchmark harness: plug any detector into the session-1 pass pipeline on the 26SE start-line footage and
report the LESSONS.md pass metrics (recall / precision / false passes after 09:18:30 vs the human laps, +80 s clock).

usage:
  # metrics only, from existing tracker output (CPU, seconds). Validation: the YOLO-World run must give
  # 241 passes, recall 0.933, precision 0.975 (LESSONS.md, WP0).
  python students/eval_26se.py --name yw_check --from-tracks out/world_0909 out/world_0956 out/world_1109 out/world_1129

  # full run with a detector: stage 1 coarse timeline (tl_fps, default 5) -> stage 2 tracking at ~track_fps (default 30,
  # as session 1) only on the foreground events of stage 1 -> stage 3 metrics
  python students/eval_26se.py --name cars_n640 --weights students/runs/cars_n640/weights/best.pt
  python students/eval_26se.py --name cars_n640 --weights ... --files 1129 --track-fps 15     # cheaper / partial

options: --imgsz (default 640 for a student, 1280 for COCO / YOLO-World as in session 1), --device 0|cpu,
--hwaccel cuda|none (ffmpeg NVDEC; default cuda when a GPU is visible), --threads N (torch CPU threads),
--crops (also save crops like lapvision/extract.py; not needed for the metrics), --ids world|none (identity metrics
borrowed from the YOLO-World VLM reads, see below), --stages timeline,track,metrics.

Outputs: out/eval_<name>/ (never out/crops_* or out/world_*): tl_<file>.csv (same format as out/tl_*.csv),
trk_<file>/tracks.csv (same as out/world_*/tracks.csv), passes.csv (per pass, GT match), missed.csv (unmatched human
laps), metrics.json, report.txt, timing.json. Resumable per file and stage: outputs are written atomically and an
existing tl_<file>.csv or trk_<file>/tracks.csv is not recomputed (delete it to redo).

Identity (--ids world, default): there are no VLM reads for a new detector. A student pass borrows the vote of the
YOLO-World pass (out/world_*, rebuilt with the current pass rules, voted as lapvision/analyze.py) that overlaps it most
in time (temporal IoU >= 0.3). This measures "would the teacher's read still land on this pass"; it is not a read of
the student's crops.

Thresholds (event conf 0.3, foreground h > 0.2 H, bottom > 0.4 H, pass rules) are session 1's. Do not tune them here:
26SE is the benchmark (TRAINING_PLAN.md).
"""
import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch  # before ultralytics: ultralytics sets OMP_NUM_THREADS=1 if it is imported first

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lapvision"))
import clock  # noqa: E402
import evaluate as E  # noqa: E402
import passes as P  # noqa: E402

OUT = ROOT / "out"
VALID_FROM = 9 * 3600 + 18 * 60 + 30


def _t(s):
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


# key -> (file, wall-clock span) from LESSONS.md "Footage facts"
FILES = {
    "0909": ("2026-10-17 09-09-26.mkv", _t("09:09:26"), _t("09:56:08")),
    "0956": ("2026-10-17 09-56-32.mkv", _t("09:56:32"), _t("11:09:02")),
    "1109": ("2026-10-17 11-09-03.mkv", _t("11:09:03"), _t("11:29:47")),
    "1129": ("2026-10-17 11-29-49.mkv", _t("11:29:49"), _t("12:10:10")),
}
ORDER = list(FILES)
TL_CONF, TL_BATCH = 0.15, 16
TRACK_CONF = 0.15


# ---------------------------------------------------------------------------------------------------- decoding
def _prefetch(gen, n=48):
    q = queue.Queue(n)
    END = object()

    def run():
        try:
            for x in gen:
                q.put(x)
        finally:
            q.put(END)

    threading.Thread(target=run, daemon=True).start()
    while True:
        x = q.get()
        if x is END:
            return
        yield x


def _ffmpeg(video, w, h, vf, hwaccel, ss=None, dur=None):
    cmd = ["ffmpeg", "-v", "error"]
    if hwaccel == "cuda":
        cmd += ["-hwaccel", "cuda"]
    if ss is not None:
        cmd += ["-ss", "%.3f" % ss]
    cmd += ["-i", str(video)]
    if dur is not None:
        cmd += ["-t", "%.3f" % dur]
    # passthrough: never duplicate / drop frames to make the raw output CFR (frame k must be input frame k*step)
    cmd += ["-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 2)
    try:
        while True:
            buf = p.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()


def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "json", str(video)],
                         capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    a, b = s["r_frame_rate"].split("/")
    return s["width"], s["height"], float(a) / float(b)


def file_offset(video):
    """Seconds-of-day at pts=0 (lapvision/detect_timeline.file_offset, CPU-only decode)."""
    f = clock.grab(video, 0.5)
    s, _ = clock.read(f)
    return s - 0.5


# ---------------------------------------------------------------------------------------------------- stages
def stage_timeline(det, video, out_csv, a):
    """Stage 1 (= lapvision/detect_timeline.py): fps a.tl_fps, frames scaled to 1920x1080."""
    W, H = 1920, 1080
    off = file_offset(video)
    prec = {"quantize": 16} if a.device != "cpu" else {}  # fp16 on GPU ('half' is deprecated in 8.4)
    rows, batch, meta, n = [], [], [], 0
    t0 = time.time()

    def flush():
        res = det.predict(batch, imgsz=a.imgsz, conf=TL_CONF, device=a.device, **prec)
        for pts, r in zip(meta, res):
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                rows.append([video.name, round(pts, 3), round(off + pts, 3), det.names[int(b.cls)],
                             round(float(b.conf), 3), int(x1), int(y1), int(x2), int(y2)])
        batch.clear()
        meta.clear()

    frames = _ffmpeg(video, W, H, "fps=%g,scale=%d:%d" % (a.tl_fps, W, H), a.hwaccel)
    for i, f in enumerate(_prefetch(frames)):
        batch.append(f)
        meta.append(i / a.tl_fps)
        n += 1
        if len(batch) == TL_BATCH:
            flush()
            if n % (TL_BATCH * 150) == 0:
                el = time.time() - t0
                print("  tl %s pts %d  %.1f img/s" % (video.name, i / a.tl_fps, n / el), flush=True)
    if batch:
        flush()
    df = pd.DataFrame(rows, columns=["file", "pts", "tod", "cls", "conf", "x1", "y1", "x2", "y2"])
    tmp = out_csv.with_suffix(".part")
    df.to_csv(tmp, index=False)
    tmp.replace(out_csv)
    return dict(frames=n, seconds=round(time.time() - t0, 1), offset=off)


def stage_track(det, video, tl_csv, outdir, a):
    """Stage 2 (= lapvision/extract.py without the crops unless --crops): ByteTrack at ~a.track_fps on the
    foreground events of the timeline. 4K input is scaled to a.track_width (passes use normalised boxes)."""
    import cv2
    from align import fg, events
    import extract

    tl = pd.read_csv(tl_csv)
    tl = tl[tl.file == video.name]
    w0, h0, fps = probe(video)
    if a.track_width and w0 > a.track_width:
        w, h = a.track_width, int(round(h0 * a.track_width / w0 / 2) * 2)
    else:
        w, h = w0, h0
    off = file_offset(video)
    step = max(1, round(fps / a.track_fps))
    prec = {"quantize": 16} if a.device != "cpu" else {}  # fp16 on GPU ('half' is deprecated in 8.4)
    evs = events(fg(tl, conf=a.event_conf), gap=extract.GAP)  # = extract.events() with a configurable conf
    outdir.mkdir(parents=True, exist_ok=True)
    rows, n, t0 = [], 0, time.time()
    for ei, (ea, eb) in enumerate(evs):
        start = max(0.0, ea - off - extract.PAD)
        dur = (eb - ea) + 2 * extract.PAD
        eid = "%s_%s" % (video.stem[-8:].replace("-", ""), clock.fmt(ea).replace(":", "").replace(".", "_"))
        if a.crops:
            (outdir / eid).mkdir(exist_ok=True)
        det.reset()
        vf = "framestep=%d,scale=%d:%d" % (step, w, h)  # keeps input frames 0, step, 2*step, ...
        for k, frame in enumerate(_prefetch(_ffmpeg(video, w, h, vf, a.hwaccel, start, dur))):
            fi = k * step
            pts = start + fi / fps
            n += 1
            r = det.track(frame, imgsz=a.imgsz, conf=TRACK_CONF, device=a.device, **prec)[0]
            if r.boxes.id is None:
                continue
            for bx, tid, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().numpy(),
                                   r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = bx
                name, sharp = "", 0.0
                if a.crops:
                    bw, bh = x2 - x1, y2 - y1
                    X1, Y1 = int(max(0, x1 - 0.05 * bw)), int(max(0, y1 - 0.05 * bh))
                    X2, Y2 = int(min(w, x2 + 0.05 * bw)), int(min(h, y2 + 0.05 * bh))
                    crop = frame[Y1:Y2, X1:X2]
                    name = "t%03d_f%05d.jpg" % (tid, fi)
                    cv2.imwrite(str(outdir / eid / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    sharp = round(extract.sharpness(crop), 1)
                rows.append(dict(event=eid, track=int(tid), frame=fi, pts=round(pts, 3), tod=round(off + pts, 3),
                                 conf=round(float(cf), 3), x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                                 W=w, H=h, sharp=sharp, crop=name))
        if ei % 10 == 0:
            print("  trk %s event %d/%d %s  %.1f frames/s" % (video.name, ei + 1, len(evs), eid,
                                                              n / max(1e-6, time.time() - t0)), flush=True)
    cols = ["event", "track", "frame", "pts", "tod", "conf", "x1", "y1", "x2", "y2", "W", "H", "sharp", "crop"]
    df = pd.DataFrame(rows, columns=cols)
    tmp = outdir / "tracks.part"
    df.to_csv(tmp, index=False)
    tmp.replace(outdir / "tracks.csv")
    return dict(events=len(evs), event_seconds=round(sum(eb - ea + 2 * extract.PAD for ea, eb in evs), 1),
                frames=n, seconds=round(time.time() - t0, 1), track_size=[w, h], step=step)


# ---------------------------------------------------------------------------------------------------- metrics
def file_key(event_id):
    """'112949_113125_353' -> '1129'."""
    return event_id[:4]


def gt_windows(keys):
    """Union of the selected files' spans; a span extends to the next file's start when that file is selected too
    (so the full set gives one window 09:18:30-12:10:10, as analyze.py)."""
    win = []
    for k in keys:
        s, e = FILES[k][1], FILES[k][2]
        i = ORDER.index(k)
        if i + 1 < len(ORDER) and ORDER[i + 1] in keys:
            e = FILES[ORDER[i + 1]][1]
        win.append((max(s, VALID_FROM), e))
    return win


def lap_file(t):
    for k in ORDER:
        if FILES[k][1] <= t <= FILES[k][2]:
            return k
    return "gap"


def load_passes(dirs):
    PS = []
    for d in dirs:
        tr = pd.read_csv(d / "tracks.csv")
        if not len(tr):
            continue
        p = P.build(tr)
        p["src"] = d.name
        p["file"] = p.event.map(file_key)
        PS.append(p)
    return pd.concat(PS, ignore_index=True) if PS else pd.DataFrame()


def world_votes():
    """YOLO-World passes (current pass rules) with their VLM vote, as lapvision/analyze.py computes them."""
    import analyze as A
    dirs = sorted(OUT.glob("world_[0-9]*"))
    ps, rd = A.load(dirs)
    rows = []
    for (src, pid), r in rd.groupby(["src", "pass_id"]):
        v, vs, vk = A.vote_vlm(r)
        rows.append(dict(src=src, pass_id=pid, vlm_car=v, vlm_share=vs))
    return ps.merge(pd.DataFrame(rows), on=["src", "pass_id"], how="left")


def borrow_ids(ps, wv):
    """Per pass: the vote of the YOLO-World pass with the highest temporal IoU (>= 0.3)."""
    ids, shares, wsrc = [], [], []
    wt0, wt1 = wv.t0.values, wv.t1.values
    for t0, t1 in zip(ps.t0.values, ps.t1.values):
        inter = np.clip(np.minimum(t1, wt1) - np.maximum(t0, wt0), 0, None)
        union = np.maximum(t1, wt1) - np.minimum(t0, wt0)
        iou = inter / np.maximum(union, 1e-6)
        j = int(np.argmax(iou)) if len(iou) else -1
        if j >= 0 and iou[j] >= 0.3:
            ids.append(wv.vlm_car.iloc[j]); shares.append(wv.vlm_share.iloc[j])
            wsrc.append("%s:%d" % (wv.src.iloc[j], wv.pass_id.iloc[j]))
        else:
            ids.append(np.nan); shares.append(np.nan); wsrc.append("")
    return ps.assign(vlm_car=ids, vlm_share=shares, world_pass=wsrc)


def det_report(ps, g):
    rep, pairs = E.report(ps, g, tcol="t0", idcol="__none__")  # time-only greedy matching (+-6 s)
    rep = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()}
    rep["false_passes"] = rep["detected"] - rep["matched"]
    rep["missed_laps"] = rep["gt_laps"] - rep["matched"]
    return rep, pairs


def metrics(dirs, outdir, a, extra=None):
    ps = load_passes(dirs)
    keys = [k for k in ORDER if k in set(ps.file)] if len(ps) else []
    if a.files:
        keys = [k for k in keys if k in a.files]
        ps = ps[ps.file.isin(keys)]
    ps = ps[ps.t0 >= VALID_FROM].sort_values("t0").reset_index(drop=True)
    win = gt_windows(keys)
    g = E.gt_laps(VALID_FROM, 86400)
    g = g[np.any([(g.t >= s) & (g.t <= e) for s, e in win], axis=0)].reset_index(drop=True)
    g["file"] = g.t.map(lap_file)

    overall, pairs = det_report(ps, g)
    ps["gt_car"], ps["gt_dt"] = np.nan, np.nan
    for i, j in pairs:
        ps.loc[i, "gt_car"] = g.loc[j, "car"]
        ps.loc[i, "gt_dt"] = round(ps.loc[i, "t0"] - g.loc[j, "t"], 3)
    g["matched"] = False
    g.loc[[j for _, j in pairs], "matched"] = True
    per_file = {}
    for k in keys:
        pk, gk = ps[ps.file == k], g[g.file == k]
        mp, mg = int(ps.loc[pk.index, "gt_car"].notna().sum()), int(gk.matched.sum())
        per_file[k] = dict(gt_laps=len(gk), detected=len(pk), matched=mg,
                           recall=round(mg / max(1, len(gk)), 4), precision=round(mp / max(1, len(pk)), 4),
                           false_passes=len(pk) - mp, missed_laps=len(gk) - mg)
    gap = g[g.file == "gap"]
    res = dict(name=a.name, files=keys, gt_windows=[[clock.fmt(s), clock.fmt(e)] for s, e in win],
               overall=overall, per_file=per_file, gt_laps_in_file_gaps=len(gap))

    if a.ids == "world":
        wv = world_votes()
        wv = wv[wv.t0 >= VALID_FROM]
        ps = borrow_ids(ps, wv)
        rep, ipairs = E.report(ps, g, tcol="t0", idcol="vlm_car")  # id-preferring matching, as analyze.py
        rep = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()}
        rep["passes_with_a_world_pass"] = int((ps.world_pass != "").sum())
        hi = [(ps.loc[i, "vlm_car"], g.loc[j, "car"]) for i, j in ipairs
              if pd.notna(ps.loc[i, "vlm_car"]) and ps.loc[i, "vlm_share"] >= 0.5]
        rep["share_ge_0.5_acc"] = round(sum(x == y for x, y in hi) / max(1, len(hi)), 4)
        rep["share_ge_0.5_coverage"] = round(len(hi) / max(1, len(ipairs)), 4)
        res["identity_borrowed_from_world"] = rep

    ps["time"] = ps.tcross.map(clock.fmt)
    ps["t0_s"] = ps.t0.map(clock.fmt)
    ps.to_csv(outdir / "passes.csv", index=False)
    miss = g[~g.matched].assign(time=lambda d: d.t.map(clock.fmt))
    miss.to_csv(outdir / "missed.csv", index=False)
    if extra:
        res["run"] = extra
    (outdir / "metrics.json").write_text(json.dumps(res, indent=1, default=str))

    lines = ["26SE pass metrics: %s  (after 09:18:30, video clock = server + %.0f s, match +-%.0f s on t0)"
             % (a.name, E.CLOCK_OFFSET, E.TOL),
             "files %s, GT windows %s" % (",".join(keys), res["gt_windows"]), ""]
    tab = pd.DataFrame({"all": overall, **{k: v for k, v in per_file.items()}}).T
    cols = [c for c in ["gt_laps", "detected", "matched", "recall", "precision", "false_passes", "missed_laps",
                        "dt_median", "dt_p10", "dt_p90"] if c in tab]
    lines.append(tab[cols].to_string())
    if "identity_borrowed_from_world" in res:
        lines += ["", "identity borrowed from YOLO-World VLM votes (id-preferring match, as analyze.py):",
                  json.dumps(res["identity_borrowed_from_world"])]
    fp = ps[ps.gt_car.isna()]
    lines += ["", "false passes (%d):" % len(fp),
              fp[["src", "pass_id", "t0_s", "n", "dwell", "exited", "hmax"]].to_string() if len(fp) else "-",
              "", "missed human laps (%d):" % len(miss),
              miss[["car", "lap", "time", "file"]].to_string() if len(miss) else "-"]
    txt = "\n".join(lines)
    (outdir / "report.txt").write_text(txt)
    print(txt)
    return res


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="-> out/eval_<name>/")
    ap.add_argument("--weights", help="detector weights (.pt / .onnx / openvino dir); student, COCO or YOLO-World")
    ap.add_argument("--mode", choices=["coco", "world", "student"], default=None, help="default: auto")
    ap.add_argument("--from-tracks", nargs="+", type=Path, help="skip detection: metrics from these tracks.csv dirs")
    ap.add_argument("--files", default=None, help="comma list of file keys (0909,0956,1109,1129); default all")
    ap.add_argument("--stages", default="timeline,track,metrics")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--tl-fps", type=float, default=5.0)
    ap.add_argument("--track-fps", type=float, default=30.0)
    ap.add_argument("--track-width", type=int, default=1920, help="downscale wider input (4K file) for tracking; 0=off")
    ap.add_argument("--event-conf", type=float, default=0.3, help="session-1 value; do not tune on 26SE")
    ap.add_argument("--device", default=None)
    ap.add_argument("--hwaccel", default=None, choices=["cuda", "none"])
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--crops", action="store_true")
    ap.add_argument("--ids", default="world", choices=["world", "none"])
    a = ap.parse_args()
    a.files = a.files.split(",") if a.files else None
    if a.name.startswith(("crops_", "world_")) or "/" in a.name or "\\" in a.name:
        sys.exit("bad --name")
    outdir = OUT / ("eval_" + a.name)
    outdir.mkdir(parents=True, exist_ok=True)

    if a.from_tracks:
        return metrics(a.from_tracks, outdir, a, extra=dict(from_tracks=[str(d) for d in a.from_tracks]))

    if not a.weights:
        sys.exit("--weights or --from-tracks is required")
    ngpu = torch.cuda.device_count()
    a.device = a.device or ("0" if ngpu else "cpu")
    a.hwaccel = a.hwaccel or ("cuda" if ngpu else "none")
    if a.threads:
        torch.set_num_threads(a.threads)
    from det import Detector
    det = Detector(a.weights, mode=a.mode)
    if a.imgsz is None:
        a.imgsz = 640 if det.mode == "student" else 1280
    stages = a.stages.split(",")
    keys = a.files or ORDER
    timing_f = outdir / "timing.json"
    timing = json.loads(timing_f.read_text()) if timing_f.exists() else {}
    settings = dict(weights=a.weights, mode=det.mode, imgsz=a.imgsz, tl_fps=a.tl_fps, track_fps=a.track_fps,
                    device=a.device, hwaccel=a.hwaccel, threads=torch.get_num_threads(), event_conf=a.event_conf)
    print(json.dumps(settings), flush=True)
    for k in keys:
        video = ROOT / FILES[k][0]
        tl_csv = outdir / ("tl_%s.csv" % k)
        if "timeline" in stages and not tl_csv.exists():
            print("timeline", k, flush=True)
            timing["tl_" + k] = dict(stage_timeline(det, video, tl_csv, a), settings=settings)
            timing_f.write_text(json.dumps(timing, indent=1))
        tdir = outdir / ("trk_%s" % k)
        if "track" in stages and tl_csv.exists() and not (tdir / "tracks.csv").exists():
            print("track", k, flush=True)
            timing["trk_" + k] = dict(stage_track(det, video, tl_csv, tdir, a), settings=settings)
            timing_f.write_text(json.dumps(timing, indent=1))
    if "metrics" in stages:
        dirs = [outdir / ("trk_%s" % k) for k in keys if (outdir / ("trk_%s" % k) / "tracks.csv").exists()]
        if not dirs:
            sys.exit("no tracks yet")
        metrics(dirs, outdir, a, extra=timing)


if __name__ == "__main__":
    main()
