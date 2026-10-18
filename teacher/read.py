"""WP4 step 2 (GPU): VLM reads of the crops from teacher/crops.py.

  plain   the session-1 number prompt (lapvision/vlm.py PROMPT, fp16, batch 4-6) on every crop -> work/<id>/reads.parquet
          (+ reads.done). Columns: path, track_id, frame_idx, h_px, raw, num (parsed int or <NA>).
  ground  the WP0 grounding prompt on 2-3 crops of tracks that reached a consensus (idlib.vote over the plain reads,
          snapped to the event roster), only crops whose own plain read equals the consensus, largest boxes first
          -> work/<id>/grounded.parquet (+ grounded.done). Columns: path, track_id, consensus, raw, json_ok,
          boxes (JSON [[label, x1, y1, x2, y2], ...] in pixels of the stored crop file), kept (JSON [[x1, y1, x2, y2]]
          = boxes whose label equals the consensus). 0..1000 coordinates are relative to the image as fed, which is the
          stored crop (<= 640 px, so no resize).

Resumable: reads are appended in parts (work/<id>/reads_parts/part_NNNNN.parquet, every PART batches); a re-run skips
crops already read. A `PAUSE` file at the repo root: finish the batch, save the part, exit with code 3.
`--sample N` (plain only): read only the crops of up to N tracks per video, stratified by box height (calibration of
the height threshold; the reads are reused by the full run). `--hmin H`: only crops with box height >= H of tracks that
still have >= 3 such crops. `--cap N` (ground): at most N grounded crops over all given videos (priority: large boxes,
then round-robin over videos and shot tags).

usage (repo root): python teacher/read.py plain [id ...] [--batch 4] [--sample N] [--hmin H]
                   python teacher/read.py ground [id ...] [--batch 4] [--cap N] [--per 3]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORK, log_error, select_videos, wdir  # noqa: E402
import idlib  # noqa: E402

PART = 50          # batches per saved part
LOG = WORK / "read.log"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_parts(d, name="reads"):
    pd_ = d / f"{name}_parts"
    fs = sorted(pd_.glob("part_*.parquet")) if pd_.exists() else []
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def save_part(d, rows, name="reads"):
    if not rows:
        return
    pd_ = d / f"{name}_parts"
    pd_.mkdir(exist_ok=True)
    n = len(list(pd_.glob("part_*.parquet")))
    tmp = pd_ / f"part_{n:05d}.tmp"
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(pd_ / f"part_{n:05d}.parquet")


def todo_crops(d, hmin=None, sample=None, seed=0):
    C = pd.read_parquet(d / "crops.parquet")
    if hmin:
        C = C[C.h_px >= hmin]
        n = C.groupby("track_id").size()
        C = C[C.track_id.isin(n[n >= 3].index)]
    if sample:
        ct = pd.read_parquet(d / "crop_tracks.parquet")
        ct = ct[ct.track_id.isin(C.track_id)]
        bins = [0, 80, 100, 130, 170, 230, 330, 2000]
        ct["hb"] = pd.cut(ct.h_cand_max, bins)
        per = max(1, sample // (len(bins) - 1))
        pick = ct.groupby("hb", observed=True, group_keys=False).apply(
            lambda g: g.sample(min(len(g), per), random_state=seed))
        C = C[C.track_id.isin(pick.track_id)]
    return C


def plain(rows, batch=4, sample=None, hmin=None):
    rd = idlib.reader()
    total_t, total_n = 0.0, 0
    for row in rows:
        vid = row["id"]
        d = wdir(vid)
        if not (d / "crops.done").exists():
            log(f"{vid}: no crops.done, skip")
            continue
        if (d / "reads.done").exists() and not sample and not hmin:
            log(f"{vid}: reads done, skip")
            continue
        try:
            C = todo_crops(d, hmin, sample)
            done = load_parts(d)
            have = set(done.path) if len(done) else set()
            C = C[~C.path.isin(have)].sort_values(["track_id", "frame_idx"])
            log(f"{vid}: {len(C)} crops to read ({len(have)} already read), batch {batch}")
            buf, t0, nb = [], time.time(), 0
            recs = C.to_dict("records")
            for bj in idlib.batched(recs, batch):
                res = rd.read([str(ROOT / r["path"]) for r in bj])
                for r, (raw, num) in zip(bj, res):
                    buf.append(dict(path=r["path"], track_id=r["track_id"], frame_idx=r["frame_idx"], h_px=r["h_px"],
                                    raw=raw, num=num))
                nb += 1
                if nb % PART == 0 or idlib.paused():
                    save_part(d, buf)
                    el = time.time() - t0
                    log(f"{vid}: {nb * batch}/{len(recs)} read, {el / (nb * batch):.3f} s/crop")
                    buf = []
                    if idlib.paused():
                        log("PAUSE: stopping cleanly")
                        return 3
            save_part(d, buf)
            el = time.time() - t0
            total_t += el
            total_n += len(recs)
            if recs:
                log(f"{vid}: read {len(recs)} crops in {el:.0f}s ({el / max(len(recs), 1):.3f} s/crop)")
            if not sample and not hmin:
                R = load_parts(d)
                R["num"] = R["num"].astype("Int64")
                R = R.drop_duplicates("path", keep="last")
                R.to_parquet(d / "reads.parquet", index=False)
                (d / "reads.done").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {len(R)} reads\n")
            gpu = d / "gpu_time.json"
            g = json.loads(gpu.read_text()) if gpu.exists() else {}
            g["plain_s"] = g.get("plain_s", 0) + el
            g["plain_n"] = g.get("plain_n", 0) + len(recs)
            gpu.write_text(json.dumps(g))
        except Exception:  # noqa: BLE001
            import traceback
            log_error(vid, "read_plain", traceback.format_exc())
            log(f"{vid}: read plain FAILED (see errors.log)")
    if total_n:
        log(f"plain total {total_n} crops {total_t:.0f}s")
    return 0


def all_reads(d):
    if (d / "reads.parquet").exists():
        return pd.read_parquet(d / "reads.parquet")
    R = load_parts(d)
    return R.drop_duplicates("path", keep="last") if len(R) else R


def consensus_table(row):
    d = wdir(row["id"])
    roster = idlib.load_roster(row["event"])
    C = pd.read_parquet(d / "crops.parquet")
    R = all_reads(d)
    if not len(R):
        return pd.DataFrame()
    R = R[R.path.isin(C.path)]
    out = []
    for tid, g in R.groupby("track_id"):
        nums = [None if pd.isna(x) else int(x) for x in g.num]
        v = idlib.vote(nums, roster)
        v["track_id"] = int(tid)
        out.append(v)
    return pd.DataFrame(out)


def ground_jobs(rows, per=3, cap=None):
    jobs = []
    for row in rows:
        d = wdir(row["id"])
        if not (d / "reads.done").exists() or (d / "grounded.done").exists():
            continue
        V = consensus_table(row)
        if not len(V):
            continue
        V = V[V.consensus]
        R = all_reads(d)
        C = pd.read_parquet(d / "crops.parquet").merge(R[["path", "num"]], on="path")
        C = C.merge(V[["track_id", "number"]], on="track_id")
        C = C[C.num == C.number]
        sh = pd.read_csv(d / "shots.csv")[["shot_id", "tag"]]
        C = C.merge(sh, on="shot_id", how="left")
        C = C.sort_values("h_px", ascending=False).groupby("track_id").head(per)
        C["vid"] = row["id"]
        C["track_h"] = C.groupby("track_id").h_px.transform("max")
        jobs.append(C)
    if not jobs:
        return pd.DataFrame()
    J = pd.concat(jobs, ignore_index=True)
    if cap and len(J) > cap:
        # priority: tracks with large boxes; round-robin over (video, tag) so every event / shot type is represented
        T = J.groupby(["vid", "track_id"]).agg(track_h=("track_h", "first"), tag=("tag", "first"),
                                               n=("path", "size")).reset_index()
        T = T.sort_values("track_h", ascending=False)
        T["rk"] = T.groupby(["vid", "tag"]).cumcount()
        T = T.sort_values(["rk", "track_h"], ascending=[True, False])
        T["cum"] = T.n.cumsum()
        T = T[T.cum <= cap]
        J = J.merge(T[["vid", "track_id"]], on=["vid", "track_id"])
    return J


def ground(rows, batch=4, per=3, cap=None):
    J = ground_jobs(rows, per, cap)
    log(f"ground: {len(J)} crops over {J.vid.nunique() if len(J) else 0} videos (cap {cap})")
    if not len(J):
        return 0
    idlib.reader()
    for vid, Jv in J.groupby("vid", sort=False):
        d = wdir(vid)
        done = load_parts(d, "grounded")
        have = set(done.path) if len(done) else set()
        Jv = Jv[~Jv.path.isin(have)]
        buf, t0, nb = [], time.time(), 0
        recs = Jv.to_dict("records")
        log(f"{vid}: {len(recs)} crops to ground ({len(have)} done)")
        try:
            for bj in idlib.batched(recs, batch):
                txt = idlib.generate([str(ROOT / r["path"]) for r in bj], idlib.GROUND, max_new_tokens=120)
                for r, t in zip(bj, txt):
                    p = idlib.parse_ground(t)
                    boxes, kept = [], []
                    for lab, b in (p or []):
                        x1, y1, x2, y2 = (b[0] / 1000 * r["cw"], b[1] / 1000 * r["ch"], b[2] / 1000 * r["cw"],
                                          b[3] / 1000 * r["ch"])
                        box = [round(min(x1, x2), 1), round(min(y1, y2), 1), round(max(x1, x2), 1),
                               round(max(y1, y2), 1)]
                        boxes.append([lab] + box)
                        if idlib.parse_num(lab) == r["number"] and box[2] > box[0] and box[3] > box[1]:
                            kept.append(box)
                    buf.append(dict(path=r["path"], track_id=r["track_id"], consensus=int(r["number"]), raw=t,
                                    json_ok=p is not None, boxes=json.dumps(boxes), kept=json.dumps(kept)))
                nb += 1
                if nb % 20 == 0 or idlib.paused():
                    save_part(d, buf, "grounded")
                    buf = []
                    log(f"{vid}: grounded {nb * batch}/{len(recs)}, {(time.time() - t0) / (nb * batch):.2f} s/crop")
                    if idlib.paused():
                        log("PAUSE: stopping cleanly")
                        return 3
            save_part(d, buf, "grounded")
            el = time.time() - t0
            G = load_parts(d, "grounded").drop_duplicates("path", keep="last")
            G.to_parquet(d / "grounded.parquet", index=False)
            (d / "grounded.done").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {len(G)} grounded, cap {cap}\n")
            gpu = d / "gpu_time.json"
            g = json.loads(gpu.read_text()) if gpu.exists() else {}
            g["ground_s"] = g.get("ground_s", 0) + el
            g["ground_n"] = g.get("ground_n", 0) + len(recs)
            gpu.write_text(json.dumps(g))
            log(f"{vid}: grounded {len(recs)} in {el:.0f}s")
        except Exception:  # noqa: BLE001
            import traceback
            log_error(vid, "read_ground", traceback.format_exc())
            log(f"{vid}: ground FAILED (see errors.log)")
    return 0


def main(argv):
    if not argv or argv[0] not in ("plain", "ground"):
        print(__doc__)
        return 2
    mode, argv = argv[0], list(argv[1:])
    opt = {"--batch": 4, "--sample": None, "--hmin": None, "--cap": None, "--per": 3}
    for k in list(opt):
        if k in argv:
            i = argv.index(k)
            opt[k] = int(argv[i + 1])
            del argv[i:i + 2]
    ids = [a for a in argv if not a.startswith("--")]
    rows = select_videos(ids or None)
    if idlib.paused():
        log("PAUSE present, not starting")
        return 3
    if mode == "plain":
        return plain(rows, opt["--batch"], opt["--sample"], opt["--hmin"])
    return ground(rows, opt["--batch"], opt["--per"], opt["--cap"])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
