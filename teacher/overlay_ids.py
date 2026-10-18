"""WP2b: identities announced by broadcast overlays -> work/<id>/overlay_ids.parquet (+ overlay_ids.done)

Which overlays carry identities (kind 'lower_third' in work/<id>/overlays.csv, after teacher/overlay_overrides.csv):
  DIN streams   lower third "#35 Carpoeira Baja" / "16 - FORJA BAJA"  -> one car (mode 'single')
  SPP streams   heat box "1 - BAJA DE GALPAO 0 / 9 - PATO BAJA 1"     -> two cars (mode 'heat')
  END streams   none (leaderboard towers, tickers, 19SU's sponsor slot, 26BR_END's "CARRO 7 | CARRO 9" onboard labels
                are not identities of the car in shot)
Per region, during its visible intervals (padded by PAD s; the intervals come from 5 s keyframes):
  1. decode the region crop at 1 fps (CPU, ffmpeg crop filter);
  2. split the samples into segments of unchanged content: dHash (32x8) of the crop, a new segment starts when the
     Hamming distance to the segment's first sample exceeds HAMMING (fraction of bits), or after a gap;
  3. read each segment once with the VLM (transcribe prompt, the crop at native size, batch of BATCH) at the
     segment's middle sample; segments shorter than MIN_SEG samples are skipped (transitions);
  4. parse '#N team' / 'N - TEAM' per line; accept a (number, team) only if the number is in the event roster AND the
     team text fuzzy-matches that roster entry (idlib.team_score >= MIN_MATCH);
  5. merge consecutive segments with the same accepted numbers (gaps <= MERGE_GAP s) -> one row.
Output rows: region, mode, t0, t1 (s, interval during which that content is shown), numbers (list; 1 for a lower
third, 2 for a heat), teams (roster names), raw (VLM text of the first read), match (min team score), n_reads, n_samples.
Rejected reads (no number, number not in the roster, team mismatch) are kept in overlay_reads.parquet for inspection.

GPU: the VLM. Honours PAUSE between batches (state is re-done from scratch on the next run; it is a short stage).
usage (repo root): python teacher/overlay_ids.py [id ...] [--force] [--dry] [--batch 4]
  --dry: CPU only, decode + segment and print the number of reads it would need.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, log_error, select_videos, tool, wdir  # noqa: E402
import idlib  # noqa: E402

FPS = 1.0
PAD = 4.0
HAMMING = 0.18
MIN_SEG = 2
MERGE_GAP = 20.0   # s; segments between two reads of the same content that were too short to read do not split it
MIN_MATCH = 0.6
BATCH = 4
MODES = {"DIN": "single", "SPP": "heat"}
# explicit exclusions by (video, region box) if a lower_third region turns out not to carry identities
EXCLUDE = {}


def id_regions(vid, prova, d):
    p = d / "overlays.csv"
    mode = MODES.get(prova)
    if mode is None or not p.exists():
        return []
    o = pd.read_csv(p)
    out = []
    for r in o.itertuples():
        if str(r.kind) != "lower_third":
            continue
        if (vid, int(r.region)) in EXCLUDE:
            continue
        iv = []
        for s in str(r.intervals).split(";"):
            if "-" in s:
                a, b = s.split("-")
                iv.append((float(a), float(b)))
        out.append(dict(region=int(r.region), box=(int(r.x1), int(r.y1), int(r.x2), int(r.y2)), mode=mode,
                        intervals=iv))
    return out


def merge_iv(iv, pad, dur):
    iv = sorted((max(0.0, a - pad), min(dur, b + pad)) for a, b in iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1] + 1.0:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def decode_crops(path, box, a, b, fps=FPS):
    """Yield (t, bgr crop) at `fps` from a to b (source pixels box)."""
    x1, y1, x2, y2 = box
    w, h = (x2 - x1) // 2 * 2, (y2 - y1) // 2 * 2
    cmd = [tool("ffmpeg"), "-v", "error", "-nostdin", "-threads", "2", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}",
           "-i", str(path), "-an", "-sn", "-vf", f"fps={fps:g},crop={w}:{h}:{x1}:{y1}", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = w * h * 3
    k = 0
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield a + k / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            k += 1
    finally:
        p.stdout.close()
        p.wait()


def segments(samples):
    """samples: list of (t, crop). Returns list of (i0, i1) index ranges of unchanged content."""
    segs = []
    ref = None
    i0 = 0
    hs = [idlib.dhash(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)) for _, c in samples]
    for i, (t, _) in enumerate(samples):
        new = ref is None or (t - samples[i - 1][0] > 1.5 / FPS) or np.mean(hs[i] != ref) > HAMMING
        if new:
            if ref is not None:
                segs.append((i0, i - 1))
            i0, ref = i, hs[i]
    if ref is not None:
        segs.append((i0, len(samples) - 1))
    return segs


LINE_RE = re.compile(r"#?\s*(\d{1,3})\s*[-–—:|.]?\s*([A-Za-zÀ-ÿ][^\n]*)")


def parse_ids(raw, roster):
    """-> list of (number, team_text, roster_team, score) for every 'N team' found, accepted or not."""
    out = []
    if not raw or raw.strip().lower().startswith("none"):
        return out
    lines = [x.strip().strip("*`").strip() for x in re.split(r"[\n\r]+", raw)]
    joined = []
    for x in lines:  # "#7" / "Imperador" on two lines -> "#7 Imperador"
        if joined and re.fullmatch(r"#?\s*\d{1,3}\s*[-–—:|.]?", joined[-1]):
            joined[-1] = joined[-1] + " " + x
        else:
            joined.append(x)
    for line in joined:
        for m in LINE_RE.finditer(line):
            num = int(m.group(1))
            team = re.sub(r"\s+\d{1,2}\s*$", "", m.group(2)).strip()  # trailing heat score
            team = re.split(r"\s{2,}|\s[|]\s", team)[0].strip()
            ent = roster.get(num)
            sc = idlib.team_score(team, ent) if ent else 0.0
            out.append((num, team, ent["equipe"] if ent else None, sc))
    return out


def build_ids(d, roster):
    """(Re-)parse overlay_reads.parquet and merge consecutive segments with the same accepted numbers
    -> overlay_ids.parquet. A heat needs both numbers accepted, a lower third exactly one."""
    RR = pd.read_parquet(d / "overlay_reads.parquet")
    rows, upd = [], []
    for reg, G in RR.groupby("region"):
        cur = None
        for r in G.sort_values("t0").itertuples():
            ids = parse_ids(r.raw, roster)
            acc = sorted({n for n, _, rt, sc in ids if rt is not None and sc >= MIN_MATCH})
            upd.append((r.Index, json.dumps([[n, tt, rt, sc] for n, tt, rt, sc in ids], ensure_ascii=False), acc))
            nums = tuple(acc)
            if len(nums) != (1 if r.mode == "single" else 2):
                cur = None
                continue
            sc = min([x[3] for x in ids if x[0] in nums] or [0.0])
            if cur and cur["numbers"] == list(nums) and r.t0 - cur["t1"] <= MERGE_GAP:
                cur["t1"] = r.t1
                cur["n_reads"] += 1
                cur["match"] = min(cur["match"], sc)
            else:
                cur = dict(region=int(reg), mode=r.mode, t0=r.t0, t1=r.t1, numbers=list(nums),
                           teams=[roster[n]["equipe"] for n in nums], raw=r.raw, match=sc, n_reads=1, crop=r.crop)
                rows.append(cur)
    for i, pj, acc in upd:
        RR.at[i, "parsed"] = pj
        RR.at[i, "numbers"] = acc
    RR.to_parquet(d / "overlay_reads.parquet", index=False)
    df = pd.DataFrame(rows, columns=["region", "mode", "t0", "t1", "numbers", "teams", "raw", "match", "n_reads",
                                     "crop"])
    df["dur"] = df.t1 - df.t0
    df.to_parquet(d / "overlay_ids.parquet", index=False)
    return df


def process(row, batch=BATCH, dry=False):
    vid = row["id"]
    d = wdir(vid)
    path = ROOT / row["path"]
    regs = id_regions(vid, row["prova"], d)
    roster = idlib.load_roster(row["event"])
    rows, reads_all = [], []
    t_dec = t_vlm = 0.0
    n_samples = n_reads = 0
    for R in regs:
        t0 = time.time()
        samples = []
        for a, b in merge_iv(R["intervals"], PAD, row["duration_s"]):
            samples.extend(decode_crops(path, R["box"], a, b))
        segs = [s for s in segments(samples) if s[1] - s[0] + 1 >= MIN_SEG]
        t_dec += time.time() - t0
        n_samples += len(samples)
        print(f"{vid} region {R['region']} ({R['mode']}): {len(samples)} samples, {len(segs)} segments to read",
              flush=True)
        if dry:
            continue
        crops_dir = d / "overlay_ids_crops"
        crops_dir.mkdir(exist_ok=True)
        jobs = []
        for i0, i1 in segs:
            im = (i0 + i1) // 2
            t, c = samples[im]
            p = crops_dir / f"r{R['region']}_{int(round(t * 10)):07d}.jpg"
            cv2.imwrite(str(p), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
            jobs.append((samples[i0][0], samples[i1][0] + 1.0 / FPS, t, p))
        t0 = time.time()
        res = []
        for bj in idlib.batched(jobs, batch):
            if idlib.paused():
                raise RuntimeError("PAUSE file present: stopped (re-run to redo this stage)")
            txt = idlib.generate([str(j[3]) for j in bj], idlib.TRANSCRIBE, max_new_tokens=40)
            res.extend(zip(bj, txt))
        t_vlm += time.time() - t0
        n_reads += len(res)
        for (s0, s1, t, p), raw in res:
            ids = parse_ids(raw, roster)
            acc = sorted({(n, rt, sc) for n, _, rt, sc in ids if rt is not None and sc >= MIN_MATCH})
            reads_all.append(dict(region=R["region"], mode=R["mode"], t0=s0, t1=s1, t_read=t,
                                  crop=str(p.relative_to(ROOT)).replace("\\", "/"), raw=raw,
                                  parsed=json.dumps([[n, tt, rt, sc] for n, tt, rt, sc in ids], ensure_ascii=False),
                                  numbers=[a[0] for a in acc]))
    if dry:
        return f"dry: {n_samples} samples"
    pd.DataFrame(reads_all).to_parquet(d / "overlay_reads.parquet", index=False)
    df = build_ids(d, roster)
    info = dict(regions=len(regs), samples=n_samples, reads=n_reads, ids=len(df),
                accepted_reads=int(pd.read_parquet(d / "overlay_reads.parquet").numbers.map(len).gt(0).sum()), decode_s=round(t_dec),
                vlm_s=round(t_vlm), distinct_numbers=int(len({n for x in df.numbers for n in x})) if len(df) else 0)
    (d / "overlay_ids.json").write_text(json.dumps(info, indent=1))
    return json.dumps(info)


def sheet(vid, n=16, seed=0):
    """Validation sheet: n random accepted ids (overlay crop read + full frame at the middle of the interval, with the
    parsed numbers / roster teams) -> out/overlay_ids_<id>.jpg."""
    from common import frame_at
    row = select_videos([vid])[0]
    d = wdir(vid)
    O = pd.read_parquet(d / "overlay_ids.parquet")
    O = O.sample(min(n, len(O)), random_state=seed).sort_values("t0")
    tiles = []
    for r in O.itertuples():
        t = (r.t0 + r.t1) / 2
        f = cv2.cvtColor(frame_at(ROOT / row["path"], t, size=(640, 360)), cv2.COLOR_RGB2BGR)
        c = cv2.imread(str(ROOT / r.crop))
        s = 360 / c.shape[0] if c.shape[1] * 360 / c.shape[0] <= 640 else 640 / c.shape[1]
        c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)))
        pad = np.zeros((360, 640, 3), np.uint8)
        pad[:c.shape[0], :c.shape[1]] = c
        txt = f"{int(r.t0)}-{int(r.t1)}s {list(r.numbers)} {' | '.join(r.teams)} m={r.match:.2f}"
        txt = txt.encode("ascii", "replace").decode()
        cv2.rectangle(f, (0, 0), (640, 24), (0, 0, 0), -1)
        cv2.putText(f, txt, (4, 17), 0, 0.5, (0, 255, 255), 1)
        tiles.append(np.hstack([f, pad]))
    out = ROOT / "out" / f"overlay_ids_{vid}.jpg"
    cv2.imwrite(str(out), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(out)


def check(vid, n=30, seed=0, batch=BATCH):
    """Change-detection check (GPU): for n random read segments, also read the first and last 1 fps sample of the
    segment and compare the accepted numbers with the middle read."""
    row = select_videos([vid])[0]
    d = wdir(vid)
    roster = idlib.load_roster(row["event"])
    RR = pd.read_parquet(d / "overlay_reads.parquet")
    RR = RR.sample(min(n, len(RR)), random_state=seed)
    box = {R["region"]: R["box"] for R in id_regions(vid, row["prova"], d)}
    tmp = d / "overlay_ids_check"
    tmp.mkdir(exist_ok=True)
    jobs = []
    for r in RR.itertuples():
        for tt in (r.t0, r.t1 - 1.0 / FPS):
            got = list(decode_crops(ROOT / row["path"], box[r.region], tt, tt + 0.99))
            if got:
                p = tmp / f"r{r.region}_{int(tt * 10)}.jpg"
                cv2.imwrite(str(p), got[0][1])
                jobs.append((r.Index, p))
    txt = []
    for bj in idlib.batched(jobs, batch):
        txt += idlib.generate([str(p) for _, p in bj], idlib.TRANSCRIBE, max_new_tokens=40)
    same = diff = 0
    for (i, p), t in zip(jobs, txt):
        mid = sorted(RR.loc[i, "numbers"])
        acc = sorted({n for n, _, rt, sc in parse_ids(t, roster) if rt is not None and sc >= MIN_MATCH})
        if acc == mid:
            same += 1
        else:
            diff += 1
            print(f"segment {RR.loc[i, 't0']:.0f}-{RR.loc[i, 't1']:.0f}: middle {mid} vs edge {acc}: {t!r}")
    print(f"{vid}: edge reads equal to the middle read: {same}/{same + diff}")


def main(argv):
    if argv and argv[0] == "reparse":
        for row in select_videos(argv[1:] or None):
            d = wdir(row["id"])
            if (d / "overlay_reads.parquet").exists():
                df = build_ids(d, idlib.load_roster(row["event"]))
                info = json.loads((d / "overlay_ids.json").read_text())
                RR = pd.read_parquet(d / "overlay_reads.parquet")
                info.update(ids=len(df), accepted_reads=int(RR.numbers.map(len).gt(0).sum()),
                            distinct_numbers=int(len({n for x in df.numbers for n in x})) if len(df) else 0)
                (d / "overlay_ids.json").write_text(json.dumps(info, indent=1))
                print(row["id"], info)
        return 0
    if argv and argv[0] in ("sheet", "check"):
        (sheet if argv[0] == "sheet" else check)(argv[1])
        return 0
    force = "--force" in argv
    dry = "--dry" in argv
    batch = BATCH
    if "--batch" in argv:
        batch = int(argv[argv.index("--batch") + 1])
    ids = [a for a in argv if not a.startswith("--") and not a.isdigit()]
    for row in select_videos(ids or None):
        d = wdir(row["id"])
        if MODES.get(row["prova"]) is None:
            continue
        if (d / "overlay_ids.done").exists() and not force and not dry:
            print(f"{row['id']}: overlay_ids done, skip")
            continue
        if not (d / "overlay.done").exists():
            print(f"{row['id']}: needs overlay.done, skip")
            continue
        if idlib.paused():
            print("PAUSE present, stop")
            return 3
        t0 = time.time()
        try:
            info = process(row, batch=batch, dry=dry)
            if not dry:
                (d / "overlay_ids.done").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {time.time() - t0:.0f}s "
                                                    f"{info}\n")
            print(f"{row['id']}: overlay_ids ok in {time.time() - t0:.0f}s {info}", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            log_error(row["id"], "overlay_ids", traceback.format_exc())
            print(f"{row['id']}: overlay_ids FAILED {e!r}", flush=True)
            if idlib.paused():
                return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
