"""Offline fusion of beam breaks with vision passes (WP-L2), for after the event.

usage:
  python rt/tools/fuse_beam.py out/lapline/2026-11-21/lapline_08-55-10.jsonl [more runs of the day ...]
         [--out fused.csv] [--window 1.0] [--search 3.0] [--offset S]
         [--dump lap_count_dump_27se.sql --video-minus-server 0.0 [--human-window 6]]

What it does:
  1. Reads rt's JSONL: vision passes (`pass_end`, plus the number from `pass_identity`) and beam breaks
     (`beam_break`); also `beam_stuck` and `camera_latency` for the summary. Several files (rt restarts during the
     day) are read together.
  2. Estimates the camera-vs-beam offset (t_cross - t_beam: the camera's centre line is not exactly at the beam)
     from CLEAN single matches: a pass with exactly one break within --search seconds, and that break with
     exactly one pass. Median, spread. --offset S skips the estimate.
  3. Matches breaks to passes one-to-one: nearest |t_beam - (t_cross - offset)| first, within --window seconds.
     What is left over is classified:
       match            a pass and its break
       shared_break     a pass whose only nearby break already belongs to another pass: two cars side by side
                        cut the beam once (one break, two passes)
       pass_no_break    a pass with no break near it (beam off / stuck / misaligned, or a false vision pass)
       break_no_pass    a break with no pass: a marshal or a person, or a car vision missed / could not read
  4. With --dump (the lap app's MySQL dump; parsed by lapvision/gt.py): compares the fused crossing times with the
     human counts (`contagem_volta`, server clock + --video-minus-server = this PC's clock), one-to-one within
     --human-window seconds, and checks the vision number against the human car number.
Outputs: a CSV (one row per pass / break / human lap) and a short summary printed and saved as <out>.summary.json.
"""
import argparse
import csv
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CSV_COLS = ["kind", "run", "pass_id", "number", "auto_confirm", "t_cross", "t_cross_iso", "break_id", "t_beam",
            "t_beam_iso", "n_edges", "duration_s", "cross_minus_beam_s", "residual_s", "t_fused", "t_fused_iso",
            "human_car", "human_lap", "t_human", "t_human_iso", "fused_minus_human_s", "number_agrees"]


def iso(t):
    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds") if t is not None else None


# ------------------------------------------------------------------------------------------------ reading
def load_runs(paths):
    """Read one or more rt JSONL files. Returns (passes, breaks, extras): passes and breaks are lists of dicts with
    a "key" = (run number, id) that is unique across files; extras holds stuck / latency events for the summary."""
    passes, breaks = {}, {}
    extras = dict(stuck=[], latency=[], pass_beam=[])
    for run, path in enumerate(paths):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue                 # a line cut short when rt was killed: skip it
                t = ev.get("type")
                if t == "pass_end":
                    key = (run, ev["pass_id"])
                    p = passes.setdefault(key, dict(key=key, run=run, pass_id=ev["pass_id"], number=None,
                                                    auto_confirm=None))
                    p.update(t_cross=ev["t_cross"], t_enter=ev.get("t_enter"), t_exit=ev.get("t_exit"))
                elif t == "pass_identity":
                    key = (run, ev["pass_id"])
                    p = passes.setdefault(key, dict(key=key, run=run, pass_id=ev["pass_id"]))
                    p.update(number=ev.get("number"), auto_confirm=ev.get("auto_confirm"))
                    p.setdefault("t_cross", ev.get("t_cross"))
                elif t == "beam_break":
                    key = (run, ev["break_id"])
                    breaks[key] = dict(key=key, run=run, break_id=ev["break_id"], t_beam=ev["t_beam"],
                                       n_edges=ev.get("n_edges"), duration_s=ev.get("duration_s"),
                                       started_blocked=bool(ev.get("started_blocked")))
                elif t == "beam_stuck":
                    extras["stuck"].append(ev)
                elif t == "camera_latency":
                    extras["latency"].append(ev)
                elif t == "pass_beam":
                    extras["pass_beam"].append(ev)
    good_passes = [p for p in passes.values() if p.get("t_cross") is not None]
    return sorted(good_passes, key=lambda p: p["t_cross"]), sorted(breaks.values(), key=lambda b: b["t_beam"]), extras


# ------------------------------------------------------------------------------------------------ offset
def estimate_offset(passes, breaks, search_s=3.0):
    """t_cross - t_beam of every CLEAN single match: the pass has exactly one break within search_s, and that break
    has exactly one pass within search_s. Returns the list of offsets (seconds)."""
    usable = [b for b in breaks if not b["started_blocked"]]
    near_pass = {p["key"]: [b for b in usable if abs(p["t_cross"] - b["t_beam"]) <= search_s] for p in passes}
    near_break = {b["key"]: [p for p in passes if abs(p["t_cross"] - b["t_beam"]) <= search_s] for b in usable}
    out = []
    for p in passes:
        cands = near_pass[p["key"]]
        if len(cands) == 1 and len(near_break[cands[0]["key"]]) == 1:
            out.append(p["t_cross"] - cands[0]["t_beam"])
    return out


def describe(values):
    """n, median, p5, p95 and MAD (median absolute deviation) of a list of seconds; None values when empty."""
    if not values:
        return dict(n=0, median=None, p5=None, p95=None, mad=None)
    v = sorted(values)
    med = statistics.median(v)

    def pct(q):
        return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]
    return dict(n=len(v), median=round(med, 4), p5=round(pct(0.05), 4), p95=round(pct(0.95), 4),
                mad=round(statistics.median(abs(x - med) for x in v), 4))


# ------------------------------------------------------------------------------------------------ matching
def fuse(passes, breaks, offset=0.0, window_s=1.0):
    """One-to-one matching (nearest first) and the classification of what is left over. Returns the rows."""
    usable = [b for b in breaks if not b["started_blocked"]]
    pairs = []
    for p in passes:
        expected = p["t_cross"] - offset
        for b in usable:
            d = b["t_beam"] - expected
            if abs(d) <= window_s:
                pairs.append((abs(d), p["key"], b["key"]))
    pairs.sort()
    b_of = {b["key"]: b for b in breaks}
    pass_to_break, break_to_pass = {}, {}
    for _, pk, bk in pairs:
        if pk in pass_to_break or bk in break_to_pass:
            continue
        pass_to_break[pk], break_to_pass[bk] = bk, pk
    rows = []
    for p in passes:
        bk = pass_to_break.get(p["key"])
        kind = "match"
        if bk is None:
            near = [(abs(b["t_beam"] - (p["t_cross"] - offset)), b["key"]) for b in usable
                    if abs(b["t_beam"] - (p["t_cross"] - offset)) <= window_s]
            if near:
                kind, bk = "shared_break", min(near)[1]       # its break was taken by a car alongside
            else:
                kind = "pass_no_break"
        rows.append(_row(kind, p, b_of.get(bk), offset))
    for b in breaks:
        if b["key"] not in break_to_pass:
            rows.append(_row("break_no_pass", None, b, offset))
    rows.sort(key=lambda r: r["t_fused"])
    return rows


def _row(kind, p, b, offset):
    r = dict(kind=kind, run=(p or b)["run"], pass_id=p["pass_id"] if p else None,
             number=p.get("number") if p else None, auto_confirm=p.get("auto_confirm") if p else None,
             t_cross=p["t_cross"] if p else None, break_id=b["break_id"] if b else None,
             t_beam=b["t_beam"] if b else None, n_edges=b["n_edges"] if b else None,
             duration_s=b["duration_s"] if b else None)
    if p and b:
        r["cross_minus_beam_s"] = round(p["t_cross"] - b["t_beam"], 4)
        r["residual_s"] = round(r["cross_minus_beam_s"] - offset, 4)
    # The best time for the crossing: the beam when there is one, else the vision time corrected by the offset.
    r["t_fused"] = b["t_beam"] if b else p["t_cross"] - offset
    return r


# ------------------------------------------------------------------------------------------------ human counts
def load_human_laps(dump_path):
    """contagem_volta rows of a lap-app dump (lapvision/gt.py parser): car, lap, ts (server clock, epoch s)."""
    sys.path.insert(0, str(ROOT / "lapvision"))
    import gt
    saved = gt.SQL
    gt.SQL = Path(dump_path)        # gt.py reads a module-level dump path: point it at this dump, then restore it
    try:
        return gt.voltas()
    finally:
        gt.SQL = saved


def compare_human(rows, laps, video_minus_server=0.0, window_s=6.0):
    """Match human laps to fused crossings one-to-one (nearest first, within window_s) and fill the human columns.
    Laps outside the time span of the run are ignored. Returns the extra rows for unmatched human laps."""
    if not rows:
        return []
    t_lo, t_hi = min(r["t_fused"] for r in rows) - 60, max(r["t_fused"] for r in rows) + 60
    laps = [dict(lap, t_video=lap["ts"] + video_minus_server) for lap in laps]
    laps = [lap for lap in laps if t_lo <= lap["t_video"] <= t_hi]
    pairs = sorted((abs(r["t_fused"] - lap["t_video"]), i, j) for i, r in enumerate(rows)
                   for j, lap in enumerate(laps) if abs(r["t_fused"] - lap["t_video"]) <= window_s)
    used_r, used_l = set(), set()
    for _, i, j in pairs:
        if i in used_r or j in used_l:
            continue
        used_r.add(i)
        used_l.add(j)
        r, lap = rows[i], laps[j]
        r.update(human_car=lap["car"], human_lap=lap["lap"], t_human=lap["t_video"],
                 fused_minus_human_s=round(r["t_fused"] - lap["t_video"], 3),
                 number_agrees=None if r.get("number") is None else int(r["number"]) == int(lap["car"]))
    return [dict(kind="human_only", run=None, human_car=lap["car"], human_lap=lap["lap"], t_human=lap["t_video"],
                 t_fused=lap["t_video"]) for j, lap in enumerate(laps) if j not in used_l]


# ------------------------------------------------------------------------------------------------ summary
def summarize(rows, offsets, offset_used, extras, human=False):
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    residuals = [abs(r["residual_s"]) for r in rows if r.get("residual_s") is not None]
    out = dict(counts=kinds, offset_used_s=round(offset_used, 4), offset_clean_matches=describe(offsets),
               abs_residual_s=describe(residuals), stuck_episodes=sum(1 for e in extras["stuck"]
                                                                       if e.get("state") == "stuck"))
    lat = [e["latency_s"] for e in extras["latency"] if e.get("clean")]
    if extras["latency"]:
        out["camera_latency_clean_s"] = dict(describe(lat), min=round(min(lat), 4) if lat else None)
    if extras["pass_beam"]:
        live = {}
        for e in extras["pass_beam"]:
            live[e["status"]] = live.get(e["status"], 0) + 1
        out["live_association"] = live
    if human:
        hum = [r for r in rows if r.get("human_lap") is not None and r["kind"] != "human_only"]
        laps = len(hum) + kinds.get("human_only", 0)
        agree = [r["number_agrees"] for r in hum if r.get("number_agrees") is not None]
        out["human"] = dict(laps=laps, matched=len(hum), with_beam=sum(1 for r in hum if r.get("t_beam") is not None),
                            number_agrees=sum(agree), number_differs=len(agree) - sum(agree),
                            fused_minus_human_s=describe([r["fused_minus_human_s"] for r in hum]))
    return out


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = dict(r)
            for k in ("t_cross", "t_beam", "t_fused", "t_human"):
                if r.get(k) is not None:
                    r[k + "_iso"] = iso(r[k])
            w.writerow(r)


def run(paths, out, window_s=1.0, search_s=3.0, offset=None, dump=None, video_minus_server=0.0, human_window_s=6.0):
    """The whole script as a function (the tests call it). Returns the summary dict."""
    passes, breaks, extras = load_runs(paths)
    offsets = estimate_offset(passes, breaks, search_s)
    if offset is None:
        offset = statistics.median(offsets) if len(offsets) >= 3 else 0.0
    rows = fuse(passes, breaks, offset, window_s)
    human = dump is not None
    if human:
        rows += compare_human(rows, load_human_laps(dump), video_minus_server, human_window_s)
        rows.sort(key=lambda r: r["t_fused"])
    write_csv(rows, out)
    summary = summarize(rows, offsets, offset, extras, human)
    summary.update(passes=len(passes), breaks=len(breaks), files=[str(p) for p in paths], csv=str(out))
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def print_summary(s, out=print):
    c = s["counts"]
    out("passes %d, beam breaks %d (%d stuck episodes)" % (s["passes"], s["breaks"], s["stuck_episodes"]))
    out("  match %d, shared_break (side by side) %d, pass_no_break %d, break_no_pass %d" % (
        c.get("match", 0), c.get("shared_break", 0), c.get("pass_no_break", 0), c.get("break_no_pass", 0)))
    o = s["offset_clean_matches"]
    if o["n"]:
        out("camera - beam offset (t_cross - t_beam), %d clean matches: median %+.3f s, p5 %+.3f, p95 %+.3f, MAD "
            "%.3f; used %+.3f s" % (o["n"], o["median"], o["p5"], o["p95"], o["mad"], s["offset_used_s"]))
    else:
        out("camera - beam offset: no clean single matches; used %+.3f s" % s["offset_used_s"])
    if "camera_latency_clean_s" in s:
        lt = s["camera_latency_clean_s"]
        out("LED camera latency (clean): n %d, min %s s, median %s s" % (lt["n"], lt["min"], lt["median"]))
    if "live_association" in s:
        out("live association: %s" % s["live_association"])
    if "human" in s:
        h = s["human"]
        d = h["fused_minus_human_s"]
        out("human laps %d: matched %d (%d with a beam time), not matched %d; number agrees %d, differs %d; "
            "fused - human median %s s" % (h["laps"], h["matched"], h["with_beam"], c.get("human_only", 0),
                                           h["number_agrees"], h["number_differs"], d["median"]))
    out("csv: %s" % s["csv"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+", help="rt run JSONL file(s)")
    ap.add_argument("--out", default=None, help="CSV path (default: <first jsonl>.fused.csv)")
    ap.add_argument("--window", type=float, default=1.0, help="match window around t_cross - offset (s)")
    ap.add_argument("--search", type=float, default=3.0, help="window for the clean matches of the offset estimate")
    ap.add_argument("--offset", type=float, default=None, help="use this t_cross - t_beam instead of estimating it")
    ap.add_argument("--dump", default=None, help="lap app MySQL dump: compare with the human counts")
    ap.add_argument("--video-minus-server", type=float, default=0.0, help="this PC's clock minus the lap server's")
    ap.add_argument("--human-window", type=float, default=6.0, help="human lap <-> crossing window (s)")
    a = ap.parse_args(argv)
    out = a.out or str(Path(a.jsonl[0]).with_suffix("")) + ".fused.csv"
    s = run([Path(p) for p in a.jsonl], out, a.window, a.search, a.offset, a.dump, a.video_minus_server,
            a.human_window)
    print_summary(s)


if __name__ == "__main__":
    main()
