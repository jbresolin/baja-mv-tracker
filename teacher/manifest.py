"""WP1: video manifest. Scans youtube/ (recursively) and writes data/videos.csv and data/events.csv.

usage: python teacher/manifest.py [--dir youtube] [--reprobe]

videos.csv: id (= file stem; runs of characters outside [A-Za-z0-9_.-] become _), path (relative to the repo), event, prova, duration_s, width, height, fps, vcodec, bitrate,
            size_bytes, has_roster, status (active | superseded), superseded_by
events.csv: event, name, year, region, n_videos (active), hours (active), has_roster, split

Idempotent: re-running picks up new files and drops rows for files that disappeared. ffprobe results are reused for
files whose size + mtime did not change (--reprobe forces a new probe).
Event/prova come from the first `<YY><BR|SE|SU|NE>_<PROVA>` match in the stem, so `26BR_END.mp4` and a later
`26BR_END_takeout.mkv` are the same EV_PROVA. When several files share an EV_PROVA, the one with the largest
(height, bitrate) is active and the others are `superseded` (stages run only on active rows).
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, ROSTERS, YOUTUBE, tool  # noqa: E402

VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".ts"}
REGIONS = {"BR": "Nacional", "SE": "Sudeste", "SU": "Sul", "NE": "Nordeste"}
TEST_EVENTS = {"24SU"}  # TRAINING_PLAN.md held-out proposal (decided at G2)
PAT = re.compile(r"(\d\d)(BR|SE|SU|NE)_([A-Z]{2,4})(?![A-Z])")
VCOLS = ["id", "path", "event", "prova", "duration_s", "width", "height", "fps", "vcodec", "bitrate", "size_bytes",
         "mtime", "has_roster", "status", "superseded_by"]
ECOLS = ["event", "name", "year", "region", "n_videos", "hours", "has_roster", "split"]


def probe(path):
    out = subprocess.run([tool("ffprobe"), "-v", "error", "-show_entries",
                          "format=duration,bit_rate:stream=codec_type,codec_name,width,height,avg_frame_rate,bit_rate",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    v = next(s for s in j["streams"] if s.get("codec_type") == "video")
    num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    br = v.get("bit_rate") or j["format"].get("bit_rate") or 0
    return {"duration_s": round(float(j["format"].get("duration", 0)), 3), "width": v.get("width", 0),
            "height": v.get("height", 0), "fps": round(fps, 3), "vcodec": v.get("codec_name", ""),
            "bitrate": int(br)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(YOUTUBE))
    ap.add_argument("--reprobe", action="store_true")
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    vpath = DATA / "videos.csv"
    old = {}
    if vpath.exists() and not a.reprobe:
        with open(vpath, newline="", encoding="utf-8") as f:
            old = {r["path"]: r for r in csv.DictReader(f)}

    files = sorted(p for p in Path(a.dir).rglob("*") if p.suffix.lower() in VIDEO_EXT and p.is_file())
    rows, seen_ids = [], set()
    for p in files:
        rel = p.resolve().relative_to(ROOT).as_posix() if p.resolve().is_relative_to(ROOT) else str(p.resolve())
        st = p.stat()
        o = old.get(rel)
        if o and o.get("size_bytes") == str(st.st_size) and o.get("mtime") == str(int(st.st_mtime)):
            info = {k: o[k] for k in ("duration_s", "width", "height", "fps", "vcodec", "bitrate")}
        else:
            try:
                info = probe(p)
            except Exception as e:  # noqa: BLE001
                print(f"ffprobe failed on {p}: {e}", file=sys.stderr)
                continue
            print(f"probed {rel}: {info}")
        vid = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.stem).strip("_")  # safe as a folder name
        if vid in seen_ids:  # same stem in two folders: keep ids unique
            vid = f"{vid}_{p.suffix[1:].lower()}"
        seen_ids.add(vid)
        m = PAT.search(p.stem)
        ev, prova = (m.group(1) + m.group(2), m.group(3)) if m else ("", "")
        if not m:
            print(f"warning: no <EV>_<PROVA> in {p.name}; event left empty", file=sys.stderr)
        rows.append({"id": vid, "path": rel, "event": ev, "prova": prova, **info, "size_bytes": st.st_size,
                     "mtime": int(st.st_mtime), "has_roster": int(bool(ev) and (ROSTERS / f"{ev}.csv").exists()),
                     "status": "active", "superseded_by": ""})

    # one active file per EV_PROVA: highest resolution, then bitrate
    groups = {}
    for r in rows:
        if r["event"]:
            groups.setdefault((r["event"], r["prova"]), []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: (int(r["height"]), float(r["bitrate"]), float(r["duration_s"])), reverse=True)
        for r in g[1:]:
            r["status"], r["superseded_by"] = "superseded", g[0]["id"]

    rows.sort(key=lambda r: (r["event"], r["prova"], r["status"], r["id"]))
    with open(vpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, VCOLS)
        w.writeheader()
        w.writerows(rows)

    evs = {}
    for r in rows:
        if not r["event"]:
            continue
        e = evs.setdefault(r["event"], {"n": 0, "s": 0.0})
        if r["status"] == "active":
            e["n"] += 1
            e["s"] += float(r["duration_s"])
    erows = []
    for ev in sorted(evs, key=lambda e: (e[:2], e[2:])):
        yy, reg = ev[:2], ev[2:]
        year = 2000 + int(yy)
        erows.append({"event": ev, "name": f"Baja SAE Brasil {REGIONS[reg]} {year}", "year": year,
                      "region": REGIONS[reg], "n_videos": evs[ev]["n"], "hours": round(evs[ev]["s"] / 3600, 2),
                      "has_roster": int((ROSTERS / f"{ev}.csv").exists()),
                      "split": "test" if ev in TEST_EVENTS else "train"})
    with open(DATA / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ECOLS)
        w.writeheader()
        w.writerows(erows)
    act = [r for r in rows if r["status"] == "active"]
    print(f"{len(rows)} files, {len(act)} active, {sum(float(r['duration_s']) for r in act) / 3600:.1f} h; "
          f"{len(erows)} events -> {vpath}, {DATA / 'events.csv'}")


if __name__ == "__main__":
    main()
