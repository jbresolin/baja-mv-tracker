"""Record the PTZ's USB H.264 output with ffmpeg `-c copy` (no re-encode), as its own process next to the zone app,
supervised by zone\\run_zone.bat. Same wall clock as the control logs: the start time is in the file name
(ptz_YYYYmmdd-HHMMSS.mkv) and in a JSON sidecar.

    python -m zone.tools.record_ptz --config my.toml

Optional: with recording.ptz_usb = false it exits 0 at once (the launcher then doesn't restart it).
Exit 4 when the file stops growing for 15 s (camera unplugged, ffmpeg stuck): the launcher restarts it.
List the camera's dshow name with:  ffmpeg -list_devices true -f dshow -i dummy
"""
import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def build_command(rc, out_dir, t_wall0):
    """-> (ffmpeg command list, output path or pattern)."""
    out_dir = Path(out_dir)
    cmd = [rc["ffmpeg"], "-hide_banner", "-nostdin", "-v", "error", "-y"] + list(rc["ptz_input"]) + \
          ["-i", rc["ptz_device"], "-map", "0:v", "-c", "copy"]
    seg = float(rc.get("ptz_segment_s", 0) or 0)
    if seg > 0:
        out = out_dir / "ptz_%Y%m%d-%H%M%S.mkv"
        cmd += ["-f", "segment", "-segment_time", "%g" % seg, "-reset_timestamps", "1", "-strftime", "1", str(out)]
    else:
        out = out_dir / ("ptz_%s.mkv" % dt.datetime.fromtimestamp(t_wall0).strftime("%Y%m%d-%H%M%S"))
        cmd += [str(out)]
    return cmd, out


def newest(out_dir):
    files = sorted(Path(out_dir).glob("ptz_*.mkv"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    from zone import config as C
    cfg = C.load(a.config, a.set)
    rc = cfg["recording"]
    if not rc["ptz_usb"]:
        print("PTZ USB recording is off (recording.ptz_usb = false)", flush=True)
        return 0
    out_dir = C.resolve_path(cfg["zone"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cmd, out = build_command(rc, out_dir, t0)
    side = out_dir / ("ptz_%s.json" % dt.datetime.fromtimestamp(t0).strftime("%Y%m%d-%H%M%S"))
    side.write_text(json.dumps(dict(start_wall=t0, start_iso=dt.datetime.fromtimestamp(t0).astimezone().isoformat(
        timespec="milliseconds"), command=cmd, output=str(out),
        note="start_wall = when ffmpeg was launched; the first frame arrives a little later"), indent=1),
        encoding="utf-8")
    print("recording PTZ: %s" % " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    last_size, last_change = -1, time.time()
    try:
        while proc.poll() is None:
            time.sleep(2.0)
            f = newest(out_dir)
            size = f.stat().st_size if f else 0
            if size != last_size:
                last_size, last_change = size, time.time()
            elif time.time() - last_change > 15:
                print("PTZ recording stalled for 15 s; restarting", flush=True)
                proc.kill()
                return 4
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait(timeout=10)
        return 0
    return proc.returncode or 1   # ffmpeg ending on its own is never "on purpose": restart


if __name__ == "__main__":
    sys.exit(main())
