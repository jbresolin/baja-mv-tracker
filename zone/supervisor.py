"""Restart-with-backoff for the zone node's processes (the app, the PTZ recorder). Used by zone\\run_zone.bat:

    python -m zone.supervisor --name zone --config my.toml -- python -m zone.app --config my.toml

Rules:
* exit code 0 = stopped on purpose (Ctrl-C, or the recorder is disabled in config): don't restart.
* anything else (crash, exit 3 = no spotter frames, exit 4 = PTZ recording stalled): wait, then restart. The wait
  starts at backoff_initial_s and doubles on each quick failure, up to backoff_max_s; a run that lasted at least
  healthy_after_s resets it.
* The app always comes back in STOPPED (it never remembers AUTO), so a restart never moves the camera by itself.
"""
import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def supervise(cmd, clock, run=None, backoff_initial_s=2.0, backoff_max_s=60.0, healthy_after_s=60.0, log=print,
              max_restarts=None):
    """Run `cmd` until it exits 0. Returns 0, or the last exit code when max_restarts (tests) is reached."""
    run = run or (lambda c: subprocess.call(c))
    delay = float(backoff_initial_s)
    restarts = 0
    while True:
        t0 = clock.monotonic()
        try:
            rc = run(cmd)
        except KeyboardInterrupt:
            return 0
        ran = clock.monotonic() - t0
        if rc == 0:
            log("%s exited normally" % _name(cmd))
            return 0
        if ran >= healthy_after_s:
            delay = float(backoff_initial_s)
        if max_restarts is not None and restarts >= max_restarts:
            return rc
        log("%s exited with code %s after %.0f s; restarting in %.0f s" % (_name(cmd), rc, ran, delay))
        clock.sleep(delay)
        restarts += 1
        delay = min(delay * 2.0, float(backoff_max_s))


def _name(cmd):
    return " ".join(str(c) for c in cmd[:4])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: python -m zone.supervisor [--name N] [--config F] -- <command ...>")
    k = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="process")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv[:k])
    cmd = argv[k + 1:]
    from zone import config as C
    from zone.clock import SystemClock
    cfg = C.load(a.config)["supervisor"]

    def log(m):
        print("[%s %s] %s" % (dt.datetime.now().strftime("%H:%M:%S"), a.name, m), flush=True)

    return supervise(cmd, SystemClock(), None, cfg["backoff_initial_s"], cfg["backoff_max_s"], cfg["healthy_after_s"],
                     log)


if __name__ == "__main__":
    sys.exit(main())
