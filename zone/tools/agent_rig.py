"""A zone node with everything simulated, to try the studio's agent API without a webcam, a PTZ or a track.

    python -m zone.tools.agent_rig prepare --dir out/agent_rig [--driver visca|sim] [--token T]
    python -m zone.tools.agent_rig camera --dir out/agent_rig            # only with --driver visca: the fake PTZ
    python -m zone.app --config out/agent_rig/site.toml                   # the zone node, agent API on 8890

`prepare` writes, in --dir:

* `spotter.mp4`: a synthetic spotter recording (white boxes driving along a racing line, a car every few seconds;
  zone/testing.py's world), played in a loop as the spotter source, with the "blobs" detector;
* `zone.json`: that world's zone, taught (5 keyframes, presets 10-14), so AUTO can run;
* `report.json`: a characterization report of the simulated PTZ (Branch A: it answers inquiries while moving, so
  the agent reports positions);
* `site.toml`: the zone config: that source, zone and report, the PTZ (`--driver visca`: the real VISCA driver
  talking to `camera` on 127.0.0.1:--visca-port; `--driver sim`: the simulator inside the zone process), recording
  off, the page and the agent API on 127.0.0.1 only.

`camera` runs the fake VISCA camera (zone/ptz/fake_camera.py) with the zone's keyframe presets stored in it, as the
real PTZ has them after a teach. It is also what the studio's direct fallback drives while the zone app is down.

The baja-studio conformance suite and the end-to-end check (zone/tools/studio_e2e.py) run against this rig; the
commands are in zone/README.md ("Agent API").
"""
import argparse
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from zone import zonefile  # noqa: E402
from zone.ptz.sim import PtzModel  # noqa: E402


def prepare(args) -> int:
    from zone.testing import Car, Scene, write_video
    out = Path(args.dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scene = Scene()
    video = out / "spotter.mp4"
    if not video.exists() or args.force:
        # a car every `gap` seconds, at slightly different speeds, for `length` seconds (then the file loops)
        cars = [Car(scene, t0, factor=1.0 + 0.1 * (k % 3)) for k, t0 in enumerate(range(1, args.length - 8,
                                                                                         args.gap))]
        print("writing %s (%d s, %d cars)..." % (video, args.length, len(cars)), flush=True)
        write_video(video, scene, cars, float(args.length), fps=10.0)
    zonefile.save(scene.make_zone(), out / "zone.json")
    report = {"verdict": {"branch": "A", "reason": "agent rig: the fake PTZ answers inquiries while moving"},
              "model": PtzModel().to_dict()}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    p = out.as_posix()
    toml = "\n".join([
        "# written by python -m zone.tools.agent_rig prepare",
        "[zone]", 'file = "%s/zone.json"' % p, 'out_dir = "%s/out"' % p, "",
        "[spotter]", 'source = "%s/spotter.mp4"' % p, 'detector = "blobs"', "",
        "[ptz]", 'driver = "%s"' % args.driver, 'host = "127.0.0.1"', "port = %d" % args.visca_port,
        'report = "%s/report.json"' % p, "",
        "[recording]", "spotter = false", "",
        "[web]", 'host = "127.0.0.1"', "port = %d" % args.page_port, "",
        "[agent]", 'host = "127.0.0.1"', "port = %d" % args.agent_port, 'token = "%s"' % args.token,
        'agent_id = "%s"' % args.agent_id, 'name = "Zone rig (simulated)"', "",
    ])
    (out / "site.toml").write_text(toml, encoding="utf-8")
    print("rig ready in %s" % out)
    if args.driver == "visca":
        print("  1. python -m zone.tools.agent_rig camera --dir %s --port %d" % (out, args.visca_port))
    print("  %s. python -m zone.app --config %s/site.toml" % ("2" if args.driver == "visca" else "1", p))
    print("  then: zone page http://127.0.0.1:%d/  agent API http://127.0.0.1:%d/agent/info"
          % (args.page_port, args.agent_port))
    return 0


def camera(args) -> int:
    from zone.clock import SystemClock
    from zone.ptz.fake_camera import FakeViscaCamera, FakeViscaUdpServer
    from zone.ptz.sim import SimulatedPtz
    sim = SimulatedPtz(PtzModel(), SystemClock())
    zone = zonefile.load(Path(args.dir) / "zone.json")
    for k in zone.keyframes():
        sim.define_preset(k.preset, k.pan, k.tilt, k.zoom)
    server = FakeViscaUdpServer(FakeViscaCamera(sim), args.host, args.port).start()
    print("fake VISCA PTZ on %s:%d with presets %s; Ctrl+C to stop"
          % (server.address[0], server.address[1], [k.preset for k in zone.keyframes()]), flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="write the rig's recording, zone, report and config")
    p.add_argument("--dir", default="out/agent_rig")
    p.add_argument("--driver", choices=["visca", "sim"], default="visca")
    p.add_argument("--visca-port", type=int, default=52399)
    p.add_argument("--agent-port", type=int, default=8890)
    p.add_argument("--page-port", type=int, default=8090)
    p.add_argument("--token", default="")
    p.add_argument("--agent-id", default="zone-rig")
    p.add_argument("--length", type=int, default=60, help="recording length (s); it loops")
    p.add_argument("--gap", type=int, default=6, help="a car every this many seconds")
    p.add_argument("--force", action="store_true", help="write the recording again even if it exists")
    c = sub.add_parser("camera", help="run the fake VISCA PTZ with the zone's keyframe presets")
    c.add_argument("--dir", default="out/agent_rig")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=52399)
    args = ap.parse_args(argv)
    return prepare(args) if args.cmd == "prepare" else camera(args)


if __name__ == "__main__":
    sys.exit(main())
