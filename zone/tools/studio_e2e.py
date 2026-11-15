"""End-to-end check: the studio (baja-studio) drives the zone's PTZ through the agent API, and falls back to VISCA.

Everything runs on this PC, on 127.0.0.1, with the agent rig (zone/tools/agent_rig.py): a synthetic spotter
recording, a taught zone, and the fake VISCA PTZ, which the zone drives through its real VISCA driver and the studio
drives directly while the zone is down.

    python -m zone.tools.studio_e2e --studio-url http://127.0.0.1:8899
    python -m zone.tools.studio_e2e --studio-python <baja-studio>/server/.venv/bin/python \\
        --studio-server <baja-studio>/server --studio-data /tmp/studio-e2e

With --studio-url, a studio already running (its own scratch data folder!) is used. With --studio-python, the script
starts one itself (`python -m baja_studio --port 8899 --data-dir <--studio-data>`) and stops it at the end. The
studio is changed only through its HTTP API, as its own tests and its rehearsal rig do: an agent "zone-e2e" and a
camera "zone-e2e-ptz" are added (left there afterwards; use a scratch data folder).

The checks, in order (each prints PASS / FAIL and what it saw):

 1. the studio's link to the agent is `ok`, and the camera is listed, routed through the agent;
 2. a move from the studio moves the PTZ (the position the agent reports changes);
 3. Auto from the studio puts the zone's director in AUTO;
 4. a studio move pauses it (auto_paused, the zone's director goes PASSIVE) and it resumes by itself (AUTO);
 5. killing the zone app: the studio sees the agent offline, then drives the PTZ directly over VISCA (route
    "fallback"), and a studio move still moves it;
 6. starting the zone app again: the studio hands the camera back (route "agent", a /resync in the agent's log), and
    the camera comes back in manual (the zone always restarts in STOPPED), with the contract's restore notice
    (mode_changed auto -> manual, reason tracker_not_ready).

Needs `websockets` (to read the studio's state), like the agent API server itself.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_ID = "zone-e2e"
CAMERA_ID = "zone-e2e-ptz"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 127.0.0.1 never goes through a proxy


# ======================================================================================================== HTTP helpers
def http(method, url, body=None, headers=None, timeout=5.0):
    """-> (status, parsed JSON or text)."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers=dict(headers or {}))
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with OPENER.open(req, timeout=timeout) as r:
            raw, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    try:
        return status, json.loads(raw or b"null")
    except ValueError:
        return status, raw.decode(errors="replace")


def wait_for(what, check, timeout_s, period_s=0.2):
    """Poll check() until it returns something truthy; -> that value, or None after timeout_s."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            value = check()
        except Exception:  # noqa: BLE001 (not up yet, connection refused...)
            value = None
        if value:
            return value
        if time.monotonic() > deadline:
            return None
        time.sleep(period_s)


class Studio:
    """The studio's HTTP API (commands, settings) and its state (a snapshot over its WebSocket)."""

    def __init__(self, url):
        self.url = url.rstrip("/")

    def command(self, module, command, args=None):
        status, body = http("POST", self.url + "/api/commands",
                            {"module": module, "command": command, "args": args or {}, "caller": "zone-e2e"})
        if status != 200:
            raise RuntimeError("studio %s %s: %s %s" % (module, command, status, body))
        return body["result"]

    def update_settings(self, section, change):
        status, current = http("GET", "%s/api/settings/%s" % (self.url, section))
        values = change(current["values"])
        status, body = http("PUT", "%s/api/settings/%s" % (self.url, section),
                            {"version": current["version"], "values": values})
        if status != 200:
            raise RuntimeError("saving %s settings: %s %s" % (section, status, body))

    def state(self):
        from websockets.sync.client import connect
        ws_url = "ws" + self.url.removeprefix("http") + "/ws"
        with connect(ws_url, open_timeout=3, proxy=None) as ws:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                message = json.loads(ws.recv(timeout=3))
                if message.get("type") == "snapshot":
                    return message["state"]
        raise RuntimeError("no state snapshot from the studio")

    def camera(self):
        return self.state().get("cameras", {}).get("cameras", {}).get(CAMERA_ID, {})

    def link(self):
        return self.state().get("agents", {}).get("agents", {}).get(AGENT_ID, {})

    def hold_move(self, seconds, pan=0.5, period=0.1):
        """Hold a pan for `seconds`, as a browser does: a move every 0.1 s with the default ttl."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.command("cameras", "move", {"camera": CAMERA_ID, "pan": pan, "tilt": 0, "zoom": 0})
            time.sleep(period)


# ======================================================================================================== processes
class Children:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.procs = {}

    def start(self, name, cmd, cwd=None):
        log = open(self.log_dir / ("%s.log" % name), "ab")
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        self.procs[name] = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, **kw)
        return self.procs[name]

    def kill(self, name):
        """Kill hard (no clean shutdown): like a laptop that dies."""
        p = self.procs.pop(name, None)
        if p and p.poll() is None:
            p.kill()
            p.wait(10)

    def stop_all(self):
        for name, p in list(self.procs.items()):
            if p.poll() is None:
                p.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
        for name, p in list(self.procs.items()):
            try:
                p.wait(8)
            except subprocess.TimeoutExpired:
                p.kill()
        self.procs.clear()


# ======================================================================================================== the run
class Run:
    def __init__(self, args):
        self.args = args
        self.results = []
        self.agent_url = "http://127.0.0.1:%d" % args.agent_port
        self.page_url = "http://127.0.0.1:%d" % args.page_port
        self.agent_headers = {"X-Agent-Token": args.token} if args.token else {}

    def check(self, name, ok, seen):
        self.results.append((name, bool(ok)))
        print("%s  %s: %s" % ("PASS" if ok else "FAIL", name, seen), flush=True)
        return ok

    def agent(self, method, path, body=None):
        return http(method, self.agent_url + path, body, self.agent_headers)

    def page(self):
        return http("GET", self.page_url + "/api/state")[1]

    def agent_position(self):
        status, body = self.agent("GET", "/cameras/cam1/position")
        return body if status == 200 else None

    def start_zone(self, children):
        children.start("zone", [sys.executable, "-m", "zone.app", "--config", str(self.rig / "site.toml")],
                       cwd=str(ROOT))
        return wait_for("agent", lambda: http("GET", self.agent_url + "/agent/info")[0] == 200, 30)

    def main(self):
        a = self.args
        self.rig = Path(a.dir).resolve()
        children = Children(self.rig / "logs")
        try:
            from zone.tools import agent_rig
            agent_rig.main(["prepare", "--dir", str(self.rig), "--driver", "visca", "--token", a.token,
                            "--agent-port", str(a.agent_port), "--page-port", str(a.page_port),
                            "--visca-port", str(a.visca_port), "--agent-id", AGENT_ID])
            children.start("camera", [sys.executable, "-m", "zone.tools.agent_rig", "camera", "--dir",
                                      str(self.rig), "--port", str(a.visca_port)], cwd=str(ROOT))
            if not self.start_zone(children):
                return self.check("zone app starts", False, "no /agent/info on %s (see %s)"
                                  % (self.agent_url, self.rig / "logs" / "zone.log"))
            studio_url = a.studio_url
            if not studio_url:
                children.start("studio", [a.studio_python, "-m", "baja_studio", "--host", "127.0.0.1", "--port",
                                          str(a.studio_port), "--data-dir", a.studio_data], cwd=a.studio_server)
                studio_url = "http://127.0.0.1:%d" % a.studio_port
            studio = Studio(studio_url)
            if not wait_for("studio", lambda: http("GET", studio.url + "/api/health")[0] == 200, 60):
                return self.check("studio starts", False, "no /api/health on %s" % studio.url)
            self.setup(studio)
            self.checks(studio, children)
        finally:
            children.stop_all()
        failed = [name for name, ok in self.results if not ok]
        print("\n%d checks, %d failed%s" % (len(self.results), len(failed), (": " + ", ".join(failed)) if failed
                                            else ""))
        return 1 if failed else 0

    def setup(self, studio):
        """The Agents page's "add" and the camera configuration page's agent camera with a direct fallback."""
        a = self.args
        agents = studio.state().get("agents", {}).get("agents", {})
        if AGENT_ID in agents:
            studio.command("agents", "remove", {"id": AGENT_ID})
        studio.command("agents", "add", {"id": AGENT_ID, "host": "127.0.0.1", "port": a.agent_port,
                                         "token": a.token})
        studio.update_settings("agents", lambda v: {**v, "direct_fallback_after_s": float(a.fallback_after)})
        camera = {"id": CAMERA_ID, "label": "Zone PTZ (e2e)", "owner": "agent", "driver": "agent",
                  "address": {"agent_id": AGENT_ID, "remote_camera_id": "cam1"},
                  "direct_fallback": {"driver": "visca_ip", "address": {"host": "127.0.0.1", "port": a.visca_port}}}
        studio.update_settings("cameras", lambda v: {**v, "cameras": [c for c in v["cameras"]
                                                                     if c["id"] != CAMERA_ID] + [camera]})

    def checks(self, studio, children):
        a = self.args
        # 1. link and camera
        link = wait_for("link", lambda: (lambda l: l if l.get("status") == "ok" else None)(studio.link()), 15)
        link = link or studio.link()
        self.check("studio link to the agent is ok", link.get("status") == "ok",
                   "status %s %s, agent %s, contract %s" % (link.get("status"), link.get("reason") or "",
                                                            (link.get("info") or {}).get("agent_id"),
                                                            (link.get("info") or {}).get("contract_version")))
        cam = wait_for("camera", lambda: (lambda c: c if c.get("status") == "ok" and c.get("route") == "agent"
                                          else None)(studio.camera()), 15)
        self.check("camera listed in the studio, routed through the agent", cam,
                   {k: (cam or studio.camera()).get(k) for k in ("status", "route", "mode", "position")})

        # 2. a studio move moves the PTZ
        before = self.agent_position()
        studio.hold_move(1.0)
        time.sleep(0.6)
        after = self.agent_position()
        moved = before and after and after["pan"] - before["pan"]
        self.check("a move from the studio moves the PTZ", moved and moved > 2.0,
                   "pan %s -> %s degrees" % (before and before["pan"], after and after["pan"]))

        # 3. Auto from the studio
        studio.command("cameras", "set_mode", {"camera": CAMERA_ID, "mode": "auto",
                                               "profile": {"resume_after_ms": a.resume_after_ms}})
        auto = wait_for("auto", lambda: self.page()["runtime"]["director"]["mode"] == "AUTO"
                        and studio.camera().get("mode") == "auto", 5)
        page = self.page()
        self.check("Auto from the studio puts the zone in AUTO", auto,
                   "zone director %s, studio mode %s" % (page["runtime"]["director"]["mode"],
                                                         studio.camera().get("mode")))
        # let it lock a car if one comes (the recording has one every few seconds)
        wait_for("tracking", lambda: self.page()["runtime"]["director"]["sub"] == "tracking", 8)

        # 4. a studio move pauses auto; it resumes by itself
        studio.hold_move(0.5, pan=-0.3)
        paused = wait_for("paused", lambda: studio.camera().get("mode") == "auto_paused", 3)
        page = self.page()
        self.check("a studio move pauses auto", paused and page["runtime"]["director"]["mode"] == "PASSIVE",
                   "studio mode %s, zone director %s, zone page shows %s by %s" % (
                       studio.camera().get("mode"), page["runtime"]["director"]["mode"], page["agent"]["mode"],
                       page["agent"]["holder"]))
        t0 = time.monotonic()
        resumed = wait_for("resumed", lambda: studio.camera().get("mode") == "auto"
                           and self.page()["runtime"]["director"]["mode"] == "AUTO", a.resume_after_ms / 1000 + 5)
        self.check("auto resumes by itself after resume_after_ms", resumed,
                   "after %.1f s (resume_after_ms %d, after the last move's ttl)" % (time.monotonic() - t0,
                                                                                      a.resume_after_ms))

        # 5. the zone laptop dies: offline, then the studio drives the PTZ directly
        t_kill = time.monotonic()
        children.kill("zone")
        offline = wait_for("offline", lambda: studio.link().get("status") == "offline", 15)
        t_offline = time.monotonic() - t_kill
        fallback = wait_for("fallback", lambda: studio.camera().get("route") == "fallback", a.fallback_after + 10)
        t_fallback = time.monotonic() - t_kill
        self.check("zone app killed: the studio sees the agent offline, then falls back to VISCA",
                   offline and fallback, "offline after %.1f s, route fallback after %.1f s (fallback_after %s s)"
                   % (t_offline, t_fallback, a.fallback_after))
        wait_for("fallback position", lambda: studio.camera().get("position"), 5)
        before = studio.camera().get("position")
        studio.hold_move(1.0)
        time.sleep(1.0)
        after = studio.camera().get("position")
        moved = before and after and after["pan"] - before["pan"]
        self.check("in fallback, a studio move moves the PTZ over VISCA", moved and moved > 2.0,
                   "studio's own VISCA position pan %s -> %s degrees" % (before and before["pan"],
                                                                         after and after["pan"]))

        # 6. the zone laptop comes back: hand back
        t_start = time.monotonic()
        up = self.start_zone(children)
        back = wait_for("back", lambda: studio.camera().get("route") == "agent"
                        and studio.link().get("status") == "ok", 20)
        log = self.agent("GET", "/agent/log?limit=50")[1] if up else []
        resynced = any(e.get("command") == "resync" for e in log or [])
        self.check("zone app back: the studio hands the camera back (route agent, /resync)", back and resynced,
                   "route %s after %.1f s, resync in the agent log: %s" % (studio.camera().get("route"),
                                                                          time.monotonic() - t_start, resynced))
        cam = studio.camera()
        page = self.page()
        notices = [e for e in studio.link().get("events") or [] if e.get("event") == "mode_changed"
                   and (e.get("detail") or {}).get("reason") == "tracker_not_ready"]
        self.check("after the restart the camera is in manual (the zone in STOPPED), and the studio was told why",
                   cam.get("mode") == "manual" and page["runtime"]["director"]["mode"] == "STOPPED" and notices,
                   "studio mode %s, zone director %s, restore notice: %s" % (
                       cam.get("mode"), page["runtime"]["director"]["mode"],
                       notices[-1]["detail"] if notices else None))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="out/studio_e2e", help="the agent rig's folder (written by this script)")
    ap.add_argument("--studio-url", default="", help="a studio already running (scratch data folder!)")
    ap.add_argument("--studio-python", default="", help="or: the studio's python, to start one")
    ap.add_argument("--studio-server", default="", help="... the studio's server folder (cwd)")
    ap.add_argument("--studio-data", default="", help="... its scratch data folder")
    ap.add_argument("--studio-port", type=int, default=8899)
    ap.add_argument("--agent-port", type=int, default=8890)
    ap.add_argument("--page-port", type=int, default=8090)
    ap.add_argument("--visca-port", type=int, default=52399)
    ap.add_argument("--token", default="e2e-token")
    ap.add_argument("--fallback-after", type=float, default=3.0, help="the studio's direct_fallback_after_s")
    ap.add_argument("--resume-after-ms", type=int, default=1500)
    a = ap.parse_args(argv)
    if not a.studio_url and not (a.studio_python and a.studio_server and a.studio_data):
        ap.error("give --studio-url, or --studio-python, --studio-server and --studio-data")
    sys.path.insert(0, str(ROOT))
    return Run(a).main()


if __name__ == "__main__":
    sys.exit(main())
