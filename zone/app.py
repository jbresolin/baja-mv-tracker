"""The zone node: one process that reads the spotter, tracks cars along the taught path, drives the PTZ, records, and
serves a local web page (teach, operation, health). Started by zone\\run_zone.bat on the laptop:

    python -m zone.app --config zone/configs/my_site.toml
    python -m zone.app --set spotter.source="out/oct_wide.mp4" --set ptz.driver=sim     # teach on a recording

Page: http://<laptop>:8090/ (pt-BR). It always starts in STOPPED (also after a crash and restart): nothing moves
until someone presses PASSIVO or AUTO.

Exit codes (the launcher restarts on anything but 0): 0 = stopped on purpose (Ctrl-C), 3 = no spotter frames for
health.no_frame_exit_s, 1 = crash.
"""
import argparse
import datetime as dt
import os
import shutil
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from zone import config as C  # noqa: E402
from zone import director as D  # noqa: E402
from zone import draw  # noqa: E402
from zone import zonefile  # noqa: E402
from zone.progress import profile_path_for  # noqa: E402
from zone.runtime import ControlLog, ZoneRuntime  # noqa: E402
from zone.teach import TeachError, TeachSession, make_pose_source  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"


def make_ptz(cfg, settings, clock, log=print):
    p = cfg["ptz"]
    if p["driver"] == "sim":
        from zone.ptz.sim import SimulatedPtz
        return SimulatedPtz(settings.report.model, clock)
    from zone.ptz.visca import ViscaPtz
    log_path = C.resolve_path(p["command_log"]) if p.get("command_log") else None
    return ViscaPtz(p["host"], int(p["port"]), float(p["timeout_s"]), clock=clock, log_path=log_path)


def seed_sim_presets(ptz, zone):
    """A simulated PTZ starts with no presets; give it the zone's keyframe presets (a real camera keeps its own)."""
    if hasattr(ptz, "define_preset"):
        for k in zone.keyframes():
            ptz.define_preset(k.preset, k.pan, k.tilt, k.zoom)


class ZoneApp:
    """Everything the page talks to. Built by `main()`, or by tests with fakes (no threads started)."""

    def __init__(self, cfg, settings, ptz, clock, zone, zone_path, out_dir, source=None, spotter=None,
                 log=print, cycle_log=True):
        self.cfg, self.settings, self.ptz, self.clock = cfg, settings, ptz, clock
        self.zone_path = Path(zone_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.source, self.spotter = source, spotter
        self.print = log
        self.t_start_wall = clock.wall()
        stamp = dt.datetime.fromtimestamp(self.t_start_wall).strftime("%Y%m%d-%H%M%S")
        self.control_log = ControlLog(self.out_dir / ("zone_%s.jsonl" % stamp), clock,
                                      cfg["log"]["flush_every"]) if cycle_log else None
        self.events = []                 # last state changes, shown on the page
        self.runtime = ZoneRuntime(settings, zone, ptz, clock, spotter, profile_path=profile_path_for(zone_path),
                                   log=self.control_log, on_event=self._on_event)
        self.pose_source = make_pose_source(settings.pose_source, ptz, clock, settings)
        self.teach = TeachSession(zone.copy(), ptz, self.pose_source, cfg["teach"]["preset_base"],
                                  on_change=self.runtime.set_zone)
        self.selected = None
        self.latest = None               # (ZoneFrame, CycleResult)
        self.frame_lock = threading.Lock()
        self.last_frame_mono = clock.monotonic()
        self.recorder = None
        self.health = dict(disk_free_gb=None, ntp_offset_s=None, ntp_server=None, clock_warnings=[],
                           ptz_reachable=None, ptz_probe_wall=None)
        self.stop_ev = threading.Event()
        self.exit_code = 0
        self._log("zone app start: branch %s (%s), pose source %s, zone %s (%d points)" % (
            settings.controller.branch, settings.report.reason or settings.report.path, settings.pose_source,
            self.zone_path, len(zone.points)))

    # ------------------------------------------------------------------------------------------- logging
    def _log(self, msg, kind="info"):
        self.print(msg)
        if self.control_log:
            self.control_log.write(dict(type=kind, msg=msg))

    def _on_event(self, ev):
        e = dict(ev)
        e["wall_iso"] = dt.datetime.fromtimestamp(self.clock.wall()).strftime("%H:%M:%S")
        self.events.append(e)
        del self.events[:-30]

    # ------------------------------------------------------------------------------------------- frame loop
    def run_loop(self):
        """Reads frames until stopped. One control cycle per frame."""
        rec_cfg = self.cfg["recording"]
        try:
            for fr in self.source.frames():
                if self.stop_ev.is_set():
                    break
                self.last_frame_mono = self.clock.monotonic()
                res = self.runtime.cycle(fr)
                with self.frame_lock:
                    self.latest = (fr, res)
                if rec_cfg["spotter"]:
                    self._record(fr)
        except Exception as e:  # noqa: BLE001
            self._log("frame loop crashed: %s: %s" % (type(e).__name__, e), "error")
            self.exit_code = 1
        finally:
            if not self.stop_ev.is_set():
                if self.exit_code == 0:
                    self._log("spotter source ended", "warning")
                    self.exit_code = 3
                self.shutdown()

    def _record(self, fr):
        if self.recorder is None:
            from zone.recorder import SpotterRecorder
            rc = self.cfg["recording"]
            fps = self.cfg["rt"]["capture"]["fps"] if rc["spotter_fps"] == "capture" else rc["spotter_fps"]
            try:
                self.recorder = SpotterRecorder(self.out_dir, fps, (fr.img.shape[1], fr.img.shape[0]), fr.t_wall,
                                                rc["ffmpeg"], rc["spotter_codec"], rc["spotter_queue"],
                                                log=lambda m: self._log(m, "warning"))
                self._log("recording spotter to %s" % self.recorder.path)
            except Exception as e:  # noqa: BLE001 (no ffmpeg: keep tracking, show it on the page)
                self._log("spotter recording failed to start: %s" % e, "warning")
                self.cfg["recording"]["spotter"] = False
                return
        self.recorder.write(fr)

    def watchdog(self):
        """No frames for too long -> exit 3, so the launcher restarts the whole process (camera unplugged...)."""
        limit = float(self.cfg["health"]["no_frame_exit_s"])
        while not self.stop_ev.wait(1.0):
            if self.clock.monotonic() - self.last_frame_mono > limit:
                self._log("no spotter frame for %.0f s: exiting so the launcher restarts" % limit, "error")
                self.exit_code = 3
                self.shutdown()
                return

    def health_loop(self, period_s=5.0):
        from rt import timesync
        rtc = self.cfg.get("rt", {}).get("clock", {})
        server = rtc.get("ntp_server", "")
        self.health["ntp_server"] = server or None
        try:
            self.health["clock_warnings"] = timesync.check(dict(clock=dict(warn_sync_age_h=24.0, ntp_server="")),
                                                           lambda m: None)
        except Exception as e:  # noqa: BLE001
            self.health["clock_warnings"] = ["time sync check failed: %s" % e]
        k = 0
        while not self.stop_ev.is_set():
            self.check_health(ntp=(server and k % 12 == 0), server=server)
            k += 1
            self.stop_ev.wait(period_s)

    def check_health(self, ntp=False, server=""):
        try:
            self.health["disk_free_gb"] = round(shutil.disk_usage(self.out_dir).free / 2 ** 30, 1)
        except OSError:
            self.health["disk_free_gb"] = None
        if ntp:
            from rt.timesync import sntp_offset
            off = sntp_offset(server)
            self.health["ntp_offset_s"] = None if off is None else round(off, 3)
        # PTZ reachable: a harmless zoom inquiry (cameras without zoom inquiry fall back to the last command result)
        try:
            z = self.ptz.inquire_zoom()
        except Exception:  # noqa: BLE001
            z = None
        self.health["ptz_reachable"] = (z is not None) or self.runtime.controller.ptz_ok
        self.health["ptz_probe_wall"] = round(self.clock.wall(), 1)

    def ptz_recording_state(self):
        files = sorted(self.out_dir.glob("ptz_*.mkv"), key=lambda p: p.stat().st_mtime)
        if not files:
            return dict(enabled=bool(self.cfg["recording"]["ptz_usb"]), file=None, growing=False)
        f = files[-1]
        st = f.stat()
        return dict(enabled=bool(self.cfg["recording"]["ptz_usb"]), file=f.name,
                    size_mb=round(st.st_size / 2 ** 20, 1), growing=time.time() - st.st_mtime < 10)

    def shutdown(self):
        if self.stop_ev.is_set():
            return
        self.stop_ev.set()
        try:
            self.runtime.big_stop()
        except Exception:  # noqa: BLE001
            pass
        if self.source:
            self.source.stop()
        if self.recorder:
            self.recorder.close()
        if self.control_log:
            self.control_log.close()

    # ------------------------------------------------------------------------------------------- page data
    def state(self):
        snap = self.runtime.snapshot()
        z = self.teach.zone
        path = self.runtime.path
        with self.frame_lock:
            fr = self.latest[0] if self.latest else None
        return dict(
            runtime=snap, events=self.events[-15:],
            zone=dict(points=[dict(i=i, **p.to_dict()) for i, p in enumerate(z.points)],
                      corridor=z.corridor_half_width_px, entry_preset=z.entry_preset, dirty=self.teach.dirty,
                      file=str(self.zone_path), keyframe_s=path.keyframe_s() if path else None),
            teach=dict(pose_source=self.pose_source.name, approximate=self.pose_source.approximate,
                       velocity_jog=self.pose_source.velocity_jog, jog_speeds=self.cfg["teach"]["jog_speeds"],
                       nudge_units=self.cfg["teach"]["nudge_units"],
                       zoom_nudge_units=self.cfg["teach"]["zoom_nudge_units"],
                       jog_zoom_speed=self.cfg["teach"]["jog_zoom_speed"], selected=self.selected,
                       allowed=self.runtime.director.mode != D.AUTO),
            frame=dict(size=list(fr.img.shape[1::-1]) if fr is not None and fr.img is not None else None,
                       age_s=round(self.clock.monotonic() - self.last_frame_mono, 1)),
            health=dict(self.health, fps=snap["fps"], dropped=self.source.dropped if self.source else 0,
                        spotter_recording=self.recorder.status() if self.recorder else None,
                        ptz_recording=self.ptz_recording_state(), ptz_driver=self.cfg["ptz"]["driver"],
                        branch=self.settings.controller.branch, report=self.settings.report.summary(),
                        source=getattr(self.source, "description", "")))

    def frame_jpeg(self):
        with self.frame_lock:
            if not self.latest or self.latest[0].img is None:
                return None
            fr, res = self.latest
            img = fr.img.copy()
        rt = self.runtime
        roi = rt.spotter.det.roi if rt.spotter is not None else None
        draw.draw_zone(img, self.teach.zone, rt.path, roi, self.selected)
        st = rt.director.state()
        draw.draw_tracks(img, res.views, st["locked_id"], res.tracker_out)
        lines = ["%s%s" % (st["mode"], (" / " + st["sub"]) if st["sub"] else "")]
        if st["locked_id"] is not None and st["locked_id"] in res.views:
            v = res.views[st["locked_id"]]
            lines.append("carro %d  s=%.2f  v=%.2f/s" % (v.track_id, v.s, v.v))
        draw.draw_status(img, lines)
        mw = int(self.cfg["web"]["frame_max_width"])
        if mw and img.shape[1] > mw:
            img = cv2.resize(img, (mw, int(img.shape[0] * mw / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg["web"]["jpeg_quality"])])
        return buf.tobytes() if ok else None

    # ------------------------------------------------------------------------------------------- page actions
    def require_teach(self):
        if self.runtime.director.mode == D.AUTO:
            raise TeachError("saia do AUTO antes de ensinar (use PARADO ou PASSIVO)")

    def save_zone(self):
        size = self.runtime.frame_size or self.teach.zone.image_size
        return self.teach.save(self.zone_path, size, report=self.settings.report.summary(),
                               cameras=dict(spotter=str(self.cfg["spotter"]["source"]),
                                            ptz="%s %s" % (self.cfg["ptz"]["driver"], self.cfg["ptz"]["host"])))


# =========================================================================================================== web
def create_web(zapp: ZoneApp):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse, Response

    web = FastAPI(title="zone")

    def guard(fn, teach=False):
        try:
            if teach:
                zapp.require_teach()
            out = fn()
            return dict(ok=True, result=out)
        except TeachError as e:
            raise HTTPException(409, str(e))
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, str(e))

    @web.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @web.get("/app.js")
    def app_js():
        return FileResponse(STATIC / "app.js", media_type="text/javascript", headers={"Cache-Control": "no-store"})

    @web.get("/api/state")
    def state():
        return JSONResponse(zapp.state())

    @web.get("/frame.jpg")
    def frame():
        jpg = zapp.frame_jpeg()
        if jpg is None:
            return Response(status_code=204)
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # ---- operation
    @web.post("/api/mode")
    def mode(body: dict = Body(...)):
        ok, msg = zapp.runtime.set_mode(body["mode"])
        if not ok:
            raise HTTPException(409, msg)
        return dict(ok=True, mode=msg)

    @web.post("/api/stop")
    def big_stop():
        zapp.runtime.big_stop()
        try:
            zapp.pose_source.jog_stop()
        except Exception:  # noqa: BLE001
            pass
        return dict(ok=True)

    # ---- teach: PTZ
    @web.post("/api/teach/jog")
    def jog(body: dict = Body(...)):
        return guard(lambda: zapp.pose_source.jog(int(body.get("pan", 0)), int(body.get("tilt", 0)),
                                                  int(body.get("zoom", 0))), teach=True)

    @web.post("/api/teach/jog_stop")
    def jog_stop():
        return guard(lambda: zapp.pose_source.jog_stop(), teach=True)

    @web.post("/api/teach/nudge")
    def nudge(body: dict = Body(...)):
        return guard(lambda: zapp.pose_source.nudge(int(body.get("pan", 0)), int(body.get("tilt", 0)),
                                                    int(body.get("zoom", 0)), int(body.get("speed", 8))), teach=True)

    @web.post("/api/teach/home")
    def home():
        return guard(lambda: zapp.pose_source.home(), teach=True)

    @web.post("/api/teach/zero_here")
    def zero_here():
        return guard(lambda: zapp.pose_source.zero_here(), teach=True)

    @web.get("/api/teach/pose")
    def pose():
        return guard(lambda: zapp.pose_source.read())

    # ---- teach: points
    @web.post("/api/teach/point")
    def add_point(body: dict = Body(...)):
        def go():
            pt = zapp.teach.add_point(float(body["x"]), float(body["y"]), bool(body.get("keyframe")),
                                      body.get("insert_at"))
            return pt.to_dict()
        return guard(go, teach=True)

    @web.post("/api/teach/move")
    def move(body: dict = Body(...)):
        return guard(lambda: zapp.teach.move_point(int(body["index"]), float(body["x"]), float(body["y"])),
                     teach=True)

    @web.post("/api/teach/delete")
    def delete(body: dict = Body(...)):
        return guard(lambda: zapp.teach.delete_point(int(body["index"])), teach=True)

    @web.post("/api/teach/reorder")
    def reorder(body: dict = Body(...)):
        return guard(lambda: zapp.teach.reorder(int(body["index"]), int(body["delta"])), teach=True)

    @web.post("/api/teach/make_keyframe")
    def make_keyframe(body: dict = Body(...)):
        return guard(lambda: zapp.teach.make_keyframe(int(body["index"])), teach=True)

    @web.post("/api/teach/unkeyframe")
    def unkeyframe(body: dict = Body(...)):
        return guard(lambda: zapp.teach.toggle_keyframe(int(body["index"])), teach=True)

    @web.post("/api/teach/reteach")
    def reteach(body: dict = Body(...)):
        return guard(lambda: zapp.teach.reteach(int(body["index"])), teach=True)

    @web.post("/api/teach/goto")
    def goto(body: dict = Body(...)):
        return guard(lambda: zapp.teach.goto(int(body["index"])), teach=True)

    @web.post("/api/teach/select")
    def select(body: dict = Body(...)):
        i = body.get("index")
        zapp.selected = None if i is None else int(i)
        return dict(ok=True)

    @web.post("/api/teach/corridor")
    def corridor(body: dict = Body(...)):
        return guard(lambda: zapp.teach.set_corridor(float(body["half_width"])), teach=True)

    @web.post("/api/teach/save")
    def save():
        return guard(lambda: str(zapp.save_zone() or ""))

    return web


# =========================================================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="user TOML (only the keys it changes)")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    a = ap.parse_args(argv)
    cfg = C.load(a.config, a.set)
    settings = C.build(cfg)
    from zone.clock import SystemClock
    from zone.source import SpotterSource
    from zone.spotter import make_spotter, rt_config
    clock = SystemClock()
    rtcfg = rt_config(cfg)
    zone_path = C.resolve_path(cfg["zone"]["file"])
    if zone_path.exists():
        zone = zonefile.load(zone_path)
    else:
        zone = zonefile.Zone(corridor_half_width_px=float(cfg["teach"]["default_corridor_px"]))
    ptz = make_ptz(cfg, settings, clock)
    seed_sim_presets(ptz, zone)
    spotter = make_spotter(cfg, rtcfg)
    source = SpotterSource(cfg, rtcfg)
    zapp = ZoneApp(cfg, settings, ptz, clock, zone, zone_path, C.resolve_path(cfg["zone"]["out_dir"]), source,
                   spotter, cycle_log=bool(cfg["log"]["cycle_log"]))
    for target in (zapp.run_loop, zapp.watchdog, zapp.health_loop):
        threading.Thread(target=target, daemon=True).start()

    import uvicorn
    web = create_web(zapp)
    server = uvicorn.Server(uvicorn.Config(web, host=cfg["web"]["host"], port=int(cfg["web"]["port"]),
                                           log_level="warning"))
    threading.Thread(target=lambda: (zapp.stop_ev.wait(), setattr(server, "should_exit", True)),
                     daemon=True).start()
    print("zone page: http://localhost:%d/" % int(cfg["web"]["port"]), flush=True)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    zapp.shutdown()
    try:
        ptz.close()
    except Exception:  # noqa: BLE001
        pass
    return zapp.exit_code


if __name__ == "__main__":
    sys.exit(main())
