"""The zone node: one process that reads the spotter, tracks cars along the taught path, drives the PTZ, records, and
serves a local web page (teach, operation, health). Started by zone\\run_zone.bat on the laptop:

    python -m zone.app --config zone/configs/my_site.toml
    python -m zone.app --set spotter.source="out/oct_wide.mp4" --set ptz.driver=sim     # teach on a recording

Page: http://<laptop>:8090/ (pt-BR). It always starts in STOPPED (also after a crash and restart): nothing moves
until someone presses PASSIVO or AUTO.

Exit codes (the launcher restarts on anything but 0): 0 = stopped on purpose (Ctrl-C), 3 = no spotter frames for
health.no_frame_exit_s, 1 = crash. When the watchdog fires and the clean shutdown doesn't finish within
health.shutdown_grace_s (e.g. the frame loop is stuck in a hung detector or PTZ call), the process is killed with
os._exit(3), so the launcher always gets to restart it.
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
from zone import direction  # noqa: E402
from zone import director as D  # noqa: E402
from zone import draw  # noqa: E402
from zone import zonefile  # noqa: E402
from zone.controller import heard_recently  # noqa: E402
from zone.progress import profile_path_for  # noqa: E402
from zone.runtime import ControlLog, ZoneRuntime  # noqa: E402
from zone.teach import TeachError, TeachSession, make_pose_source  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"


def make_ptz(cfg, settings, clock, log=print):
    p = cfg["ptz"]
    if p["driver"] == "sim":
        from zone.ptz.locked import LockedPtz
        from zone.ptz.sim import SimulatedPtz
        # the simulator isn't thread-safe, and the control loop, the health check, the page and the agent API all
        # call it: one call at a time (the real driver has its own lock)
        return LockedPtz(SimulatedPtz(settings.report.model, clock))
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
                 log=print, cycle_log=True, hard_exit=os._exit, wait_event=None):
        self.cfg, self.settings, self.ptz, self.clock = cfg, settings, ptz, clock
        # how the watchdog kills a stuck process, and how it waits for the shutdown: injected by the tests (a fake
        # exit that only records the code, and an instant wait), real os._exit / Event.wait otherwise
        self.hard_exit = hard_exit
        self.wait_event = wait_event or (lambda ev, timeout: ev.wait(timeout))
        self.exited_ev = threading.Event()   # set by main() just before the process returns
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
        self.event_listeners = []        # callables that also get every event (the agent API: car locked / let go)
        self.agent = None                # the studio's agent API (zone/agent), when [agent] enabled: attached later
        self.agent_error = None          # why the agent API is down (port taken, websockets missing...): on the page
        self.runtime = ZoneRuntime(settings, zone, ptz, clock, spotter, profile_path=profile_path_for(zone_path),
                                   log=self.control_log, on_event=self._on_event)
        self.pose_source = make_pose_source(settings.pose_source, ptz, clock, settings)
        self.teach = TeachSession(zone.copy(), ptz, self.pose_source, cfg["teach"]["preset_base"],
                                  on_change=self.runtime.set_zone)
        self.selected = None
        # "Modo ensino": the page may move the PTZ (jog, nudge, home, test a keyframe) only while this is on. Off at
        # start, turned off by the big STOP, by PARADO and by entering AUTO, so a STOP also cancels a held jog (the
        # page repeats the jog every 150 ms while a button is held: those repeats are refused from then on).
        self.teach_mode = False
        self.latest = None               # (ZoneFrame, CycleResult)
        self.frame_lock = threading.Lock()
        self.last_frame_mono = clock.monotonic()
        self.recorder = None
        # ptz_reachable: None = unknown (nothing has answered yet), True / False; ptz_reachable_how: "probe" (an
        # inquiry made by the health check), "probe_reply" (the probe got an ACK / completion but no data) or
        # "last_command" (the result of the controller's last command)
        self.health = dict(disk_free_gb=None, ntp_offset_s=None, ntp_server=None, clock_warnings=[],
                           ptz_reachable=None, ptz_reachable_how=None, ptz_probe_wall=None)
        self._probe_answered_once = False    # has this camera ever answered the health probe?
        self.stop_ev = threading.Event()
        self.exit_code = 0
        self._log("zone app start: branch %s (%s), pose source %s, pan_sign %+d (%s), tilt_sign %+d (%s), zone %s "
                  "(%d points)" % (
                      settings.controller.branch, settings.report.reason or settings.report.path, settings.pose_source,
                      settings.controller.pan_sign, (settings.sign_how or {}).get("pan"),
                      settings.controller.tilt_sign, (settings.sign_how or {}).get("tilt"),
                      self.zone_path, len(zone.points)))
        for w in self.runtime.sign_warnings:
            self._log("WARNING: " + w, "warning")

    # ------------------------------------------------------------------------------------------- logging
    def _log(self, msg, kind="info"):
        self.print(msg)
        if self.control_log:
            try:
                self.control_log.write(dict(type=kind, msg=msg))
            except Exception:  # noqa: BLE001 (logging must never take the process down)
                pass

    def _on_event(self, ev):
        e = dict(ev)
        e["wall_iso"] = dt.datetime.fromtimestamp(self.clock.wall()).strftime("%H:%M:%S")
        self.events.append(e)
        del self.events[:-30]
        for listener in list(self.event_listeners):
            try:
                listener(ev)
            except Exception as err:  # noqa: BLE001 (a listener must never break the control loop)
                self.print("event listener failed: %s: %s" % (type(err).__name__, err))

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
        """No frames for too long -> exit 3, so the launcher restarts the whole process (camera unplugged, hung
        detector or PTZ call...)."""
        while not self.stop_ev.wait(1.0):
            if self.check_frames():
                return

    def check_frames(self):
        """One watchdog check. -> True when it fired (and the process is on its way out)."""
        limit = float(self.cfg["health"]["no_frame_exit_s"])
        if self.clock.monotonic() - self.last_frame_mono <= limit:
            return False
        self._log("no spotter frame for %.0f s: exiting so the launcher restarts" % limit, "error")
        self.exit_code = 3
        self.shutdown_or_exit(3)
        return True

    def shutdown_or_exit(self, code):
        """Clean shutdown, but never a hang: shutdown() runs in a helper thread, and if it hasn't finished within
        health.shutdown_grace_s the process is killed (os._exit) with `code`. A frame loop stuck inside a cycle can
        block the shutdown (it holds locks, the PTZ call never returns...); without this the process would sit there
        forever and the launcher would never restart it. After a clean shutdown, main() must also return within the
        same grace, or the process is killed anyway."""
        grace = float(self.cfg["health"].get("shutdown_grace_s", 5.0))
        done = threading.Event()

        def run():
            try:
                self.shutdown()
            finally:
                done.set()
        threading.Thread(target=run, daemon=True, name="shutdown").start()
        if not self.wait_event(done, grace):
            self._log("shutdown did not finish within %.0f s: forcing exit %d" % (grace, code), "error")
            self.hard_exit(code)
            return
        if not self.wait_event(self.exited_ev, grace):
            self.print("process still running %.0f s after shutdown: forcing exit %d" % (grace, code))
            self.hard_exit(code)

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
        self.health["ptz_reachable"], self.health["ptz_reachable_how"] = self.ptz_reachability()
        self.health["ptz_probe_wall"] = round(self.clock.wall(), 1)

    def ptz_reachability(self):
        """-> (reachable, how). reachable: True / False / None (unknown).

        While AUTO is driving the camera (tracking a car or returning to the entry), no probe is sent: an inquiry
        from this thread would compete with the control loop for the driver (one request on the wire at a time) and
        could delay a velocity refresh up to the TTL. The result of the controller's last command is used instead.

        Otherwise a harmless inquiry is sent (zoom, then pan/tilt if the camera has no zoom inquiry). If neither is
        answered: when this camera has answered the probe before, it is unreachable now; when it never has (it may
        simply not support inquiries), fall back to the last command's result, which is None until something was
        sent: "unknown", never a false "reachable".

        A camera that replied *something* during the probe (an ACK, a completion) without the data is reachable
        ("probe_reply"): the PTZ probed on 2026-11-15 sends the data for only ~17 % of its inquiries, and a missing
        answer must not make it look unplugged (the studio would see it offline)."""
        ctrl = self.runtime.controller
        d = self.runtime.director
        last_command = ctrl.ptz_ok
        if last_command is False and heard_recently(self.ptz, self.clock.monotonic()):
            last_command = True  # it timed out, but the camera is talking (a late reply): see heard_recently
        if d.mode == D.AUTO and d.sub in (D.TRACKING, D.RETURNING):
            return last_command, "last_command"
        answer = None
        for inquiry in (self.ptz.inquire_zoom, self.ptz.inquire_pan_tilt):
            try:
                answer = inquiry()
            except Exception:  # noqa: BLE001
                answer = None
            if answer is not None:
                break
        if answer is not None:
            self._probe_answered_once = True
            return True, "probe"
        if heard_recently(self.ptz, self.clock.monotonic()):
            return True, "probe_reply"
        if self._probe_answered_once:
            return False, "probe"
        return last_command, "last_command"

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
            agent=(self.agent.page_state() if self.agent is not None
                   else dict(enabled=False, error=self.agent_error)),
            zone=dict(points=[dict(i=i, **p.to_dict()) for i, p in enumerate(z.points)],
                      corridor=z.corridor_half_width_px, entry_preset=z.entry_preset, dirty=self.teach.dirty,
                      file=str(self.zone_path), keyframe_s=path.keyframe_s() if path else None),
            teach=dict(pose_source=self.pose_source.name, approximate=self.pose_source.approximate,
                       velocity_jog=self.pose_source.velocity_jog, jog_speeds=self.cfg["teach"]["jog_speeds"],
                       nudge_units=self.cfg["teach"]["nudge_units"],
                       zoom_nudge_units=self.cfg["teach"]["zoom_nudge_units"],
                       jog_zoom_speed=self.cfg["teach"]["jog_zoom_speed"], selected=self.selected,
                       allowed=self.runtime.director.mode != D.AUTO, teach_mode=self.teach_mode,
                       zoom_unknown=getattr(self.pose_source, "zoom_readable", True) is False,
                       zoom_limits=list(self.settings.report.model.zoom_limits)),
            frame=dict(size=list(fr.img.shape[1::-1]) if fr is not None and fr.img is not None else None,
                       age_s=round(self.clock.monotonic() - self.last_frame_mono, 1)),
            health=dict(self.health, fps=snap["fps"], dropped=self.source.dropped if self.source else 0,
                        spotter_recording=self.recorder.status() if self.recorder else None,
                        ptz_recording=self.ptz_recording_state(), ptz_driver=self.cfg["ptz"]["driver"],
                        branch=self.settings.controller.branch, report=self.settings.report.summary(),
                        # pan/tilt sign and where it came from (zone/direction.py), + warnings ("não medido"...)
                        ptz_signs=direction.summary(self.settings) if self.settings.sign_how else None,
                        sign_warnings=list(self.runtime.sign_warnings),
                        source=getattr(self.source, "description", ""),
                        # which car detector runs; bench=True (COCO) makes the page show a "bench detector" warning
                        detector=getattr(self.spotter, "detector_info", None)))

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
    def require_teach(self, moves_ptz=False):
        """Refuse teach actions in AUTO; refuse anything that moves the PTZ unless teach mode is on."""
        if self.runtime.director.mode == D.AUTO:
            raise TeachError("saia do AUTO antes de ensinar (use PARADO ou PASSIVO)")
        if moves_ptz and not self.teach_mode:
            raise TeachError("ligue o 'Modo ensino' para mover a PTZ")

    def set_teach_mode(self, on):
        if on and self.runtime.director.mode == D.AUTO:
            raise TeachError("saia do AUTO antes de ligar o modo ensino")
        was = self.teach_mode
        self.teach_mode = bool(on)
        if was and not self.teach_mode:
            self._cancel_jog()
        self._on_event(dict(type="teach_mode", new="ligado" if self.teach_mode else "desligado"))
        return self.teach_mode

    def _cancel_jog(self):
        try:
            self.pose_source.jog_stop()
        except Exception:  # noqa: BLE001 (a dead PTZ must not break the STOP)
            pass

    def set_mode(self, mode):
        """Mode buttons. Entering AUTO or PARADO ends teach mode (and any held jog). With the agent API on, the
        buttons also set the studio's view of the camera (PARADO / PASSIVO = manual, AUTO = auto; AUTO is refused
        while the studio holds the camera): see zone/agent/modes.py."""
        if self.agent is not None:
            return self.agent.local_mode(mode)   # -> agent's link -> runtime.set_mode + teach_after_mode
        ok, msg = self.runtime.set_mode(mode)
        if ok:
            self.teach_after_mode(msg)
        return ok, msg

    def teach_after_mode(self, mode):
        """After a mode change: AUTO and STOPPED end teach mode; STOPPED also cancels a held jog."""
        mode = str(mode).upper()
        if mode in (D.AUTO, D.STOPPED) and self.teach_mode:
            self.teach_mode = False
            if mode == D.STOPPED:
                self._cancel_jog()

    def big_stop(self):
        """The big red button: teach mode off first (so held-jog repeats are refused from now on), the runtime's
        immediate stop, then the teach jog's own stop (zoom included). For the studio it is also a stop: its
        velocity is forgotten and auto (or a pause) ends."""
        self.teach_mode = False
        self.runtime.big_stop()
        self._cancel_jog()
        if self.agent is not None:
            self.agent.local_stop()

    def save_zone(self):
        size = self.runtime.frame_size or self.teach.zone.image_size
        return self.teach.save(self.zone_path, size, report=self.settings.report.summary(),
                               cameras=dict(spotter=str(self.cfg["spotter"]["source"]),
                                            ptz="%s %s" % (self.cfg["ptz"]["driver"], self.cfg["ptz"]["host"])))


# =========================================================================================================== web

def plain_json(value):
    """Copy `value` with every numpy number, bool or array turned into the plain Python equivalent.
    The page's state is built from numbers numpy computed; the JSON encoder refuses numpy types (2026-11-15 bench:
    'Object of type bool is not JSON serializable' made every /api/state request fail and froze the page)."""
    if isinstance(value, dict):
        return {key: plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_json(item) for item in value]
    if hasattr(value, "tolist") and type(value).__module__ == "numpy":
        return plain_json(value.tolist())       # numpy scalar -> int/float/bool; numpy array -> nested lists
    return value

def create_web(zapp: ZoneApp):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse, Response

    web = FastAPI(title="zone")

    def guard(fn, teach=False, moves_ptz=False):
        try:
            if teach:
                zapp.require_teach(moves_ptz)
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
        return JSONResponse(plain_json(zapp.state()))

    @web.get("/frame.jpg")
    def frame():
        jpg = zapp.frame_jpeg()
        if jpg is None:
            return Response(status_code=204)
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # ---- operation
    @web.post("/api/mode")
    def mode(body: dict = Body(...)):
        ok, msg = zapp.set_mode(body["mode"])
        if not ok:
            raise HTTPException(409, msg)
        return dict(ok=True, mode=msg)

    @web.post("/api/stop")
    def big_stop():
        zapp.big_stop()
        return dict(ok=True)

    @web.post("/api/agent/release_hold")
    def release_hold(body: dict = Body(...)):
        """"Liberar HOLD": the person at the laptop takes the camera back from a studio that holds it. The page asks
        for a confirmation first, and sends {"confirm": true}."""
        if body.get("confirm") is not True:
            raise HTTPException(400, "confirme para liberar o HOLD do estúdio")
        if zapp.agent is None:
            raise HTTPException(409, "a API do estúdio não está ligada")
        ok, msg = zapp.agent.local_release_hold()
        if not ok:
            raise HTTPException(409, msg)
        return dict(ok=True, mode=msg)

    # ---- teach mode (needed to move the PTZ from the page)
    @web.post("/api/teach/mode")
    def teach_mode(body: dict = Body(...)):
        return guard(lambda: zapp.set_teach_mode(bool(body.get("on"))))

    # ---- teach: PTZ
    @web.post("/api/teach/jog")
    def jog(body: dict = Body(...)):
        return guard(lambda: zapp.pose_source.jog(int(body.get("pan", 0)), int(body.get("tilt", 0)),
                                                  int(body.get("zoom", 0))), teach=True, moves_ptz=True)

    @web.post("/api/teach/jog_stop")
    def jog_stop():
        return guard(lambda: zapp.pose_source.jog_stop(), teach=True)

    @web.post("/api/teach/nudge")
    def nudge(body: dict = Body(...)):
        return guard(lambda: zapp.pose_source.nudge(int(body.get("pan", 0)), int(body.get("tilt", 0)),
                                                    int(body.get("zoom", 0)), int(body.get("speed", 8))), teach=True,
                     moves_ptz=True)

    @web.post("/api/teach/home")
    def home():
        return guard(lambda: zapp.pose_source.home(), teach=True, moves_ptz=True)

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
                                      body.get("insert_at"), zoom=body.get("zoom"))
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
        return guard(lambda: zapp.teach.make_keyframe(int(body["index"]), body.get("zoom")), teach=True)

    @web.post("/api/teach/unkeyframe")
    def unkeyframe(body: dict = Body(...)):
        return guard(lambda: zapp.teach.toggle_keyframe(int(body["index"])), teach=True)

    @web.post("/api/teach/reteach")
    def reteach(body: dict = Body(...)):
        return guard(lambda: zapp.teach.reteach(int(body["index"]), body.get("zoom")), teach=True)

    @web.post("/api/teach/keyframe_zoom")
    def keyframe_zoom(body: dict = Body(...)):
        return guard(lambda: zapp.teach.set_keyframe_zoom(int(body["index"]), body["zoom"]), teach=True)

    @web.post("/api/teach/goto")
    def goto(body: dict = Body(...)):
        return guard(lambda: zapp.teach.goto(int(body["index"])), teach=True, moves_ptz=True)

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


def stop_at_startup(ptz, log=print):
    """Stop the PTZ when the process starts. A VISCA camera has no dead-man of its own: the TTL lives in this process
    (ViscaPtz's watchdog thread), so a crash or a kill mid-move leaves the camera moving. The zone restarts in
    STOPPED, which sends nothing; this stop is what ends that motion. Never fails the start (a camera that doesn't
    answer yet is the health check's business)."""
    for stop in (ptz.stop, ptz.zoom_stop):
        try:
            stop()
        except Exception as e:  # noqa: BLE001
            log("stop at startup failed: %s: %s" % (type(e).__name__, e))


# =========================================================================================================== agent
def agent_server_problem(find_spec=None):
    """What would keep the agent API from serving the studio, checked at startup: uvicorn (the server) and a
    WebSocket library for it (`websockets`, or `wsproto`). Without one, uvicorn answers /agent/ws with a plain HTTP
    error and the studio never links. -> None, or the problem as a sentence for the page and the console."""
    import importlib.util
    find_spec = find_spec or importlib.util.find_spec
    if find_spec("uvicorn") is None:
        return "uvicorn não instalado (pip install uvicorn websockets)"
    if find_spec("websockets") is None and find_spec("wsproto") is None:
        return "websockets não instalado: o estúdio não consegue se conectar (pip install websockets)"
    return None


def agent_down(zapp, why):
    """The agent API is down: shown on the page ("API DO ESTÚDIO FORA DO AR") and shouted on the console."""
    zapp.agent_error = why
    zapp._log("!" * 100 + "\n!!! API DO ESTÚDIO FORA DO AR (agent API down): %s\n" % why + "!" * 100, "error")


def start_agent(zapp, cfg):
    """The studio's agent API (zone/agent) on its own port, in this process: same PTZ driver, same director. Runs in
    background threads (its server, and its ticker). -> the uvicorn server (to stop it), or None if it can't start;
    the zone itself keeps running either way, and the page says the studio's API is down."""
    problem = agent_server_problem()
    if problem and problem.startswith("uvicorn"):
        agent_down(zapp, problem)
        return None
    import uvicorn
    from zone.agent.server import create_agent_app
    from zone.agent.service import ZoneAgent
    a = cfg["agent"]
    try:
        agent = ZoneAgent(zapp, a, log=zapp.print).attach()
    except Exception as e:  # noqa: BLE001 (the zone must run without the studio)
        agent_down(zapp, "%s: %s" % (type(e).__name__, e))
        return None
    if problem:
        agent_down(zapp, problem)       # HTTP works, the studio's WebSocket won't: still down for the studio
    threading.Thread(target=agent.run_ticker, args=(zapp.stop_ev,), daemon=True, name="agent-tick").start()
    server = uvicorn.Server(uvicorn.Config(create_agent_app(agent), host=a["host"], port=int(a["port"]),
                                           log_level="warning", ws_ping_interval=None))

    def serve():
        try:
            server.run()
        except BaseException as e:  # noqa: BLE001 (port taken: uvicorn exits its thread with SystemExit)
            agent_down(zapp, "porta %s: %s: %s (outro programa usa a porta?)" % (a["port"], type(e).__name__, e))
            return
        if not zapp.stop_ev.is_set():
            agent_down(zapp, "o servidor parou (porta %s ocupada?)" % a["port"])
    threading.Thread(target=serve, daemon=True, name="agent-api").start()
    zapp._log("agent API (studio): http://<this laptop>:%d/agent/info, agent_id %s, camera %s, token %s" % (
        int(a["port"]), agent.agent_id, agent.camera_id, "set" if agent.token else "none"))
    return server


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
    stop_at_startup(ptz, print)
    seed_sim_presets(ptz, zone)
    spotter = make_spotter(cfg, rtcfg)
    source = SpotterSource(cfg, rtcfg)
    zapp = ZoneApp(cfg, settings, ptz, clock, zone, zone_path, C.resolve_path(cfg["zone"]["out_dir"]), source,
                   spotter, cycle_log=bool(cfg["log"]["cycle_log"]))
    agent_server = start_agent(zapp, cfg) if cfg["agent"]["enabled"] else None
    for target in (zapp.run_loop, zapp.watchdog, zapp.health_loop):
        threading.Thread(target=target, daemon=True).start()

    import uvicorn
    web = create_web(zapp)
    server = uvicorn.Server(uvicorn.Config(web, host=cfg["web"]["host"], port=int(cfg["web"]["port"]),
                                           log_level="warning"))
    def stop_servers():
        zapp.stop_ev.wait()
        server.should_exit = True
        if agent_server is not None:
            agent_server.should_exit = True
    threading.Thread(target=stop_servers, daemon=True).start()
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
    zapp.exited_ev.set()           # the watchdog's "forced exit" no longer needs to fire
    return zapp.exit_code


if __name__ == "__main__":
    sys.exit(main())
