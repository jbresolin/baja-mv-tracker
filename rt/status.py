"""Read-only status page of a running rt (WP-L1): a tiny HTTP server inside the rt process (Python stdlib only, no
frontend build), meant to be opened from a phone on the LAN.

  http://<laptop>:8765/             the page (refreshes itself every 2 s)
  http://<laptop>:8765/status.json  the same data as JSON (the launcher rt/supervise.py reads `processing.fps`)
  http://<laptop>:8765/thumb/<pass_id>.jpg  a pass's thumbnail

What it shows: processing fps and dropped frames; camera state and the image settings read back; the clock offset
(red above 0.5 s); recording state and free disk; the beam input when it runs (port, line level, edges/min, last
10 breaks, poll rate, stuck warning); the last 20 passes with a thumbnail, the number read, confidence and t_cross.
The page itself has no controls. The one exception is the Beam setup panel (/beam, rt/beam/setup_page.py): its
few POST endpoints (BEAM_POST_ROUTES) let the owner find and set up the beam sensor on site from a phone; they
only ever touch the beam (rt/beam/control.py). Every other path answers GET only. POSTs must be JSON
(Content-Type application/json: a plain cross-site form cannot send that), at most 8 KB, and, when
[status] setup_token is set, carry it in the X-Setup-Token header (default: no token, LAN only).

Pieces: `StatusBoard` holds the state (the detection loop calls on_frame(), the event sink feeds passes);
`StatusSink` is the rt/sinks.py sink that feeds it; `TimeChecker` re-measures the clock offset once a minute in the
background; `StatusServer` serves the page.
"""
import collections
import datetime as dt
import hmac
import html
import json
import socket
import sys
import threading
from bisect import bisect_left
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from .backoff import SystemClock
from .sinks import Sink

OFFSET_RED_S = 0.5


def iso(t):
    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds") if t is not None else None


class StatusBoard:
    def __init__(self, cfg, clock=None):
        s = cfg["status"]
        self.clock = clock or SystemClock()
        self.target_fps = float(cfg["capture"]["fps"])
        self.window_s = float(s.get("fps_window_s", 10))
        self.n_passes = int(s.get("passes", 20))
        self.thumb_w = int(s.get("thumb_width", 320))
        self.thumb_every_s = 1.0 / float(s.get("thumb_fps", 5))
        self.ring_s = float(s.get("thumb_ring_s", 20))
        self.lock = threading.RLock()                 # re-entrant: on_event holds it while it calls helpers
        self.frame_monos = collections.deque()        # monotonic times of processed frames, last window_s
        self.frames = 0
        self.last_frame_t = None
        self.ring = collections.deque()               # (frame wall time, JPEG bytes) of the last ring_s seconds
        self.last_thumb_mono = None
        self.passes = collections.OrderedDict()       # pass_id -> dict, the last n_passes
        self.thumbs = {}                              # pass_id -> (frame time, JPEG bytes)
        self.capture = None                           # set by run.py: the capture object (status(), dropped)
        self.recorder = None                          # set by run.py when recording
        self.beam = None                              # set by run.py when the beam input runs (rt/beam/, WP-L2)
        self.time = dict(offset_s=None, source="not checked yet", warnings=[], checked=None)
        self.started_mono = self.clock.mono()
        self.started_wall = self.clock.wall()
        self.info = {}                                # static info: event, version, source
        # Notices: warnings set once at start that stay for the whole run (no roster for the event, ffmpeg missing so
        # no recording). Shown in yellow at the top of the page: rt works, but something should be fixed. run.py
        # appends plain strings.
        self.notices = []
        self.setup_token = str(s.get("setup_token", "") or "")   # Beam setup POSTs need it when set

    # ------------------------------------------------------------------ inputs
    def on_frame(self, fr):
        """Called by the detection loop after each processed frame."""
        now = self.clock.mono()
        with self.lock:
            self.frames += 1
            self.last_frame_t = fr.t
            self.frame_monos.append(now)
            while self.frame_monos and now - self.frame_monos[0] > self.window_s:
                self.frame_monos.popleft()
        if self.last_thumb_mono is None or now - self.last_thumb_mono >= self.thumb_every_s:
            self.last_thumb_mono = now
            jpg = self.make_thumb(fr.img)
            with self.lock:
                self.ring.append((fr.t, jpg))
                while self.ring and fr.t - self.ring[0][0] > self.ring_s:
                    self.ring.popleft()

    def make_thumb(self, img):
        h, w = img.shape[:2]
        if w > self.thumb_w:
            img = cv2.resize(img, (self.thumb_w, max(1, int(round(h * self.thumb_w / w)))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else b""

    def nearest_thumb(self, t, max_dt=0.5):
        """The ring frame closest to time t, if one is within max_dt seconds: (frame time, JPEG) or None."""
        with self.lock:
            ring = list(self.ring)
        if not ring or t is None:
            return None
        times = [x[0] for x in ring]
        i = bisect_left(times, t)
        best = min((j for j in (i - 1, i) if 0 <= j < len(ring)), key=lambda j: abs(times[j] - t))
        return ring[best] if abs(times[best] - t) <= max_dt else None

    def on_event(self, ev):
        """Called by StatusSink for every rt event."""
        kind = ev["type"]
        if kind not in ("pass_start", "pass_provisional", "pass_end", "pass_identity"):
            return
        pass_id = ev["pass_id"]
        with self.lock:                     # the web server thread reads these dicts: change them under the lock
            entry = self.passes.get(pass_id)
            if entry is None:
                entry = self.passes[pass_id] = dict(pass_id=pass_id, state="open", number=None, confidence=None,
                                                    auto_confirm=None, provisional=None, t_enter=None, t_cross=None,
                                                    team=None, thumb_t=None)
                while len(self.passes) > self.n_passes:
                    old, _ = self.passes.popitem(last=False)
                    self.thumbs.pop(old, None)
            if kind == "pass_start":
                entry["t_enter"] = ev.get("t_enter")
                thumb = self.nearest_thumb(ev.get("t_video"))     # the frame being processed when the pass started
                if thumb:
                    self._set_thumb(pass_id, thumb)
            elif kind == "pass_provisional":
                entry["provisional"] = ev.get("number")
            elif kind in ("pass_end", "pass_identity"):
                entry["t_enter"] = ev.get("t_enter", entry["t_enter"])
                entry["t_cross"] = ev.get("t_cross", entry["t_cross"])
                if kind == "pass_end":
                    entry["state"] = "closed"
                    thumb = self.nearest_thumb(entry["t_cross"])  # better: the frame where the car crosses the centre
                    if thumb:
                        self._set_thumb(pass_id, thumb)
                else:
                    entry.update(state="read", number=ev.get("number"), confidence=ev.get("confidence"),
                                 auto_confirm=ev.get("auto_confirm"), team=ev.get("team"))

    def _set_thumb(self, pid, th):
        with self.lock:
            if pid in self.passes:
                self.thumbs[pid] = th
                self.passes[pid]["thumb_t"] = th[0]

    def thumb(self, pid):
        with self.lock:
            th = self.thumbs.get(pid)
        return th[1] if th else None

    # ------------------------------------------------------------------ output
    def fps(self):
        now = self.clock.mono()
        with self.lock:
            n_recent = sum(1 for m in self.frame_monos if now - m <= self.window_s)
        span = min(self.window_s, max(1e-9, now - self.started_mono))
        return round(n_recent / span, 2) if n_recent else 0.0

    def payload(self):
        cap = self.capture
        cam = cap.status() if cap is not None and hasattr(cap, "status") else dict(state="file" if cap else "none")
        rec = self.recorder.status() if self.recorder is not None else dict(state="off")
        fps = self.fps()
        warnings = []
        off = self.time.get("offset_s")
        clock_bad = off is not None and abs(off) > OFFSET_RED_S
        if clock_bad:
            warnings.append("clock offset %+.2f s (> %.1f s): resync the time" % (off, OFFSET_RED_S))
        warnings += ["clock: " + w for w in self.time.get("warnings", [])]
        if fps < 0.8 * self.target_fps:
            warnings.append("processing %.1f fps, below %.0f" % (fps, self.target_fps))
        if cam.get("state") not in ("streaming", "file"):
            warnings.append("camera %s %s" % (cam.get("state"), cam.get("detail", "")))
        if rec.get("disk_state") == "low":
            warnings.append("disk low: %.1f GB free" % rec["free_gb"])
        if rec.get("disk_state") == "stopped":
            warnings.append("recording STOPPED: %.1f GB free" % rec["free_gb"])
        if str(rec.get("state", "")).startswith(("error", "failed")):
            warnings.append("recording " + rec["state"])
        beam = self.beam.status() if self.beam is not None else dict(enabled=False)
        warnings += beam_warnings(beam)
        with self.lock:
            passes = [dict(p, t_enter_iso=iso(p["t_enter"]), t_cross_iso=iso(p["t_cross"]), thumb_iso=iso(p["thumb_t"]),
                           thumb_url=("/thumb/%d.jpg?v=%.3f" % (p["pass_id"], p["thumb_t"])) if p["thumb_t"] else None)
                      for p in reversed(self.passes.values())]   # p = one pass entry
        return dict(now=iso(self.clock.wall()), uptime_s=round(self.clock.mono() - self.started_mono, 1),
                    info=self.info,
                    processing=dict(fps=fps, target_fps=self.target_fps, frames=self.frames,
                                    dropped=getattr(cap, "dropped", 0) if cap is not None else 0,
                                    last_frame_iso=iso(self.last_frame_t)),
                    camera=cam, clock=dict(self.time, bad=clock_bad, red_above_s=OFFSET_RED_S),
                    recording=rec, beam=beam, passes=passes, warnings=warnings, notices=list(self.notices))


def beam_warnings(beam):
    """Red boxes for the beam: a setup error, setup mode, port not open (missing / unplugged), beam stuck, poll loop
    too slow."""
    out = []
    if beam.get("error"):
        out.append(beam["error"])
    if beam.get("setup_mode"):
        out.append("beam setup mode: the beam driver is paused (Beam setup page)")
    if not beam.get("enabled"):
        return out
    if beam.get("state") != "open":
        out.append("beam %s: %s" % (beam.get("state"), beam.get("detail", "")))
    if beam.get("stuck"):
        out.append("beam STUCK for %.0f s (blocked or flickering): someone in the beam, misaligned, or wrong "
                   "polarity" % beam.get("stuck_for_s", 0))
    if beam.get("queue_dropped"):
        out.append("beam: %d queued items dropped (the beam dispatcher fell behind)" % beam["queue_dropped"])
    rate = (beam.get("poll") or {}).get("rate_hz")
    if beam.get("driver") == "modem-line" and rate is not None and rate < 200:
        out.append("beam poll rate only %.0f Hz (timing worse than 5 ms)" % rate)
    return out


class StatusSink(Sink):
    def __init__(self, board):
        self.board = board

    def event(self, ev):
        self.board.on_event(ev)


class TimeChecker(threading.Thread):
    """Measures the clock offset (rt/timesync.py) now and then every `period_s`, into board.time."""

    def __init__(self, cfg, board, period_s=60.0, measure=None, check=None):
        super().__init__(daemon=True)
        from . import timesync
        self.cfg, self.board, self.period = cfg, board, float(period_s)
        self.measure = measure or timesync.measure_offset
        self.check = check or timesync.check
        self.halt = threading.Event()

    def once(self):
        try:
            off, src = self.measure(self.cfg)
        except Exception as e:  # noqa: BLE001
            off, src = None, "error: %s" % e
        try:
            warns = self.check(self.cfg, lambda m: None)
        except Exception as e:  # noqa: BLE001
            warns = ["time check failed: %s" % e]
        self.board.time = dict(offset_s=None if off is None else round(off, 4), source=src, warnings=warns,
                               checked=iso(self.board.clock.wall()))

    def run(self):
        while True:
            self.once()
            if self.halt.wait(self.period):
                return


# ------------------------------------------------------------------------------------------------ page
CSS = """body{font-family:system-ui,sans-serif;margin:0 16px;background:#fff;color:#111}
h1{font-size:1.2em}h2{font-size:1em;margin:1em 0 .3em}table{border-collapse:collapse;width:100%;font-size:.9em}
td,th{border-bottom:1px solid #ddd;padding:3px 6px;text-align:left;vertical-align:middle}
.ok{color:#070}.warn{color:#a60}.bad{color:#fff;background:#c00;padding:2px 6px;border-radius:3px}
.box{background:#fdd;border:1px solid #c00;padding:6px;margin:6px 0}
.notice{background:#ffc;border:1px solid #c90;padding:6px;margin:6px 0}img{width:160px;max-width:40vw;display:block}
.num{font-size:1.4em;font-weight:bold}small{color:#666}"""


def _e(x):
    return html.escape("" if x is None else str(x))


def _t(s):
    return s[11:23] if s else ""


def render_html(p):
    pr, cam, ck, rec = p["processing"], p["camera"], p["clock"], p["recording"]
    fps_cls = "ok" if pr["fps"] >= 0.8 * pr["target_fps"] else "bad"
    off = ck.get("offset_s")
    if off is None:
        off_html = '<span class="warn">unknown (%s)</span>' % _e(ck.get("source"))
    else:
        off_html = '<span class="%s">%+.3f s</span> <small>%s</small>' % ("bad" if ck.get("bad") else "ok", off,
                                                                          _e(ck.get("source")))
    cam_cls = "ok" if cam.get("state") in ("streaming", "file") else "bad"
    info = cam.get("info") or {}
    settings = "".join("<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (
        _e(k), _e(v.get("requested")), _e(v.get("read_back")) + ("" if v.get("accepted") else " (not accepted)"))
        for k, v in (info.get("settings") or {}).items())
    rec_cls = {"ok": "ok", "low": "warn", "stopped": "bad"}.get(rec.get("disk_state"), "ok")
    if str(rec.get("state", "")).startswith(("error", "failed")):
        rec_cls = "bad"
    rows = []
    for x in p["passes"]:
        img = '<img src="%s" alt="">' % _e(x["thumb_url"]) if x.get("thumb_url") else ""
        num = x["number"] if x["number"] is not None else (("(%s)" % x["provisional"]) if x["provisional"] else "?")
        conf = "" if x["confidence"] is None else "%.2f%s" % (x["confidence"], " AUTO" if x["auto_confirm"] else "")
        rows.append('<tr><td>%s<small>%s</small></td><td class="num">%s</td><td>%s<br><small>%s</small></td>'
                    '<td>%s<br><small>pass %s, %s</small></td></tr>' % (
                        img, _e(_t(x.get("thumb_iso"))), _e(num), _e(conf), _e(x.get("team") or ""),
                        _e(_t(x.get("t_cross_iso")) or _t(x.get("t_enter_iso"))), _e(x["pass_id"]), _e(x["state"])))
    warn = "".join('<div class="box">%s</div>' % _e(w) for w in p["warnings"])
    warn += "".join('<div class="notice">%s</div>' % _e(n) for n in p.get("notices", []))
    return """<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="2">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>rt status</title><style>%s</style></head>
<body><h1>rt lap line &middot; %s</h1><small>%s &middot; up %.0f s &middot; %s</small>%s
<h2>Processing</h2><span class="%s">%.1f fps</span> (target %.0f) &middot; %d frames &middot; %d dropped
<h2>Camera</h2><span class="%s">%s</span> %s<br><small>%s %sx%s @ %s fps, read mode %s, %s camera frames, %s failed
reads, %s corrupt, %s reopens, last frame %s s ago</small><table>%s</table>
<h2>Clock offset</h2>%s
<h2>Recording</h2><span class="%s">%s</span> %s<br><small>%s GB free (warn &lt; %s, stop &lt; %s) &middot; %s frames
&middot; %s dropped &middot; %s</small>
%s
<h2>Last passes</h2><table><tr><th>thumbnail</th><th>number</th><th>confidence</th><th>t_cross</th></tr>%s</table>
</body></html>""" % (
        CSS, _e(p["info"].get("event", "")), _e(p["now"]), p["uptime_s"], _e(p["info"].get("version", "")), warn,
        fps_cls, pr["fps"], pr["target_fps"], pr["frames"], pr["dropped"],
        cam_cls, _e(cam.get("state")), _e(cam.get("detail", "")), _e(info.get("fourcc", "")), _e(info.get("width")),
        _e(info.get("height")), _e(info.get("fps")), _e(cam.get("read_mode")), _e(cam.get("camera_frames")),
        _e(cam.get("failed_reads")), _e(cam.get("corrupt")), _e(cam.get("reopens")), _e(cam.get("last_frame_age_s")),
        settings,
        off_html,
        rec_cls, _e(rec.get("state")), _e(rec.get("file") or ""), _e(rec.get("free_gb")), _e(rec.get("warn_free_gb")),
        _e(rec.get("stop_free_gb")), _e(rec.get("frames")), _e(rec.get("dropped")), _e(rec.get("encoder") or ""),
        render_beam(p.get("beam") or {}), "".join(rows))


def render_beam(b):
    """The Beam section of the page (one line with the setup link when the beam input is off)."""
    if not b.get("enabled"):
        return '<h2>Beam</h2>off &middot; <a href="/beam">Beam setup</a>'
    cls = "ok" if b.get("state") == "open" and not b.get("stuck") else "bad"
    level = {1: "ACTIVE (blocked)", 0: "clear", None: "unknown"}.get(b.get("level"), b.get("level"))
    poll = b.get("poll") or {}
    poll_txt = ("%s Hz, max gap %s ms, %s late polls" % (poll.get("rate_hz"), poll.get("max_gap_ms"),
                                                         poll.get("late_polls")) if poll else "not measured yet")
    rows = "".join("<tr><td>%s</td><td>%s</td><td>%s s</td><td>%s</td></tr>" % (
        _e(_t(x.get("t_beam_iso"))), _e(x.get("n_edges")), _e(x.get("duration_s")),
        _e("" if x.get("pass_id") is None else "pass %s %s" % (x["pass_id"], "#%s" % x["number"]
                                                                if x.get("number") is not None else "")))
        for x in b.get("last_breaks", []))
    a, led = b.get("association") or {}, b.get("led") or {}
    led_txt = ("LED: %s measurements, last %s s, min %s s, ROI mean %s" % (
        led.get("measurements"), led.get("last_latency_s"), led.get("min_latency_s"), led.get("last_mean"))
        if led.get("enabled") else "LED latency off")
    return """<h2>Beam</h2><span class="%s">%s</span> %s &middot; <a href="/beam">Beam setup</a>
<br><small>%s on %s, line %s &middot; level <b>%s</b> &middot;
%s edges/min (%s total) &middot; %s breaks &middot; poll %s &middot; stuck episodes %s &middot; passes matched %s /
none %s / ambiguous %s &middot; %s</small>
<table><tr><th>break (t_beam)</th><th>edges</th><th>duration</th><th>matched pass</th></tr>%s</table>""" % (
        cls, _e("STUCK" if b.get("stuck") else b.get("state")), _e(b.get("detail", "")), _e(b.get("driver")),
        _e(b.get("port")), _e(b.get("line")), _e(level), _e(b.get("edges_per_min")), _e(b.get("edges_total")),
        _e(b.get("breaks_total")), _e(poll_txt), _e(b.get("stuck_episodes")), _e(a.get("matched")), _e(a.get("none")),
        _e(a.get("ambiguous")), _e(led_txt), rows)


# ------------------------------------------------------------------------------------------------ beam setup actions
BEAM_POST_ROUTES = ("/beam/setup/start", "/beam/setup/stop", "/beam/monitor/clear", "/beam/suggest", "/beam/apply",
                    "/beam/off")
MAX_POST_BYTES = 8192


def handle_beam_post(controller, path, data):
    """One Beam setup action on the BeamController (rt/beam/control.py). Returns (HTTP code, JSON answer)."""
    if path == "/beam/setup/start":
        ok, errors = controller.start_setup(data)
        return (200, dict(ok=True, message="monitoring: wave through the beam, then Suggest")) if ok else \
            (400, dict(ok=False, errors=errors))
    if path == "/beam/setup/stop":
        left = controller.leave_setup()
        return 200, dict(ok=True, message="setup closed: beam restarted with the saved settings" if left else
                         "not in setup mode")
    if path == "/beam/monitor/clear":
        return (200, dict(ok=True, message="cleared")) if controller.clear_monitor() else \
            (409, dict(ok=False, errors=["start the monitor first"]))
    if path == "/beam/suggest":
        suggestion = controller.suggest()
        if suggestion is None:
            return 409, dict(ok=False, errors=["start the monitor first, then wave through the beam"])
        return 200, dict(ok=True, suggestion=suggestion)
    if path == "/beam/apply":
        ok, errors = controller.apply(data)
        return (200, dict(ok=True, message="applied and saved: beam restarted")) if ok else \
            (400, dict(ok=False, errors=errors))
    if path == "/beam/off":
        ok, errors = controller.turn_off()
        return (200, dict(ok=True, message="beam turned off (saved)")) if ok else (400, dict(ok=False, errors=errors))
    return 404, dict(ok=False, errors=["unknown action"])


class StatusHTTPServer(ThreadingHTTPServer):
    """The stdlib HTTP server, but on Windows it refuses a port another program already uses. Python's HTTPServer
    sets SO_REUSEADDR, and on Windows that lets a second program bind a port that is already listening (both
    "own" it and requests go to either), so a busy port would go unnoticed. SO_EXCLUSIVEADDRUSE makes the bind
    fail instead, as it does on Linux."""

    daemon_threads = True
    if sys.platform == "win32":
        allow_reuse_address = False

        def server_bind(self):
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()


def start_status_server(board, bind, port, required, log=print, warn=print):
    """Start the status page. If the port cannot be opened (another rt, or another program, uses it):
      * required (rt started by the launcher, which watches rt through this page): say why and exit with
        EXIT_STATUS_PORT_BUSY. Running on would be pointless: the launcher would see no page, call rt stalled and
        kill it every ~90 s, forever, without saying why.
      * not required (a manual run): warn and run without the page. Returns the server, or None."""
    from .config import EXIT_STATUS_PORT_BUSY
    try:
        server = StatusServer(board, bind, int(port)).start()
    except OSError as e:
        msg = ("status page cannot listen on port %s (%s): another program, or another rt, is using it. Close it, "
               "or set status.port to a free port in the config (the launcher reads the same setting)" % (port, e))
        if required:
            warn(msg + ". Exiting: the launcher needs the status page to watch rt.")
            print("rt: " + msg, file=sys.stderr, flush=True)
            raise SystemExit(EXIT_STATUS_PORT_BUSY)
        warn(msg + ". Running without the status page.")
        return None
    log("status page: http://%s:%d/" % ("localhost" if bind in ("0.0.0.0", "") else bind, server.port))
    return server


class StatusServer:
    """Serves the board on (bind, port) from a background thread. stop() closes it."""

    def __init__(self, board, bind="0.0.0.0", port=8765, log=print):
        self.board = board
        handler = self._handler()
        self.httpd = StatusHTTPServer((bind, int(port)), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.log = log

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _handler(self):
        board = self.board

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):   # keep the console for rt's own messages
                pass

            def _send(self, code, body, ctype, cache=False):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code, obj):
                return self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json")

            def do_POST(self):  # noqa: N802 (http.server's name)
                """Only the Beam setup actions. Anything wrong is a clear JSON error; nothing here can stop rt."""
                path = self.path.split("?", 1)[0]
                if path not in BEAM_POST_ROUTES:
                    return self._json(405, dict(ok=False, errors=["only the Beam setup actions accept POST"]))
                controller = board.beam
                if controller is None or not hasattr(controller, "apply"):
                    return self._json(404, dict(ok=False, errors=["beam setup is not available in this rt"]))
                if board.setup_token and not hmac.compare_digest(self.headers.get("X-Setup-Token", ""),
                                                                 board.setup_token):
                    return self._json(403, dict(ok=False, errors=["wrong or missing setup token"]))
                if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                    return self._json(415, dict(ok=False, errors=["send JSON (Content-Type: application/json)"]))
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = -1
                if not 0 <= length <= MAX_POST_BYTES:
                    return self._json(413, dict(ok=False, errors=["request too large"]))
                try:
                    data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                except (ValueError, UnicodeDecodeError):
                    return self._json(400, dict(ok=False, errors=["the body is not valid JSON"]))
                if not isinstance(data, dict):
                    return self._json(400, dict(ok=False, errors=["the body must be a JSON object"]))
                try:
                    code, answer = handle_beam_post(controller, path, data)
                except Exception as e:  # noqa: BLE001 (a bug in a setup action must not take the page down)
                    code, answer = 500, dict(ok=False, errors=["%s: %s" % (type(e).__name__, e)])
                return self._json(code, answer)

            def do_GET(self):  # noqa: N802 (http.server's name)
                path = self.path.split("?", 1)[0]
                try:
                    if path in ("/beam", "/beam/"):
                        from .beam.setup_page import render_setup_page
                        return self._send(200, render_setup_page().encode("utf-8"), "text/html; charset=utf-8")
                    if path == "/beam/setup.json":
                        controller = board.beam
                        if controller is None or not hasattr(controller, "setup_state"):
                            return self._json(404, dict(ok=False, errors=["beam setup is not available"]))
                        return self._json(200, controller.setup_state())
                    if path == "/status.json":
                        return self._send(200, json.dumps(board.payload(), default=str).encode(), "application/json")
                    if path in ("/", "/index.html"):
                        return self._send(200, render_html(board.payload()).encode("utf-8"),
                                          "text/html; charset=utf-8")
                    if path.startswith("/thumb/") and path.endswith(".jpg"):
                        jpg = board.thumb(int(path[len("/thumb/"):-4]))
                        if jpg:
                            return self._send(200, jpg, "image/jpeg", cache=True)
                    return self._send(404, b"not found", "text/plain")
                except (ValueError, KeyError):
                    return self._send(404, b"not found", "text/plain")

        return Handler
