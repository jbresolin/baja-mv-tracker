"""Read-only status page of a running rt (WP-L1): a tiny HTTP server inside the rt process (Python stdlib only, no
frontend build), meant to be opened from a phone on the LAN.

  http://<laptop>:8765/             the page (refreshes itself every 2 s)
  http://<laptop>:8765/status.json  the same data as JSON (the launcher rt/supervise.py reads `processing.fps`)
  http://<laptop>:8765/thumb/<pass_id>.jpg  a pass's thumbnail

What it shows: processing fps and dropped frames; camera state and the image settings read back; the clock offset
(red above 0.5 s); recording state and free disk; the last 20 passes with a thumbnail, the number read, confidence
and t_cross. There are no controls.

Pieces: `StatusBoard` holds the state (the detection loop calls on_frame(), the event sink feeds passes);
`StatusSink` is the rt/sinks.py sink that feeds it; `TimeChecker` re-measures the clock offset once a minute in the
background; `StatusServer` serves the page.
"""
import collections
import datetime as dt
import html
import json
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
        self.lock = threading.Lock()
        self.frame_monos = collections.deque()        # monotonic times of processed frames, last window_s
        self.frames = 0
        self.last_frame_t = None
        self.ring = collections.deque()               # (frame wall time, JPEG bytes) of the last ring_s seconds
        self.last_thumb_mono = None
        self.passes = collections.OrderedDict()       # pass_id -> dict, the last n_passes
        self.thumbs = {}                              # pass_id -> (frame time, JPEG bytes)
        self.capture = None                           # set by run.py: the capture object (status(), dropped)
        self.recorder = None                          # set by run.py when recording
        self.time = dict(offset_s=None, source="not checked yet", warnings=[], checked=None)
        self.started_mono = self.clock.mono()
        self.started_wall = self.clock.wall()
        self.info = {}                                # static info: event, version, source

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
        t = ev["type"]
        if t not in ("pass_start", "pass_provisional", "pass_end", "pass_identity"):
            return
        pid = ev["pass_id"]
        with self.lock:
            p = self.passes.get(pid)
            if p is None:
                p = self.passes[pid] = dict(pass_id=pid, state="open", number=None, confidence=None,
                                            auto_confirm=None, provisional=None, t_enter=None, t_cross=None,
                                            team=None, thumb_t=None)
                while len(self.passes) > self.n_passes:
                    old, _ = self.passes.popitem(last=False)
                    self.thumbs.pop(old, None)
        if t == "pass_start":
            p["t_enter"] = ev.get("t_enter")
            th = self.nearest_thumb(ev.get("t_video"))        # the frame being processed when the pass started
            if th:
                self._set_thumb(pid, th)
        elif t == "pass_provisional":
            p["provisional"] = ev.get("number")
        elif t in ("pass_end", "pass_identity"):
            p["t_enter"] = ev.get("t_enter", p["t_enter"])
            p["t_cross"] = ev.get("t_cross", p["t_cross"])
            if t == "pass_end":
                p["state"] = "closed"
                th = self.nearest_thumb(p["t_cross"])          # better: the frame where the car crosses the centre
                if th:
                    self._set_thumb(pid, th)
            else:
                p.update(state="read", number=ev.get("number"), confidence=ev.get("confidence"),
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
            n = sum(1 for m in self.frame_monos if now - m <= self.window_s)
        span = min(self.window_s, max(1e-9, now - self.started_mono))
        return round(n / span, 2) if n else 0.0

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
        with self.lock:
            passes = [dict(p, t_enter_iso=iso(p["t_enter"]), t_cross_iso=iso(p["t_cross"]), thumb_iso=iso(p["thumb_t"]),
                           thumb_url=("/thumb/%d.jpg?v=%.3f" % (p["pass_id"], p["thumb_t"])) if p["thumb_t"] else None)
                      for p in reversed(self.passes.values())]
        return dict(now=iso(self.clock.wall()), uptime_s=round(self.clock.mono() - self.started_mono, 1),
                    info=self.info,
                    processing=dict(fps=fps, target_fps=self.target_fps, frames=self.frames,
                                    dropped=getattr(cap, "dropped", 0) if cap is not None else 0,
                                    last_frame_iso=iso(self.last_frame_t)),
                    camera=cam, clock=dict(self.time, bad=clock_bad, red_above_s=OFFSET_RED_S),
                    recording=rec, passes=passes, warnings=warnings)


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
.box{background:#fdd;border:1px solid #c00;padding:6px;margin:6px 0}img{width:160px;max-width:40vw;display:block}
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
    return """<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="2">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>rt status</title><style>%s</style></head>
<body><h1>rt lap line &middot; %s</h1><small>%s &middot; up %.0f s &middot; %s</small>%s
<h2>Processing</h2><span class="%s">%.1f fps</span> (target %.0f) &middot; %d frames &middot; %d dropped
<h2>Camera</h2><span class="%s">%s</span> %s<br><small>%s %sx%s @ %s fps, read mode %s, %s camera frames, %s failed
reads, %s corrupt, %s reopens, last frame %s s ago</small><table>%s</table>
<h2>Clock offset</h2>%s
<h2>Recording</h2><span class="%s">%s</span> %s<br><small>%s GB free (warn &lt; %s, stop &lt; %s) &middot; %s frames
&middot; %s dropped &middot; %s</small>
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
        "".join(rows))


class StatusServer:
    """Serves the board on (bind, port) from a background thread. stop() closes it."""

    def __init__(self, board, bind="0.0.0.0", port=8765, log=print):
        self.board = board
        handler = self._handler()
        self.httpd = ThreadingHTTPServer((bind, int(port)), handler)
        self.httpd.daemon_threads = True
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

            def do_GET(self):  # noqa: N802 (http.server's name)
                path = self.path.split("?", 1)[0]
                try:
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
