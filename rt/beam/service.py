"""BeamService (WP-L2): runs a beam driver next to rt, groups its edges into breaks, writes the beam events into rt's
JSONL, links breaks to vision passes, and feeds the status page. Shadow mode: nothing goes to the lap app.

Two threads, so that the beam can never slow the detection pipeline (and the pipeline can never delay a beam time
stamp):

  reader thread      opens the port (retries with backoff 1 s, 2 s, ... 30 s when it is missing), polls the driver
                     (~1 kHz for modem-line), and puts every stamped edge into a queue. It does nothing else: no
                     file writes, no locks shared with the detection loop.
  dispatcher thread  takes edges from the queue and: writes `beam_edge` events, groups edges into `beam_break` /
                     `beam_stuck` events (rt/beam/grouping.py), answers `pass_end` events with a `pass_beam` event
                     (rt/beam/associate.py), matches LED transitions to edges (`camera_latency`), and keeps the
                     numbers the status page shows.

The detection loop only calls on_frame(frame) (a few microseconds; nothing at all when the LED feature is off),
and rt's event emitter gives every event to PassBeamSink, which only puts pass_end / pass_identity into the queue.

Events written (rt/README.md "Event schema" has every field):
  beam          the port's state changed (opening / open / retrying / lost / stopped)
  beam_edge     every raw edge: line, level, raw value, both clocks
  beam_break    one per car: first edge = crossing time (t_beam), last edge, edge count, duration
  beam_stuck    the beam stayed blocked (or kept flickering) longer than stuck_s: state "stuck", later "cleared"
  beam_poll     every poll_report_s: achieved poll rate and the largest gap between polls (jitter)
  pass_beam     for every pass_end: matched (with t_beam) / none / ambiguous
  camera_latency  LED feature only: frame time - beam edge time

Testing: reader_step() and dispatch_step() each do one round of work, so tests call them directly with a fake
clock and a fake source instead of starting threads.
"""
import collections
import queue
import threading
import time

from ..backoff import Backoff, SystemClock
from ..sinks import Sink, iso
from .associate import LedWatcher, associate_pass
from .grouping import BreakGrouper
from .sources import BeamPortError, make_source


class BeamService:
    def __init__(self, cfg, emit, source=None, clock=None, sleep=None, serial_factory=None):
        b = self.conf = cfg["beam"]
        self.emit = emit
        self.clock = clock or SystemClock()
        self.sleep = sleep or time.sleep
        self.source = source or make_source(cfg, clock=self.clock, sleep=sleep, serial_factory=serial_factory)
        if self.source is None:
            raise ValueError('beam.driver is "off": there is nothing to run')
        # Same correction rt adds to frame times (clock.offset_s), so t_beam and t_cross share a time base. The
        # camera latency (capture.live_latency_s) is NOT subtracted: it belongs to the camera, not to the beam.
        self.offset = float(cfg["clock"]["offset_s"])
        self.grouper = BreakGrouper(float(b.get("group_gap_s", 0.25)), float(b.get("stuck_s", 5.0)))
        self.backoff = Backoff(float(b.get("reopen_first_s", 1)), 2.0, float(b.get("reopen_max_s", 30)))
        self.assoc_window = float(b.get("assoc_window_s", 1.0))
        self.assoc_offset = float(b.get("assoc_offset_s", 0.0))
        self.max_edge_events = int(b.get("max_edge_events_per_s", 100))
        # The dispatcher never lets the grouper's time run past what the reader has surely queued (see _advance).
        self.advance_guard = float(b.get("advance_guard_s", 0.02))
        self.poll_report_s = float(b.get("poll_report_s", 60))
        self.led = LedWatcher(b.get("led_roi") or [], float(b.get("led_threshold", 128)))
        self.led_max_latency = float(b.get("led_max_latency_s", 0.5))
        self.q = queue.SimpleQueue()          # reader thread / detection loop / event sink -> dispatcher
        self.halt = threading.Event()
        self.threads = []
        self.lock = threading.Lock()          # guards the status numbers below (read by the status page thread)

        # reader-thread state (only the reader touches these)
        self.port_open = False
        self.next_open_mono = None
        self.polled_since_open = False
        self.last_poll_report = None
        # Monotonic time up to which every edge the reader has seen is already in the queue. Written by the reader
        # (one float assignment, atomic in Python) AFTER it queued a poll's edges; read by the dispatcher.
        self.reader_horizon_mono = None

        # dispatcher-thread state
        self.state, self.detail, self.state_since = "starting", "", self.clock.wall()
        self.opens = 0
        self.edges_total = 0
        self.edge_monos = collections.deque()         # monotonic times of the edges of the last 60 s
        self.edge_log_window = None                   # (start of the current 1 s window, events logged in it)
        self.edges_not_logged = 0
        self.edges_not_logged_total = 0
        self.level = None
        self.recent_breaks = collections.deque(maxlen=2000)   # for association (newest last)
        self.last_breaks = collections.deque(maxlen=10)       # for the status page
        self.break_passes = {}                                # break_id -> [pass ids matched to it]
        self.pass_numbers = {}                                # pass_id -> number read (status page only)
        self.breaks_total = 0
        self.stuck_episodes = 0
        self.pending_passes = []
        self.assoc_counts = dict(matched=0, none=0, ambiguous=0)
        self.recent_active_edges = collections.deque(maxlen=200)
        self.pending_led = []
        self.led_stats = dict(measurements=0, unmatched=0, last_latency_s=None, min_latency_s=None)
        self.poll = {}

    # ================================================================== reader thread
    def reader_step(self):
        """One round of the reader: (re)open the port when it is time, or poll it once."""
        if not self.port_open:
            now = self.clock.mono()
            if self.next_open_mono is not None and now < self.next_open_mono:
                self.sleep(min(0.2, self.next_open_mono - now))
                return
            try:
                info = self.source.open()
            except BeamPortError as e:
                self.source.close()
                self._schedule_reopen("retrying", str(e))
                return
            self.port_open, self.polled_since_open = True, False
            self.last_poll_report = self.clock.mono()
            self.q.put(("opened", info, self.clock.mono(), self.clock.wall()))
            return
        try:
            edges = self.source.poll()
        except BeamPortError as e:
            self.source.close()
            self.port_open = False
            self._schedule_reopen("lost", str(e))
            return
        if not self.polled_since_open:
            self.polled_since_open = True
            self.backoff.reset()          # the port works: the next failure starts again at the short wait
        for e in edges:
            self.q.put(("edge", e))
        now = self.clock.mono()
        # Every edge stamped up to `now` is queued: any later edge will be stamped later than this.
        self.reader_horizon_mono = now
        if now - self.last_poll_report >= self.poll_report_s:
            self.last_poll_report = now
            self.q.put(("poll", self.source.poll_stats.report(now)))

    def _schedule_reopen(self, state, why):
        wait = self.backoff.next()
        now = self.clock.mono()
        self.next_open_mono = now + wait
        self.q.put((state, "%s; next try in %.0f s" % (why, wait), now, self.clock.wall()))

    def _reader_loop(self):
        while not self.halt.is_set():
            try:
                self.reader_step()
            except Exception as e:  # noqa: BLE001 (an unexpected driver error must not kill the beam thread)
                self.q.put(("error", "beam reader: %s: %s" % (type(e).__name__, e)))
                try:
                    self.source.close()
                except Exception:  # noqa: BLE001
                    pass
                self.port_open = False
                self._schedule_reopen("retrying", "unexpected error")

    # ================================================================== inputs from rt
    def on_frame(self, fr):
        """Called by the detection loop for every processed frame. Only does work when the LED ROI is set: the mean
        brightness of a small region. Never raises."""
        if self.led.roi is None:
            return
        try:
            mean = self.led.update(fr.img)
            if mean is not None:
                self.q.put(("led", fr.t_cap, fr.t, fr.idx, mean))
        except Exception as e:  # noqa: BLE001
            self.q.put(("error", "beam LED ROI: %s" % e))

    def on_event(self, ev):
        """Called (through PassBeamSink) for every rt event, from inside rt's emitter: only queue, never emit here
        (the emitter's lock is held)."""
        if ev.get("type") == "pass_end":
            self.q.put(("pass", ev))
        elif ev.get("type") == "pass_identity":
            self.q.put(("number", ev.get("pass_id"), ev.get("number")))

    # ================================================================== dispatcher thread
    def dispatch_step(self, timeout=0.05):
        """Handle everything queued (waiting up to `timeout` s for the first item), then let time pass."""
        try:
            item = self.q.get(timeout=timeout) if timeout > 0 else self.q.get_nowait()
        except queue.Empty:
            item = None
        # Read the reader's horizon BEFORE draining the queue: the reader queues its edges first and moves the
        # horizon after, so every edge up to this horizon is in the queue now and gets handled below.
        horizon = self.reader_horizon_mono
        while item is not None:
            self._handle(item)
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                item = None
        self._advance(horizon)

    def _dispatch_loop(self):
        while not self.halt.is_set():
            try:
                self.dispatch_step(0.05)
            except Exception as e:  # noqa: BLE001 (never stop handling beam events)
                self._warn("beam dispatcher: %s: %s" % (type(e).__name__, e))
                self.sleep(0.05)

    def _handle(self, item):
        kind = item[0]
        if kind == "edge":
            self._on_edge(item[1])
        elif kind == "opened":
            _, info, t_mono, t_wall = item
            self.opens += 1
            self.level = info.get("level")
            self._set_state("open", ", ".join("%s=%s" % kv for kv in info.items()))
            self._info("beam: %s %s open (%s)" % (self.source.driver, self.source.port, self.detail))
            if info.get("level") == 1:
                self._warn("beam: the beam is ACTIVE right after opening the port: something blocks it, or the "
                           "polarity (beam.active) is the wrong way round")
                self.grouper.start_blocked(t_mono, t_wall)
        elif kind in ("retrying", "lost"):
            _, detail, t_mono, t_wall = item
            if kind == "lost":
                self._warn("beam: port lost: %s" % detail)
                self.level = None
                for rec in self.grouper.disconnect(t_mono, t_wall):
                    self._on_record(rec)
            elif self.state != "retrying":
                self._warn("beam: %s" % detail)
            self._set_state(kind, detail)
        elif kind == "poll":
            with self.lock:
                self.poll = item[1]
            self.emit(dict(type="beam_poll", driver=self.source.driver, **item[1]))
        elif kind == "pass":
            self.pending_passes.append((item[1], self.clock.mono()))
        elif kind == "number":
            with self.lock:
                self.pass_numbers[item[1]] = item[2]
                while len(self.pass_numbers) > 200:
                    self.pass_numbers.pop(next(iter(self.pass_numbers)))
        elif kind == "led":
            self.pending_led.append(item[1:])
        elif kind == "error":
            self._warn(item[1])

    def _on_edge(self, e):
        now = e.t_mono
        with self.lock:
            self.edges_total += 1
            self.edge_monos.append(now)
            while self.edge_monos and now - self.edge_monos[0] > 60.0:
                self.edge_monos.popleft()
            if not e.momentary:
                self.level = e.level
        if e.level == 1:
            self.recent_active_edges.append(e)
        self._log_edge(e)
        for rec in self.grouper.feed(e):
            self._on_record(rec)

    def _log_edge(self, e):
        """A `beam_edge` event per raw edge, at most max_edge_events_per_s per second (a bouncing contact or a noisy
        line could produce thousands; they still all count for grouping, only their individual lines are skipped,
        and a warning says how many)."""
        start, n = self.edge_log_window or (e.t_mono, 0)
        if e.t_mono - start >= 1.0:
            self._flush_not_logged()
            start, n = e.t_mono, 0
        if n >= self.max_edge_events:
            self.edges_not_logged += 1
            self.edges_not_logged_total += 1
            self.edge_log_window = (start, n)
            return
        self.edge_log_window = (start, n + 1)
        self.emit(dict(type="beam_edge", edge_no=self.edges_total, line=e.line, level=e.level, raw=e.raw,
                       t_edge=e.t_wall + self.offset, t_wall=round(e.t_wall, 6), t_mono=round(e.t_mono, 6),
                       poll_gap_ms=None if e.poll_gap_s is None else round(1000.0 * e.poll_gap_s, 3)))

    def _flush_not_logged(self):
        """One warning per 1 s window in which edges were not written one by one."""
        if self.edges_not_logged:
            self._warn("beam: %d edges in one second were not written one by one (more than %d/s)" % (
                self.edges_not_logged, self.max_edge_events))
            self.edges_not_logged = 0

    def _on_record(self, rec):
        if rec["kind"] == "break":
            ev = dict(type="beam_break", break_id=rec["break_id"], t_beam=rec["first_wall"] + self.offset,
                      t_beam_mono=round(rec["first_mono"], 6), t_last=rec["last_wall"] + self.offset,
                      t_last_mono=round(rec["last_mono"], 6), n_edges=rec["n_edges"], n_pulses=rec["n_pulses"],
                      duration_s=rec["duration_s"], started_blocked=rec["started_blocked"],
                      interrupted=rec["interrupted"])
            self.emit(ev)
            self.recent_breaks.append(ev)
            with self.lock:
                self.breaks_total += 1
                self.last_breaks.append(dict(break_id=ev["break_id"], t_beam=ev["t_beam"], n_edges=ev["n_edges"],
                                             duration_s=ev["duration_s"], pass_id=None))
        else:
            ev = dict(type="beam_stuck", state=rec["state"], t_since=rec["since_wall"] + self.offset,
                      since_mono=round(rec["since_mono"], 6), n_edges=rec["n_edges"],
                      started_blocked=rec["started_blocked"])
            if rec["state"] == "cleared":
                ev.update(duration_s=rec["duration_s"], reason=rec.get("reason", "beam clear again"))
            else:
                with self.lock:
                    self.stuck_episodes += 1
                self._warn("beam: blocked / flickering for more than %.0f s: not counted as cars until it clears "
                           "(someone in the beam? sensor misaligned? wrong polarity?)" % self.grouper.stuck_s)
            self.emit(ev)

    def _grouping_time(self, now_mono, reader_horizon):
        """How far the grouper may let time pass. Not simply "now": the reader may have stamped an edge a moment ago
        and not queued it yet; closing a break at "now" could then split one car into two breaks. So: at most
        now - advance_guard_s, and never past the reader's horizon (everything up to it is queued). A reader that
        has published nothing for over 1 s (port closed, PC stalled) cannot hold the grouping back forever."""
        t = now_mono - self.advance_guard
        if reader_horizon is not None and now_mono - reader_horizon < 1.0:
            t = min(t, reader_horizon)
        return t

    def _advance(self, reader_horizon=None):
        now_mono, now_wall = self.clock.mono(), self.clock.wall()
        if self.edge_log_window and now_mono - self.edge_log_window[0] >= 1.0:
            self._flush_not_logged()          # the noisy second is over even if no edge followed it
        t_group = self._grouping_time(now_mono, reader_horizon)
        for rec in self.grouper.advance(t_group, now_wall - (now_mono - t_group)):
            self._on_record(rec)
        self._associate_ready(now_wall + self.offset)
        self._match_led(now_mono)

    # ------------------------------------------------------------------ association
    def _associate_ready(self, beam_now):
        """Decide the pending passes whose window is over (all breaks that could match are closed). pass_end comes
        ~10 s after the car left, so in practice this is immediate."""
        still = []
        now_mono = self.clock.mono()
        for p, arrived_mono in self.pending_passes:
            if p.get("t_cross") is None:
                continue
            # A break that could still match must have started by t_cross - offset + window, and is closed
            # group_gap_s after its last edge. Wait for that, but never more than window + 5 s after the pass_end
            # arrived (so a wall-clock jump cannot keep a pass waiting forever).
            ready_at = p["t_cross"] - self.assoc_offset + self.assoc_window + self.grouper.gap + 0.05
            waiting = beam_now < ready_at or (self.grouper.open is not None and
                                              self.grouper.open["first_wall"] + self.offset <= ready_at)
            if waiting and now_mono - arrived_mono < self.assoc_window + 5.0:
                still.append((p, arrived_mono))
                continue
            self._emit_pass_beam(p)
        self.pending_passes = still

    def _emit_pass_beam(self, p):
        status, b, cands = associate_pass(p["t_cross"], self.recent_breaks, self.assoc_window, self.assoc_offset)
        ev = dict(type="pass_beam", pass_id=p["pass_id"], status=status, t_cross=p["t_cross"],
                  n_candidates=len(cands), candidates=[c["break_id"] for c in cands],
                  window_s=self.assoc_window, offset_s=self.assoc_offset)
        if b is not None:
            shared = list(self.break_passes.get(b["break_id"], []))
            ev.update(t_beam=b["t_beam"], break_id=b["break_id"],
                      beam_minus_cross_s=round(b["t_beam"] - p["t_cross"], 4), break_also_matched_to=shared)
            self.break_passes.setdefault(b["break_id"], []).append(p["pass_id"])
            with self.lock:
                for lb in self.last_breaks:
                    if lb["break_id"] == b["break_id"]:
                        lb["pass_id"] = p["pass_id"] if lb["pass_id"] is None else lb["pass_id"]
        with self.lock:
            self.assoc_counts[status] += 1
        self.emit(ev)

    # ------------------------------------------------------------------ LED latency
    def _match_led(self, now_mono):
        """Pair each LED off->on frame with the latest active beam edge before it (at most led_max_latency_s
        earlier). `clean` = that was the only active edge in the window (a multi-pulse car makes the pairing
        uncertain; use clean ones, and the minimum over many crossings, as the latency)."""
        still = []
        for t_cap, frame_t, idx, mean in self.pending_led:
            window = [e for e in self.recent_active_edges if 0.0 <= t_cap - e.t_mono <= self.led_max_latency]
            if not window:
                if now_mono - t_cap > self.led_max_latency + 0.5:   # the edge never came: give up on this frame
                    with self.lock:
                        self.led_stats["unmatched"] += 1
                else:
                    still.append((t_cap, frame_t, idx, mean))
                continue
            e = max(window, key=lambda x: x.t_mono)
            latency = t_cap - e.t_mono
            with self.lock:
                st = self.led_stats
                st["measurements"] += 1
                st["last_latency_s"] = round(latency, 4)
                st["min_latency_s"] = round(latency if st["min_latency_s"] is None else min(latency,
                                                                                         st["min_latency_s"]), 4)
            self.emit(dict(type="camera_latency", latency_s=round(latency, 4), frame_t=frame_t, frame_idx=idx,
                           frame_mono=round(t_cap, 6), t_edge=e.t_wall + self.offset, edge_mono=round(e.t_mono, 6),
                           residual_s=round(frame_t - (e.t_wall + self.offset), 4), clean=len(window) == 1,
                           roi_mean=round(mean, 1)))
        self.pending_led = still

    # ------------------------------------------------------------------ helpers
    def _set_state(self, state, detail=""):
        if (state, detail) == (self.state, self.detail):
            return
        with self.lock:
            self.state, self.detail, self.state_since = state, detail, self.clock.wall()
        self.emit(dict(type="beam", state=state, detail=detail, driver=self.source.driver, port=self.source.port))

    def _info(self, msg):
        self.emit(dict(type="info", msg=msg))

    def _warn(self, msg):
        self.emit(dict(type="warning", msg=msg))

    # ================================================================== status page
    def status(self):
        now = self.clock.mono()
        with self.lock:
            per_min = sum(1 for m in self.edge_monos if now - m <= 60.0)
            breaks = [dict(b, t_beam_iso=iso(b["t_beam"]), number=self.pass_numbers.get(b["pass_id"]))
                      for b in reversed(self.last_breaks)]
            out = dict(enabled=True, driver=self.source.driver, port=self.source.port, state=self.state,
                       detail=self.detail, since_iso=iso(self.state_since), connected=self.state == "open",
                       line=getattr(self.source, "line", "bytes" if self.source.driver == "byte-stream" else None),
                       level=self.level, edges_total=self.edges_total, edges_per_min=per_min,
                       edges_not_logged=self.edges_not_logged_total, breaks_total=self.breaks_total,
                       last_breaks=breaks, poll=dict(self.poll), stuck_episodes=self.stuck_episodes,
                       association=dict(self.assoc_counts, window_s=self.assoc_window, offset_s=self.assoc_offset),
                       led=dict(self.led_stats, enabled=self.led.roi is not None, roi=self.led.roi,
                                last_mean=None if self.led.last_mean is None else round(self.led.last_mean, 1)),
                       opens=self.opens)
        out.update(self.grouper.status(now))
        return out

    # ================================================================== threads
    def start(self):
        for name, target in (("beam-reader", self._reader_loop), ("beam-dispatch", self._dispatch_loop)):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self.threads.append(t)
        return self

    def stop(self, timeout=2.0):
        """Stop both threads, write what is still queued, finish an open break, close the port."""
        self.halt.set()
        for t in self.threads:
            t.join(timeout)
        try:
            self.source.close()
        except Exception:  # noqa: BLE001
            pass
        self.dispatch_step(timeout=0)
        for rec in self.grouper.disconnect(self.clock.mono(), self.clock.wall()):
            self._on_record(rec)
        for p, _ in self.pending_passes:
            self._emit_pass_beam(p)
        self.pending_passes = []
        self._set_state("stopped")


class PassBeamSink(Sink):
    """The rt/sinks.py sink that hands pass_end / pass_identity events to the BeamService (queue only)."""

    def __init__(self, service):
        self.service = service

    def event(self, ev):
        self.service.on_event(ev)
