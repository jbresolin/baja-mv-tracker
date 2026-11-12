"""Beam sources (WP-L2): the drivers that turn the timing sensor's signal into time-stamped edges.

An *edge* is a change of the beam signal: the beam got blocked (level 1, "active") or cleared (level 0). Each edge
is stamped, right after the driver sees it, on the SAME two clocks rt uses for camera frames (rt.backoff.SystemClock,
the clock object rt/livecam.py stamps grab() with):
  t_wall = time.time()        epoch seconds; what every rt event uses; can jump when NTP corrects it
  t_mono = time.monotonic()   never jumps; good for durations and for comparing with frame `t_cap`
So a beam edge and a frame can be compared directly (beam vs camera latency, beam vs vision crossing time).

Drivers (chosen by `beam.driver` in the config):
  * ModemLineSource ("modem-line"): the sensor drives one RS232 modem-status input of the USB adapter (CTS, DSR,
    DCD or RI). A thread polls that line ~1000 times a second and reports every change. `active` says which line
    state means "beam blocked" ("high" = line asserted, "low" = line not asserted).
  * ByteStreamSource ("byte-stream"): the adapter sends serial bytes for each pulse. Every receipt of bytes (or
    every occurrence of `byte_pattern`) is an edge, stamped on receipt; the raw bytes are kept in the edge.
  * FakeBeamSource ("fake"): edges from a script on an injected clock, for tests and dry runs without hardware.

All three share one small interface (BeamSource): open() -> info dict, poll() -> new edges (waits at most about one
poll period), close(). A missing or unplugged port raises BeamPortError; the BeamService (rt/beam/service.py)
catches it, shows it on the status page and reopens the port with backoff. Nothing here ever touches the detection
pipeline.
"""
import time
from dataclasses import dataclass, field

from ..backoff import SystemClock

MODEM_LINES = ("cts", "dsr", "cd", "ri")     # pyserial attribute names: Serial.cts, .dsr, .cd (= DCD), .ri


class BeamPortError(Exception):
    """The serial port cannot be opened, or it disappeared (USB adapter unplugged)."""


@dataclass
class RawEdge:
    """One change of the beam signal, as seen by a driver."""
    t_wall: float                 # time.time() right after the change was seen
    t_mono: float                 # time.monotonic() of the same instant
    line: str                     # "cts" / "dsr" / "cd" / "ri", "bytes" (byte-stream) or "fake"
    level: int                    # 1 = active (beam blocked / a pulse), 0 = inactive (beam clear)
    raw: object = None            # modem-line: the line as read (True = asserted); byte-stream: the bytes, as hex
    poll_gap_s: float = None      # time since the previous poll: the real change happened somewhere in this gap
    momentary: bool = False       # True = a pulse with no "release" (byte-stream without a release pattern):
                                  # the beam is not left "blocked" after it


class PollStats:
    """How regularly a driver polls: the achieved rate and the largest gap between two polls (jitter).

    note(t_mono) is called once per poll. report(now) returns the numbers since the last report and starts a new
    window. On Windows, Python 3.11+ sleeps with a high-resolution timer, so 1 kHz is reachable; a max gap of
    tens of ms means the PC was busy (or a USB hiccup) and an edge in that gap was stamped late by up to that much."""

    LATE_S = 0.005               # a gap above 5 ms counts as a "late" poll

    def __init__(self):
        self.window_start = None
        self.last = None
        self.count = 0
        self.max_gap = 0.0
        self.late = 0

    def note(self, t_mono):
        if self.window_start is None:
            self.window_start = t_mono
        if self.last is not None:
            gap = t_mono - self.last
            self.max_gap = max(self.max_gap, gap)
            if gap > self.LATE_S:
                self.late += 1
        self.last = t_mono
        self.count += 1

    def report(self, now):
        span = (now - self.window_start) if self.window_start is not None else 0.0
        out = dict(rate_hz=round(self.count / span, 1) if span > 0 else 0.0, polls=self.count,
                   mean_gap_ms=round(1000.0 * span / self.count, 3) if self.count else None,
                   max_gap_ms=round(1000.0 * self.max_gap, 3), late_polls=self.late, window_s=round(span, 3))
        self.window_start, self.count, self.max_gap, self.late = now, 0, 0.0, 0
        return out


def default_serial_factory(port, baud, timeout):
    """Open a real serial port with pyserial. Tests pass their own factory that returns a fake serial object."""
    try:
        import serial
    except ImportError as e:
        raise BeamPortError("pyserial is not installed (pip install pyserial)") from e
    return serial.Serial(port=port, baudrate=int(baud), timeout=timeout)


class BeamSource:
    """The interface every driver implements. `poll_stats` is read by the service for the status page."""
    driver = "base"
    port = ""

    def __init__(self):
        self.poll_stats = PollStats()

    def open(self):
        """Open the device. Returns a dict of what was found (shown in the log). Raises BeamPortError."""
        raise NotImplementedError

    def poll(self):
        """Wait at most about one poll period, return a list of RawEdge (often empty). Raises BeamPortError when
        the device is gone."""
        raise NotImplementedError

    def close(self):
        pass

    def level(self):
        """The current beam level (1 active, 0 inactive), or None when the driver cannot tell."""
        return None


# ------------------------------------------------------------------------------------------------ modem-line
class ModemLineSource(BeamSource):
    """The beam sensor drives one modem-status input (CTS/DSR/DCD/RI) of the serial adapter; poll it at `poll_hz`.

    The line is read, the clocks are read right after, and a change of level becomes an edge. Polling is on a fixed
    grid of due times (1 ms apart at 1 kHz), so the loop's own work does not slow the rate down; if the loop falls
    behind by more than one period (PC busy) the grid restarts instead of polling in a burst.

    DTR and RTS are output lines of the adapter. pyserial asserts both when it opens a port; they are set here from
    the config because some sensors / pull-ups are powered from them (see rt/README.md "Beam input")."""
    driver = "modem-line"

    def __init__(self, port, line="cts", active="high", poll_hz=1000.0, baud=9600, set_dtr=True, set_rts=True,
                 clock=None, sleep=None, serial_factory=None):
        super().__init__()
        line = str(line).lower()
        if line == "dcd":
            line = "cd"                       # pyserial calls DCD "cd"
        if line not in MODEM_LINES:
            raise ValueError("beam.line must be one of cts, dsr, cd (dcd), ri; got %r" % line)
        if str(active).lower() not in ("high", "low"):
            raise ValueError('beam.active must be "high" or "low"; got %r' % active)
        self.port, self.line, self.baud = str(port), line, baud
        self.active_high = str(active).lower() == "high"
        self.period = 1.0 / float(poll_hz)
        self.set_dtr, self.set_rts = bool(set_dtr), bool(set_rts)
        self.clock = clock or SystemClock()
        self.sleep = sleep or time.sleep
        self.serial_factory = serial_factory or default_serial_factory
        self.ser = None
        self.last_level = None
        self.last_poll_mono = None
        self.next_due = None

    def _read_line(self):
        """The line as the adapter reports it: True = asserted. Raises BeamPortError if the adapter is gone."""
        try:
            return bool(getattr(self.ser, self.line))
        except (OSError, ValueError, AttributeError) as e:   # pyserial: SerialException is an OSError
            raise BeamPortError("%s: reading %s failed: %s" % (self.port, self.line.upper(), e)) from e

    def _level_of(self, asserted):
        return int(asserted == self.active_high)

    def open(self):
        try:
            self.ser = self.serial_factory(self.port, self.baud, 0)
            self.ser.dtr = self.set_dtr
            self.ser.rts = self.set_rts
        except (OSError, ValueError) as e:
            self.close()
            raise BeamPortError("cannot open %s: %s" % (self.port, e)) from e
        asserted = self._read_line()
        self.last_level = self._level_of(asserted)
        self.last_poll_mono = self.clock.mono()
        self.next_due = self.last_poll_mono + self.period
        return dict(port=self.port, line=self.line, active="high" if self.active_high else "low",
                    line_asserted=asserted, level=self.last_level, poll_hz=round(1.0 / self.period, 1))

    def poll(self):
        # 1. Wait for the next due time on the grid.
        now = self.clock.mono()
        if self.next_due - now > 0:
            self.sleep(self.next_due - now)
        # 2. Read the line, then BOTH clocks immediately: this is the edge's time stamp.
        asserted = self._read_line()
        t_wall, t_mono = self.clock.wall(), self.clock.mono()
        gap = t_mono - self.last_poll_mono
        self.last_poll_mono = t_mono
        self.poll_stats.note(t_mono)
        # 3. Next due time; restart the grid after a stall (no burst of catch-up polls).
        self.next_due += self.period
        if t_mono - self.next_due > self.period:
            self.next_due = t_mono + self.period
        level = self._level_of(asserted)
        if level == self.last_level:
            return []
        self.last_level = level
        return [RawEdge(t_wall, t_mono, self.line, level, raw=asserted, poll_gap_s=gap)]

    def level(self):
        return self.last_level

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:  # noqa: BLE001 (closing an unplugged port can fail; it is gone anyway)
                pass
        self.ser = None


# ------------------------------------------------------------------------------------------------ byte-stream
def parse_hex(text):
    """'0D 0A' / '0d0a' / '' -> bytes. Config patterns are written in hex so any byte can be given."""
    text = "".join(str(text or "").split())
    return bytes.fromhex(text) if text else b""


class ByteStreamSource(BeamSource):
    """The adapter sends serial bytes for each pulse. The bytes are read as they arrive and stamped on receipt.

    * No `pattern`: every receipt of bytes is one edge (a "pulse": level 1, momentary). Bytes that arrive together
      in one read (one short message) are one edge, not one edge per byte.
    * With `pattern` (hex): every occurrence of it in the stream is an active edge. With `release_pattern` too,
      every occurrence of that is an inactive edge (the adapter reports both "blocked" and "clear"), and the beam
      level is known; without it, pattern edges are momentary pulses.
    The raw bytes of the read are kept in each edge (hex), so the JSONL shows exactly what the adapter sent."""
    driver = "byte-stream"

    def __init__(self, port, baud=9600, pattern="", release_pattern="", read_timeout_s=0.002, clock=None,
                 serial_factory=None):
        super().__init__()
        self.port, self.baud = str(port), int(baud)
        self.pattern, self.release = parse_hex(pattern), parse_hex(release_pattern)
        self.timeout = float(read_timeout_s)
        self.clock = clock or SystemClock()
        self.serial_factory = serial_factory or default_serial_factory
        self.ser = None
        self.buffer = b""             # unmatched tail kept between reads (a pattern can be split over two reads)
        self.last_level = None
        self.last_poll_mono = None
        self.bytes_total = 0

    def open(self):
        try:
            self.ser = self.serial_factory(self.port, self.baud, self.timeout)
        except (OSError, ValueError) as e:
            self.close()
            raise BeamPortError("cannot open %s at %d baud: %s" % (self.port, self.baud, e)) from e
        self.buffer = b""
        self.last_level = 0 if self.release else None
        self.last_poll_mono = self.clock.mono()
        return dict(port=self.port, baud=self.baud, pattern=self.pattern.hex(), release_pattern=self.release.hex())

    def poll(self):
        try:
            data = self.ser.read(1)                      # waits at most read_timeout_s for the first byte
            t_wall, t_mono = self.clock.wall(), self.clock.mono()   # stamp on receipt
            if data:
                waiting = self.ser.in_waiting
                if waiting:
                    data += self.ser.read(waiting)       # the rest of the message, already in the buffer
        except (OSError, ValueError, AttributeError) as e:
            raise BeamPortError("%s: read failed: %s" % (self.port, e)) from e
        gap = t_mono - self.last_poll_mono
        self.last_poll_mono = t_mono
        self.poll_stats.note(t_mono)
        if not data:
            return []
        self.bytes_total += len(data)
        raw = data.hex()
        if not self.pattern:
            return [RawEdge(t_wall, t_mono, "bytes", 1, raw=raw, poll_gap_s=gap, momentary=True)]
        return [RawEdge(t_wall, t_mono, "bytes", level, raw=raw, poll_gap_s=gap, momentary=not self.release)
                for level in self._match(data)]

    def _match(self, data):
        """Levels (1 for pattern, 0 for release_pattern) of every complete occurrence in the stream, in order."""
        self.buffer += data
        found = []
        i = 0
        while True:
            hits = [(self.buffer.find(p, i), p, lvl) for p, lvl in ((self.pattern, 1), (self.release, 0)) if p]
            hits = [h for h in hits if h[0] >= 0]
            if not hits:
                break
            pos, p, lvl = min(hits, key=lambda h: h[0])
            found.append(lvl)
            i = pos + len(p)
        keep = max(len(self.pattern), len(self.release)) - 1   # a pattern may continue in the next read
        rest = self.buffer[i:]
        self.buffer = rest[-keep:] if keep > 0 else b""
        if found and self.release:
            self.last_level = found[-1]
        return found

    def level(self):
        return self.last_level

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:  # noqa: BLE001
                pass
        self.ser = None


# ------------------------------------------------------------------------------------------------ fake
@dataclass
class FakeFailures:
    """Problems a FakeBeamSource can pretend to have (tests): the first `open_fails` opens fail (port missing);
    `unplugged` = True makes the next poll fail (adapter pulled out)."""
    open_fails: int = 0
    unplugged: bool = False
    opens: int = field(default=0)


class FakeBeamSource(BeamSource):
    """Edges from a script, on an injected clock: for tests and for a dry run of the whole chain without hardware.

    `script` is an iterable of (t_mono, level) in time order (a list in tests, an endless generator in dry runs:
    see synthetic_cars()). poll() returns every scripted edge whose time has come (t_mono <= clock.mono()), each
    stamped with its scripted monotonic time and the matching wall time. `sleep` is called for one poll period
    (tests pass a fake that moves the fake clock instead of sleeping)."""
    driver = "fake"

    def __init__(self, script, clock=None, sleep=None, poll_s=0.001, initial_level=0, failures=None, port="fake"):
        super().__init__()
        self.script = iter(script)
        self.clock = clock or SystemClock()
        self.sleep = sleep or time.sleep
        self.poll_s = float(poll_s)
        self.failures = failures or FakeFailures()
        self.port = port
        self.initial_level = int(initial_level)
        self.current = None
        self.pending = None            # the next scripted edge, not due yet
        self.last_poll_mono = None

    def open(self):
        self.failures.opens += 1
        if self.failures.open_fails > 0:
            self.failures.open_fails -= 1
            raise BeamPortError("cannot open fake: port not found (simulated)")
        self.failures.unplugged = False
        if self.current is None:
            self.current = self.initial_level
        self.last_poll_mono = self.clock.mono()
        return dict(port=self.port, level=self.current)

    def poll(self):
        if self.failures.unplugged:
            raise BeamPortError("fake: device disconnected (simulated)")
        self.sleep(self.poll_s)
        now = self.clock.mono()
        self.poll_stats.note(now)
        wall_minus_mono = self.clock.wall() - now
        edges = []
        while True:
            if self.pending is None:
                self.pending = next(self.script, None)
                if self.pending is None:
                    break
            t, level = self.pending
            if t > now:
                break
            self.pending = None
            gap = t - self.last_poll_mono if self.last_poll_mono is not None else None
            edges.append(RawEdge(t + wall_minus_mono, t, "fake", int(level), raw=int(level), poll_gap_s=gap))
            self.current = int(level)
        self.last_poll_mono = now
        return edges

    def level(self):
        return self.current


def synthetic_cars(start_mono, every_s=20.0, pulses=3, pulse_s=0.03, pulse_gap_s=0.05):
    """An endless script for the fake driver: one "car" every `every_s` seconds, each blocking the beam `pulses`
    times (a tube-frame car: the beam sees through the gaps between tubes). Used by driver = "fake" dry runs."""
    t = float(start_mono) + every_s
    while True:
        for k in range(int(pulses)):
            on = t + k * (pulse_s + pulse_gap_s)
            yield (on, 1)
            yield (on + pulse_s, 0)
        t += every_s


def make_source(cfg, clock=None, sleep=None, serial_factory=None):
    """The driver named by beam.driver, built from the [beam] config. Returns None for driver = "off"."""
    b = cfg["beam"]
    driver = str(b.get("driver", "off")).lower()
    if driver == "off":
        return None
    if driver == "modem-line":
        return ModemLineSource(b["port"], b.get("line", "cts"), b.get("active", "high"), float(b.get("poll_hz", 1000)),
                               int(b.get("baud", 9600)), bool(b.get("set_dtr", True)), bool(b.get("set_rts", True)),
                               clock=clock, sleep=sleep, serial_factory=serial_factory)
    if driver == "byte-stream":
        return ByteStreamSource(b["port"], int(b.get("baud", 9600)), b.get("byte_pattern", ""),
                                b.get("byte_release_pattern", ""), float(b.get("read_timeout_s", 0.002)),
                                clock=clock, serial_factory=serial_factory)
    if driver == "fake":
        clock = clock or SystemClock()
        return FakeBeamSource(synthetic_cars(clock.mono(), float(b.get("fake_every_s", 20)),
                                             int(b.get("fake_pulses", 3))), clock=clock, sleep=sleep)
    raise ValueError('beam.driver must be "off", "modem-line", "byte-stream" or "fake"; got %r' % driver)
