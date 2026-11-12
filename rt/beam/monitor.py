"""Beam monitor (on-site setup): a built-in probe that runs INSIDE rt, for the status page's Beam setup panel.

The owner only gets the sensor and the USB adapter at the event, and sets the beam up from his phone. While the setup
panel is in "monitor" mode, this opens the chosen COM port (the beam driver is stopped meanwhile: only one program
can hold a COM port) and records, like rt/tools/beam_probe.py does:
  * the four modem input lines (CTS, DSR, DCD, RI): their current state and every change, with times;
  * every message of bytes received (bytes less than 3 character times apart = one message), in hex, with times.
The page shows the lines (refreshing ~5 times a second) and the last messages; "Suggest" runs the probe's guess
(rt.tools.beam_probe.guess_config) on everything seen since the monitor started (or since "clear").

It reads the port in its own thread at ~500 Hz, never touches the detection pipeline, and survives a missing or
unplugged adapter (it shows the error and tries again every second). step() does one round of work, so the tests
call it directly with a fake serial port and a fake clock.
"""
import collections
import copy
import threading
import time

from ..backoff import SystemClock
from ..tools.beam_probe import ProbeLog, guess_config, read_lines
from .sources import MODEM_LINES, MessageAssembler, default_serial_factory, merge_gap_s, windows_modem_status_ok

LINE_LABELS = dict(cts="CTS", dsr="DSR", cd="DCD", ri="RI")


class BeamMonitor:
    def __init__(self, port, baud=9600, set_dtr=True, set_rts=True, clock=None, sleep=None, serial_factory=None,
                 poll_hz=500.0, max_messages=40, max_changes=40, max_logged_chunks=5000,
                 modem_status_check=None):
        self.port, self.baud = str(port), int(baud)
        self.set_dtr, self.set_rts = bool(set_dtr), bool(set_rts)
        self.clock = clock or SystemClock()
        self.sleep = sleep or time.sleep
        self.serial_factory = serial_factory or default_serial_factory
        self.modem_status_check = modem_status_check or windows_modem_status_ok
        self.period = 1.0 / float(poll_hz)
        self.max_logged_chunks = int(max_logged_chunks)
        self.lock = threading.Lock()          # guards everything the page reads (snapshot / suggest)
        self.halt = threading.Event()
        self.thread = None
        self.ser = None
        self.state, self.detail = "opening", ""
        self.next_open_mono = None
        self.lines_now = {}                   # line -> True (asserted) / False
        self.messages = collections.deque(maxlen=int(max_messages))   # newest last: dict(t_wall, hex, text)
        self.changes = collections.deque(maxlen=int(max_changes))     # newest last: dict(t_wall, line, asserted)
        self.n_messages = 0
        self.assembler = MessageAssembler(merge_gap_s(self.baud))
        self.log = None                       # ProbeLog since the monitor started (or since clear())
        self.mono_to_wall = 0.0

    # ------------------------------------------------------------------ the reading loop
    def step(self):
        """One round: open the port if needed, else read the lines and the bytes once. Never raises."""
        if self.ser is None:
            now = self.clock.mono()
            if self.next_open_mono is not None and now < self.next_open_mono:
                self.sleep(min(0.1, self.next_open_mono - now))
                return
            self._open()
            return
        try:
            states = read_lines(self.ser)
            waiting = self.ser.in_waiting            # fails when the adapter is gone (Windows: the lines do not)
            chunk = self.ser.read(waiting) if waiting else b""
            if self.modem_status_check(self.ser) is False:
                raise OSError("GetCommModemStatus failed")
        except Exception as e:  # noqa: BLE001 (unplugged, or anything else: show it, retry)
            self._lost("port lost (adapter unplugged?): %s" % e)
            return
        t_mono = self.clock.mono()
        t_wall = self.clock.wall()
        self.mono_to_wall = t_wall - t_mono
        with self.lock:
            for line in self.log.lines(t_mono, states):
                self.changes.append(dict(t_wall=t_wall, line=LINE_LABELS[line], asserted=states[line]))
            self.lines_now = states
            self._record(self.assembler.feed(t_mono, chunk))
        self.sleep(self.period)

    def _open(self):
        try:
            ser = self.serial_factory(self.port, self.baud, 0)
            ser.dtr = self.set_dtr
            ser.rts = self.set_rts
            states = read_lines(ser)
        except Exception as e:  # noqa: BLE001
            self._set_state("retrying", "cannot open %s: %s" % (self.port, e))
            self.next_open_mono = self.clock.mono() + 1.0
            return
        self.ser = ser
        t_mono = self.clock.mono()
        with self.lock:
            self.assembler = MessageAssembler(merge_gap_s(self.baud))
            if self.log is None:
                self.log = ProbeLog(t_mono)
            self.log.lines(t_mono, states)
            self.lines_now = states
        self._set_state("open", "%s at %d baud" % (self.port, self.baud))

    def _lost(self, why):
        self._close_port()
        with self.lock:
            self.lines_now = {}
        self._set_state("lost", why + "; trying again every second")
        self.next_open_mono = self.clock.mono() + 1.0

    def _record(self, messages):
        """Called under the lock with finished byte messages: keep them for the page and for Suggest."""
        for t_msg, msg in messages:
            self.n_messages += 1
            self.log.data(t_msg, msg)
            if len(self.log.chunks) > self.max_logged_chunks:          # bounded memory, however long the setup
                del self.log.chunks[: len(self.log.chunks) // 2]
            self.messages.append(dict(t_wall=t_msg + self.mono_to_wall, hex=msg.hex(" "),
                                      text=msg.decode("latin-1").encode("unicode_escape").decode("ascii")))

    def _set_state(self, state, detail):
        with self.lock:
            self.state, self.detail = state, detail

    def _close_port(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:  # noqa: BLE001 (an unplugged port can fail to close; it is gone anyway)
                pass
        self.ser = None

    # ------------------------------------------------------------------ what the page asks
    def clear(self):
        """Forget what was seen (start a fresh observation for Suggest)."""
        with self.lock:
            self.log = ProbeLog(self.clock.mono())
            if self.lines_now:
                self.log.lines(self.clock.mono(), dict(self.lines_now))
            self.messages.clear()
            self.changes.clear()
            self.n_messages = 0

    def snapshot(self):
        """The live view for the page: state, the four lines now, recent changes and messages (newest first)."""
        with self.lock:
            return dict(port=self.port, baud=self.baud, state=self.state, detail=self.detail,
                        set_dtr=self.set_dtr, set_rts=self.set_rts,
                        lines={LINE_LABELS[k]: self.lines_now.get(k) for k in MODEM_LINES},
                        changes=list(reversed(self.changes)), messages=list(reversed(self.messages)),
                        n_messages=self.n_messages,
                        transitions={LINE_LABELS[k]: (self.log.transitions[k] if self.log else 0)
                                     for k in MODEM_LINES})

    def suggest(self):
        """The probe's guess on what was seen: dict(driver, settings (dict for the form), config (TOML lines),
        notes). Works on a copy, so the monitor keeps running."""
        with self.lock:
            if self.log is None:
                return dict(driver=None, settings={}, config=[], notes=["the port has not been opened yet"])
            log = copy.deepcopy(self.log)
        log.finish(self.clock.mono())
        guess = guess_config(log.summary(), self.baud)
        settings = {}
        for line in guess["config"]:
            key, _, value = line.partition("=")
            settings[key.strip()] = value.strip().strip('"')
        if "baud" in settings:
            settings["baud"] = int(settings["baud"])
        if guess["driver"]:
            settings["port"] = self.port
        return dict(guess, settings=settings)

    # ------------------------------------------------------------------ thread
    def start(self):
        self.thread = threading.Thread(target=self._loop, name="beam-monitor", daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while not self.halt.is_set():
            self.step()

    def stop(self, timeout=2.0):
        self.halt.set()
        if self.thread is not None:
            self.thread.join(timeout)
        self._close_port()
        self._set_state("stopped", "")
