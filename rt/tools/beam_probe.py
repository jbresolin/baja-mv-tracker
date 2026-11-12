"""Beam probe (WP-L2): find out how the USB adapter delivers the timing sensor's pulse, before configuring rt.

usage (from the repo folder, with the adapter plugged in):
  python -m rt.tools.beam_probe --list                      which serial ports exist (CP2102 = USB id 10C4:EA60)
  python -m rt.tools.beam_probe --port COM5                 watch COM5 at 9600 baud until Ctrl+C
  python -m rt.tools.beam_probe --port COM5 --baud 115200 --seconds 60

While it runs, break the beam several times (wave a hand / walk through it, slowly and quickly). It prints, live and
time-stamped:
  * every change of the four modem input lines (CTS, DSR, DCD, RI), read ~1000 times a second;
  * every chunk of bytes received, in hex and as text.
At the end (Ctrl+C or --seconds) it prints a summary and its GUESS of the [beam] config to use: driver (modem-line or
byte-stream), the line, the polarity (`active`), or the byte pattern. Paste that into rt/configs/lapline.toml.

The guess is only as good as what happened during the probe: break the beam at least 3 times. rt itself does not need
to be running (and must NOT be running: only one program can open a COM port).
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rt.beam.sources import MODEM_LINES, default_serial_factory  # noqa: E402

LINE_NAMES = dict(cts="CTS", dsr="DSR", cd="DCD", ri="RI")


class ProbeLog:
    """Everything seen during the probe, and the summary the guess is made from. Pure bookkeeping (no I/O), so the
    tests can feed it by hand."""

    def __init__(self, t_start):
        self.t_start = t_start
        self.initial = {}                                  # line -> first state read (True = asserted)
        self.state = {}                                    # line -> current state
        self.changed_at = {}                               # line -> time of the last change
        self.transitions = Counter()                       # line -> number of changes
        self.asserted_s = Counter()                        # line -> total time spent asserted
        self.chunks = []                                   # (time, bytes) of every receipt
        self.t_end = t_start

    def lines(self, t, states):
        """Record one reading of the four lines (dict line -> bool). Returns the list of lines that changed."""
        changed = []
        for line, v in states.items():
            if line not in self.state:
                self.initial[line] = self.state[line] = v
                self.changed_at[line] = t
                continue
            if v != self.state[line]:
                if self.state[line]:
                    self.asserted_s[line] += t - self.changed_at[line]
                self.state[line] = v
                self.changed_at[line] = t
                self.transitions[line] += 1
                changed.append(line)
        self.t_end = t
        return changed

    def data(self, t, chunk):
        self.chunks.append((t, bytes(chunk)))
        self.t_end = t

    def finish(self, t):
        """Close the time accounting at the end of the probe."""
        for line, v in self.state.items():
            if v:
                self.asserted_s[line] += t - self.changed_at[line]
                self.changed_at[line] = t
        self.t_end = t

    def summary(self):
        dur = max(1e-9, self.t_end - self.t_start)
        lines = {line: dict(initial=self.initial.get(line), transitions=self.transitions[line],
                            asserted_share=round(self.asserted_s[line] / dur, 3)) for line in self.state}
        msgs = Counter(c for _, c in self.chunks)
        return dict(duration_s=round(dur, 3), lines=lines, n_chunks=len(self.chunks),
                    n_bytes=sum(len(c) for _, c in self.chunks),
                    messages=[(m.hex(), n) for m, n in msgs.most_common(5)],
                    order=[c.hex() for _, c in self.chunks[:50]])


def guess_config(summary, baud=9600):
    """From a probe summary, the most likely [beam] config. Returns dict(driver=..., config=[TOML lines],
    notes=[plain-language remarks]). driver None = nothing usable was seen."""
    notes = []
    toggled = {line: s for line, s in summary["lines"].items() if s["transitions"] >= 2}
    if toggled:
        order = {line: i for i, line in enumerate(MODEM_LINES)}
        line = max(toggled, key=lambda k: (toggled[k]["transitions"], -order.get(k, 9)))
        s = toggled[line]
        # The idle state is where the line spends most of its time; "active" (beam blocked) is the other one.
        idle_asserted = s["asserted_share"] > 0.5
        active = "low" if idle_asserted else "high"
        breaks = s["transitions"] // 2
        notes.append("%s changed %d times (~%d beam breaks); idle it is %s, so a break makes it %s -> active = \"%s\""
                     % (LINE_NAMES[line], s["transitions"], breaks, "asserted" if idle_asserted else "not asserted",
                        "drop" if idle_asserted else "rise", active))
        others = [LINE_NAMES[k] for k in toggled if k != line]
        if others:
            notes.append("other lines changed too (%s): check the wiring; the busiest line was picked" %
                         ", ".join(others))
        if summary["n_chunks"]:
            notes.append("bytes were received as well (%d chunks); modem-line is preferred (no serial framing, "
                         "stamped at the line change)" % summary["n_chunks"])
        if breaks < 3:
            notes.append("fewer than 3 breaks seen: run the probe again and break the beam a few more times")
        return dict(driver="modem-line", config=['driver = "modem-line"', 'line = "%s"' % line,
                                                  'active = "%s"' % active], notes=notes)
    if summary["n_chunks"]:
        msgs = summary["messages"]
        cfg = ['driver = "byte-stream"', "baud = %d" % baud]
        distinct = [m for m, _ in msgs]
        all_bytes = bytes.fromhex("".join(summary["order"]))
        junk = sum(b in (0x00, 0xFF) for b in all_bytes) / max(1, len(all_bytes))
        if junk > 0.5:
            notes.append("most bytes are 00 / FF: often the wrong baud rate (or the sensor is on the RX pin as a "
                         "plain level). Try the probe with --baud 115200, 19200, 4800, 1200")
        if len(distinct) == 1:
            cfg.append('byte_pattern = "%s"' % distinct[0])
            notes.append("one message type (%s) seen %d times: each one = a pulse" % (distinct[0], msgs[0][1]))
        elif len(distinct) == 2 and _alternates(summary["order"]):
            first = summary["order"][0]
            second = distinct[1] if distinct[0] == first else distinct[0]
            cfg += ['byte_pattern = "%s"' % first, 'byte_release_pattern = "%s"' % second]
            notes.append("two messages alternate (%s, %s): taken as blocked / clear" % (first, second))
        else:
            cfg.append('byte_pattern = ""')
            notes.append("several different messages: every receipt of bytes is taken as a pulse; look at the hex "
                         "above and set byte_pattern if one message clearly marks the beam break")
        return dict(driver="byte-stream", config=cfg, notes=notes)
    stuck = [LINE_NAMES[k] for k, s in summary["lines"].items() if s["initial"]]
    notes.append("nothing changed and no bytes arrived. Break the beam during the probe; check the port (--list), "
                 "the wiring and the sensor's power")
    if stuck:
        notes.append("asserted the whole time: %s (a sensor output that never changes, or a line driven by the "
                     "adapter itself)" % ", ".join(stuck))
    return dict(driver=None, config=[], notes=notes)


def _alternates(order):
    return all(order[i] != order[i + 1] for i in range(len(order) - 1))


def read_lines(ser):
    return {line: bool(getattr(ser, line)) for line in MODEM_LINES}


def run_probe(ser, seconds=None, poll_hz=1000.0, clock=time.monotonic, wall=time.time, sleep=time.sleep, out=print,
              stop=lambda: False):
    """Watch an open serial object (pyserial, or a fake in tests) until `seconds` pass, Ctrl+C or stop() is true.
    Prints every change; returns the ProbeLog."""
    t0 = clock()
    log = ProbeLog(t0)
    first = read_lines(ser)
    log.lines(t0, first)
    out("start %s  lines: %s" % (_hms(wall()), "  ".join("%s=%d" % (LINE_NAMES[k], v) for k, v in first.items())))
    period = 1.0 / float(poll_hz)
    try:
        while not stop():
            now = clock()
            if seconds is not None and now - t0 >= seconds:
                break
            states = read_lines(ser)
            t = clock()
            for line in log.lines(t, states):
                out("%s  +%9.3f s  %-3s -> %s" % (_hms(wall()), t - t0, LINE_NAMES[line],
                                                   "asserted (1)" if states[line] else "not asserted (0)"))
            waiting = ser.in_waiting
            if waiting:
                chunk = ser.read(waiting)
                t = clock()
                log.data(t, chunk)
                out("%s  +%9.3f s  bytes %s  %r" % (_hms(wall()), t - t0, chunk.hex(" "), chunk))
            sleep(period)
    except KeyboardInterrupt:
        out("(stopped)")
    log.finish(clock())
    return log


def _hms(t):
    return time.strftime("%H:%M:%S", time.localtime(t)) + (".%03d" % int((t % 1) * 1000))


def report(log, baud, out=print):
    s = log.summary()
    out("\n=== summary (%.1f s) ===" % s["duration_s"])
    for line, d in s["lines"].items():
        out("  %-3s  start %s, %d changes, asserted %.0f%% of the time" % (
            LINE_NAMES[line], "asserted" if d["initial"] else "not asserted", d["transitions"],
            100 * d["asserted_share"]))
    out("  bytes: %d in %d chunks; most common: %s" % (s["n_bytes"], s["n_chunks"],
                                                        ", ".join("%s x%d" % m for m in s["messages"]) or "-"))
    g = guess_config(s, baud)
    out("\n=== guess ===")
    for n in g["notes"]:
        out("  * " + n)
    if g["driver"]:
        out("\nPut this in rt/configs/lapline.toml (plus port = \"...\"):")
        out("[beam]")
        for line in g["config"]:
            out(line)
    return g


def list_ports(out=print):
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        out("pyserial is not installed: pip install pyserial")
        return
    ports = list(lp.comports())
    if not ports:
        out("no serial ports found")
    for p in ports:
        vid = "%04X:%04X" % (p.vid, p.pid) if p.vid is not None else "-"
        tag = "  <- Silicon Labs CP210x" if (p.vid, p.pid) == (0x10C4, 0xEA60) else ""
        out("%-8s %s  [%s]%s" % (p.device, p.description, vid, tag))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port, e.g. COM5 (Windows) or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long (default: until Ctrl+C)")
    ap.add_argument("--list", action="store_true", help="list the serial ports and exit")
    a = ap.parse_args(argv)
    if a.list or not a.port:
        list_ports()
        if not a.port:
            print("\nthen: python -m rt.tools.beam_probe --port <port>")
        return
    try:
        ser = default_serial_factory(a.port, a.baud, 0)
    except Exception as e:  # noqa: BLE001 (pyserial missing, port not found, port busy: say it plainly)
        raise SystemExit("cannot open %s: %s\n(is rt or another program using it? python -m rt.tools.beam_probe "
                         "--list shows the ports)" % (a.port, e))
    try:
        print("watching %s at %d baud: break the beam a few times; Ctrl+C to finish" % (a.port, a.baud))
        log = run_probe(ser, a.seconds)
    finally:
        ser.close()
    report(log, a.baud)


if __name__ == "__main__":
    main()
