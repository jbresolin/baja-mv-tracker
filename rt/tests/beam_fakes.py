"""Fakes for the beam tests: a pretend serial port and a pretend clock that moves only when the code "sleeps".

Why fakes: the real beam input needs a USB serial adapter (a CP2102) and a light-beam sensor, and real time. The
tests must run anywhere, fast, with nothing plugged in. So:

  * FakeSerial behaves like a pyserial `Serial` object: it has the four modem input lines (.cts .dsr .cd .ri) that
    a test can switch on and off, a byte buffer that a test can fill (as if the adapter had sent bytes), and it can
    be "unplugged". Unplugged, it behaves like REAL pyserial on Windows: reading a modem line does NOT fail, it just
    says "not asserted" (False), because pyserial ignores the error of the Windows call behind it; only
    `in_waiting` and `read()` fail (with an OSError, like pyserial's SerialException). That is exactly the trap the
    driver has to handle. `lines_raise_when_unplugged=True` gives the Linux behaviour (the line read fails too).
  * SerialPortWorld is "the PC's USB port": it decides whether COM5 exists right now, and hands out the FakeSerial
    when the code opens the port (this is the `serial_factory` the drivers accept instead of pyserial).
  * TickingClock is rt's FakeClock (rt/backoff.py) plus a sleep() that advances the clock instead of waiting. So a
    driver that "sleeps 1 ms" between polls makes fake time move by exactly 1 ms, and a test that runs 5 seconds of
    beam activity takes a fraction of a real second.
"""
from rt.backoff import FakeClock


class FakeSerial:
    def __init__(self, port, baud, timeout, clock=None, lines_raise_when_unplugged=False):
        self.port, self.baudrate, self.timeout = port, baud, timeout
        self.clock = clock
        self.lines_raise_when_unplugged = lines_raise_when_unplugged
        self.lines = dict(cts=False, dsr=False, cd=False, ri=False)   # False = not asserted
        self.incoming = bytearray()           # bytes "sent by the adapter", not read yet
        self.unplugged = False
        self.closed = False
        self.dtr = self.rts = None            # the code sets these (outputs of the adapter)

    def _check(self):
        if self.unplugged:
            raise OSError("ClearCommError failed (PermissionError(13, 'The device does not recognize the command.'))")

    def _line(self, name):
        """One modem input line. Unplugged on Windows: False, no error (see the module comment)."""
        if self.unplugged:
            if self.lines_raise_when_unplugged:
                self._check()
            return False
        return self.lines[name]

    # The four modem input lines, as pyserial names them.
    @property
    def cts(self):
        return self._line("cts")

    @property
    def dsr(self):
        return self._line("dsr")

    @property
    def cd(self):
        return self._line("cd")

    @property
    def ri(self):
        return self._line("ri")

    @property
    def in_waiting(self):
        self._check()
        return len(self.incoming)

    def read(self, n=1):
        """Like pyserial: up to n bytes; when nothing is waiting, wait `timeout` (fake time) and return b""."""
        self._check()
        if not self.incoming:
            if self.clock is not None and self.timeout:
                self.clock.advance(self.timeout)
            return b""
        out = bytes(self.incoming[:n])
        del self.incoming[:n]
        return out

    def close(self):
        self.closed = True


class SerialPortWorld:
    """The PC's serial ports, faked. `present` = is the adapter plugged in. factory() is what the drivers call to
    open a port; it fails like pyserial does when the port does not exist."""

    def __init__(self, clock=None, present=True, lines=None, lines_raise_when_unplugged=False):
        self.clock = clock
        self.lines_raise_when_unplugged = lines_raise_when_unplugged
        self.present = present
        self.initial_lines = dict(lines or {})   # line states of a freshly opened port, e.g. {"cts": True}
        self.opened = []                      # every FakeSerial handed out, newest last

    def factory(self, port, baud, timeout):
        if not self.present:
            raise OSError("could not open port '%s': FileNotFoundError(2, 'The system cannot find the file "
                          "specified.', None, 2)" % port)
        s = FakeSerial(port, baud, timeout, self.clock, self.lines_raise_when_unplugged)
        s.lines.update(self.initial_lines)
        self.opened.append(s)
        return s

    @property
    def serial(self):
        return self.opened[-1]

    def unplug(self):
        """Pull the adapter out: the open port starts failing, and opening it again fails until plug_in()."""
        self.present = False
        if self.opened:
            self.opened[-1].unplugged = True

    def plug_in(self):
        self.present = True


class TickingClock(FakeClock):
    """FakeClock whose sleep() moves time forward instead of waiting. `sleeps` keeps every requested sleep."""

    def __init__(self, wall=1_800_000_000.0, mono=1000.0):
        super().__init__(wall, mono)
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.advance(max(0.0, seconds))
