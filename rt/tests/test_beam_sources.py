"""Tests of the beam drivers (rt/beam/sources.py): how the sensor's signal becomes time-stamped "edges".

Background for a beginner:
  * The timing sensor is a light beam across the finish line. When a car is in the beam, the sensor's output changes.
  * That output reaches the laptop through a USB serial adapter. Either it switches one of the adapter's four
    "modem input lines" (CTS, DSR, DCD, RI: wires that are either asserted = on, or not), or the adapter sends
    some bytes. We do not know which yet, so there is a driver for each.
  * An EDGE is a change: beam becomes blocked (level 1, "active") or clear again (level 0).
  * "Polarity" (`active = "high"` or `"low"`) says whether "blocked" is the line ON or the line OFF. Sensors exist
    both ways, so it is a setting.
  * Every edge gets two time stamps, read right after the change is seen: the wall clock (time.time()) and the
    monotonic clock (time.monotonic()). rt stamps camera frames with exactly these two clocks.

No real serial port is used: FakeSerial (rt/tests/beam_fakes.py) pretends to be one, and the clock is fake too.
"""
import pytest

from rt.beam.sources import (BeamPortError, ByteStreamSource, FakeBeamSource, FakeFailures, ModemLineSource,
                             PollStats, make_source, parse_hex, synthetic_cars)

from .beam_fakes import SerialPortWorld, TickingClock


def modem(world, clock, line="cts", active="high", poll_hz=1000):
    """A modem-line driver wired to the fake port and the fake clock."""
    return ModemLineSource("COM5", line, active, poll_hz, clock=clock, sleep=clock.sleep, serial_factory=world.factory)


# ------------------------------------------------------------------------------------------------ modem-line
def test_active_high_line_on_means_beam_blocked():
    # active = "high": the line going ON (asserted) means the beam is blocked (level 1); going OFF means clear (0).
    clock = TickingClock()
    world = SerialPortWorld(clock)
    src = modem(world, clock, active="high")
    info = src.open()
    assert info["level"] == 0                       # line is off at the start: beam clear
    assert src.poll() == []                         # nothing changed: no edge
    world.serial.lines["cts"] = True                # a car cuts the beam
    (e,) = src.poll()
    assert (e.line, e.level, e.raw) == ("cts", 1, True)
    world.serial.lines["cts"] = False               # beam clear again
    (e,) = src.poll()
    assert (e.level, e.raw) == (0, False)


def test_active_low_line_off_means_beam_blocked():
    # active = "low": the opposite. Many sensors pull the line DOWN (off) when the beam is broken.
    clock = TickingClock()
    world = SerialPortWorld(clock, lines={"cts": True})   # idle: line on = beam clear
    src = modem(world, clock, active="low")
    assert src.open()["level"] == 0
    world.serial.lines["cts"] = False                     # a car: the line drops
    (e,) = src.poll()
    assert (e.level, e.raw) == (1, False)
    world.serial.lines["cts"] = True                      # clear again
    (e,) = src.poll()
    assert (e.level, e.raw) == (0, True)


@pytest.mark.parametrize("line", ["cts", "dsr", "cd", "ri"])
def test_each_of_the_four_lines_can_be_used(line):
    # The probe tells which line the adapter uses; every one of them must work, and only that one counts.
    clock = TickingClock()
    world = SerialPortWorld(clock)
    src = modem(world, clock, line=line)
    src.open()
    other = "dsr" if line != "dsr" else "cts"
    world.serial.lines[other] = True                # a change on ANOTHER line: ignored
    assert src.poll() == []
    world.serial.lines[line] = True
    (e,) = src.poll()
    assert e.line == line and e.level == 1


def test_dcd_is_accepted_as_a_name_and_bad_settings_are_refused():
    clock = TickingClock()
    world = SerialPortWorld(clock)
    assert modem(world, clock, line="DCD").line == "cd"     # pyserial calls DCD "cd"; both spellings work
    with pytest.raises(ValueError):
        modem(world, clock, line="txd")                     # not an input line
    with pytest.raises(ValueError):
        modem(world, clock, active="sideways")


def test_edge_is_stamped_with_both_clocks_right_after_the_read():
    # The fake clock's wall and monotonic times start at different values (1.8e9 and 1000) on purpose: the edge
    # must carry each one in the right field, read at the same instant.
    clock = TickingClock(wall=1_800_000_000.0, mono=1000.0)
    world = SerialPortWorld(clock)
    src = modem(world, clock)
    src.open()
    for _ in range(10):
        src.poll()                                  # 10 quiet polls = 10 ms of fake time
    world.serial.lines["cts"] = True
    (e,) = src.poll()
    assert e.t_mono == pytest.approx(clock.mono())
    assert e.t_wall == pytest.approx(clock.wall())
    assert e.t_wall - e.t_mono == pytest.approx(1_800_000_000.0 - 1000.0)
    # The real change happened somewhere since the previous poll: that gap (1 ms) is recorded as the uncertainty.
    assert e.poll_gap_s == pytest.approx(0.001)


def test_polls_on_a_1_ms_grid_and_measures_the_rate():
    # At poll_hz = 1000 the driver should read the line every 1 ms. PollStats reports the achieved rate.
    clock = TickingClock()
    world = SerialPortWorld(clock)
    src = modem(world, clock, poll_hz=1000)
    src.open()
    start = clock.mono()
    for _ in range(2000):
        src.poll()
    assert clock.mono() - start == pytest.approx(2.0, abs=0.002)   # 2000 polls took 2 s of fake time
    rep = src.poll_stats.report(clock.mono())
    assert rep["rate_hz"] == pytest.approx(1000, rel=0.01)
    assert rep["max_gap_ms"] == pytest.approx(1.0, abs=0.01)
    assert rep["late_polls"] == 0


def test_a_stall_shows_up_as_jitter_and_the_grid_restarts():
    # The PC is busy for 50 ms (Windows did something else): the max gap shows it, and the driver does NOT then
    # poll 50 times in a burst to "catch up"; it simply carries on 1 ms later.
    clock = TickingClock()
    world = SerialPortWorld(clock)
    src = modem(world, clock)
    src.open()
    for _ in range(10):
        src.poll()
    clock.advance(0.050)                              # the stall
    src.poll()
    n_sleeps = len(clock.sleeps)
    src.poll()
    assert clock.sleeps[n_sleeps] == pytest.approx(0.001)   # next poll is 1 ms later, not immediately
    rep = src.poll_stats.report(clock.mono())
    assert rep["max_gap_ms"] >= 50
    assert rep["late_polls"] == 1


def test_dtr_and_rts_are_set_from_the_config():
    # DTR / RTS are adapter OUTPUTS. Some wiring powers a pull-up or the sensor from them, so they are settings.
    clock = TickingClock()
    world = SerialPortWorld(clock)
    src = ModemLineSource("COM5", set_dtr=True, set_rts=False, clock=clock, sleep=clock.sleep,
                          serial_factory=world.factory)
    src.open()
    assert world.serial.dtr is True and world.serial.rts is False


def test_missing_port_raises_a_clear_error():
    # The adapter is not plugged in: opening fails with BeamPortError (the service retries; nothing crashes).
    clock = TickingClock()
    world = SerialPortWorld(clock, present=False)
    src = modem(world, clock)
    with pytest.raises(BeamPortError, match="cannot open COM5"):
        src.open()


def test_unplugged_adapter_on_linux_raises_on_the_next_poll():
    # On Linux, reading a modem line of an unplugged adapter fails at once.
    clock = TickingClock()
    world = SerialPortWorld(clock, lines_raise_when_unplugged=True)
    src = modem(world, clock)
    src.open()
    src.poll()
    world.unplug()
    with pytest.raises(BeamPortError, match="reading CTS failed"):
        src.poll()
    src.close()                                       # closing an unplugged port must not raise either
    assert world.opened[0].closed


def test_unplug_on_windows_is_detected_within_a_second_although_the_lines_just_read_false():
    # The Windows trap: after the adapter is pulled out, pyserial's .cts just returns False (no error), which looks
    # exactly like "beam clear". Only `in_waiting` fails. The driver checks it every ~100 ms, so the unplug must
    # be noticed well within 1 second of (fake) time.
    clock = TickingClock()
    world = SerialPortWorld(clock, lines={"cts": True})   # line asserted while plugged in
    src = modem(world, clock, active="low")              # asserted = beam clear (so "False" would look like a car)
    src.open()
    world.unplug()
    t_unplug = clock.mono()
    edges = []
    with pytest.raises(BeamPortError, match="unplugged"):
        for _ in range(2000):                             # up to 2 s of 1 ms polls
            edges += src.poll()
    assert clock.mono() - t_unplug < 1.0
    # The dead line reading False must NOT have been reported as a car: the driver checks the port before it
    # reports any change.
    assert edges == []


def test_windows_modem_status_check_failing_means_unplugged():
    # On Windows the driver can also ask GetCommModemStatus itself (pyserial ignores its result). Here a fake
    # check says "the call failed" (False): that alone must count as unplugged.
    clock = TickingClock()
    world = SerialPortWorld(clock)
    answers = [True]                                     # the check passes at open ...
    src = ModemLineSource("COM5", "cts", "high", 1000, clock=clock, sleep=clock.sleep, serial_factory=world.factory,
                          modem_status_check=lambda ser: answers[-1])
    src.open()
    answers.append(False)                                # ... then the Windows call starts failing
    with pytest.raises(BeamPortError, match="GetCommModemStatus failed"):
        for _ in range(200):
            src.poll()


def test_windows_modem_status_check_is_skipped_off_windows():
    # On this test machine (not Windows) the optional ctypes check answers None = "cannot check": no error.
    from rt.beam.sources import windows_modem_status_ok
    import sys
    if sys.platform != "win32":
        assert windows_modem_status_ok(object()) is None


def test_bytes_arriving_in_modem_line_mode_are_thrown_away():
    # In modem-line mode bytes are not used; the health check reads them away so the driver's buffer never fills.
    clock = TickingClock()
    world = SerialPortWorld(clock)
    src = modem(world, clock)
    src.open()
    world.serial.incoming += b"\x01\x02\x03"
    for _ in range(150):                                 # 150 ms: at least one health check
        src.poll()
    assert src.bytes_discarded == 3 and not world.serial.incoming


# ------------------------------------------------------------------------------------------------ byte-stream
def bytes_source(world, clock, pattern="", release=""):
    return ByteStreamSource("COM5", 9600, pattern, release, read_timeout_s=0.002, clock=clock,
                            serial_factory=world.factory)


def test_any_received_bytes_are_one_edge_with_the_raw_bytes_kept():
    # No pattern configured: whatever the adapter sends is a pulse. A message of several bytes that arrives in one
    # go is ONE edge (not one per byte), and the bytes are kept (hex) so the JSONL shows what was sent.
    clock = TickingClock()
    world = SerialPortWorld(clock)
    src = bytes_source(world, clock)
    src.open()
    assert src.poll() == []                          # nothing waiting: the read times out (2 ms of fake time)
    world.serial.incoming += b"T1\r\n"
    (e,) = src.poll()
    assert (e.line, e.level, e.raw, e.momentary) == ("bytes", 1, b"T1\r\n".hex(), True)
    assert e.t_mono == pytest.approx(clock.mono())   # stamped on receipt
    assert src.bytes_total == 4


def test_byte_pattern_marks_each_pulse_even_split_over_two_reads():
    # With byte_pattern = "0D0A" (CR LF), every CR LF is a pulse, even if the CR arrives in one read and the LF in
    # the next.
    clock = TickingClock()
    world = SerialPortWorld(clock)
    src = bytes_source(world, clock, pattern="0D 0A")
    src.open()
    world.serial.incoming += b"A\r"
    assert src.poll() == []                          # CR alone: not yet a match
    world.serial.incoming += b"\nB\r\nC\r\n"
    edges = src.poll()
    assert [e.level for e in edges] == [1, 1, 1]     # three CR LF in total
    assert all(e.momentary for e in edges)           # no release pattern: pulses


def test_release_pattern_gives_blocked_and_clear_edges():
    # Some adapters send one message for "blocked" and another for "clear": then the beam level is known.
    clock = TickingClock()
    world = SerialPortWorld(clock)
    src = bytes_source(world, clock, pattern="31", release="30")      # "1" = blocked, "0" = clear
    src.open()
    assert src.level() == 0
    world.serial.incoming += b"1"
    (e,) = src.poll()
    assert e.level == 1 and not e.momentary and src.level() == 1
    world.serial.incoming += b"0"
    (e,) = src.poll()
    assert e.level == 0 and src.level() == 0


def test_byte_stream_unplug_and_missing_port():
    clock = TickingClock()
    world = SerialPortWorld(clock, present=False)
    src = bytes_source(world, clock)
    with pytest.raises(BeamPortError, match="9600 baud"):
        src.open()
    world.plug_in()
    src.open()
    world.unplug()
    with pytest.raises(BeamPortError, match="read failed"):
        src.poll()


def test_parse_hex_accepts_spaces_and_lowercase():
    assert parse_hex("0D 0a") == b"\r\n"
    assert parse_hex("") == b""


# ------------------------------------------------------------------------------------------------ fake + factory
def test_fake_source_plays_its_script_on_the_fake_clock():
    # The fake driver gives the scripted edges once their time has come, stamped at their scripted time.
    clock = TickingClock(mono=1000.0)
    src = FakeBeamSource([(1000.010, 1), (1000.030, 0)], clock=clock, sleep=clock.sleep, poll_s=0.005)
    src.open()
    got = []
    for _ in range(10):
        got += src.poll()
    assert [(e.t_mono, e.level) for e in got] == [(1000.010, 1), (1000.030, 0)]
    assert got[0].t_wall - got[0].t_mono == pytest.approx(clock.wall() - clock.mono())


def test_fake_source_can_pretend_to_fail():
    clock = TickingClock()
    failures = FakeFailures(open_fails=2)
    src = FakeBeamSource([], clock=clock, sleep=clock.sleep, failures=failures)
    for _ in range(2):
        with pytest.raises(BeamPortError):
            src.open()
    src.open()                                        # third try works
    failures.unplugged = True
    with pytest.raises(BeamPortError):
        src.poll()


def test_synthetic_cars_make_multi_pulse_cars_at_a_fixed_period():
    gen = synthetic_cars(0.0, every_s=20.0, pulses=3)
    first_car = [next(gen) for _ in range(6)]         # 3 pulses = 3 x (on, off)
    assert [lvl for _, lvl in first_car] == [1, 0, 1, 0, 1, 0]
    assert first_car[0][0] == 20.0
    assert next(gen)[0] == 40.0                       # the next car 20 s later


def test_make_source_picks_the_driver_from_the_config(cfg):
    assert cfg["beam"]["driver"] == "off"             # default: no beam at all
    assert make_source(cfg) is None
    clock = TickingClock()
    world = SerialPortWorld(clock)
    cfg["beam"].update(driver="modem-line", line="dsr", active="low")
    s = make_source(cfg, clock=clock, sleep=clock.sleep, serial_factory=world.factory)
    assert isinstance(s, ModemLineSource) and s.line == "dsr" and not s.active_high
    cfg["beam"].update(driver="byte-stream", baud=19200, byte_pattern="0d")
    s = make_source(cfg, clock=clock, serial_factory=world.factory)
    assert isinstance(s, ByteStreamSource) and s.baud == 19200 and s.pattern == b"\r"
    cfg["beam"]["driver"] = "fake"
    assert isinstance(make_source(cfg, clock=clock, sleep=clock.sleep), FakeBeamSource)
    cfg["beam"]["driver"] = "laser-telepathy"
    with pytest.raises(ValueError):
        make_source(cfg)


def test_poll_stats_with_no_polls_does_not_divide_by_zero():
    rep = PollStats().report(5.0)
    assert rep["rate_hz"] == 0.0 and rep["polls"] == 0


def test_lapline_example_config_has_the_beam_off_and_valid_keys():
    # The example config the owner copies for the event: the beam is OFF until the adapter has been probed, and
    # switching it to modem-line with the documented keys builds a working driver.
    from rt import config as C
    cfg = C.load(str(C.ROOT / "rt" / "configs" / "lapline.example.toml"))
    assert cfg["beam"]["driver"] == "off"
    assert make_source(cfg) is None
    clock = TickingClock()
    world = SerialPortWorld(clock)
    cfg["beam"].update(driver="modem-line")
    src = make_source(cfg, clock=clock, sleep=clock.sleep, serial_factory=world.factory)
    assert src.open()["port"] == "COM5"
