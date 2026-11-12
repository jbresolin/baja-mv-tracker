"""Tests of the beam probe tool (rt/tools/beam_probe.py): the program the owner runs with the adapter plugged in, to
find out HOW the adapter delivers the sensor's pulse, and which [beam] settings to use.

How the probe decides (the "guess"), in plain words:
  * If one of the four modem input lines (CTS, DSR, DCD, RI) changed while the beam was being broken, that is the
    line: driver = "modem-line". The state the line sits in most of the time is "beam clear"; the other state is
    "beam blocked", which gives the polarity (`active`). "Most of the time" = the state the line was in when the
    probe STARTED (the beam is clear then); if the line then spends most of the probe in the other state, the guess
    warns (someone may have been standing in the beam).
  * Else, if bytes arrived: driver = "byte-stream". One repeated message -> that is the pattern; two messages that
    alternate -> "blocked" and "clear" patterns. Mostly 00 / FF bytes -> probably the wrong baud rate.
  * Else nothing happened: check the wiring (and break the beam during the probe!).
The tests feed made-up observations, and one runs the whole probe loop on a fake serial port.
"""
from rt.tools.beam_probe import ProbeLog, guess_config, report, run_probe

from .beam_fakes import SerialPortWorld, TickingClock

ALL_OFF = dict(cts=False, dsr=False, cd=False, ri=False)


def probe_with_pulses(line, idle, pulses=4, pulse_s=0.05, every_s=1.0):
    """A ProbeLog of `pulses` beam breaks on `line`, which sits at `idle` (True = asserted) between breaks."""
    log = ProbeLog(0.0)
    log.lines(0.0, dict(ALL_OFF, **{line: idle}))
    t = 0.0
    for _ in range(pulses):
        t += every_s
        log.lines(t, dict(ALL_OFF, **{line: not idle}))
        log.lines(t + pulse_s, dict(ALL_OFF, **{line: idle}))
    log.finish(t + every_s)
    return log


def test_line_that_rises_during_breaks_is_active_high():
    g = guess_config(probe_with_pulses("dsr", idle=False).summary())
    assert g["driver"] == "modem-line"
    assert g["config"] == ['driver = "modem-line"', 'line = "dsr"', 'active = "high"']


def test_line_that_drops_during_breaks_is_active_low():
    # Idle asserted (on) most of the time, drops when the beam is broken: an NPN sensor pulling the line down.
    g = guess_config(probe_with_pulses("cts", idle=True).summary())
    assert g["config"] == ['driver = "modem-line"', 'line = "cts"', 'active = "low"']
    assert any("drop" in n for n in g["notes"])


def test_dcd_line_is_written_as_cd_in_the_config():
    g = guess_config(probe_with_pulses("cd", idle=False).summary())
    assert 'line = "cd"' in g["config"]


def test_too_few_breaks_asks_for_more():
    g = guess_config(probe_with_pulses("ri", idle=False, pulses=1).summary())
    assert g["driver"] == "modem-line"
    assert any("fewer than 3 breaks" in n for n in g["notes"])


def test_one_repeated_message_is_a_byte_pattern():
    log = ProbeLog(0.0)
    log.lines(0.0, ALL_OFF)
    for k in range(5):
        log.data(1.0 + k, b"P\r\n")
    log.finish(7.0)
    g = guess_config(log.summary(), baud=19200)
    assert g["config"] == ['driver = "byte-stream"', "baud = 19200", 'byte_pattern = "500d0a"']


def test_two_alternating_messages_are_blocked_and_clear():
    log = ProbeLog(0.0)
    log.lines(0.0, ALL_OFF)
    for k in range(3):
        log.data(1.0 + k, b"1")                      # "blocked"
        log.data(1.1 + k, b"0")                      # "clear"
    log.finish(5.0)
    g = guess_config(log.summary())
    assert 'byte_pattern = "31"' in g["config"] and 'byte_release_pattern = "30"' in g["config"]


def test_garbage_bytes_suggest_the_wrong_baud_rate():
    log = ProbeLog(0.0)
    log.lines(0.0, ALL_OFF)
    for k in range(4):
        log.data(1.0 + k, b"\x00\xff\x00" if k % 2 else b"\xff\x00")
    log.finish(6.0)
    g = guess_config(log.summary())
    assert g["driver"] == "byte-stream"
    assert any("wrong baud rate" in n for n in g["notes"])


def test_a_line_wins_over_bytes_when_both_change():
    log = probe_with_pulses("cts", idle=False)
    log.data(1.0, b"x")
    g = guess_config(log.summary())
    assert g["driver"] == "modem-line"
    assert any("bytes were received as well" in n for n in g["notes"])


def test_nothing_seen_gives_no_config_and_advice():
    log = ProbeLog(0.0)
    log.lines(0.0, dict(ALL_OFF, dsr=True))          # DSR on the whole time, nothing changes
    log.finish(30.0)
    g = guess_config(log.summary())
    assert g["driver"] is None and g["config"] == []
    assert any("Break the beam" in n for n in g["notes"])
    assert any("DSR" in n for n in g["notes"])       # it mentions the line that is asserted the whole time


def test_asserted_time_share_is_measured():
    # CTS on for 0.5 s of a 2 s probe = 25 %.
    log = ProbeLog(0.0)
    log.lines(0.0, ALL_OFF)
    log.lines(1.0, dict(ALL_OFF, cts=True))
    log.lines(1.5, ALL_OFF)
    log.finish(2.0)
    assert log.summary()["lines"]["cts"] == dict(initial=False, transitions=2, asserted_share=0.25)


def test_whole_probe_loop_on_a_fake_port_prints_changes_and_guesses():
    # The real loop, on a fake serial port and a fake clock: CTS pulses 3 times and the adapter sends one byte.
    # Every change is printed with its time, and the final guess is CTS / active high.
    clock = TickingClock(mono=0.0)
    world = SerialPortWorld(clock)
    ser = world.factory("COM5", 9600, 0)
    script = {0.100: True, 0.150: False, 0.300: True, 0.350: False, 0.500: True, 0.550: False}

    def sleep(dt):
        clock.advance(dt)
        for t, v in script.items():          # the "sensor": switch CTS when its time comes
            if abs(clock.mono() - t) < 0.0005:
                ser.lines["cts"] = v
        if abs(clock.mono() - 0.2) < 0.0005:
            ser.incoming += b"\x55"

    printed = []
    log = run_probe(ser, seconds=0.7, clock=clock.mono, wall=clock.wall, sleep=sleep, out=printed.append)
    assert sum("CTS -> asserted" in line for line in printed) == 3
    assert any("bytes 55" in line for line in printed)
    g = report(log, 9600, out=printed.append)
    assert g["config"] == ['driver = "modem-line"', 'line = "cts"', 'active = "high"']
    assert any(line.startswith("[beam]") for line in printed)


# ------------------------------------------------------------------------------------------------ polarity from idle
def test_polarity_comes_from_the_state_at_the_start_even_if_the_beam_was_mostly_blocked():
    # CTS is off at the start (beam clear). Then someone stands in the beam for a long time: CTS is ON for most of
    # the probe. The idle state is still "off at the start", so active = "high" - and a warning says the time
    # share disagrees.
    log = ProbeLog(0.0)
    log.lines(0.0, ALL_OFF)                          # start: beam clear, CTS off
    log.lines(1.0, dict(ALL_OFF, cts=True))          # blocked ...
    log.lines(1.2, ALL_OFF)                          # ... a short car
    log.lines(2.0, dict(ALL_OFF, cts=True))          # someone stands in the beam for 20 s
    log.lines(22.0, ALL_OFF)
    log.finish(23.0)
    g = guess_config(log.summary())
    assert 'active = "high"' in g["config"]
    assert any(n.startswith("WARNING") and "opposite of its state at the start" in n for n in g["notes"])


def test_no_polarity_warning_when_start_and_majority_agree():
    log = probe_with_pulses("dsr", idle=True)        # DSR on at the start and most of the time: active low
    g = guess_config(log.summary())
    assert 'active = "low"' in g["config"]
    assert not any(n.startswith("WARNING") for n in g["notes"])


# ------------------------------------------------------------------------------------------------ split messages
def test_probe_joins_a_message_split_into_pieces_at_9600_baud():
    # The adapter sends "B1\r\n" for blocked and "B0\r\n" for clear, but each arrives in two pieces 1 ms apart (as
    # USB can do). The probe must see 2 kinds of MESSAGE that alternate, not 4 kinds of fragment, and so guess a
    # blocked / clear pattern pair.
    clock = TickingClock(mono=0.0)
    world = SerialPortWorld(clock)
    ser = world.factory("COM5", 9600, 0)
    pieces = {}
    for k in range(3):                                   # 3 cars: blocked at 0.1 + k, clear 0.2 s later
        for t0, msg in ((0.1 + k, b"B1\r\n"), (0.3 + k, b"B0\r\n")):
            pieces[round(t0, 3)] = msg[:2]
            pieces[round(t0 + 0.001, 3)] = msg[2:]

    def sleep(dt):
        clock.advance(dt)
        piece = pieces.pop(round(clock.mono(), 3), None)
        if piece:
            ser.incoming += piece

    printed = []
    log = run_probe(ser, seconds=3.0, clock=clock.mono, wall=clock.wall, sleep=sleep, out=printed.append)
    summary = log.summary()
    assert summary["n_chunks"] == 6                      # 6 whole messages, not 12 pieces
    g = guess_config(summary, baud=9600)
    assert 'byte_pattern = "42310d0a"' in g["config"]
    assert 'byte_release_pattern = "42300d0a"' in g["config"]


def test_alternation_needs_at_least_one_pair():
    from rt.tools.beam_probe import _alternates
    assert _alternates(["31", "30", "31", "30"])
    assert not _alternates(["31", "31", "30"])
    assert not _alternates(["31"])
