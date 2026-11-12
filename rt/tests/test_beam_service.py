"""Tests of the BeamService (rt/beam/service.py): the beam input running next to rt.

What the service does, in plain words:
  * a "reader" opens the serial port (and keeps retrying, 1 s, 2 s, 4 s ... if it is missing), reads the sensor and
    time-stamps every change ("edge");
  * a "dispatcher" turns the edges into JSONL events: `beam_edge` (every raw change), `beam_break` (one per car),
    `beam_stuck` (beam blocked too long), and answers every vision pass (`pass_end`) with a `pass_beam` event that
    says whether exactly one beam break matches it;
  * optionally it measures the camera's delay with an LED that the camera sees (`camera_latency`);
  * it keeps the numbers the status page shows.

In real use these are two threads. In the tests the same work is done step by step: reader_step() then
dispatch_step(), with a fake clock that only moves when the code "sleeps". So 10 seconds of beam activity run in a
few milliseconds and the results are exactly repeatable.
"""
import numpy as np
import pytest

from rt.beam.service import BeamService, PassBeamSink
from rt.beam.sources import FakeBeamSource, FakeFailures, ModemLineSource
from rt.capture import Frame
from rt.sinks import Emitter, Sink
from rt.status import StatusBoard, render_html

from .beam_fakes import SerialPortWorld, TickingClock

T = 1000.0                  # the fake monotonic clock starts here
WALL0 = 1_800_000_000.0     # ... and the fake wall clock here


def make_service(cfg, script=(), failures=None, clock=None, **beam_settings):
    """A BeamService on the fake driver: `script` = [(monotonic time, level), ...]. Returns (service, the list
    that receives every emitted event, the clock)."""
    clock = clock or TickingClock(wall=WALL0, mono=T)
    cfg["beam"].update(driver="fake", **beam_settings)
    src = FakeBeamSource(script, clock=clock, sleep=clock.sleep, poll_s=0.01, failures=failures)
    events = []
    svc = BeamService(cfg, events.append, source=src, clock=clock, sleep=clock.sleep)
    return svc, events, clock


def run_for(svc, clock, seconds):
    """Do what the two threads would do during `seconds` of (fake) time."""
    end = clock.mono() + seconds
    while clock.mono() < end:
        svc.reader_step()
        svc.dispatch_step(timeout=0)


def of_type(events, kind):
    return [e for e in events if e["type"] == kind]


def car(t, pulses=2):
    """A car crossing at monotonic time t: `pulses` short blocks 0.1 s apart."""
    out = []
    for k in range(pulses):
        out += [(t + 0.1 * k, 1), (t + 0.1 * k + 0.03, 0)]
    return out


def wall_of(t_mono):
    return t_mono - T + WALL0


# ------------------------------------------------------------------------------------------------ events
def test_edges_and_one_break_per_car_are_written(cfg):
    svc, events, clock = make_service(cfg, car(T + 0.5, pulses=2))
    run_for(svc, clock, 2.0)
    assert of_type(events, "beam")[0]["state"] == "open"
    edges = of_type(events, "beam_edge")
    assert [e["level"] for e in edges] == [1, 0, 1, 0]
    # Both clocks are in every edge: the wall clock (what rt's events use) and the monotonic clock.
    assert edges[0]["t_mono"] == pytest.approx(T + 0.5)
    assert edges[0]["t_wall"] == pytest.approx(wall_of(T + 0.5))
    assert edges[0]["t_edge"] == pytest.approx(wall_of(T + 0.5))      # clock.offset_s = 0: same as t_wall
    (b,) = of_type(events, "beam_break")
    assert b["t_beam"] == pytest.approx(wall_of(T + 0.5))             # the crossing time = the first edge
    assert b["n_pulses"] == 2 and b["n_edges"] == 4
    assert b["duration_s"] == pytest.approx(0.13)


def test_clock_offset_is_applied_like_for_frames(cfg):
    # rt adds clock.offset_s to every frame time; the beam times get the same correction, so they stay comparable.
    # The raw wall clock is kept untouched in t_wall.
    cfg["clock"]["offset_s"] = 0.5
    svc, events, clock = make_service(cfg, car(T + 0.5, pulses=1))
    run_for(svc, clock, 1.0)
    e = of_type(events, "beam_edge")[0]
    assert e["t_edge"] == pytest.approx(e["t_wall"] + 0.5)
    assert of_type(events, "beam_break")[0]["t_beam"] == pytest.approx(wall_of(T + 0.5) + 0.5)


def test_stuck_beam_is_one_warning_not_endless_breaks(cfg):
    # Someone stands in the beam from T+1 to T+31: one beam_stuck "stuck", one "cleared", no beam_break.
    svc, events, clock = make_service(cfg, [(T + 1, 1), (T + 31, 0)])
    run_for(svc, clock, 10.0)
    assert svc.status()["stuck"] is True
    run_for(svc, clock, 25.0)
    assert [e["state"] for e in of_type(events, "beam_stuck")] == ["stuck", "cleared"]
    assert of_type(events, "beam_break") == []
    assert svc.status()["stuck"] is False and svc.status()["stuck_episodes"] == 1


def test_noisy_line_edges_are_capped_in_the_log_but_all_grouped(cfg):
    # A bouncing contact: 300 edges within one second. At most max_edge_events_per_s (here 50) get their own JSONL
    # line; the rest are counted and a warning says so. The grouping still sees all of them.
    script = [(T + 0.5 + k * 0.003, k % 2 == 0) for k in range(300)]
    svc, events, clock = make_service(cfg, [(t, int(v)) for t, v in script], max_edge_events_per_s=50)
    run_for(svc, clock, 3.0)
    assert len(of_type(events, "beam_edge")) == 50
    assert svc.status()["edges_total"] == 300
    assert svc.status()["edges_not_logged"] == 250
    assert any("not written one by one" in e["msg"] for e in of_type(events, "warning"))


def test_poll_rate_is_reported_regularly(cfg):
    svc, events, clock = make_service(cfg, [], poll_report_s=5)
    run_for(svc, clock, 11.0)
    polls = of_type(events, "beam_poll")
    assert len(polls) == 2                               # at 5 s and 10 s
    assert polls[0]["rate_hz"] == pytest.approx(100, rel=0.05)   # the fake polls every 10 ms
    assert svc.status()["poll"]["rate_hz"] == polls[-1]["rate_hz"]


# ------------------------------------------------------------------------------------------------ robustness
def test_port_missing_at_start_is_retried_with_backoff(cfg):
    # The adapter is not plugged in for the first 3 tries. The service must not crash: it says "retrying", waits
    # 1 s, 2 s, 4 s between tries, and opens the port at the 4th try (~7 s after start).
    svc, events, clock = make_service(cfg, [], failures=FakeFailures(open_fails=3))
    run_for(svc, clock, 0.5)
    st = svc.status()
    assert st["state"] == "retrying" and st["connected"] is False
    assert "next try in 1 s" in st["detail"]
    run_for(svc, clock, 5.0)
    assert svc.status()["state"] == "retrying"           # at 5.5 s: tries at 0, 1, 3 failed; next at 7
    run_for(svc, clock, 2.0)
    assert svc.status()["state"] == "open"
    assert svc.source.failures.opens == 4
    states = [e["state"] for e in of_type(events, "beam")]
    assert states[0] == "retrying" and states[-1] == "open"
    # One warning for the first failure, not one per retry.
    assert sum("cannot open fake" in e["msg"] for e in of_type(events, "warning")) == 1


def test_unplug_during_a_car_interrupts_the_break_and_reopens(cfg):
    failures = FakeFailures()
    svc, events, clock = make_service(cfg, [(T + 1.0, 1)], failures=failures)
    run_for(svc, clock, 1.1)                             # beam blocked at T+1 ...
    failures.unplugged = True                            # ... and the adapter is pulled out
    run_for(svc, clock, 0.1)
    assert svc.status()["state"] == "lost"
    (b,) = of_type(events, "beam_break")
    assert b["interrupted"] is True
    assert any("port lost" in e["msg"] for e in of_type(events, "warning"))
    run_for(svc, clock, 1.5)                             # the fake can be reopened at once: after the 1 s wait
    assert svc.status()["state"] == "open" and svc.status()["opens"] == 2


def test_backoff_starts_short_again_after_the_port_worked(cfg):
    # Unplugged twice, an hour apart: both times the first retry is after 1 s (the port worked in between).
    failures = FakeFailures()
    svc, events, clock = make_service(cfg, [], failures=failures)
    run_for(svc, clock, 1.0)
    for _ in range(2):
        failures.unplugged = True
        run_for(svc, clock, 0.05)
        assert "next try in 1 s" in svc.status()["detail"]
        run_for(svc, clock, 3.0)
        assert svc.status()["state"] == "open"


def test_real_modem_driver_unplug_and_replug_with_a_fake_serial_port(cfg):
    # The real ModemLineSource on the fake serial port: unplugged -> "lost"; still missing -> "retrying" with
    # growing waits; plugged back in -> "open", and edges flow again.
    clock = TickingClock(wall=WALL0, mono=T)
    world = SerialPortWorld(clock)
    cfg["beam"].update(driver="modem-line", line="cts", active="high")
    src = ModemLineSource("COM5", "cts", "high", 1000, clock=clock, sleep=clock.sleep, serial_factory=world.factory)
    events = []
    svc = BeamService(cfg, events.append, source=src, clock=clock, sleep=clock.sleep)
    run_for(svc, clock, 0.1)
    assert svc.status()["state"] == "open"
    world.unplug()
    # Like real pyserial on Windows: the CTS line now just reads False (no error); the driver's ~100 ms port check
    # (in_waiting) notices the unplug within a second.
    run_for(svc, clock, 1.0)
    assert svc.status()["state"] == "lost"
    run_for(svc, clock, 5.0)
    assert svc.status()["state"] == "retrying"           # COM5 does not exist any more
    world.plug_in()
    run_for(svc, clock, 10.0)
    assert svc.status()["state"] == "open"
    world.serial.lines["cts"] = True                     # a car on the new connection
    run_for(svc, clock, 0.01)
    assert of_type(events, "beam_edge")[-1]["level"] == 1


def test_an_unexpected_driver_error_does_not_kill_the_reader(cfg):
    # A bug in a driver (anything other than a port error) is reported as a warning and the port is reopened.
    svc, events, clock = make_service(cfg, [])

    def broken_poll():
        svc.halt.set()                                   # stop the loop after this round
        raise RuntimeError("driver bug")

    svc.reader_step()                                    # open
    svc.source.poll = broken_poll
    svc._reader_loop()                                   # must return normally, not raise
    svc.dispatch_step(timeout=0)
    assert any("driver bug" in e["msg"] for e in of_type(events, "warning"))
    assert svc.status()["state"] == "retrying"


# ------------------------------------------------------------------------------------------------ association
def pass_end(pass_id, t_cross_mono):
    """A vision pass_end whose crossing time is the wall time of monotonic time t_cross_mono."""
    t = wall_of(t_cross_mono)
    return dict(type="pass_end", pass_id=pass_id, t_enter=t - 1, t_cross=t, t_exit=t + 1)


def test_pass_with_exactly_one_break_gets_t_beam(cfg):
    svc, events, clock = make_service(cfg, car(T + 2.0))
    run_for(svc, clock, 3.0)
    svc.on_event(pass_end(7, T + 2.3))                   # vision crossing 0.3 s after the beam
    run_for(svc, clock, 1.0)                             # (until the pass's 1 s window is over)
    (pb,) = of_type(events, "pass_beam")
    assert pb["status"] == "matched" and pb["pass_id"] == 7
    assert pb["t_beam"] == pytest.approx(wall_of(T + 2.0))
    assert pb["beam_minus_cross_s"] == pytest.approx(-0.3)
    assert svc.status()["last_breaks"][0]["pass_id"] == 7


def test_pass_with_no_break_is_none_and_with_two_is_ambiguous(cfg):
    svc, events, clock = make_service(cfg, car(T + 2.0, 1) + car(T + 2.6, 1))   # two cars 0.6 s apart
    run_for(svc, clock, 4.0)
    svc.on_event(pass_end(1, T + 2.3))                   # both breaks within 1 s of T+2.3: can't tell which
    svc.on_event(pass_end(2, T + 3.5))                   # only the second car (T+2.6, 0.9 s away) is within 1 s
    svc.on_event(pass_end(3, T + 7.0))                   # nothing near T+7
    run_for(svc, clock, 5.0)
    by_pass = {e["pass_id"]: e for e in of_type(events, "pass_beam")}
    assert by_pass[1]["status"] == "ambiguous" and by_pass[1]["n_candidates"] == 2
    assert by_pass[2]["status"] == "matched"             # only the second car (0.9 s) is within 1 s
    assert by_pass[3]["status"] == "none" and "t_beam" not in by_pass[3]
    assert svc.status()["association"]["ambiguous"] == 1


def test_the_expected_camera_offset_shifts_the_window(cfg):
    # If the camera's centre line is 1.5 s after the beam (assoc_offset_s = 1.5), a pass at T+3.5 looks for a break
    # around T+2.0, and finds it; with offset 0 it would find nothing.
    svc, events, clock = make_service(cfg, car(T + 2.0, 1), assoc_offset_s=1.5, assoc_window_s=0.5)
    run_for(svc, clock, 4.0)
    svc.on_event(pass_end(1, T + 3.5))
    run_for(svc, clock, 1.0)
    assert of_type(events, "pass_beam")[0]["status"] == "matched"


def test_a_pass_waits_until_its_window_is_over(cfg):
    # If a pass_end arrives while a break inside its window could still be coming, the decision waits.
    svc, events, clock = make_service(cfg, car(T + 2.0, 1))
    run_for(svc, clock, 1.0)
    svc.on_event(pass_end(1, T + 1.8))                   # window T+0.8 .. T+2.8; now = T+1
    run_for(svc, clock, 0.5)
    assert of_type(events, "pass_beam") == []            # not decided yet
    run_for(svc, clock, 2.0)
    assert of_type(events, "pass_beam")[0]["status"] == "matched"


def test_two_passes_on_one_break_both_match_and_say_so(cfg):
    # Two cars side by side cut the beam once. Live, both passes get the same break; the second one says the break
    # was already given to the first (offline fusion decides properly).
    svc, events, clock = make_service(cfg, car(T + 2.0, 2))
    run_for(svc, clock, 3.0)
    svc.on_event(pass_end(1, T + 2.1))
    svc.on_event(pass_end(2, T + 2.2))
    run_for(svc, clock, 1.0)
    a, b = of_type(events, "pass_beam")
    assert a["break_also_matched_to"] == [] and b["break_also_matched_to"] == [1]


def test_association_through_rts_real_event_emitter(cfg):
    # In rt, pass_end comes from the detection loop through the Emitter, which holds a lock while it calls the
    # sinks. The service must only queue there (emitting from inside would deadlock). Then pass_beam must come out
    # through the same emitter, into the same JSONL as everything else.
    class ListSink(Sink):
        def __init__(self):
            self.events = []

        def event(self, ev):
            self.events.append(ev)

    out = ListSink()
    em = Emitter([out])
    clock = TickingClock(wall=WALL0, mono=T)
    cfg["beam"]["driver"] = "fake"
    src = FakeBeamSource(car(T + 1.0, 1), clock=clock, sleep=clock.sleep, poll_s=0.01)
    svc = BeamService(cfg, em.emit, source=src, clock=clock, sleep=clock.sleep)
    em.sinks.append(PassBeamSink(svc))
    run_for(svc, clock, 2.0)
    em.emit(pass_end(4, T + 1.2))                        # what the pass logic does
    em.emit(dict(type="pass_identity", pass_id=4, number=14, t_cross=wall_of(T + 1.2)))
    run_for(svc, clock, 1.0)
    types = [e["type"] for e in out.events]
    assert "pass_beam" in types and "beam_break" in types
    pb = [e for e in out.events if e["type"] == "pass_beam"][0]
    assert pb["t_beam_iso"] is not None                  # the emitter adds the readable time, like for t_cross
    assert svc.status()["last_breaks"][0]["number"] == 14   # the status page shows the number read


# ------------------------------------------------------------------------------------------------ LED latency
def led_frame(idx, t_mono, bright):
    """A small processed frame whose top-left 10x10 corner is the LED: bright (on) or dark (off). Its time stamps
    are what the camera capture gives: t = wall time, t_cap = monotonic time of the grab."""
    img = np.zeros((90, 160, 3), np.uint8)
    if bright:
        img[0:10, 0:10] = 250
    return Frame(img=img, t=wall_of(t_mono), pts=0.0, idx=idx, t_cap=t_mono)


def test_led_latency_frame_minus_beam_edge(cfg):
    # The beam goes active at T+1.000; the first processed frame that shows the LED on was grabbed at T+1.080.
    # camera_latency = 0.080 s.
    svc, events, clock = make_service(cfg, [(T + 1.0, 1), (T + 1.2, 0)], led_roi=[0, 0, 10, 10], led_threshold=128)
    run_for(svc, clock, 1.05)
    svc.on_frame(led_frame(0, T + 0.98, False))          # LED off
    svc.on_frame(led_frame(1, T + 1.08, True))           # LED on: a transition
    svc.on_frame(led_frame(2, T + 1.18, True))           # still on: not a transition
    run_for(svc, clock, 0.1)
    (lat,) = of_type(events, "camera_latency")
    assert lat["latency_s"] == pytest.approx(0.08)
    assert lat["residual_s"] == pytest.approx(0.08)      # frame time - beam time on the wall clock (no live_latency)
    assert lat["clean"] is True                          # only one active edge before it: a sure pairing
    assert svc.status()["led"]["measurements"] == 1


def test_led_on_without_a_beam_edge_is_counted_unmatched(cfg):
    svc, events, clock = make_service(cfg, [], led_roi=[0, 0, 10, 10])
    run_for(svc, clock, 1.0)
    svc.on_frame(led_frame(0, T + 0.9, False))
    svc.on_frame(led_frame(1, T + 1.0, True))
    run_for(svc, clock, 2.0)
    assert of_type(events, "camera_latency") == []
    assert svc.status()["led"]["unmatched"] == 1


def test_led_feature_is_off_by_default_and_costs_nothing(cfg):
    svc, events, clock = make_service(cfg, [])
    assert cfg["beam"]["led_roi"] == []
    svc.on_frame(led_frame(0, T, True))
    assert svc.q.empty()                                 # nothing was even queued
    assert svc.status()["led"]["enabled"] is False


# ------------------------------------------------------------------------------------------------ status
def test_status_payload_for_the_page(cfg):
    svc, events, clock = make_service(cfg, car(T + 1.0) + car(T + 3.0) + [(T + 5.0, 1)])
    run_for(svc, clock, 5.5)
    svc.on_event(dict(type="pass_identity", pass_id=9, number=22))
    svc.on_event(pass_end(9, T + 3.1))
    run_for(svc, clock, 0.1)
    st = svc.status()
    assert st["driver"] == "fake" and st["port"] == "fake" and st["connected"] is True
    assert st["level"] == 1                              # blocked right now (since T+5)
    assert st["edges_per_min"] == 9 and st["breaks_total"] == 2
    newest, older = st["last_breaks"]
    assert newest["pass_id"] == 9 and newest["number"] == 22   # newest first, with the matched pass and its number
    assert older["pass_id"] is None


def test_status_page_shows_the_beam_section_and_warnings(cfg):
    # The status page (rt/status.py) gets the beam numbers, red warnings for a lost port / a stuck beam, and an
    # HTML section. Without a beam the section is absent.
    board = StatusBoard(cfg, TickingClock())
    assert board.payload()["beam"] == dict(enabled=False)
    assert "<h2>Beam</h2>" not in render_html(board.payload())
    svc, events, clock = make_service(cfg, [(T + 1, 1)], failures=FakeFailures())
    board.beam = svc
    run_for(svc, clock, 8.0)                             # blocked since T+1: stuck
    p = board.payload()
    assert p["beam"]["stuck"] is True
    assert any("beam STUCK" in w for w in p["warnings"])
    html = render_html(p)
    assert "<h2>Beam</h2>" in html and "STUCK" in html
    svc.source.failures.unplugged = True
    svc.source.failures.open_fails = 99
    run_for(svc, clock, 0.05)
    assert any(w.startswith("beam lost") for w in board.payload()["warnings"])


def test_slow_poll_rate_is_a_warning_for_modem_line():
    from rt.status import beam_warnings
    ok = dict(enabled=True, state="open", driver="modem-line", poll=dict(rate_hz=990.0))
    slow = dict(ok, poll=dict(rate_hz=64.0))
    assert beam_warnings(ok) == []
    assert "poll rate only 64 Hz" in beam_warnings(slow)[0]


# ------------------------------------------------------------------------------------------------ threads, run.py
def test_real_threads_start_and_stop_cleanly(cfg):
    # The only test with real threads and the real clock: start, let the fake driver run ~50 ms, stop. Everything
    # must stop, and the open break must be written at stop.
    import time
    cfg["beam"]["driver"] = "fake"
    now = time.monotonic()
    src = FakeBeamSource([(now + 0.005, 1)], poll_s=0.001)
    events = []
    svc = BeamService(cfg, events.append, source=src).start()
    time.sleep(0.05)
    svc.stop()
    assert all(not t.is_alive() for t in svc.threads)
    assert svc.status()["state"] == "stopped"
    assert of_type(events, "beam_break")[0]["interrupted"] is True


def test_stop_answers_waiting_passes(cfg):
    svc, events, clock = make_service(cfg, car(T + 1.0, 1))
    run_for(svc, clock, 1.1)
    svc.on_event(pass_end(1, T + 1.0))
    svc.stop()
    assert of_type(events, "pass_beam")[0]["pass_id"] == 1


def test_run_py_starts_the_beam_only_when_configured(cfg):
    from rt.run import start_beam

    class FakeEmitter:
        def __init__(self):
            import threading
            self.lock, self.sinks = threading.Lock(), []
            self.emit = lambda ev: None

    warnings = []
    assert start_beam(cfg, FakeEmitter(), None, True, print, warnings.append) is None       # driver = "off"
    cfg["beam"].update(driver="modem-line", line="txd")                                       # a typo in the config
    assert start_beam(cfg, FakeEmitter(), None, True, print, warnings.append) is None
    assert "beam input NOT started" in warnings[0]


# ------------------------------------------------------------------------------------------------ thread race
class RacySource(FakeBeamSource):
    """A fake driver that lets the test run "the dispatcher thread" at the worst possible moment: after the reader
    has stamped an edge but before that edge is in the queue. In real life the two threads can interleave exactly
    like that, for a fraction of a millisecond."""

    def __init__(self, *args, between=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.between = between          # called inside poll(), after stamping, before returning the edges

    def poll(self):
        edges = super().poll()
        if edges and self.between is not None:
            self.between()
        return edges


def test_dispatcher_does_not_close_a_break_before_the_reader_queued_its_next_edge(cfg):
    # One car: pulses at T+0.50 and T+0.70 (0.2 s apart, less than group_gap_s = 0.25: ONE break).
    # The second pulse is stamped at T+0.70; then, before it reaches the queue, the clock jumps 0.1 s (the PC was
    # busy) and the dispatcher runs. Measured from "now" (T+0.80) the break looks quiet for 0.3 s and would be
    # closed, splitting the car into two breaks. The dispatcher must only advance up to what the reader queued.
    clock = TickingClock(wall=WALL0, mono=T)
    cfg["beam"].update(driver="fake")
    svc_box = []

    def dispatcher_runs_now():
        clock.advance(0.1)                      # the reader thread is delayed between stamping and queueing
        svc_box[0].dispatch_step(timeout=0)     # ... and the dispatcher runs meanwhile

    src = RacySource([(T + 0.50, 1), (T + 0.53, 0), (T + 0.70, 1), (T + 0.73, 0)], clock=clock, sleep=clock.sleep,
                     poll_s=0.01)
    events = []
    svc = BeamService(cfg, events.append, source=src, clock=clock, sleep=clock.sleep)
    svc_box.append(svc)
    run_for(svc, clock, 0.65)                   # first pulse handled normally
    src.between = dispatcher_runs_now           # the next poll that returns an edge triggers the race
    run_for(svc, clock, 2.0)
    breaks = of_type(events, "beam_break")
    assert len(breaks) == 1                     # one car = one break, the race did not split it
    assert breaks[0]["n_pulses"] == 2


def test_grouping_time_is_behind_now_by_the_guard(cfg):
    # Without any reader information, the grouper is advanced to now - advance_guard_s (20 ms by default).
    svc, _, clock = make_service(cfg, [])
    now = clock.mono()
    assert svc._grouping_time(now, None) == pytest.approx(now - 0.02)
    # With a reader horizon 5 ms ago: that one (it is earlier).
    assert svc._grouping_time(now, now - 0.005) == pytest.approx(now - 0.02)
    assert svc._grouping_time(now, now - 0.5) == pytest.approx(now - 0.5)
    # A reader silent for more than 1 s does not hold the grouping back.
    assert svc._grouping_time(now, now - 5.0) == pytest.approx(now - 0.02)


# ------------------------------------------------------------------------------------------------ small rules
def test_a_pass_without_t_cross_gets_pass_beam_none_not_silence(cfg):
    # A pass_end without a crossing time cannot be matched, but it must still get its pass_beam answer ("none",
    # with the reason), so every pass_end has exactly one pass_beam in the JSONL.
    svc, events, clock = make_service(cfg, [])
    run_for(svc, clock, 0.1)
    svc.on_event(dict(type="pass_end", pass_id=7, t_cross=None))
    run_for(svc, clock, 0.1)
    (pb,) = of_type(events, "pass_beam")
    assert pb["pass_id"] == 7 and pb["status"] == "none" and pb["reason"] == "pass has no t_cross"
    assert svc.status()["association"]["none"] == 1


def test_queue_is_bounded_and_drops_the_oldest(cfg):
    # If the dispatcher ever stopped, the queue must not grow without limit for a whole race day: with
    # queue_max = 5, putting 8 items keeps the newest 5 and counts 3 dropped.
    svc, events, clock = make_service(cfg, [], queue_max=5)
    for k in range(8):
        svc._put(("number", k, str(k)))
    assert svc.q.qsize() == 5
    assert svc.status()["queue_dropped"] == 3
    assert svc.q.get_nowait() == ("number", 3, "3")   # 0, 1, 2 were dropped (the oldest)


def test_break_to_pass_memory_is_trimmed(cfg):
    # break_passes remembers which passes matched which break; over a race day it must not grow forever.
    svc, events, clock = make_service(cfg, [])
    for k in range(2500):
        svc.recent_breaks.clear()
        svc.recent_breaks.append(dict(break_id=k, t_beam=float(k)))
        svc._emit_pass_beam(dict(pass_id=k, t_cross=float(k)))
    assert len(svc.break_passes) == 2000
    assert 2499 in svc.break_passes and 0 not in svc.break_passes


def test_stop_skips_the_final_dispatch_while_a_thread_is_still_alive(cfg):
    # stop() writes what is still queued from the CALLING thread. If the dispatcher thread were still running,
    # two threads would handle the queue at once. Here a fake "thread" that never ends: stop() must warn and
    # leave the queue alone.
    svc, events, clock = make_service(cfg, [])

    class NeverEndingThread:
        name = "beam-dispatch"

        def join(self, timeout):
            pass

        def is_alive(self):
            return True

    svc.threads = [NeverEndingThread()]
    svc._put(("number", 1, "12"))
    svc.stop(timeout=0.01)
    assert svc.q.qsize() == 1                          # not dispatched by stop()
    assert any("did not stop" in e["msg"] for e in of_type(events, "warning"))
