"""Tests of the edge grouping (rt/beam/grouping.py): many raw beam edges -> one "break" per car.

Background for a beginner:
  * A Baja car is a frame of steel tubes. Crossing the beam, the light is blocked by the nose, gets through a gap
    between tubes, is blocked by the driver, gets through again, is blocked by the rear... ONE car = SEVERAL pulses.
  * So the edges are grouped in software: a pulse that starts less than `group_gap_s` (0.25 s) after the previous
    edge belongs to the same car. The FIRST edge of the group is the car's crossing time.
  * A "stuck" beam (someone standing in it, a car stopped on the line, a sensor knocked out of line) must not make
    thousands of fake cars: after `stuck_s` (5 s) it becomes ONE `stuck` record, then a `cleared` one.

These tests feed hand-made edges with made-up times (seconds on the monotonic clock, starting at 100) to the grouper
and look at what comes out. No clock, no thread, no waiting.
"""
import pytest

from rt.beam.grouping import BreakGrouper
from rt.beam.sources import RawEdge

WALL_MINUS_MONO = 1_799_999_000.0      # wall = mono + this, for the made-up edges


def edge(t, level, momentary=False):
    """A raw edge at monotonic time t (seconds)."""
    return RawEdge(t + WALL_MINUS_MONO, t, "cts", level, raw=bool(level), momentary=momentary)


def pulse(t, length=0.03):
    """One short block of the beam: an active edge at t and the clear edge `length` later."""
    return [edge(t, 1), edge(t + length, 0)]


def run(grouper, edges, until):
    """Feed the edges in order, then let time pass in 10 ms steps up to `until`. Returns every record produced."""
    out = []
    for e in edges:
        out += grouper.feed(e)
    t = edges[-1].t_mono if edges else 0.0
    while t < until:
        t += 0.01
        out += grouper.advance(t, t + WALL_MINUS_MONO)
    return out


def breaks_of(records):
    return [r for r in records if r["kind"] == "break"]


def test_one_short_pulse_is_one_break_closed_after_the_gap():
    g = BreakGrouper(group_gap_s=0.25)
    out = []
    for e in pulse(100.0):
        out += g.feed(e)
    assert g.advance(100.2, 0) == []                 # only 0.17 s quiet since the clear edge: maybe more pulses
    out += g.advance(100.29, 100.29 + WALL_MINUS_MONO)   # 0.26 s quiet: the car is through
    (b,) = breaks_of(out)
    assert b["first_mono"] == 100.0                  # crossing time = the FIRST edge
    assert b["first_wall"] == 100.0 + WALL_MINUS_MONO
    assert b["n_pulses"] == 1 and b["n_edges"] == 2
    assert b["duration_s"] == pytest.approx(0.03)


def test_tube_frame_car_with_four_pulses_is_one_break():
    # Nose, driver, firewall, rear wheel: four pulses within 0.4 s, gaps of ~0.1 s.
    edges = pulse(100.00) + pulse(100.12) + pulse(100.25) + pulse(100.40, 0.05)
    (b,) = breaks_of(run(BreakGrouper(0.25), edges, 101.0))
    assert b["first_mono"] == 100.00                 # the nose
    assert b["last_mono"] == pytest.approx(100.45)   # the end of the last pulse
    assert b["n_pulses"] == 4 and b["n_edges"] == 8
    assert b["duration_s"] == pytest.approx(0.45)


def test_nose_to_tail_cars_just_beyond_the_gap_are_two_breaks():
    # Car B's nose cuts the beam 0.30 s after car A's tail cleared it (gap setting 0.25 s): two cars.
    car_a = pulse(100.00) + pulse(100.10)            # A clears at 100.13
    car_b = pulse(100.43) + pulse(100.55)            # 0.30 s later
    a, b = breaks_of(run(BreakGrouper(0.25), car_a + car_b, 101.5))
    assert a["first_mono"] == 100.00 and b["first_mono"] == 100.43
    assert a["n_pulses"] == 2 and b["n_pulses"] == 2


def test_cars_closer_than_the_gap_merge_into_one_break():
    # The limit of the rule, written down on purpose: 0.20 s apart (< 0.25 s) cannot be told apart by the beam.
    # Vision still counts two passes; the offline fusion (fuse_beam.py) reports such a pair as a shared break.
    car_a = pulse(100.00)
    car_b = pulse(100.23)                            # 0.20 s after A cleared
    (b,) = breaks_of(run(BreakGrouper(0.25), car_a + car_b, 101.0))
    assert b["n_pulses"] == 2


def test_a_break_stays_open_while_the_beam_is_blocked():
    # A long car (or a slow one) blocks the beam for 1.5 s without any edge: that is still ONE break, and it must not
    # be closed just because 0.25 s passed without edges.
    g = BreakGrouper(0.25, stuck_s=5.0)
    out = g.feed(edge(100.0, 1))
    for k in range(1, 150):
        out += g.advance(100.0 + k * 0.01, 0)
    assert breaks_of(out) == []                      # still blocked at 101.49
    out += run(g, [edge(101.5, 0)], 102.0)
    (b,) = breaks_of(out)
    assert b["duration_s"] == pytest.approx(1.5)


def test_person_standing_in_the_beam_gives_one_stuck_episode_not_breaks():
    # Someone stands in the beam for 60 s. Expected: ONE "stuck" record after 5 s, no break at all, and ONE
    # "cleared" record (with the duration) when the beam has been clear for the gap again.
    g = BreakGrouper(0.25, stuck_s=5.0)
    out = run(g, [edge(100.0, 1)], 160.0)
    stuck = [r for r in out if r["kind"] == "stuck"]
    assert len(stuck) == 1 and stuck[0]["state"] == "stuck"
    assert stuck[0]["since_mono"] == 100.0
    assert g.status(160.0) == dict(stuck=True, stuck_for_s=60.0)
    out = run(g, [edge(160.0, 0)], 161.0)
    (cleared,) = [r for r in out if r["kind"] == "stuck"]
    assert cleared["state"] == "cleared" and cleared["duration_s"] == pytest.approx(60.0)
    assert breaks_of(out) == []
    assert g.status(161.0) == dict(stuck=False)


def test_someone_moving_inside_the_beam_flickering_is_also_stuck():
    # A person walking around in the beam makes it flicker: an edge every 0.1 s for 20 s. All gaps are below 0.25 s,
    # so it would be one endless "car". After 5 s it is declared stuck; no break comes out of it.
    edges = []
    for k in range(100):
        edges += pulse(100.0 + k * 0.2, 0.1)
    out = run(BreakGrouper(0.25, 5.0), edges, 121.0)
    assert breaks_of(out) == []
    states = [r["state"] for r in out if r["kind"] == "stuck"]
    assert states == ["stuck", "cleared"]


def test_after_a_stuck_episode_the_next_car_is_a_normal_break():
    g = BreakGrouper(0.25, 5.0)
    run(g, [edge(100.0, 1)], 107.0)                  # stuck from 105
    run(g, [edge(107.0, 0)], 108.0)                  # cleared
    (b,) = breaks_of(run(g, pulse(110.0), 111.0))     # the next car
    assert b["first_mono"] == 110.0


def test_byte_mode_pulses_without_release_are_grouped_by_the_gap():
    # The byte-stream driver without a release pattern gives "momentary" pulses: the beam is never left "blocked".
    edges = [edge(100.0, 1, momentary=True), edge(100.1, 1, momentary=True), edge(100.5, 1, momentary=True)]
    a, b = breaks_of(run(BreakGrouper(0.25), edges, 101.0))
    assert a["n_pulses"] == 2 and b["n_pulses"] == 1


def test_beam_blocked_when_the_port_opens_is_flagged():
    # The line already reads "blocked" when the port opens (someone in the beam, or the polarity set the wrong way).
    # That break has no real crossing time, so it is flagged `started_blocked` (association ignores it).
    g = BreakGrouper(0.25, 5.0)
    g.start_blocked(100.0, 100.0 + WALL_MINUS_MONO)
    (b,) = breaks_of(run(g, [edge(100.5, 0)], 101.0))
    assert b["started_blocked"] is True


def test_wrong_polarity_looks_like_a_stuck_beam():
    # With the polarity reversed the beam reads "blocked" all the time: it must become stuck, not a flood of cars.
    g = BreakGrouper(0.25, 5.0)
    g.start_blocked(100.0, 100.0 + WALL_MINUS_MONO)
    out = []
    t = 100.0
    while t < 110.0:
        t += 0.01
        out += g.advance(t, t + WALL_MINUS_MONO)
    assert [r["state"] for r in out if r["kind"] == "stuck"] == ["stuck"]
    assert breaks_of(out) == []


def test_port_lost_in_the_middle_of_a_break_finishes_it_as_interrupted():
    g = BreakGrouper(0.25, 5.0)
    g.feed(edge(100.0, 1))
    (b,) = g.disconnect(100.1, 100.1 + WALL_MINUS_MONO)
    assert b["kind"] == "break" and b["interrupted"] is True
    assert g.open is None and g.blocked is False


def test_port_lost_during_a_stuck_episode_clears_it():
    g = BreakGrouper(0.25, 5.0)
    run(g, [edge(100.0, 1)], 106.0)
    (s,) = g.disconnect(106.0, 106.0 + WALL_MINUS_MONO)
    assert s["state"] == "cleared" and s["reason"] == "port lost"


def test_a_lone_clear_edge_does_nothing():
    # A "clear" edge with no break open (e.g. right after the port opened) is not a car.
    assert run(BreakGrouper(0.25), [edge(100.0, 0)], 101.0) == []


def test_break_ids_count_up():
    g = BreakGrouper(0.25)
    ids = [b["break_id"] for b in breaks_of(run(g, pulse(100.0) + pulse(101.0) + pulse(102.0), 103.0))]
    assert ids == [0, 1, 2]


def test_status_reads_the_stuck_episode_once_so_a_concurrent_clear_cannot_crash_it():
    # The status page runs in its own thread. While it builds the page, the dispatcher thread may clear the stuck
    # episode (set `stuck` to None). If status() read `self.stuck` twice ("is it None?" then "since when?") the
    # second read could get None and crash. Here a grouper whose `stuck` changes to None right after the first
    # read imitates that unlucky timing: status() must still answer.
    class VanishingStuckGrouper(BreakGrouper):
        reads = 0

        @property
        def stuck(self):
            VanishingStuckGrouper.reads += 1
            # first read: a stuck episode since t=100; every later read: already cleared
            return dict(since_mono=100.0) if VanishingStuckGrouper.reads == 1 else None

        @stuck.setter
        def stuck(self, value):
            pass                     # BreakGrouper.__init__ sets it; ignored here

    g = VanishingStuckGrouper()
    VanishingStuckGrouper.reads = 0
    assert g.status(130.0) == dict(stuck=True, stuck_for_s=30.0)
    assert VanishingStuckGrouper.reads == 1
