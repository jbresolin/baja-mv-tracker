"""Tests of the bench script: its arithmetic, then whole runs against simulated cameras.

The bench only talks to the `Ptz` interface and a clock, so a full run can use `SimulatedPtz` (or the real driver
on top of the fake camera) with a `FakeClock`. A run of 5-6 simulated minutes takes well under a second. Because
the simulated camera's true numbers are known, the tests check that the bench measures them back.
"""
import json

import pytest

from zone.clock import FakeClock
from zone.ptz import visca_packets as packets
from zone.ptz.fake_camera import FakeTransport, FakeViscaCamera
from zone.ptz.sim import PtzModel, SimulatedPtz
from zone.ptz.visca import ViscaPtz
from zone.tools import bench_analysis as ba
from zone.tools.characterize import BenchOptions, Characterizer, Poll, ScriptedUI, parse_args, summarize

# --- arithmetic -------------------------------------------------------------------------------------------------


def test_fit_rate_is_the_slope():
    samples = [(t / 10, 5.0 + 30.0 * t / 10) for t in range(20)]
    assert ba.fit_rate(samples) == pytest.approx(30.0)
    assert ba.fit_rate(samples[:2]) is None


def test_smoothness_is_zero_for_even_motion_and_large_for_stepped():
    even = [(t * 0.1, t * 3.0) for t in range(10)]
    stepped = [(t * 0.1, (t // 3) * 9.0) for t in range(10)]  # freezes, then jumps
    assert ba.smoothness_cv(even) == pytest.approx(0.0)
    assert ba.smoothness_cv(stepped) > 1.0
    assert ba.count_jumps(stepped) > 0


def test_percentile():
    assert ba.percentile([3, 1, 2], 50) == 2
    assert ba.percentile([0, 10], 90) == pytest.approx(9.0)
    assert ba.percentile([], 50) is None


def test_first_and_last_change():
    samples = [(0.0, 5), (0.1, 5), (0.2, 6), (0.3, 8), (0.4, 8)]
    assert ba.first_change(samples, baseline=5, after=0.05) == 0.2
    assert ba.last_change(samples, after=0.0) == 0.3


def test_edge_time_uses_send_time_for_silent_polls():
    polls = [Poll(0.0, 0.002, (0, 0)), Poll(0.05, 0.052, (0, 0)), Poll(0.10, 0.30, None), Poll(0.30, 0.50, None),
             Poll(0.50, 0.502, (9, 0))]
    # went silent between the poll at 0.05 (seen at 0.051) and the one sent at 0.10
    assert ba.edge_time(polls, 0.0, answered_before=True) == pytest.approx((0.051 + 0.10) / 2)
    # answered again between the silent poll sent at 0.30 and the answer seen at 0.501
    assert ba.edge_time(polls, 0.2, answered_before=False) == pytest.approx((0.30 + 0.501) / 2)


def test_feedback_classification():
    assert ba.classify_feedback(1.0, changed=True) == "yes"
    assert ba.classify_feedback(1.0, changed=False) == "frozen"
    assert ba.classify_feedback(0.0, changed=False) == "no"


def test_verdict_rule():
    assert ba.verdict("yes", 10.0)[0] == "A"
    assert ba.verdict("yes", 5.0)[0] == "B"
    assert ba.verdict("frozen", 50.0)[0] == "B"
    assert ba.verdict("no", None)[0] == "B"
    assert ba.verdict(None, None)[0] == "B"


# --- B1: is the feedback really live? (answers vs. updates) ------------------------------------------------------
#
# A camera can answer a position inquiry 50 times a second and still be useless for feedback if the number it
# answers with is only refreshed twice a second (a "stale cache"). These tests feed made-up (time, position) samples
# into the rule that decides whether a poll rate is "ok", so we can see exactly which camera passes and why.


def samples_from(rate_hz, duration_s, true_position):
    """Poll an imaginary camera at `rate_hz` for `duration_s`: a list of (time, answered position)."""
    count = int(duration_s * rate_hz)
    return [(k / rate_hz, true_position(k / rate_hz)) for k in range(count)]


def test_a_live_camera_passes_at_50_hz():
    # The camera turns at 100 units/s and every answer is the position right now, rounded to whole units.
    samples = samples_from(50, 1.5, lambda t: round(100 * t))
    judged = ba.judge_feedback_rate(50, sent=len(samples), samples=samples)
    assert judged["ok"], judged["why"]
    assert judged["answers_hz"] == pytest.approx(50, abs=0.5)
    assert judged["updates_hz"] == pytest.approx(50, abs=0.5)  # the value changes on every answer


def test_a_stale_cache_camera_fails_even_though_it_answers_at_50_hz():
    # Same motion, but the camera only refreshes the number it reports every 0.5 s (2 Hz). Between refreshes it
    # keeps repeating the old value. It answers every inquiry, so "did it answer?" alone would call this live.
    samples = samples_from(50, 1.5, lambda t: round(100 * (int(t * 2) / 2)))
    judged = ba.judge_feedback_rate(50, sent=len(samples), samples=samples)
    assert not judged["ok"]
    assert judged["answers_hz"] == pytest.approx(50, abs=0.5)  # it does answer fast...
    assert judged["updates_hz"] == pytest.approx(2, abs=0.2)  # ...but the value changes only twice a second
    assert "stale" in judged["why"]
    assert judged["jumps"] > 0  # each refresh catches up 50 units at once


def test_a_slow_but_live_camera_is_not_mistaken_for_a_stale_one():
    # Moving only 20 units/s, polled at 50 Hz: 0.4 units per poll, so even a perfect, live camera reports the same
    # whole number several times in a row. The rule expects that (expected change fraction 0.4), so it still passes.
    samples = samples_from(50, 1.5, lambda t: round(20 * t))
    judged = ba.judge_feedback_rate(50, sent=len(samples), samples=samples)
    assert judged["expected_change_fraction"] == pytest.approx(0.4, abs=0.01)
    assert judged["ok"], judged["why"]


def test_unanswered_inquiries_fail_the_rate():
    # Only every other inquiry gets an answer: 50% answered is below the 90% rule.
    samples = samples_from(25, 1.5, lambda t: round(100 * t))
    judged = ba.judge_feedback_rate(50, sent=2 * len(samples), samples=samples)
    assert not judged["ok"] and "answered" in judged["why"]


def test_verdict_needs_10_updates_per_second():
    # Polled at 20 Hz and passing, but the value changed only 6 times a second: not enough for Branch A.
    branch, reason = ba.verdict("yes", 20.0, update_rate_hz=6.0)
    assert branch == "B" and "6 Hz" in reason
    assert ba.verdict("yes", 20.0, update_rate_hz=20.0)[0] == "A"
    assert ba.verdict("yes", 10.0, update_rate_hz=9.7)[0] == "A"  # within the 5% counting tolerance
    assert ba.verdict("yes", None, refresh_hz=2.0)[0] == "B"


# --- whole runs -------------------------------------------------------------------------------------------------


def run_on_sim(model: PtzModel, options: BenchOptions | None = None, ui=None, tmp_path=None):
    clock = FakeClock()
    sim = SimulatedPtz(model, clock)
    bench = Characterizer(sim, clock, ui or ScriptedUI(), options or BenchOptions(quick=True), tmp_path,
                          {"ip": "sim", "port": 0})
    return bench.run(), sim


def run_through_driver(model: PtzModel, options: BenchOptions | None = None, **camera_options):
    """The real driver, the fake VISCA camera and the simulator, all on one fake clock."""
    clock = FakeClock()
    sim = SimulatedPtz(model, clock)
    camera = FakeViscaCamera(sim)
    for name, value in camera_options.items():
        setattr(camera, name, value)
    ptz = ViscaPtz("fake", clock=clock, transport=FakeTransport(camera, clock), watchdog=False)
    report = Characterizer(ptz, clock, ScriptedUI(), options or BenchOptions(quick=True)).run()
    return report, sim, camera


def test_camera_with_live_feedback_gives_branch_a_and_measures_the_speed_table(tmp_path):
    truth = PtzModel()
    report, _ = run_on_sim(truth, tmp_path=tmp_path)
    assert report["verdict"]["branch"] == "A"
    assert report["feedback"]["while_moving"] == "yes"
    assert report["feedback"]["max_rate_hz"] >= 10
    measured = PtzModel.from_report(tmp_path / "report.json")
    for step in (1, 8, 16):
        assert measured.pan_speed_table[step] == pytest.approx(truth.pan_speed_table[step], rel=0.03)
        assert measured.tilt_speed_table[step] == pytest.approx(truth.tilt_speed_table[step], rel=0.03)
    assert measured.pan_sign == 1 and measured.inquiry_mode == "always"
    lat = report["latency"]
    assert lat["method"] == "polling"
    assert lat["latency_s_p50"] == pytest.approx(truth.command_latency_s, abs=0.06)
    assert summarize(report).rstrip().splitlines()[-1].startswith("VERDICT: Branch A")


def test_camera_answering_only_at_rest_gives_branch_b():
    truth = PtzModel(inquiry_mode="at_rest")
    report, _, _ = run_through_driver(truth)
    assert report["verdict"]["branch"] == "B"
    assert report["feedback"]["while_moving"] == "no"
    assert report["speed_table"]["method"] == "timed"
    model = PtzModel.from_dict(report["model"])
    assert model.inquiry_mode == "at_rest"
    assert model.pan_speed_table[8] == pytest.approx(truth.pan_speed_table[8], rel=0.05)
    assert report["latency"]["method"] == "inquiry_silence"
    assert model.command_latency_s == pytest.approx(truth.command_latency_s, abs=0.06)


def test_frozen_feedback_gives_branch_b():
    report, _ = run_on_sim(PtzModel(inquiry_mode="frozen_while_moving"), BenchOptions(quick=True, skip=("zoom",)))
    assert report["feedback"]["while_moving"] == "frozen"
    assert report["verdict"]["branch"] == "B"
    assert report["model"]["inquiry_mode"] == "frozen_while_moving"


def test_camera_without_inquiry_still_gets_a_report():
    report, _, _ = run_through_driver(PtzModel(inquiry_mode="never", zoom_inquiry_mode="never"))
    assert report["feedback"]["pan_tilt_at_rest"]["supported"] is False
    assert report["speed_table"]["method"] == "unsupported"
    assert report["zoom"]["inquiry_at_rest"] is False
    assert report["verdict"]["branch"] == "B"
    # No section failed ("driver_stats" is left out: it always has an "errors" counter).
    assert "error" not in json.dumps({k: v for k, v in report.items() if k not in ("model", "driver_stats")})


def test_unsupported_absolute_positioning_is_recorded():
    report, _, _ = run_through_driver(PtzModel(supports_absolute=False, supports_relative=False),
                                      BenchOptions(quick=True, skip=("zoom", "repeatability")))
    assert report["absolute"]["supported"] is False
    assert report["absolute"]["relative"]["supported"] is False
    assert report["model"]["supports_absolute"] is False


def test_a_failing_section_is_recorded_and_the_run_continues():
    class BrokenZoom(SimulatedPtz):
        def zoom_velocity(self, step, ttl=None):  # only the zoom section uses it
            raise RuntimeError("zoom board on fire")

    clock = FakeClock()
    sim = BrokenZoom(PtzModel(), clock)
    report = Characterizer(sim, clock, ScriptedUI(), BenchOptions(quick=True)).run()
    assert "zoom board on fire" in report["zoom"]["error"]
    assert "recalls" in report["repeatability"]  # sections after the failure still ran
    assert report["verdict"]["branch"] == "A"


def test_nothing_moves_without_confirmation():
    clock = FakeClock()
    sim = SimulatedPtz(PtzModel(), clock)
    report = Characterizer(sim, clock, ScriptedUI(confirm=False), BenchOptions()).run()
    assert report["aborted"]
    moving_commands = [c for c in sim.commands if c[1] not in ("stop", "zoom_stop")]
    assert moving_commands == []


def test_units_from_two_marks():
    clock = FakeClock()
    sim = SimulatedPtz(PtzModel(), clock)

    def operator_aims_at_mark_b(message):
        if "mark B (pan)" in message:
            sim.pan.position += 1200.0  # as if João turned the camera 90 degrees with the joystick

    ui = ScriptedUI(numbers=[90.0, None], on_pause=operator_aims_at_mark_b)
    options = BenchOptions(quick=True, skip=("feedback", "speed_table", "latency", "absolute", "zoom", "repeatability"))
    report = Characterizer(sim, clock, ui, options).run()
    assert report["units"]["pan_units_per_degree"] == pytest.approx(1200 / 90, rel=1e-3)
    assert report["units"]["tilt_source"] == "skipped"
    assert report["model"]["pan_units_per_degree"] == pytest.approx(1200 / 90, rel=1e-3)


def test_documented_units_are_used_as_given():
    options = BenchOptions(quick=True, pan_units_per_degree=14.4, skip=("feedback", "speed_table", "latency",
                                                                         "absolute", "zoom", "repeatability"))
    report, _ = run_on_sim(PtzModel(), options)
    assert report["units"]["pan_units_per_degree"] == 14.4


def test_fov_stops_pause_for_screenshots():
    ui = ScriptedUI()
    options = BenchOptions(quick=True, skip=("feedback", "speed_table", "latency", "absolute", "repeatability", "units"))
    report, _ = run_on_sim(PtzModel(), options, ui=ui)
    assert len(report["zoom"]["fov_positions"]) == 9
    assert sum("screenshot" in m for m in ui.messages) == 9
    assert report["zoom"]["range"] == [0, 16384]


def test_repeatability_measures_spread_and_settle_time():
    report, _ = run_on_sim(PtzModel(), BenchOptions(quick=True, skip=("speed_table", "latency", "absolute", "zoom")))
    rep = report["repeatability"]
    assert rep["recalls"] == 10 and len(rep["arrivals"]) == 10
    assert rep["pan_spread"] == 0  # the simulator is perfectly repeatable
    assert report["model"]["preset_settle_s"] == rep["settle_s_p90"] > 0


def test_report_is_written_after_every_section(tmp_path):
    run_on_sim(PtzModel(), BenchOptions(quick=True, skip=("speed_table", "latency", "absolute", "zoom")),
               tmp_path=tmp_path)
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["schema"] == "zone.ptz.characterization/1"
    assert {"connection", "feedback", "repeatability", "model", "verdict"} <= set(saved)


def test_command_line_defaults():
    args = parse_args(["--ip", "10.0.0.5", "--out", "reports/a"])
    assert (args.port, args.timeout, args.home_preset, args.quick, args.yes) == (52381, 0.2, None, False, False)


# --- B1 whole runs: stale cache vs. live ---------------------------------------------------------------------------


def test_stale_cache_camera_gives_branch_b(tmp_path):
    # A simulated camera that answers every inquiry at once, but refreshes the position it reports only twice a
    # second. Before the fix the bench said "Branch A at 50 Hz" for this camera; now it must say Branch B.
    stale = PtzModel(inquiry_refresh_hz=2.0, inquiry_poll_hz=None)
    report, _ = run_on_sim(stale, BenchOptions(quick=True, skip=("zoom", "repeatability")), tmp_path=tmp_path)
    feedback = report["feedback"]
    assert report["verdict"]["branch"] == "B"
    assert feedback["max_rate_hz"] is None  # no poll rate passed
    assert feedback["refresh_limited"] is True
    assert feedback["update_rate_hz"] == pytest.approx(2.0, abs=0.3)
    fifty = feedback["rates"]["50"]
    assert fifty["answers_hz"] >= 45 and fifty["updates_hz"] < 3 and not fifty["ok"]
    # The simulator model learned from this report reproduces the stale cache.
    assert report["model"]["inquiry_refresh_hz"] == pytest.approx(2.0, abs=0.3)
    # Without live feedback, the speed table falls back to timed moves read at rest.
    assert report["speed_table"]["method"] == "timed"
    # The summary shows, per rate, answers/s, updates/s, cv, jumps and why it passed or failed.
    summary = (summarize(report))
    line_50 = next(line for line in summary.splitlines() if line.strip().startswith("50 Hz:"))
    for word in ("answers/s", "updates/s", "cv", "jumps", "NOT ok", "stale"):
        assert word in line_50
    assert "refreshes" in summary.rstrip().splitlines()[-1]  # the verdict line says why


def test_stale_cache_camera_gives_branch_b_through_the_real_driver():
    # Same, but through the VISCA driver and the fake VISCA camera, as on the bench tomorrow.
    report, _, _ = run_through_driver(PtzModel(inquiry_refresh_hz=2.0, inquiry_poll_hz=None),
                                      BenchOptions(quick=True, skip=("zoom", "repeatability", "absolute")))
    assert report["verdict"]["branch"] == "B"
    assert report["feedback"]["rates"]["50"]["answers_hz"] >= 40


def test_live_camera_gives_branch_a_through_the_real_driver():
    # The control case: a camera whose every answer is fresh gets Branch A, with updates/s close to answers/s.
    report, _, _ = run_through_driver(PtzModel(inquiry_poll_hz=None),
                                      BenchOptions(quick=True, skip=("zoom", "repeatability", "absolute")))
    assert report["verdict"]["branch"] == "A"
    best = report["feedback"]["rates"][str(int(report["feedback"]["max_rate_hz"]))]
    assert best["ok"] and best["updates_hz"] >= 0.9 * best["answers_hz"]
    # The driver's own counters end up in the report and the summary (S3: unmatched replies among them).
    assert report["driver_stats"]["unmatched_replies"] == 0
    assert report["driver_stats"]["sequence_echo"] is True
    assert "0 unmatched replies" in summarize(report)


def test_a_camera_refreshing_at_20_hz_passes_at_20_but_not_at_50():
    # Refreshing 20 times a second is fast enough for Branch A, but polling it at 50 Hz is wasted: the bench must
    # say so (50 Hz not ok) and settle on 20 Hz.
    report, _ = run_on_sim(PtzModel(inquiry_refresh_hz=20.0, inquiry_poll_hz=None),
                           BenchOptions(quick=True, skip=("zoom", "repeatability", "absolute")))
    assert report["feedback"]["rates"]["50"]["ok"] is False
    assert report["feedback"]["max_rate_hz"] == 20.0
    assert report["verdict"]["branch"] == "A"


# --- S2: the bench retries its stops -------------------------------------------------------------------------------


def test_safe_stop_retries_a_lost_stop():
    # The network loses the first two stop packets. The bench's safe_stop tries up to 3 times, so the third gets
    # through and the camera stops, even without the driver's dead-man thread (switched off in these tests).
    clock = FakeClock()
    sim = SimulatedPtz(PtzModel(), clock)
    camera = FakeViscaCamera(sim)
    ptz = ViscaPtz("fake", clock=clock, transport=FakeTransport(camera, clock), watchdog=False)
    bench = Characterizer(ptz, clock, ScriptedUI(), BenchOptions(quick=True))
    ptz.pan_tilt_velocity(10, 0)
    camera.lose_next[packets.pan_tilt_stop()] = 2
    bench.safe_stop()
    clock.advance(1.0)
    assert len(camera.lost) == 2
    assert not sim.is_moving()


# --- S4: top speed steps -------------------------------------------------------------------------------------------


def test_a_camera_that_does_not_answer_max_speed_gets_the_safe_defaults():
    # The camera ignores the max-speed inquiry. The bench then uses pan 24 / tilt 20 (0x14) and never sends a
    # tilt speed above 20, so it doesn't probe steps the camera may refuse.
    report, _, camera = run_through_driver(
        PtzModel(), BenchOptions(quick=True, skip=("feedback", "latency", "absolute", "zoom", "repeatability", "units")),
        behaviour={"inquire_max_speed": "silent"})
    connection = report["connection"]
    assert (connection["pan_max_step_used"], connection["tilt_max_step_used"]) == (24, 20)
    tilt_speeds = [payload[5] for name, payload in camera.received if name == "pan_tilt_drive"]
    assert max(tilt_speeds) == 20
    assert "20" in report["speed_table"]["tilt"] and "23" not in report["speed_table"]["tilt"]
