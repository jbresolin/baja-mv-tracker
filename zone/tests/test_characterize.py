"""Tests of the bench script: its arithmetic, then whole runs against simulated cameras.

The bench only talks to the `Ptz` interface and a clock, so a full run can use `SimulatedPtz` (or the real driver
on top of the fake camera) with a `FakeClock`. A run of 5-6 simulated minutes takes well under a second. Because
the simulated camera's true numbers are known, the tests check that the bench measures them back.
"""
import json

import pytest

from zone.clock import FakeClock
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
    assert "error" not in json.dumps({k: v for k, v in report.items() if k != "model"})


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
