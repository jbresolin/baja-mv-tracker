"""Tests of the simulated PTZ's kinematics, inquiry behaviour and model loading.

The sim only moves when its clock does, so each test sends a command, moves the fake clock forward by a chosen
amount, and checks the position against a value worked out by hand from the model's numbers.

A worked example used below: speed table 100 units/s, latency 0.1 s, ramp 0.2 s (linear). One second after the
command, the camera has
  - waited 0.1 s                                   ->  0 units
  - ramped 0 -> 100 units/s over 0.2 s (avg 50)    -> 10 units
  - run at 100 units/s for the remaining 0.7 s     -> 70 units
  = 80 units.
"""
import json

import pytest

from zone.clock import FakeClock
from zone.ptz.interface import ERROR
from zone.ptz.sim import PtzModel, SimulatedPtz


def simple_model(**overrides) -> PtzModel:
    """A model with round numbers so positions can be worked out by hand."""
    values = dict(
        pan_speed_table={1: 10.0, 2: 100.0},
        tilt_speed_table={1: 10.0, 2: 50.0},
        zoom_speed_table={1: 1000.0},
        command_latency_s=0.1, stop_latency_s=0.1, ramp_s=0.2, stop_ramp_s=0.2,
        pan_limits=(-1000.0, 1000.0), tilt_limits=(-200.0, 200.0), zoom_limits=(0.0, 4000.0),
        preset_move_speed=500.0,
    )
    values.update(overrides)
    return PtzModel(**values)


def make(**overrides) -> tuple[FakeClock, SimulatedPtz]:
    clock = FakeClock()
    return clock, SimulatedPtz(simple_model(**overrides), clock)


def test_needs_a_clock():
    with pytest.raises(ValueError):
        SimulatedPtz(PtzModel())


# --- velocity, latency, ramp ------------------------------------------------------------------------------------


def test_nothing_happens_before_the_latency():
    clock, sim = make()
    sim.pan_tilt_velocity(2, 0)
    clock.advance(0.1)
    assert sim.true_pan_tilt() == (0.0, 0.0)


def test_worked_example_latency_ramp_steady():
    clock, sim = make()
    sim.pan_tilt_velocity(2, 0)
    clock.advance(1.0)
    assert sim.true_pan_tilt()[0] == pytest.approx(80.0)


def test_negative_step_moves_the_other_way():
    clock, sim = make()
    sim.pan_tilt_velocity(0, -2)
    clock.advance(1.0)
    assert sim.true_pan_tilt() == pytest.approx((0.0, -40.0))  # tilt step 2 = 50 units/s


def test_stop_has_latency_and_ramp_too():
    clock, sim = make()
    sim.pan_tilt_velocity(2, 0)
    clock.advance(1.0)  # at 80 units, moving at 100 units/s
    sim.stop()
    clock.advance(2.0)
    # 0.1 s more at full speed (10) + ramp down 100 -> 0 over 0.2 s (10)
    assert sim.true_pan_tilt()[0] == pytest.approx(100.0)
    assert not sim.is_moving()


def test_changing_speed_ramps_from_the_current_speed():
    clock, sim = make()
    sim.pan_tilt_velocity(1, 0)  # 10 units/s
    clock.advance(1.0)  # 0.1 latency + 0.2 ramp (1 unit) + 0.7 * 10 = 8 units
    start = sim.true_pan_tilt()[0]
    assert start == pytest.approx(8.0)
    sim.pan_tilt_velocity(2, 0)  # to 100 units/s
    clock.advance(1.0)
    # 0.1 s at 10 (1) + ramp 10 -> 100 over 0.2 s (11) + 0.7 s at 100 (70) = 82
    assert sim.true_pan_tilt()[0] - start == pytest.approx(82.0)


def test_steps_beyond_the_table_use_the_top_speed():
    clock, sim = make(ramp_s=0.0, command_latency_s=0.0)
    sim.pan_tilt_velocity(24, 0)
    clock.advance(1.0)
    assert sim.true_pan_tilt()[0] == pytest.approx(100.0)


def test_missing_steps_are_interpolated():
    clock, sim = make(pan_speed_table={1: 10.0, 3: 30.0}, ramp_s=0.0, command_latency_s=0.0)
    sim.pan_tilt_velocity(2, 0)
    clock.advance(1.0)
    assert sim.true_pan_tilt()[0] == pytest.approx(20.0)


def test_asymmetric_speed_table():
    clock, sim = make(pan_speed_table_negative={2: 60.0}, ramp_s=0.0, command_latency_s=0.0)
    sim.pan_tilt_velocity(-2, 0)
    clock.advance(1.0)
    assert sim.true_pan_tilt()[0] == pytest.approx(-60.0)


def test_travel_limits_stop_the_axis():
    clock, sim = make()
    sim.pan_tilt_velocity(0, 2)
    clock.advance(10.0)
    assert sim.true_pan_tilt()[1] == pytest.approx(200.0)
    assert not sim.is_moving()


def test_reported_sign_can_be_inverted():
    clock, sim = make(pan_sign=-1, ramp_s=0.0, command_latency_s=0.0)
    sim.pan_tilt_velocity(2, 0)
    clock.advance(1.0)
    assert sim.inquire_pan_tilt() == (-100, 0)


# --- the dead-man (TTL) -----------------------------------------------------------------------------------------


def test_ttl_stops_the_camera():
    clock, sim = make()
    sim.pan_tilt_velocity(2, 0, ttl=0.5)
    clock.advance(5.0)
    # The TTL runs out at 0.5 s; the stop takes effect 0.1 s later. So:
    #   0.1-0.3 ramp up (10 units), 0.3-0.6 full speed (30), 0.6-0.8 ramp down (10) = 50
    assert sim.true_pan_tilt()[0] == pytest.approx(50.0)
    assert not sim.is_moving()


def test_renewed_ttl_keeps_moving():
    clock, sim = make()
    for _ in range(20):
        sim.pan_tilt_velocity(2, 0, ttl=0.5)
        clock.advance(0.25)
    sim.tick()
    assert sim.is_moving()


def test_ttl_fires_on_tick_without_other_calls():
    clock, sim = make()
    sim.pan_tilt_velocity(2, 0, ttl=0.3)
    clock.advance(2.0)
    sim.tick()
    assert not sim.is_moving()


def test_zoom_ttl():
    clock, sim = make()
    sim.zoom_velocity(1, ttl=0.3)
    clock.advance(3.0)
    assert not sim.is_moving()
    assert 0 < sim.true_zoom() < 4000


# --- position moves ---------------------------------------------------------------------------------------------


def test_preset_store_and_recall():
    clock, sim = make()
    sim.pan_tilt_velocity(2, 1)
    clock.advance(2.0)
    sim.stop()
    clock.advance(1.0)
    stored = sim.true_pan_tilt()
    sim.preset_store(3)
    sim.home()
    clock.advance(3.0)
    assert sim.true_pan_tilt() == (0.0, 0.0)
    sim.preset_recall(3)
    clock.advance(0.1 + stored[0] / 500.0 - 0.05)  # latency + almost all the travel at preset_move_speed
    assert sim.is_moving()
    clock.advance(0.1)
    assert sim.true_pan_tilt() == pytest.approx(stored)
    assert not sim.is_moving()


def test_define_preset_stores_without_moving():
    clock, sim = make(pan_sign=-1)
    sim.define_preset(5, pan=-200, tilt=40, zoom=1000)  # raw, as the camera would report it
    clock.advance(1.0)
    assert sim.true_pan_tilt() == (0.0, 0.0)
    sim.preset_recall(5)
    clock.advance(5.0)
    assert sim.inquire_pan_tilt() == (-200, 40)
    assert sim.true_zoom() == pytest.approx(1000.0)


def test_absolute_and_relative():
    clock, sim = make()
    sim.absolute_position(300, -50, 2, 2)
    clock.advance(0.1 + 3.0 + 0.01)  # 300 units at 100 units/s
    assert sim.true_pan_tilt() == pytest.approx((300.0, -50.0))
    sim.relative_position(-100, 20, 2, 2)
    clock.advance(2.0)
    assert sim.true_pan_tilt() == pytest.approx((200.0, -30.0))


def test_unsupported_commands_return_an_error():
    clock, sim = make(supports_absolute=False, supports_zoom_direct=False)
    assert sim.absolute_position(1, 1, 1, 1).status == ERROR
    assert sim.zoom_direct(100).status == ERROR


def test_zoom_velocity_and_direct():
    clock, sim = make(ramp_s=0.0)
    sim.zoom_velocity(1)
    clock.advance(1.1)
    assert sim.true_zoom() == pytest.approx(1000.0)
    sim.zoom_direct(3000)
    clock.advance(5.0)
    assert sim.true_zoom() == pytest.approx(3000.0)
    sim.zoom_velocity(1)
    clock.advance(5.0)
    assert sim.true_zoom() == pytest.approx(4000.0)  # zoom limit


# --- inquiry behaviour ------------------------------------------------------------------------------------------


def moving_sim(mode: str) -> tuple[FakeClock, SimulatedPtz]:
    clock, sim = make(inquiry_mode=mode, zoom_inquiry_mode=mode)
    sim.pan_tilt_velocity(2, 0)
    clock.advance(1.0)
    return clock, sim


def test_inquiry_always():
    _, sim = moving_sim("always")
    assert sim.inquire_pan_tilt() == (80, 0)


def test_inquiry_at_rest_only():
    clock, sim = moving_sim("at_rest")
    assert sim.inquire_pan_tilt() is None
    assert sim.inquire_zoom() == 0  # zoom isn't moving
    sim.stop()
    clock.advance(1.0)
    assert sim.inquire_pan_tilt() == (100, 0)


def test_inquiry_never():
    _, sim = moving_sim("never")
    assert sim.inquire_pan_tilt() is None
    assert sim.last_inquiry.status == "timeout"


def test_inquiry_frozen_while_moving():
    clock, sim = moving_sim("frozen_while_moving")
    assert sim.inquire_pan_tilt() == (0, 0)  # the last at-rest value
    sim.stop()
    clock.advance(1.0)
    assert sim.inquire_pan_tilt() == (100, 0)


def test_inquiry_latency_returns_an_older_pose():
    clock, sim = make(inquiry_latency_s=0.5, ramp_s=0.0, command_latency_s=0.0)
    sim.pan_tilt_velocity(2, 0)
    clock.advance(1.0)
    assert sim.inquire_pan_tilt() == (50, 0)  # where it was 0.5 s ago


def test_invalid_inquiry_mode_is_refused():
    with pytest.raises(ValueError):
        PtzModel(inquiry_mode="sometimes")


# --- determinism and model files --------------------------------------------------------------------------------


def test_same_commands_give_the_same_result():
    results = []
    for _ in range(2):
        clock, sim = make()
        sim.pan_tilt_velocity(2, -1, ttl=0.7)
        clock.advance(0.33)
        sim.zoom_velocity(1)
        clock.advance(1.234)
        results.append((sim.true_pan_tilt(), sim.true_zoom()))
    assert results[0] == results[1]


def test_model_round_trips_through_json(tmp_path):
    model = simple_model(inquiry_mode="at_rest", pan_speed_table_negative={1: 9.0}, inquiry_poll_hz=None)
    text = json.dumps({"schema": "x", "model": model.to_dict(), "verdict": {"branch": "B"}})
    assert json.loads(text)["model"]["pan_speed_table"] == {"1": 10.0, "2": 100.0}  # string keys in JSON
    path = tmp_path / "report.json"
    path.write_text(text)
    assert PtzModel.from_report(path) == model
    assert PtzModel.from_report(json.loads(text)) == model


def test_missing_and_unknown_keys():
    model = PtzModel.from_dict({"command_latency_s": 0.3, "something_new": 1})
    assert model.command_latency_s == 0.3
    assert model.ramp_s == PtzModel().ramp_s


def test_default_model_looks_like_a_visca_camera():
    model = PtzModel()
    assert model.pan_step_max == 24 and model.tilt_step_max == 23 and model.zoom_step_max == 7
    rates = [model.pan_speed_table[s] for s in range(1, 25)]
    assert rates == sorted(rates)
