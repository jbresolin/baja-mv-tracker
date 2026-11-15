"""Unit tests for the pan/tilt sign (zone/direction.py) and how it is measured while teaching (zone/teach.py).

For a first-time reader: a camera "reports" its pan as a number. On some cameras that number grows when the camera
turns right (sign +1), on others it shrinks (sign -1). The zone must know which, or it turns the camera the wrong
way. These tests check where the sign comes from (config, teach measurement, bench report, default) and that holding
the jog buttons measures it. The end-to-end version (real VISCA packets) is test_direction_e2e.py.
"""
from zone import config as C
from zone import direction
from zone.clock import FakeClock
from zone.ptz.sim import PtzModel, SimulatedPtz
from zone.teach import InquirySource, TeachSession
from zone.zonefile import Keyframe, PathPoint, Zone


def build(sets=(), model=None, raw=None, path="r.json"):
    """Settings as the app builds them, with a made-up report (`raw` = the report file's contents)."""
    cfg = C.load(None, list(sets))
    return C.build(cfg, C.Report(model or PtzModel(), "B", "test", path, raw or {}))


def zone_with(source="inquiry", signs=None):
    kf = lambda p: Keyframe(p, 0, 0, 10 + p, False, source)  # noqa: E731 (tiny helper)
    return Zone(points=[PathPoint(0, 0, kf(0)), PathPoint(100, 0, kf(1))], ptz_signs=dict(signs or {}))


# ------------------------------------------------------------------------------------------------ where it comes from
def test_without_anything_the_sign_is_plus_one_and_marked_default():
    st = build(path="")
    assert st.controller.pan_sign == 1 and st.sign_base_how == {"pan": "default", "tilt": "default"}


def test_a_report_counts_only_if_the_bench_measured_the_sign():
    # the report's model always says pan_sign = 1 (its default): that alone is not a measurement
    assert build(raw={"model": {"pan_sign": 1}}).sign_base_how["pan"] == "default"
    # the speed table's own pan_sign is written only when the bench measured rates
    st = build(model=PtzModel(pan_sign=-1), raw={"speed_table": {"pan_sign": -1, "tilt_sign": 1}})
    assert (st.controller.pan_sign, st.sign_base_how["pan"]) == (-1, "report")
    assert st.sign_base_how["tilt"] == "report"


def test_the_config_wins_over_everything():
    st = build(["ptz.pan_sign=-1"], raw={"speed_table": {"pan_sign": 1}})
    assert (st.controller.pan_sign, st.sign_base_how["pan"]) == (-1, "config")
    # and over the teach measurement too, with a warning when they disagree
    warnings = direction.apply_zone(st, zone_with(signs={"pan": 1}))
    assert st.controller.pan_sign == -1 and any("força -1" in w for w in warnings)


def test_a_bad_config_value_is_refused():
    import pytest
    with pytest.raises(ValueError):
        build(["ptz.pan_sign=2"])


def test_the_teach_measurement_beats_the_report_and_says_so():
    st = build(model=PtzModel(pan_sign=-1), raw={"speed_table": {"pan_sign": -1}})
    warnings = direction.apply_zone(st, zone_with(signs={"pan": 1}))
    assert st.controller.pan_sign == 1 and st.sign_how["pan"] == "teach"
    assert any("relatório" in w for w in warnings)


def test_an_unmeasured_sign_is_a_warning_only_when_the_keyframes_depend_on_it():
    # inquiry / commanded keyframes are numbers in the camera's own frame: they need the right sign
    assert any("NÃO MEDIDO" in w for w in direction.apply_zone(build(path=""), zone_with("inquiry")))
    assert any("NÃO MEDIDO" in w for w in direction.apply_zone(build(path=""), zone_with("commanded")))
    # dead-reckoned keyframes were made with the sign the controller uses: consistent whatever the camera does
    assert direction.apply_zone(build(path=""), zone_with("dead_reckoning")) == []


def test_vote():
    assert direction.vote_sign([]) == (None, True)
    assert direction.vote_sign([-1, -1]) == (-1, True)
    assert direction.vote_sign([1, -1, 1]) == (1, False)     # majority, flagged inconsistent
    assert direction.vote_sign([1, -1]) == (None, False)     # a tie decides nothing


def test_the_zone_file_keeps_the_measured_signs():
    z = zone_with(signs={"pan": -1, "pan_votes": 3, "how": "teach_jog"})
    assert Zone.from_dict(z.to_dict()).ptz_signs == {"pan": -1, "pan_votes": 3, "how": "teach_jog"}
    assert Zone.from_dict({"points": []}).ptz_signs == {}   # old zone files: nothing measured


# ------------------------------------------------------------------------------------------------ measured by the jogs
def jog(source, clock, pan, tilt, seconds):
    """Hold a jog button for `seconds` (the page repeats it every 150 ms), then let go."""
    for _ in range(int(seconds / 0.15)):
        source.jog(pan, tilt, 0)
        clock.advance(0.15)
    source.jog_stop()
    clock.advance(0.5)


def test_jogs_measure_the_sign_of_a_reversed_camera():
    clock = FakeClock(0.0)
    sim = SimulatedPtz(PtzModel(pan_sign=-1, tilt_sign=1), clock)   # pan counts DOWN when turning right
    src = InquirySource(sim, clock)
    jog(src, clock, 6, 0, 1.5)       # right
    jog(src, clock, -6, 0, 1.5)      # left
    jog(src, clock, 0, 6, 1.0)       # up
    assert src.sign_votes == {"pan": [-1, -1], "tilt": [1]}
    assert src.measured_signs() == {"pan": -1, "pan_votes": 2, "pan_consistent": True,
                                    "tilt": 1, "tilt_votes": 1, "tilt_consistent": True}


def test_a_tap_too_short_to_move_the_camera_is_not_a_vote():
    clock = FakeClock(0.0)
    src = InquirySource(SimulatedPtz(PtzModel(), clock), clock)
    src.jog(1, 0, 0)                 # one tap at the slowest speed, released at once
    src.jog_stop()
    assert src.sign_votes == {"pan": [], "tilt": []}


def test_the_teach_session_saves_the_measured_sign_with_the_next_edit():
    clock = FakeClock(0.0)
    sim = SimulatedPtz(PtzModel(pan_sign=-1), clock)
    src = InquirySource(sim, clock)
    seen = []
    session = TeachSession(Zone(), sim, src, on_change=seen.append)
    jog(src, clock, 12, 0, 1.0)
    session.add_point(10, 10, keyframe=True)
    assert session.zone.ptz_signs["pan"] == -1 and session.zone.ptz_signs["how"] == "teach_jog"
    assert seen[-1].ptz_signs["pan"] == -1          # the runtime gets it at once (on_change)
