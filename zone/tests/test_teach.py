"""Tests for zone/teach.py: building a zone by jogging the PTZ and clicking, and the three ways of knowing the pose."""
import pytest

from zone.clock import FakeClock
from zone.ptz.sim import PtzModel, SimulatedPtz
from zone.teach import (CommandedSource, DeadReckoningSource, InquirySource, TeachError, TeachSession,
                        make_pose_source)
from zone.tests.z2_fixtures import settings
from zone.zonefile import Zone


def rig(inquiry_mode="always", zoom_inquiry_mode="always"):
    clock = FakeClock(0.0)
    sim = SimulatedPtz(PtzModel(inquiry_mode=inquiry_mode, zoom_inquiry_mode=zoom_inquiry_mode), clock)
    return sim, clock


def jog(source, clock, sim, pan=6, tilt=0, zoom=0, seconds=1.0):
    """Hold a jog button for `seconds` (the page repeats the jog every 150 ms), then let go."""
    t = 0.0
    while t < seconds:
        source.jog(pan, tilt, zoom)
        clock.advance(0.15)
        t += 0.15
    source.jog_stop()
    clock.advance(1.0)          # let it come to rest
    sim.tick()


def test_teach_a_zone_with_inquiry():
    sim, clock = rig()
    changes = []
    s = TeachSession(Zone(), sim, InquirySource(sim, clock), preset_base=10, on_change=changes.append)
    s.add_point(100, 500)                               # path point upstream
    jog(s.source, clock, sim, pan=-6)
    kf1 = s.add_point(200, 500, keyframe=True)          # entry keyframe: preset 10, pose read from the camera
    jog(s.source, clock, sim, pan=8)
    kf2 = s.add_point(800, 480, keyframe=True)          # preset 11
    assert [kf1.keyframe.preset, kf2.keyframe.preset] == [10, 11]
    assert kf1.keyframe.pan < 0 < kf2.keyframe.pan      # the camera was jogged left, then right
    assert kf2.keyframe.pan == pytest.approx(sim.true_pan_tilt()[0], abs=1)
    assert not kf1.keyframe.approximate and kf1.keyframe.pose_source == "inquiry"
    assert s.zone.entry_preset == 10
    assert len(changes) == 3 and s.dirty                # every edit was pushed to the runtime


def test_the_preset_is_stored_in_the_camera():
    sim, clock = rig()
    s = TeachSession(Zone(), sim, InquirySource(sim, clock))
    jog(s.source, clock, sim, pan=8)
    s.add_point(10, 10, keyframe=True)
    pose = sim.true_pan_tilt()
    jog(s.source, clock, sim, pan=-8, seconds=2)        # move away
    s.goto(0)                                           # "Testar keyframe": recall its preset
    clock.advance(5.0)
    assert sim.true_pan_tilt()[0] == pytest.approx(pose[0], abs=1)


def test_reteach_keeps_the_preset_number_and_takes_the_new_framing():
    sim, clock = rig()
    s = TeachSession(Zone(), sim, InquirySource(sim, clock))
    s.add_point(10, 10, keyframe=True)
    old = s.zone.points[0].keyframe
    jog(s.source, clock, sim, pan=10)
    s.reteach(0)
    new = s.zone.points[0].keyframe
    assert new.preset == old.preset and new.pan > old.pan


def test_edit_operations():
    sim, clock = rig()
    s = TeachSession(Zone(), sim, InquirySource(sim, clock))
    for x in (100, 200, 300):
        s.add_point(x, 500)
    s.move_point(1, 250, 510)
    assert (s.zone.points[1].x, s.zone.points[1].y) == (250, 510)
    s.reorder(2, -1)                                    # swap the last two
    assert [p.x for p in s.zone.points] == [100, 300, 250]
    s.delete_point(0)
    assert [p.x for p in s.zone.points] == [300, 250]
    s.add_point(50, 500, insert_at=0)
    assert s.zone.points[0].x == 50
    s.make_keyframe(1)                                  # attach the current framing to an existing point
    assert s.zone.points[1].keyframe is not None
    s.toggle_keyframe(1)
    assert s.zone.points[1].keyframe is None
    with pytest.raises(TeachError):
        s.reorder(0, -1)                                # can't move the first point further up
    with pytest.raises(TeachError):
        s.delete_point(9)
    with pytest.raises(TeachError):
        s.reteach(0)                                    # not a keyframe


def test_corridor_width_is_validated():
    sim, clock = rig()
    s = TeachSession(Zone(), sim, InquirySource(sim, clock))
    s.set_corridor(80)
    assert s.zone.corridor_half_width_px == 80
    with pytest.raises(TeachError):
        s.set_corridor(2)


def test_save_writes_the_file_and_a_backup(tmp_path):
    sim, clock = rig()
    s = TeachSession(Zone(), sim, InquirySource(sim, clock))
    s.add_point(1, 2)
    f = tmp_path / "zone.json"
    assert s.save(f, image_size=(1920, 1080)) is None
    s.add_point(3, 4)
    backup = s.save(f)
    assert backup.exists() and not s.dirty


def test_keyframe_fails_cleanly_when_the_pose_is_unknown():
    sim, clock = rig(inquiry_mode="never")
    s = TeachSession(Zone(), sim, InquirySource(sim, clock))
    with pytest.raises(TeachError):
        s.add_point(1, 2, keyframe=True)
    assert s.zone.points == []                          # nothing half-added


def test_zoom_without_inquiry_is_asked_for_never_guessed():
    # Some cameras answer the pan/tilt inquiry but not the zoom one. The keyframe's zoom used to be stored as 0
    # without a word (0 = fully wide: the controller would then zoom all the way out at that keyframe). Now the
    # keyframe is refused until the operator types a zoom on the page, and the typed value is recorded as such.
    sim, clock = rig(zoom_inquiry_mode="never")
    src = InquirySource(sim, clock)
    s = TeachSession(Zone(), sim, src)
    with pytest.raises(TeachError, match="zoom"):
        s.add_point(1, 2, keyframe=True)                 # no zoom typed: refused...
    assert s.zone.points == []                           # ...and nothing half-added
    assert src.zoom_readable is False                    # the page shows its "type the zoom" warning from this
    p = s.add_point(1, 2, keyframe=True, zoom=3000)      # with the operator's zoom: accepted
    assert p.keyframe.zoom == 3000 and p.keyframe.pose_source == "inquiry+zoom_manual"
    # it can be corrected later without re-teaching the framing
    s.set_keyframe_zoom(0, "3500")
    assert s.zone.points[0].keyframe.zoom == 3500
    with pytest.raises(TeachError):
        s.set_keyframe_zoom(0, "muito")                  # not a number: refused, value unchanged
    assert s.zone.points[0].keyframe.zoom == 3500


def test_a_camera_that_reads_its_zoom_ignores_a_typed_one():
    sim, clock = rig()                                   # zoom inquiry works
    s = TeachSession(Zone(), sim, InquirySource(sim, clock))
    p = s.add_point(1, 2, keyframe=True, zoom=9999)
    assert p.keyframe.zoom == sim.true_zoom() != 9999   # the camera's own value wins
    assert p.keyframe.pose_source == "inquiry"


# ------------------------------------------------------------------------------------------------ pose sources
def test_commanded_source_needs_home_then_knows_the_pose_by_construction():
    sim, clock = rig(inquiry_mode="never")
    src = CommandedSource(sim, clock, limits=dict(pan=(-2000, 2000), tilt=(-400, 1200), zoom=(0, 16384)))
    assert src.read() is None
    with pytest.raises(TeachError):
        src.nudge(10, 0, 0, 8)                          # unknown start: refuse
    src.home()
    src.nudge(300, 50, 1000, 12)
    src.nudge(-100, 0, 0, 12)
    assert src.read() == (200, 50, 1000)
    clock.advance(10.0)
    pan, tilt = sim.true_pan_tilt()
    assert (round(pan), round(tilt)) == (200, 50)       # the camera really is there
    src.nudge(5000, 0, 0, 12)                           # clamped to the travel limit
    assert src.read()[0] == 2000


def test_dead_reckoning_integrates_the_jog_and_honours_the_ttl():
    sim, clock = rig(inquiry_mode="never")
    st = settings("B")
    src = make_pose_source("dead_reckoning", sim, clock, st)
    assert isinstance(src, DeadReckoningSource) and src.approximate
    src.zero_here()
    src.jog(10, 0, 0)                                   # one jog command and then nothing: the TTL (0.3 s) ends it
    clock.advance(2.0)
    pan = src.read()[0]
    assert pan == pytest.approx(st.pan_map.rate_of(10) * 0.3, abs=1)   # whole raw units
    # the real camera moved about the same (it also has latency and a ramp, hence the tolerance)
    assert sim.true_pan_tilt()[0] == pytest.approx(pan, rel=0.35)


def test_dead_reckoning_resets_on_recall():
    sim, clock = rig(inquiry_mode="never")
    s = TeachSession(Zone(), sim, make_pose_source("dead_reckoning", sim, clock, settings("B")))
    s.source.zero_here()
    s.add_point(1, 1, keyframe=True)
    assert s.zone.points[0].keyframe.approximate
    jog(s.source, clock, sim, pan=8)
    s.goto(0)
    assert s.source.read()[:2] == (0, 0)


def test_pose_source_is_chosen_from_the_report():
    from zone.config import Report, pose_source_for
    assert pose_source_for("auto", Report(PtzModel(inquiry_mode="at_rest"), "B", "", "r.json")) == "inquiry"
    assert pose_source_for("auto", Report(PtzModel(inquiry_mode="never", supports_absolute=True), "B", "",
                                          "r.json")) == "commanded"
    assert pose_source_for("auto", Report(PtzModel(inquiry_mode="never", supports_absolute=False), "B", "",
                                          "r.json")) == "dead_reckoning"
    assert pose_source_for("commanded", Report(PtzModel(), "A", "", "r.json")) == "commanded"   # config wins
