"""End-to-end: synthetic cars move along a synthetic racing line; a perfect detector feeds rt's ByteTrack; the zone
runtime (progress -> director -> controller) drives the simulated PTZ on a fake clock. We check the framing.

The fake world (zone/testing.py): a racing line ~105 m long with a bend; the zone is 60 m of it; cars do ~10 m/s and
brake to 5 m/s in the bend. The PTZ stands 22-35 m away, so it must pan ~130 degrees during a pass."""
import json
import math

import numpy as np
import pytest

from rt.tracker import ByteTracker
from zone import director as D
from zone.clock import FakeClock
from zone.ptz.sim import PtzModel, SimulatedPtz
from zone.runtime import ControlLog, ZoneRuntime
from zone.source import ZoneFrame
from zone.testing import UNITS_PER_DEGREE, Car, Scene, boxes_at
from zone.tests.z2_fixtures import settings, straight_zone


def world(branch, cars_fn, t_end, inquiry_mode=None, log=None, sets=()):
    scene = Scene()
    zone = scene.make_zone()
    model = PtzModel(inquiry_mode=inquiry_mode or ("always" if branch == "A" else "never"),
                     pan_units_per_degree=UNITS_PER_DEGREE, tilt_units_per_degree=UNITS_PER_DEGREE)
    st = settings(branch, model=model, sets=sets)
    clock = FakeClock(0.0)
    ptz = SimulatedPtz(model, clock)
    for k in zone.keyframes():
        ptz.define_preset(k.preset, k.pan, k.tilt, k.zoom)
    rt = ZoneRuntime(st, zone, ptz, clock, log=log(clock) if log else None)
    cars = cars_fn(scene)
    trk = ByteTracker()
    rt.set_mode(D.AUTO)
    errors = []          # (car index, s_true, error in degrees) while a car is followed
    for k in range(int(t_end * 10)):
        t = k * 0.1
        clock.advance(max(0.0, t + 0.05 - clock.monotonic()))       # 50 ms of detection time
        b, s = boxes_at(scene, cars, t)
        res = rt.cycle(ZoneFrame(None, t, 1.8e9 + t, k), trk.update(b, s))
        st_d = rt.director.state()
        if st_d["sub"] == D.TRACKING and res.command is not None:
            now = clock.monotonic()
            tv = res.views[st_d["locked_id"]]
            # which real car is that track? the one whose ground point is nearest to the track's
            best = min(((abs(scene.image_of(scene.world_at(c.arc_at(now)))[0] - tv.ground[0]), i, c.arc_at(now))
                        for i, c in enumerate(cars) if c.arc_at(now) is not None))
            _, i, arc = best
            want = scene.true_pose(scene.world_at(arc))
            pan, tilt = ptz.true_pan_tilt()
            errors.append((i, scene.s_of_arc(arc), math.hypot(pan - want[0], tilt - want[1]) / UNITS_PER_DEGREE))
    return rt, ptz, errors


@pytest.mark.parametrize("branch", ["A", "B"])
def test_follows_cars_through_the_zone(branch):
    rt, ptz, errors = world(branch, lambda sc: [Car(sc, 2.0), Car(sc, 20.0, factor=1.25)], 40)
    assert rt.director.passes_followed == 2
    inside = [e for _, s, e in errors if 0.0 < s < 0.95]
    # with the default ease-in: mean error under 2 degrees, never more than 4
    assert np.mean(inside) < 2.0, np.mean(inside)
    assert max(inside) < 4.0, max(inside)
    # back at the entry afterwards
    clock_pose = ptz.true_pan_tilt()
    entry = rt.entry_pose()
    assert clock_pose[0] == pytest.approx(entry.pan, abs=2)


def test_without_ease_in_the_camera_starts_behind():
    # zone.ease_in = 0 (and s_lock = 0): the brief's literal rule, the camera holds the entry pose until s = 0. The
    # car then crosses the entry at full speed while the camera is still standing: it starts several degrees behind.
    # This is why ease_in is on by default.
    _, _, errors = world("B", lambda sc: [Car(sc, 2.0, factor=1.25)], 15,
                         sets=["zone.ease_in=0", "director.s_lock=0"])
    _, _, eased = world("B", lambda sc: [Car(sc, 2.0, factor=1.25)], 15)
    worst = max(e for _, s, e in errors if 0.0 < s < 0.95)
    worst_eased = max(e for _, s, e in eased if 0.0 < s < 0.95)
    assert worst > 6.0 and worst_eased < worst / 2


def test_a_car_already_deep_in_the_zone_is_not_chased():
    # car 2 starts 1.5 s after car 1: when the camera is back at the entry, car 2 is mid-zone -> skipped
    rt, _, errors = world("B", lambda sc: [Car(sc, 2.0), Car(sc, 3.5)], 30)
    assert rt.director.passes_followed == 1
    assert {i for i, _, _ in errors} == {0}


def test_a_car_that_stops_is_dropped():
    sc_cars = lambda sc: [Car(sc, 2.0, stop_at=sc.arc_entry + 25)]   # breaks down mid-zone
    rt, _, _ = world("B", sc_cars, 20)
    assert rt.director.last_drop_reason == "stopped"
    assert rt.director.sub in (D.RETURNING, D.WAITING)


def test_the_control_log_has_one_line_per_cycle(tmp_path):
    f = tmp_path / "zone.jsonl"
    rt, _, _ = world("B", lambda sc: [Car(sc, 1.0)], 12, log=lambda clock: ControlLog(f, clock, 1))
    rt.log.close()
    lines = [json.loads(x) for x in f.read_text().splitlines()]
    cycles = [x for x in lines if x["type"] == "cycle"]
    assert len(cycles) == 120
    c = next(x for x in cycles if x["cmd"])
    # every cycle carries the grab time, the tracks, the director state, and (when following) the commands
    assert {"mono", "wall", "t_grab", "t_grab_wall", "tracks", "director"} <= set(c)
    assert {"target", "steps", "pan_now", "s", "v"} <= set(c["cmd"])
    assert any(x["type"] == "director" and x["event"] == "lock" for x in lines)


# ------------------------------------------------------------------------------------------------ runtime rules
def make_runtime(zone=None):
    clock = FakeClock(0.0)
    ptz = SimulatedPtz(PtzModel(), clock)
    return ZoneRuntime(settings("B"), zone or straight_zone(), ptz, clock), ptz, clock


def test_runtime_starts_stopped():
    rt, _, _ = make_runtime()
    assert rt.director.mode == D.STOPPED


def test_auto_needs_a_complete_zone():
    from zone.zonefile import Zone
    rt, _, _ = make_runtime(Zone())
    ok, msg = rt.set_mode(D.AUTO)
    assert not ok and "keyframes" in msg
    assert rt.director.mode == D.STOPPED


def test_editing_the_zone_into_an_incomplete_one_leaves_auto():
    from zone.zonefile import Zone
    rt, _, _ = make_runtime()
    rt.set_mode(D.AUTO)
    rt.set_zone(Zone())
    assert rt.director.mode == D.STOPPED


def test_big_stop_stops_the_camera_immediately():
    rt, ptz, clock = make_runtime()
    ptz.pan_tilt_velocity(10, 0)            # something (say, a teach jog) left the camera moving
    clock.advance(0.5)
    assert ptz.is_moving()
    rt.big_stop()
    clock.advance(0.5)
    assert not ptz.is_moving()
    assert rt.director.mode == D.STOPPED
