"""Per-point corridor width: each path point may carry its own `half_width_px`.

Background for a first-time reader:
* The "corridor" is a band drawn around the taught path on the spotter image. Only detections whose ground point
  (bottom centre of the box) falls inside it count as cars of the zone. Until now it had ONE width for the whole path
  (`corridor_half_width_px`, the page's "Corredor" box). Now a point can have its own width: e.g. narrow where the
  track passes close to another stretch, wide where the cars spread out.
* Between two points that have their own width, the width changes linearly with the distance along the path. Before
  the first / after the last such point it stays at that point's value. With no point having one, every point uses
  the global value: old zone.json files behave exactly as before.
* "half-width" = from the path line to the edge of the band, in pixels (the band is twice as wide).

Every test builds a small zone by hand; no camera, no clock.
"""
import json

import numpy as np
import pytest

from zone import draw, zonefile
from zone.path import PathModel
from zone.progress import ProgressSettings, ProgressTracker
from zone.teach import TeachError, TeachSession
from zone.tests.z2_fixtures import box_from_ground
from zone.zonefile import Keyframe, PathPoint, Zone


def straight(widths=(None, None, None, None), global_width=60.0):
    """A straight path along y = 500: points at x = 100, 200 (entry keyframe), 600 and 1100 (exit keyframe), with the
    given own widths (None = use the global). Arc length from the first point = x - 100."""
    xs = (100.0, 200.0, 600.0, 1100.0)
    kfs = (None, Keyframe(0, 0, 0, 10), None, Keyframe(100, 0, 0, 11))
    return Zone(points=[PathPoint(x, 500.0, k, w) for x, k, w in zip(xs, kfs, widths)],
                corridor_half_width_px=global_width)


# ------------------------------------------------------------------------------------------------ the file
def test_an_old_zone_file_loads_unchanged_and_saves_without_the_new_field(tmp_path):
    # A zone.json written before this change has no "half_width_px" anywhere. It must load, every point must use the
    # global width, and saving it again must not add the new key (so old and new files look the same).
    old = straight().to_dict()
    assert all("half_width_px" not in p for p in old["points"])
    f = tmp_path / "zone.json"
    f.write_text(json.dumps(old))
    z = zonefile.load(f)
    assert [p.half_width_px for p in z.points] == [None] * 4
    assert z.point_half_widths() == [60.0] * 4
    assert z.to_dict() == old


def test_a_point_width_is_saved_and_loaded_back(tmp_path):
    z = straight(widths=(None, 35.0, None, None))
    f = tmp_path / "zone.json"
    zonefile.save(z, f)
    raw = json.loads(f.read_text())
    assert raw["points"][1]["half_width_px"] == 35.0       # only the point that has one carries the key
    assert "half_width_px" not in raw["points"][0]
    assert zonefile.load(f).points[1].half_width_px == 35.0


# ------------------------------------------------------------------------------------------------ interpolation
def test_no_point_width_means_the_global_width_everywhere():
    path = PathModel(straight(global_width=70.0))
    for arc in (0.0, 300.0, 1000.0):
        assert path.half_width_at_arc(arc) == 70.0


def test_widths_are_interpolated_along_the_path_and_held_past_the_ends():
    # point 1 (x = 200, arc 100) has 20 px, point 3 (x = 1100, arc 1000) has 110 px; points 0 and 2 have none.
    z = straight(widths=(None, 20.0, None, 110.0))
    # point 0 is before the first explicit value: it holds 20. Point 2 (arc 500) is 400/900 of the way from 20 to 110.
    assert z.point_half_widths() == pytest.approx([20.0, 20.0, 20.0 + 90.0 * 400 / 900, 110.0])
    path = PathModel(z)
    assert path.half_width_at_arc(0.0) == pytest.approx(20.0)           # before the first explicit value: held
    assert path.half_width_at_arc(550.0) == pytest.approx(65.0)         # halfway between arc 100 and arc 1000
    assert path.half_width_at_arc(5000.0) == pytest.approx(110.0)       # past the end: held
    assert path.half_width_at_s(0.0) == pytest.approx(20.0)             # s = 0 is the entry keyframe (arc 100)


def test_a_single_point_width_holds_for_the_whole_path():
    # only one point has a width: it is "the nearest explicit value" for every other point
    z = straight(widths=(None, None, 45.0, None), global_width=60.0)
    assert z.point_half_widths() == [45.0] * 4


def test_in_corridor_uses_the_width_where_the_detection_is():
    # narrow (20 px) at the entry, wide (110 px) at the exit. A detection 50 px beside the path is outside near the
    # entry and inside near the exit.
    path = PathModel(straight(widths=(None, 20.0, None, 110.0)))
    near_entry = path.project(250.0, 550.0)
    near_exit = path.project(1050.0, 550.0)
    assert not path.in_corridor(near_entry)
    assert path.in_corridor(near_exit)
    # corridor_factor still multiplies the local width: 3 x 20 = 60 > 50
    assert path.in_corridor(near_entry, factor=3.0)


def test_progress_ignores_a_detection_outside_the_narrow_part_only():
    # The same two detections fed to the progress tracker: only the one beside the wide part starts a track.
    path = PathModel(straight(widths=(None, 20.0, None, 110.0)))
    prog = ProgressTracker(path, ProgressSettings())
    views = prog.update(1.0, [(1, box_from_ground(250.0, 550.0), 0.9), (2, box_from_ground(1050.0, 550.0), 0.9)])
    assert list(views) == [2]


# ------------------------------------------------------------------------------------------------ detector ROI
def test_the_roi_covers_the_widest_point():
    # The detector only looks inside this box. With one point 200 px wide, the box must reach 200 px (+ margin)
    # around that point; with no own widths it is the old box (60 px around every point).
    old = PathModel(straight()).corridor_bbox(margin_px=10)
    assert old == (100 - 70, 500 - 70, 1100 + 70, 500 + 70)
    wide = PathModel(straight(widths=(None, None, 200.0, None))).corridor_bbox(margin_px=10)
    # every point holds 200 (one explicit value), so the box is 210 px around the whole line
    assert wide == (100 - 210, 500 - 210, 1100 + 210, 500 + 210)
    mixed = PathModel(straight(widths=(20.0, None, None, 150.0))).corridor_bbox(margin_px=0)
    assert mixed[0] == 100 - 20                  # the narrow start
    assert mixed[2] == 1100 + 150                # the wide end
    assert mixed[1] == 500 - 150 and mixed[3] == 500 + 150


def test_the_runtime_gives_the_detector_the_variable_width_roi():
    # ZoneRuntime.set_zone hands the spotter its region of interest: it must come from the variable-width band.
    from zone.runtime import ZoneRuntime
    from zone.tests.z2_fixtures import settings, sim

    class FakeSpotter:
        roi = None

        def set_roi(self, roi):
            self.roi = roi

    ptz, clock = sim()
    spotter = FakeSpotter()
    z = straight(widths=(None, None, None, 150.0))
    z.image_size = [1920, 1080]
    ZoneRuntime(settings(), z, ptz, clock, spotter=spotter)
    margin = 40                                   # spotter.roi_margin_px default
    assert spotter.roi == (0, 500 - 150 - margin, 1100 + 150 + margin, 500 + 150 + margin)


# ------------------------------------------------------------------------------------------------ drawing
def test_the_drawn_band_follows_the_width():
    # Draw on a black picture and look at two pixels 50 px below the path: green-ish (the band) near the wide end,
    # untouched (black) near the narrow start.
    z = straight(widths=(None, 20.0, None, 110.0))
    img = np.zeros((1080, 1920, 3), np.uint8)
    draw.draw_zone(img, z)
    assert img[550, 1050].sum() > 0               # inside the 110 px band
    assert img[550, 250].sum() == 0               # outside the 20 px band
    quads, discs = draw.corridor_polygons(z)
    assert len(quads) == 3 and len(discs) == 4


# ------------------------------------------------------------------------------------------------ teaching
class _NoPtz:
    """Teaching a width never touches the camera: any call would fail the test."""


def test_teach_sets_and_clears_a_point_width():
    changes = []
    s = TeachSession(straight(), _NoPtz(), source=None, on_change=changes.append)
    s.set_point_width(2, 40)
    assert s.zone.points[2].half_width_px == 40.0
    assert s.dirty and len(changes) == 1          # marks the zone dirty and rebuilds the runtime, like other edits
    s.set_point_width(2, None)                    # empty = back to the global width
    assert s.zone.points[2].half_width_px is None
    s.set_point_width(2, "")                      # the page sends "" for an empty box too
    assert s.zone.points[2].half_width_px is None
    with pytest.raises(TeachError):
        s.set_point_width(2, 2)                   # too narrow (5..1000)
    with pytest.raises(TeachError):
        s.set_point_width(9, 40)                  # no such point
    with pytest.raises(TeachError):
        s.set_point_width(1, "abc")


def test_moving_and_reordering_a_point_keeps_its_width():
    s = TeachSession(straight(widths=(None, None, 33.0, None)), _NoPtz(), source=None)
    s.move_point(2, 650, 510)
    assert s.zone.points[2].half_width_px == 33.0
    s.reorder(2, -1)
    assert s.zone.points[1].half_width_px == 33.0


# ------------------------------------------------------------------------------------------------ page API
fastapi = pytest.importorskip("fastapi")


def test_point_width_from_the_api(tmp_path):
    # POST /api/teach/point_width {index, half_width_px}: works in PARADO / PASSIVO, refused in AUTO (409), null
    # resets it. The page's state shows the point's own width; the global "Corredor" still works alongside.
    from zone.tests.test_app import make
    client, app, _, _ = make(tmp_path)
    r = client.post("/api/teach/point_width", json={"index": 2, "half_width_px": 35})
    assert r.status_code == 200
    state = client.get("/api/state").json()
    assert state["zone"]["points"][2]["half_width_px"] == 35.0
    assert state["zone"]["dirty"] is True
    assert app.runtime.path.point_widths[2] == 35.0          # the running tracker uses it at once
    assert client.post("/api/teach/corridor", json={"half_width": 80}).status_code == 200
    assert app.runtime.path.half_width == 80.0               # the global value (used where no point has one)
    assert client.post("/api/teach/point_width", json={"index": 2, "half_width_px": None}).status_code == 200
    assert "half_width_px" not in client.get("/api/state").json()["zone"]["points"][2]
    assert client.post("/api/teach/point_width", json={"index": 2, "half_width_px": 1}).status_code == 409
    client.post("/api/mode", json={"mode": "PASSIVE"})
    assert client.post("/api/teach/point_width", json={"index": 2, "half_width_px": 40}).status_code == 200
    client.post("/api/mode", json={"mode": "AUTO"})
    r = client.post("/api/teach/point_width", json={"index": 2, "half_width_px": 50})
    assert r.status_code == 409 and "AUTO" in r.json()["detail"]
    assert app.teach.zone.points[2].half_width_px == 40.0    # unchanged by the refused request
