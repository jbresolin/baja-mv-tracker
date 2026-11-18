"""Small builders shared by the WP-Z2 tests (zone tracker, teach, controller)."""
from zone import config as C
from zone.clock import FakeClock
from zone.ptz.sim import PtzModel, SimulatedPtz
from zone.zonefile import Keyframe, PathPoint, Zone


def straight_zone(n_keyframes=3, length=800.0, y=500.0, upstream=100.0, half_width=60.0):
    """A straight horizontal path: one path-only point `upstream` px before the first keyframe, then keyframes evenly
    spread over `length` px. Pan grows 100 units per keyframe, so pose(s) is easy to predict."""
    pts = [PathPoint(100.0, y)]
    for k in range(n_keyframes):
        x = 100.0 + upstream + length * k / (n_keyframes - 1)
        pts.append(PathPoint(x, y, Keyframe(pan=100 * k, tilt=-10 * k, zoom=1000 + 500 * k, preset=10 + k)))
    return Zone(points=pts, corridor_half_width_px=half_width, image_size=[1920, 1080])


def settings(branch="B", inquiry_mode=None, sets=(), model=None):
    """Resolved settings as if a report with this branch had been loaded."""
    model = model or PtzModel(inquiry_mode=inquiry_mode or ("always" if branch == "A" else "never"))
    cfg = C.load(None, list(sets))
    return C.build(cfg, C.Report(model, branch, "test report", "test.json", {}))


def sim(model=None, start=0.0):
    clock = FakeClock(start)
    return SimulatedPtz(model or PtzModel(), clock), clock


def crossing_zone(half_width=100.0):
    """A path that loops back and CROSSES ITSELF right after the entry (pixels, y grows downwards):

        approach (100,500) -> entry keyframe (300,500) -> (1100,500) -> up to (1100,150) -> left to (200,150)
        -> down to (200,350) -> diagonal down-right to the exit keyframe (500,650)

    The last, diagonal stretch crosses the entry stretch at (350,500). There a detection is inside the corridor of
    two stretches far apart along the path: s ~ 0.02 (just after the entry) and s ~ 0.92 (just before the exit).
    s = 0 at x = 300 on the entry stretch, s = 1 at (500,650); the whole path is ~2874 px long, so the zone (s 0..1)
    is ~2674 px of path."""
    pts = [PathPoint(100.0, 500.0),
           PathPoint(300.0, 500.0, Keyframe(pan=0, tilt=0, zoom=1000, preset=10)),
           PathPoint(1100.0, 500.0),
           PathPoint(1100.0, 150.0, Keyframe(pan=300, tilt=0, zoom=1500, preset=11)),
           PathPoint(200.0, 150.0),
           PathPoint(200.0, 350.0),
           PathPoint(500.0, 650.0, Keyframe(pan=600, tilt=0, zoom=2000, preset=12))]
    return Zone(points=pts, corridor_half_width_px=half_width, image_size=[1920, 1080])


def box_from_ground(gx, gy, height=40.0, width=40.0):
    """A detection box (x1, y1, x2, y2) whose bottom centre (the car's 'ground point') is at (gx, gy)."""
    return (gx - width / 2.0, gy - height, gx + width / 2.0, gy)
