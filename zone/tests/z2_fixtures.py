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
