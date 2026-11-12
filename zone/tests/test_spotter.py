"""Tests for zone/spotter.py: cropping to the corridor before detecting, tiling, merging cut boxes."""
import numpy as np
import pytest

from zone.spotter import RoiDetector, Spotter, ground_point, merge_boxes, tile_grid
from zone.testing import BlobDetector, Scene, render


def image_with(boxes, size=(1920, 1080)):
    return render(Scene(image_size=size), boxes)


def test_ground_point_is_bottom_centre():
    assert ground_point((10, 20, 30, 60)) == (20, 60)


def test_tiles_cover_the_roi_with_overlap():
    tiles = tile_grid((100, 200, 1900, 800), 640, 384, 0.2)
    xs = sorted({t[0] for t in tiles})
    ys = sorted({t[1] for t in tiles})
    assert tiles[0][:2] == (100, 200) and max(t[2] for t in tiles) == 1900 and max(t[3] for t in tiles) == 800
    assert all(t[2] - t[0] <= 640 and t[3] - t[1] <= 384 for t in tiles)
    assert all(b - a <= 640 * 0.8 + 1 for a, b in zip(xs, xs[1:]))       # neighbours overlap
    assert len(ys) == 2


def test_downscale_mode_runs_the_detector_on_the_crop_only():
    det = BlobDetector()
    rd = RoiDetector(det, "downscale", roi=(500, 300, 1500, 900))
    boxes, _ = rd(image_with([(700, 500, 790, 560), (100, 100, 190, 160)]))   # the second car is outside the ROI
    assert det.calls == [(600, 1000)]                    # one inference, on the 1000 x 600 crop
    assert boxes.tolist() == [[700, 500, 790, 560]]      # back in full-frame pixels


def test_full_mode_sees_everything():
    rd = RoiDetector(BlobDetector(), "full", roi=(500, 300, 1500, 900))
    boxes, _ = rd(image_with([(700, 500, 790, 560), (100, 100, 190, 160)]))
    assert len(boxes) == 2


def test_tile_mode_finds_a_car_cut_by_a_tile_edge_once():
    det = BlobDetector()
    rd = RoiDetector(det, "tile", roi=(50, 435, 1893, 950))
    car = (411, 783, 501, 843)                           # straddles a vertical AND a horizontal tile edge
    boxes, _ = rd(image_with([car]))
    assert len(det.calls) == 8
    assert boxes.shape == (1, 4)
    assert boxes[0].tolist() == pytest.approx(list(car), abs=1)


def test_merge_boxes_unions_pieces_and_keeps_separate_cars():
    full = (100, 100, 200, 160)
    piece = (150, 100, 200, 160)                         # right half of the same car
    other = (400, 100, 500, 160)
    b, s = merge_boxes(np.array([piece, full, other], np.float32), np.array([0.9, 0.6, 0.8], np.float32))
    assert sorted(b.tolist()) == [list(full), list(other)]
    assert max(s) == pytest.approx(0.9)


def test_roi_outside_the_image_falls_back_to_the_full_frame():
    rd = RoiDetector(BlobDetector(), "downscale", roi=(5000, 5000, 6000, 6000))
    assert rd.regions((1080, 1920)) == [(0, 0, 1920, 1080)]


def test_bad_mode_is_refused():
    with pytest.raises(ValueError):
        RoiDetector(BlobDetector(), "zoom")


def test_spotter_tracks_across_frames():
    from rt.tracker import ByteTracker
    sp = Spotter(RoiDetector(BlobDetector(), "downscale", roi=(0, 300, 1920, 900)), ByteTracker())
    ids = set()
    for k in range(10):
        out = sp.process(image_with([(300 + 20 * k, 500, 390 + 20 * k, 560)]))
        ids |= {tid for tid, _, _ in out}
    assert ids == {1}                                    # one car, one track id
    assert sp.last_ms >= 0
