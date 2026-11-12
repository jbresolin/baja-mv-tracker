"""Tests for the bench-only COCO detector (zone/bench_detector.py) and how the spotter picks its detector.

Background, for a first-time reader:
* The zone tracker normally uses the "baja" detector: a small model trained only on Baja buggies. For a home test on
  a public street the owner can switch to a "coco" detector: a stock YOLO11 model trained on the COCO dataset, which
  knows 80 kinds of objects (person, car, truck, bus, dog, ...).
* A YOLO11 model answers with one big array of shape (1, 84, N): for each of N candidate boxes ("anchors"), 4 numbers
  for the box (centre x, centre y, width, height, in input pixels) and 80 scores, one per COCO class.
* No real model runs here. `FakeSession` pretends to be ONNX Runtime: whatever image it is given, it returns an
  array that we wrote by hand, with exactly the boxes and scores each test needs. pytest's `monkeypatch` swaps it in
  for the real ONNX Runtime session for the duration of one test only.
* `tmp_path` is a fresh empty folder pytest gives each test.
"""
import numpy as np
import pytest

import rt.models
from zone import config as C
from zone.bench_detector import (COCO_NAMES, BenchDetectorError, CocoCarDetector, coco_class_ids,
                                 make_coco_detector)
from zone.spotter import detector_kind, make_spotter, rt_config

INPUT_H, INPUT_W = 384, 640          # the input size the fake model says it has (same as the Baja detector)


# ------------------------------------------------------------------------------------------------ the fake model
class _Input:
    """What ONNX Runtime says about a model input: its name and shape."""
    def __init__(self, shape):
        self.name, self.shape = "images", shape


class FakeSession:
    """Stands in for onnxruntime.InferenceSession. `anchors` is a list of (class name, score, (cx, cy, w, h)):
    one candidate box each. `rows` = 84 makes a COCO-shaped output; another value makes a wrong model."""

    def __init__(self, anchors=(), rows=84, n_anchors=10):
        self.anchors, self.rows, self.n_anchors = list(anchors), rows, n_anchors
        self.calls = 0

    def get_inputs(self):
        return [_Input([1, 3, INPUT_H, INPUT_W])]       # a fixed-shape export, like `yolo export` makes by default

    def run(self, _output_names, feeds):
        self.calls += 1
        out = np.zeros((1, self.rows, self.n_anchors), np.float32)   # every score 0 = "nothing here"
        for i, (name, score, (cx, cy, w, h)) in enumerate(self.anchors):
            out[0, 0:4, i] = (cx, cy, w, h)                           # rows 0-3: the box
            # `name` is one class name, or a dict {class name: score} when one box gets scores for several classes
            for class_name, class_score in (name.items() if isinstance(name, dict) else [(name, score)]):
                out[0, 4 + COCO_NAMES.index(class_name), i] = class_score   # row 4 + class id: that class's score
        return [out]


def install_fake(monkeypatch, session):
    """Make every model the code opens use `session` instead of real ONNX Runtime (for this test only)."""
    monkeypatch.setattr(rt.models, "make_session", lambda path, threads, spinning=False: session)


def model_file(tmp_path):
    """An empty file standing in for the .onnx (the fake session never reads it; it only has to exist)."""
    p = tmp_path / "yolo11n_coco.onnx"
    p.write_bytes(b"")
    return p


def coco_cfg(tmp_path, *extra):
    """The full zone config with the bench detector switched on and pointing at the fake model file."""
    return C.load(None, ["spotter.detector=coco", "spotter.coco_model=%s" % model_file(tmp_path).as_posix(),
                         *extra])


def street_image():
    """A blank image exactly the model's input size, so a box in the image = the same box in model pixels."""
    return np.zeros((INPUT_H, INPUT_W, 3), np.uint8)


# Four objects in different places, none overlapping: a car, a truck, a pedestrian and a dog.
CAR = ("car", 0.9, (100, 200, 80, 40))         # box x 60-140, y 180-220
TRUCK = ("truck", 0.8, (300, 200, 120, 60))    # box x 240-360, y 170-230
PERSON = ("person", 0.9, (500, 200, 20, 60))
DOG = ("dog", 0.9, (600, 300, 30, 20))


# ------------------------------------------------------------------------------------------------ class filter + map
def test_vehicles_become_cars_and_everything_else_is_ignored(tmp_path, monkeypatch):
    # Default classes: car, truck, bus, motorcycle. The car and the truck come out, both as the tracker's class 0
    # ("car"); the pedestrian and the dog are not vehicles, so they are dropped.
    install_fake(monkeypatch, FakeSession([CAR, TRUCK, PERSON, DOG]))
    det, input_wh = make_coco_detector(coco_cfg(tmp_path)["spotter"], log=lambda m: None)
    boxes, scores, classes = det(street_image(), conf=0.25, iou=0.7)
    assert input_wh == (INPUT_W, INPUT_H)                       # (width, height), as the spotter wants it
    assert sorted(scores.tolist()) == pytest.approx([0.8, 0.9])
    assert classes.tolist() == [0, 0]                           # car AND truck are both "car" for the tracker
    boxes = sorted(boxes.tolist())                              # left to right: the car, then the truck
    assert boxes[0] == pytest.approx([60, 180, 140, 220]) and boxes[1] == pytest.approx([240, 170, 360, 230])


def test_person_can_be_added_for_a_walk_test(tmp_path, monkeypatch):
    # With "person" in coco_classes, someone walking past the webcam is tracked like a car.
    install_fake(monkeypatch, FakeSession([CAR, PERSON, DOG]))
    cfg = coco_cfg(tmp_path, 'spotter.coco_classes=["car", "person"]')
    det, _ = make_coco_detector(cfg["spotter"], log=lambda m: None)
    _, scores, classes = det(street_image(), conf=0.25, iou=0.7)
    assert len(scores) == 2 and classes.tolist() == [0, 0]     # the car and the person; still no dog


def test_one_vehicle_seen_as_car_and_truck_comes_out_once(tmp_path, monkeypatch):
    # YOLO sometimes gives the same pickup two boxes: one "car" and one "truck". YOLO's own clean-up only removes
    # duplicates of the same class, so both would survive and the tracker would see two cars. Since both are "car"
    # now, the weaker of two strongly overlapping boxes is removed.
    same_spot_truck = ("truck", 0.6, (102, 201, 80, 40))
    install_fake(monkeypatch, FakeSession([CAR, same_spot_truck]))
    det, _ = make_coco_detector(coco_cfg(tmp_path)["spotter"], log=lambda m: None)
    _, scores, _ = det(street_image(), conf=0.25, iou=0.7)
    assert scores.tolist() == pytest.approx([0.9])             # only the stronger box (the "car" one) is left


# Two boxes of the same size (80 x 40 pixels) overlap by IoU ("intersection over union") = the area they share
# divided by the area they cover together. 1.0 = the same box, 0 = not touching.
def iou_of(a, b):
    """IoU of two (cx, cy, w, h) boxes, computed here so each test can show the overlap it is about."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def test_car_and_truck_boxes_of_one_pickup_that_only_partly_overlap_are_merged(tmp_path, monkeypatch):
    # The realistic case: the "car" box and the "truck" box of one pickup are not identical, they are shifted by a
    # quarter of the car's width, which gives IoU 0.6. YOLO's own threshold (0.7) would keep both and the tracker
    # would follow two cars. spotter.coco_agnostic_iou (default 0.5) is what decides here, so they are merged.
    shifted_truck = ("truck", 0.6, (120, 200, 80, 40))          # 20 px right of CAR
    assert iou_of(CAR[2], shifted_truck[2]) == pytest.approx(0.6)
    install_fake(monkeypatch, FakeSession([CAR, shifted_truck]))
    det, _ = make_coco_detector(coco_cfg(tmp_path)["spotter"], log=lambda m: None)
    assert det.agnostic_iou == 0.5                              # the default from zone.example.toml
    _, scores, _ = det(street_image(), conf=0.25, iou=0.7)
    assert scores.tolist() == pytest.approx([0.9])             # one vehicle: the stronger ("car") box

    # Proof that the setting is the one used: raised to 0.7 (YOLO's value), the same pair comes out twice.
    det, _ = make_coco_detector(coco_cfg(tmp_path, "spotter.coco_agnostic_iou=0.7")["spotter"], log=lambda m: None)
    _, scores, _ = det(street_image(), conf=0.25, iou=0.7)
    assert len(scores) == 2


def test_two_cars_side_by_side_are_both_kept(tmp_path, monkeypatch):
    # Two real cars in neighbouring lanes (or bumper to bumper) overlap a little on screen: IoU about 0.2. That is
    # well below 0.5, so both stay: the duplicate filter must not eat a real second car.
    next_car = ("car", 0.7, (153, 200, 80, 40))                 # 53 px right of CAR
    assert iou_of(CAR[2], next_car[2]) == pytest.approx(0.2, abs=0.01)
    install_fake(monkeypatch, FakeSession([CAR, next_car]))
    det, _ = make_coco_detector(coco_cfg(tmp_path)["spotter"], log=lambda m: None)
    _, scores, _ = det(street_image(), conf=0.25, iou=0.7)
    assert sorted(scores.tolist()) == pytest.approx([0.7, 0.9])


# ------------------------------------------------------------------------------------------------ best selected class
# A van that the model finds a bit "train"-like: its scores are train 0.7, car 0.5. YOLO's usual rule takes the best
# of all 80 classes first ("train") and only then applies the class filter, so a car/truck filter drops the van.
VAN = ({"train": 0.7, "car": 0.5}, None, (300, 200, 120, 60))


def test_rt_default_keeps_its_old_rule_the_van_is_dropped(tmp_path, monkeypatch):
    # rt.models.YoloOnnx's default behaviour (used by the Baja detector and rt/) is unchanged: best of all classes.
    install_fake(monkeypatch, FakeSession([VAN]))
    yolo = rt.models.YoloOnnx(str(model_file(tmp_path)), (INPUT_H, INPUT_W), classes=[2, 7], log=lambda m: None)
    boxes, _, _ = yolo(street_image(), conf=0.25, iou=0.7)
    assert len(boxes) == 0
    # and without a class filter it is reported as what it scored best: a train (COCO class 6)
    yolo = rt.models.YoloOnnx(str(model_file(tmp_path)), (INPUT_H, INPUT_W), log=lambda m: None)
    _, scores, classes = yolo(street_image(), conf=0.25, iou=0.7)
    assert classes.tolist() == [COCO_NAMES.index("train")] and scores.tolist() == pytest.approx([0.7])


def test_best_of_selected_keeps_the_van_as_a_car(tmp_path, monkeypatch):
    # The opt-in rule (best_of_selected=True): only the selected classes compete, so the van is a "car" at 0.5.
    install_fake(monkeypatch, FakeSession([VAN, DOG]))
    yolo = rt.models.YoloOnnx(str(model_file(tmp_path)), (INPUT_H, INPUT_W), classes=[2, 7], best_of_selected=True,
                              log=lambda m: None)
    boxes, scores, classes = yolo(street_image(), conf=0.25, iou=0.7)
    assert classes.tolist() == [COCO_NAMES.index("car")] and scores.tolist() == pytest.approx([0.5])
    assert boxes.tolist()[0] == pytest.approx([240, 170, 360, 230])     # the dog is still not a vehicle


def test_the_bench_detector_uses_best_of_selected(tmp_path, monkeypatch):
    # The bench detector switches the opt-in rule on, so the van is tracked.
    install_fake(monkeypatch, FakeSession([VAN]))
    det, _ = make_coco_detector(coco_cfg(tmp_path)["spotter"], log=lambda m: None)
    _, scores, classes = det(street_image(), conf=0.25, iou=0.7)
    assert scores.tolist() == pytest.approx([0.5]) and classes.tolist() == [0]


def test_unknown_class_name_is_an_error():
    # a typo in coco_classes must stop the program at startup, not silently detect nothing
    assert coco_class_ids(["car", "truck", "bus", "motorcycle"]) == [2, 7, 5, 3]
    with pytest.raises(BenchDetectorError, match="carr"):
        coco_class_ids(["carr"])
    with pytest.raises(BenchDetectorError, match="empty"):
        coco_class_ids([])


# ------------------------------------------------------------------------------------------------ threshold
def test_the_coco_detector_uses_its_own_confidence_threshold(tmp_path, monkeypatch):
    # coco_conf is separate from rt.detection.conf (0.15, tuned for the Baja detector). A weak car (score 0.2)
    # is dropped at the default coco_conf 0.25 and kept when coco_conf is lowered to 0.1.
    weak_car = ("car", 0.2, (100, 200, 80, 40))
    install_fake(monkeypatch, FakeSession([weak_car, TRUCK]))
    for extra, expected in (([], 1), (["spotter.coco_conf=0.1"], 2)):
        cfg = coco_cfg(tmp_path, *extra)
        sp = make_spotter(cfg, rt_config(cfg), log=lambda m: None)
        assert sp.det.conf == cfg["spotter"]["coco_conf"]       # the ROI detector got coco_conf, not rt's 0.15
        boxes, _ = sp.det(street_image())
        assert len(boxes) == expected, extra


# ------------------------------------------------------------------------------------------------ startup errors
def test_a_model_that_is_not_80_class_coco_is_refused_at_startup(tmp_path, monkeypatch):
    # Pointing coco_model at the wrong file (e.g. the 1-class Baja detector, output (1, 5, N)) must fail right away
    # with a clear message, not track nonsense later.
    install_fake(monkeypatch, FakeSession(rows=5))
    with pytest.raises(BenchDetectorError, match="80-class"):
        make_coco_detector(coco_cfg(tmp_path)["spotter"], log=lambda m: None)


def test_a_missing_model_file_is_a_clear_error(tmp_path, monkeypatch):
    install_fake(monkeypatch, FakeSession([CAR]))
    cfg = C.load(None, ["spotter.detector=coco", "spotter.coco_model=%s" % (tmp_path / "nope.onnx").as_posix()])
    with pytest.raises(BenchDetectorError, match="not found") as err:
        make_spotter(cfg, rt_config(cfg), log=lambda m: None)
    assert "Bench detector" in str(err.value)                   # it says where the export instructions are


def test_the_startup_check_also_works_without_a_real_session():
    # CocoCarDetector itself only needs something callable like YoloOnnx: here a tiny stand-in that returns a car
    # and a truck. Their COCO class numbers (2, 7) become the tracker's class 0.
    class StandIn:
        imgsz, desc = (384, 640), "stand-in"

        def __call__(self, img, conf, iou):
            return (np.array([[0, 0, 10, 10], [50, 50, 60, 60]], np.float32), np.array([0.9, 0.5], np.float32),
                    np.array([2, 7]))
    det = CocoCarDetector(StandIn(), [2, 7], ["car", "truck"])
    _, _, classes = det(street_image())
    assert classes.tolist() == [0, 0] and "car/truck" in det.desc


# ------------------------------------------------------------------------------------------------ config + profiles
def test_default_detector_stays_baja(monkeypatch):
    # Nothing in the default config changes: the race-day detector is the Baja one, with rt's threshold, and the
    # page gets no bench warning. (The real Baja model isn't loaded here: make_detector is replaced by a stand-in.)
    import zone.spotter as spotter_module
    from zone.testing import BlobDetector
    called = []
    monkeypatch.setattr(spotter_module, "make_detector",
                        lambda rtcfg, log=print: (called.append(1) or BlobDetector(), (640, 384)))
    cfg = C.load(None, [])
    assert cfg["spotter"]["detector"] == "baja" and detector_kind(cfg["spotter"]) == "baja"
    rtcfg = rt_config(cfg)
    sp = make_spotter(cfg, rtcfg)
    assert called == [1]                                        # the Baja detector was built
    assert sp.det.conf == rtcfg["detection"]["conf"]
    assert sp.detector_info["kind"] == "baja" and sp.detector_info["bench"] is False


def test_the_old_name_rt_still_means_baja_and_nonsense_is_refused():
    assert detector_kind({"detector": "rt"}) == "baja"          # site.toml files written before the rename
    assert detector_kind({}) == "baja"
    with pytest.raises(ValueError, match="coco"):               # the message lists the valid choices
        detector_kind({"detector": "yolo"})


def test_bench_profile_switches_to_coco_with_the_default_model_path():
    cfg = C.load(None, ["profile=bench"])
    assert cfg["spotter"]["detector"] == "coco"
    assert cfg["spotter"]["coco_model"] == "deploy/models/yolo11n_coco.onnx"
    assert cfg["spotter"]["coco_classes"] == ["car", "truck", "bus", "motorcycle"]
    assert cfg["controller"]["k_quant"] == 1.5                  # bench changes the detector only


def test_bench_strict_profile_is_bench_plus_strict():
    cfg = C.load(None, ["profile=bench_strict"])
    assert cfg["spotter"]["detector"] == "coco"
    assert cfg["controller"]["dead_reckoning_trust_s"] == 0.0 and cfg["controller"]["k_quant"] == 0.0


def test_a_coco_model_path_from_site_toml_survives_the_bench_profiles(tmp_path):
    # Order of config layers: defaults < site.toml < profile < --set. If the profiles repeated coco_model, they
    # would silently replace the path the owner wrote in site.toml. They don't, so the owner's path stays.
    site = tmp_path / "site.toml"
    site.write_text('[spotter]\ncoco_model = "D:/models/my_yolo11s.onnx"\n', encoding="utf-8")
    for profile in ("bench", "bench_strict"):
        cfg = C.load(site, ["profile=%s" % profile])
        assert cfg["spotter"]["detector"] == "coco", profile
        assert cfg["spotter"]["coco_model"] == "D:/models/my_yolo11s.onnx", profile


def test_set_wins_over_the_bench_profile():
    # e.g. a walk test: the bench profile, plus people counted as cars
    cfg = C.load(None, ["profile=bench", 'spotter.coco_classes=["person"]'])
    assert cfg["spotter"]["detector"] == "coco" and cfg["spotter"]["coco_classes"] == ["person"]


def test_coco_spotter_reports_itself_as_a_bench_detector(tmp_path, monkeypatch):
    install_fake(monkeypatch, FakeSession([CAR]))
    cfg = coco_cfg(tmp_path)
    messages = []
    sp = make_spotter(cfg, rt_config(cfg), log=messages.append)
    assert sp.detector_info["kind"] == "coco" and sp.detector_info["bench"] is True
    assert "COCO car/truck/bus/motorcycle" in sp.detector_info["desc"]
    assert any("bench detector active" in m for m in messages)  # also warned in the console / log at startup


# ------------------------------------------------------------------------------------------------ the page
def test_health_line_shows_the_detector_and_warns_for_the_bench_one(tmp_path):
    # The page's health data carries spotter.detector_info; the page script turns bench=True into a red line and
    # an amber "DETECTOR DE BANCADA" banner in the header, so it isn't left on at the race.
    pytest.importorskip("fastapi")
    from pathlib import Path
    from fastapi.testclient import TestClient
    from zone.app import ZoneApp, create_web
    from zone.clock import FakeClock
    from zone.ptz.sim import PtzModel, SimulatedPtz
    from zone.tests.z2_fixtures import straight_zone

    class SpotterStub:
        """Only what the app asks of a spotter here: which detector it runs (and where its ROI is)."""
        def __init__(self, info):
            self.detector_info = info

        def set_roi(self, roi):
            pass

    def health_with(spotter):
        cfg = C.load(None, ["ptz.driver=sim"])
        clock = FakeClock(0.0)
        app = ZoneApp(cfg, C.build(cfg), SimulatedPtz(PtzModel(), clock), clock, straight_zone(),
                      tmp_path / "zone.json", tmp_path / "out", spotter=spotter, log=lambda m: None, cycle_log=False)
        return TestClient(create_web(app)).get("/api/state").json()["health"]

    bench = dict(kind="coco", desc="COCO car/truck", bench=True)
    assert health_with(SpotterStub(bench))["detector"] == bench
    assert health_with(SpotterStub(dict(kind="baja", desc="x", bench=False)))["detector"]["bench"] is False
    assert health_with(None)["detector"] is None                # no spotter (tests): the page shows "–"

    static = Path(__file__).resolve().parent.parent / "static"
    js = (static / "app.js").read_text(encoding="utf-8")
    html = (static / "index.html").read_text(encoding="utf-8")
    assert "DETECTOR DE BANCADA" in js and "det.bench" in js and "#benchWarn" in js
    assert 'id="benchWarn"' in html and "DETECTOR DE BANCADA (COCO)" in html


# ------------------------------------------------------------------------------------- ROI: back to image pixels
class WhiteBlobSession(FakeSession):
    """A fake model that really "looks" at its input: every white rectangle in the image it is given is reported as
    a car, with its box in model-input pixels. So the test can check the whole way back: camera frame -> ROI crop
    (-> tiles) -> letterbox (shrink + grey padding) -> model -> boxes in camera-frame pixels."""

    def run(self, _output_names, feeds):
        import cv2
        self.calls += 1
        image = feeds["images"][0]                               # (3, H, W), values 0..1
        white = (image.mean(0) > 0.5).astype(np.uint8)           # 1 where the pixel is (almost) white
        n, _labels, stats, _ = cv2.connectedComponentsWithStats(white)
        out = np.zeros((1, 84, self.n_anchors), np.float32)
        for i in range(1, n):                                    # label 0 is the background
            x, y, w, h = stats[i, :4]
            out[0, 0:4, i - 1] = (x + w / 2, y + h / 2, w, h)     # centre + size, as YOLO writes it
            out[0, 4 + COCO_NAMES.index("car"), i - 1] = 0.9
        return [out]


FRAME_W, FRAME_H = 1920, 1080                                    # a full-HD spotter frame
CORRIDOR_ROI = (200, 300, 1500, 900)                             # the corridor box: 1300 x 600 pixels


def frame_with_cars(*boxes):
    """A black full-HD frame with a white rectangle (x0, y0, x1, y1) for each car."""
    img = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    for x0, y0, x1, y1 in boxes:
        img[y0:y1, x0:x1] = 255
    return img


@pytest.mark.parametrize("mode", ["downscale", "tile"])
def test_coco_boxes_come_back_in_camera_frame_pixels_in_both_roi_modes(tmp_path, monkeypatch, mode):
    # "downscale": the 1300 x 600 corridor crop is shrunk about 2x to fit the 640 x 384 input, so the model sees a
    #   smaller car with an offset; the box must be scaled up and shifted back. A few pixels of error are normal
    #   (the shrink blurs the edges).
    # "tile": the corridor is cut into 640 x 384 tiles (no shrink); each tile's boxes are shifted by the tile's
    #   corner, and pieces of one car seen by two overlapping tiles are merged.
    # Two cars: one well inside the corridor, one near its far corner. A third white box OUTSIDE the corridor must
    # not be seen at all (the model is never shown that part of the frame).
    car_a, car_b, outside = (400, 450, 520, 510), (1300, 780, 1460, 860), (1650, 100, 1800, 180)
    session = WhiteBlobSession()
    install_fake(monkeypatch, session)
    cfg = coco_cfg(tmp_path)
    sp = make_spotter(cfg, rt_config(cfg), roi=CORRIDOR_ROI, mode=mode, log=lambda m: None)
    boxes, scores = sp.det(frame_with_cars(car_a, car_b, outside))

    if mode == "downscale":
        assert sp.det.last_n_inferences == 1                    # one crop, one model run
        tolerance = 3.0
    else:
        assert sp.det.last_n_inferences > 1                     # several tiles
        tolerance = 1.0                                          # no shrink: (almost) exact
    found = sorted(boxes.tolist())                               # left to right
    assert len(found) == 2 and scores.tolist() == pytest.approx([0.9, 0.9])
    assert found[0] == pytest.approx(list(car_a), abs=tolerance)
    assert found[1] == pytest.approx(list(car_b), abs=tolerance)
