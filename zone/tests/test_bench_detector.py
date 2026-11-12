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
            out[0, 4 + COCO_NAMES.index(name), i] = score             # row 4 + class id: that class's score
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
