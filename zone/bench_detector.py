"""Bench-only detector: a general COCO-trained YOLO11 (80 classes) in place of the Baja car detector.

Why: the deployed rt/ car detector was distilled on Baja buggies. On a home bench (spotter webcam on a public street,
the PTZ following ordinary passing cars) it may barely see normal cars. A stock COCO model (yolo11n.pt exported to
ONNX by the owner, see zone/README.md "Bench detector") sees them fine. COCO detectors are unreliable on muddy Baja
cars (LESSONS.md), so this is for the bench only: the zone page shows a warning while it is active.

How it plugs in: `CocoCarDetector` has the same call signature as rt.models.YoloOnnx,
`det(img, conf=, iou=) -> (boxes xyxy, scores, classes)`, so RoiDetector, ByteTrack, progress, director and controller
don't know the difference. Inside, the selected COCO classes (car, truck, bus, motorcycle; optionally person for walk
tests) all become the tracker's single "car" class (class 0).

Picked with `spotter.detector = "coco"` (or the `bench` profile); `spotter.detector = "baja"` (default) is unchanged.
"""
from pathlib import Path

import numpy as np

# The 80 COCO class names in the order a stock YOLO11 model outputs them (index = class id).
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
N_COCO_CLASSES = len(COCO_NAMES)          # 80
YOLO_OUTPUT_ROWS = 4 + N_COCO_CLASSES     # a YOLO11 head outputs (1, 4 box values + 80 class scores, N anchors)
TRACKER_CAR_CLASS = 0                     # the zone tracker has a single class: "car"


class BenchDetectorError(RuntimeError):
    """The bench detector can't start (missing model file, wrong model). Raised at startup, never mid-race."""


def coco_class_ids(names):
    """Class names from the config (e.g. ["car", "truck"]) -> COCO class ids (e.g. [2, 7]). Unknown name = error."""
    ids = []
    for name in names:
        key = str(name).strip().lower()
        if key not in COCO_NAMES:
            raise BenchDetectorError("spotter.coco_classes: %r is not a COCO class name (e.g. car, truck, bus, "
                                     "motorcycle, person)" % name)
        ids.append(COCO_NAMES.index(key))
    if not ids:
        raise BenchDetectorError("spotter.coco_classes is empty: the bench detector would never see anything")
    return ids


def agnostic_nms(boxes, scores, iou):
    """Non-maximum suppression that ignores the class. YOLO's own NMS is per class, so one vehicle can come out
    twice, once as "car" and once as "truck". Since both now mean "car" to the tracker, the weaker of two boxes that
    overlap by more than `iou` is dropped."""
    from rt.models import nms
    if len(scores) < 2:
        return np.arange(len(scores))
    return nms(np.asarray(boxes, np.float32), np.asarray(scores, np.float32), iou)


class CocoCarDetector:
    """A COCO YOLO11 detector (rt.models.YoloOnnx, built with its `classes` filter) whose selected classes are all
    reported as the tracker's "car" class."""

    def __init__(self, yolo, class_ids, class_names=(), agnostic_iou=0.5):
        self.yolo = yolo                      # rt.models.YoloOnnx (already filters to class_ids)
        self.class_ids = list(class_ids)
        # IoU above which two boxes (of any selected classes) are one vehicle. Lower than YOLO's own per-class 0.7:
        # the "car" and "truck" boxes of one pickup often overlap only 0.5-0.7, and would become two tracks.
        self.agnostic_iou = float(agnostic_iou)
        self.class_names = list(class_names)
        self.imgsz = yolo.imgsz               # (h, w), like YoloOnnx
        self.desc = "COCO %s (%s)" % ("/".join(self.class_names) or "classes %s" % self.class_ids,
                                      getattr(yolo, "desc", ""))

    def __call__(self, img, conf=0.25, iou=0.7):
        boxes, scores, _coco_classes = self.yolo(img, conf=conf, iou=iou)
        keep = agnostic_nms(boxes, scores, self.agnostic_iou)   # not `iou`: that is YOLO's per-class 0.7
        boxes, scores = np.asarray(boxes, np.float32)[keep], np.asarray(scores, np.float32)[keep]
        classes = np.full(len(scores), TRACKER_CAR_CLASS, int)    # car / truck / bus / ... -> "car"
        return boxes.reshape(-1, 4), scores, classes


def check_output_shape(yolo):
    """Run the model once on a blank image of its input size and check it is an 80-class YOLO11: output
    (1, 84, N). A wrong file (e.g. the Baja car detector, 1 class -> (1, 5, N)) stops the program here, at startup."""
    h, w = yolo.imgsz
    blank = np.zeros((1, 3, int(h), int(w)), np.float32)
    out = np.asarray(yolo.runner.run(blank))
    if out.ndim != 3 or out.shape[1] != YOLO_OUTPUT_ROWS:
        raise BenchDetectorError(
            "%s is not an 80-class COCO YOLO11 model: its output shape is %s, expected (1, %d, N). Export yolo11n.pt "
            "as in zone/README.md (\"Bench detector\")." % (yolo.runner.path, tuple(out.shape), YOLO_OUTPUT_ROWS))


def make_coco_detector(spotter_cfg, threads=2, log=print):
    """The bench detector from the [spotter] config. -> (CocoCarDetector, (input_w, input_h))."""
    from zone.config import resolve_path
    from rt.models import YoloOnnx
    path = resolve_path(spotter_cfg["coco_model"])
    if not Path(path).is_file():
        raise BenchDetectorError(
            "bench detector model not found: %s. Export it on your PC (zone/README.md, \"Bench detector\") and copy "
            "it there, or set spotter.coco_model." % path)
    names = list(spotter_cfg.get("coco_classes") or [])
    ids = coco_class_ids(names)
    imgsz = spotter_cfg.get("coco_imgsz", [384, 640])            # a fixed-shape export overrides it
    # best_of_selected: a box is scored by its best class among the selected ones, so a van whose top COCO class
    # is e.g. "train" still counts as a car/truck (the Baja detector path keeps rt's default behaviour)
    yolo = YoloOnnx(str(path), imgsz if isinstance(imgsz, int) else tuple(imgsz), int(threads), classes=ids,
                    log=log, best_of_selected=True)
    check_output_shape(yolo)
    det = CocoCarDetector(yolo, ids, names, float(spotter_cfg.get("coco_agnostic_iou", 0.5)))
    h, w = det.imgsz
    log("WARNING: bench detector active (%s): COCO models are unreliable on Baja cars, don't race with it"
        % det.desc)
    return det, (w, h)
