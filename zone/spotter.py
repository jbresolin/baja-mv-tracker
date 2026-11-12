"""Spotter detection: car detector on the corridor region of interest (ROI), then ByteTrack. Both come from rt/.

Why crop: the rt car detector takes a 640x384 input. A whole 1920x1080 spotter frame is shrunk 3x to fit, so a car
95 px wide becomes ~32 px, which is where small-car recall falls apart. The zone only cares about the corridor
around the taught path, so we cut out its bounding box (plus a margin) and run the detector on that:

* "downscale": the ROI crop goes to the detector as is (the detector's letterbox shrinks it if it's bigger than the
  input). One inference per frame. Good when the corridor box is not much bigger than 640x384.
* "tile": the ROI is cut into detector-sized tiles that overlap a little; each tile is one inference (no shrinking),
  and the boxes are merged. More CPU, more detail.
* "full": no crop (the rt default), for comparison in the detection check.

The point used per car is the **bottom centre of its box**: where it touches the ground, the point that follows the
racing line the taught path was clicked on.
"""
import math
import time

import numpy as np



def ground_point(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, y2


def tile_grid(roi, tile_w, tile_h, overlap):
    """Tiles (x0, y0, x1, y1) covering roi = (x0, y0, x1, y1), each at most tile_w x tile_h, overlapping by about
    `overlap` of a tile. Tiles are spread evenly so the last one doesn't hang over the ROI edge."""
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    out = []

    def starts(length, size):
        if length <= size:
            return [0]
        n = int(math.ceil((length - size * overlap) / (size * (1 - overlap))))
        n = max(2, n)
        step = (length - size) / (n - 1)
        return [int(round(i * step)) for i in range(n)]

    for sy in starts(h, tile_h):
        for sx in starts(w, tile_w):
            out.append((x0 + sx, y0 + sy, x0 + min(w, sx + tile_w), y0 + min(h, sy + tile_h)))
    return out


def merge_boxes(boxes, scores, contain=0.6):
    """Merge detections from overlapping tiles. A car cut by a tile edge gives a partial box in one tile and a full
    (or another partial) box in the neighbouring one; each piece lies mostly inside the car's real box. So boxes whose
    intersection covers more than `contain` of the smaller one are clustered, and each cluster becomes the union of
    its boxes with the best score. (Plain NMS is wrong here: with similar scores it can keep a partial box and
    throw the full one away.)"""
    boxes = [list(map(float, b)) for b in boxes]
    scores = [float(s) for s in scores]
    merged = True
    while merged and len(boxes) > 1:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                iw = min(a[2], b[2]) - max(a[0], b[0])
                ih = min(a[3], b[3]) - max(a[1], b[1])
                if iw <= 0 or ih <= 0:
                    continue
                small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                if iw * ih / max(small, 1e-9) > contain:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    scores[i] = max(scores[i], scores[j])
                    del boxes[j], scores[j]
                    merged = True
                    break
            if merged:
                break
    return np.array(boxes, np.float32).reshape(-1, 4), np.array(scores, np.float32)


class RoiDetector:
    """Wraps a detector `det(img, conf=, iou=) -> (boxes xyxy, scores, classes)` (rt.models.YoloOnnx or a fake)
    with the ROI crop. Boxes come back in full-frame pixels."""

    def __init__(self, det, mode="downscale", roi=None, input_wh=(640, 384), tile_overlap=0.2, tile_scale=1.0,
                 conf=0.15, iou=0.7):
        if mode not in ("downscale", "tile", "full"):
            raise ValueError("roi_mode must be downscale, tile or full, not %r" % mode)
        self.det, self.mode, self.roi = det, mode, roi
        self.input_wh = (int(input_wh[0]), int(input_wh[1]))
        self.tile_overlap, self.tile_scale = float(tile_overlap), float(tile_scale)
        self.conf, self.iou = float(conf), float(iou)
        self.last_n_inferences = 0

    def regions(self, img_shape):
        h, w = img_shape[:2]
        if self.mode == "full" or self.roi is None:
            return [(0, 0, w, h)]
        x0, y0, x1, y1 = self.roi
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return [(0, 0, w, h)]
        if self.mode == "downscale":
            return [(x0, y0, x1, y1)]
        tw = int(self.input_wh[0] * self.tile_scale)
        th = int(self.input_wh[1] * self.tile_scale)
        return tile_grid((x0, y0, x1, y1), tw, th, self.tile_overlap)

    def __call__(self, img):
        regs = self.regions(img.shape)
        self.last_n_inferences = len(regs)
        all_b, all_s = [], []
        for (x0, y0, x1, y1) in regs:
            crop = img[y0:y1, x0:x1]
            b, s, _ = self.det(crop, conf=self.conf, iou=self.iou)
            if len(b):
                b = np.asarray(b, np.float32).copy()
                b[:, [0, 2]] += x0
                b[:, [1, 3]] += y0
                all_b.append(b)
                all_s.append(np.asarray(s, np.float32))
        if not all_b:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        boxes, scores = np.concatenate(all_b), np.concatenate(all_s)
        if len(regs) > 1:
            boxes, scores = merge_boxes(boxes, scores)
        return boxes, scores


class Spotter:
    """detector (with ROI) + tracker. `process(img)` -> list of (track_id, xyxy, score), as rt's ByteTracker."""

    def __init__(self, roi_detector: RoiDetector, tracker):
        self.det = roi_detector
        self.tracker = tracker
        self.last_dets = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
        self.last_ms = 0.0
        # which detector runs, for the page's health line: dict(kind="baja" | "coco" | "blobs", desc=..., bench=bool).
        # bench = True means a bench-only detector (COCO) is active: the page shows a warning.
        self.detector_info = None

    def set_roi(self, roi):
        self.det.roi = roi

    def process(self, img):
        t0 = time.perf_counter()
        boxes, scores = self.det(img)
        out = self.tracker.update(boxes, scores)
        self.last_dets = (boxes, scores)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return out


# =============================================================================================== building the real one
def rt_config(cfg):
    """The rt/ config, with the zone config's [rt.*] tables applied on top (same names as rt/config.toml)."""
    from rt import config as rtc
    sets = []
    for sec, kv in (cfg.get("rt") or {}).items():
        for k, v in kv.items():
            sets.append("%s.%s=%r" % (sec, k, v))
    return rtc.load(None, sets)


def make_tracker(rtcfg):
    from rt.tracker import ByteTracker
    t = rtcfg["tracker"]
    return ByteTracker(**{k: t[k] for k in ("track_high_thresh", "track_low_thresh", "new_track_thresh",
                                            "track_buffer", "match_thresh", "fuse_score")})


def make_detector(rtcfg, log=print):
    """The rt car detector (ONNX). Returns (detector, (input_w, input_h))."""
    from rt.models import YoloOnnx
    m, r = rtcfg["models"], rtcfg["runtime"]
    det = YoloOnnx(m["car_detector"], m["car_imgsz"], int(r["threads_detector"]), spinning=bool(r["ort_spinning"]),
                   backend=r["car_backend"], device=r["car_device"], log=log)
    h, w = det.imgsz
    return det, (w, h)


# spotter.detector values. "rt" is the old name of "baja" (kept so existing site.toml files still work).
DETECTOR_KINDS = ("baja", "coco", "blobs")
DETECTOR_ALIASES = {"rt": "baja"}


def detector_kind(spotter_cfg):
    """spotter.detector -> "baja" (default), "coco" or "blobs". Anything else is an error at startup."""
    kind = str(spotter_cfg.get("detector", "baja")).strip().lower()
    kind = DETECTOR_ALIASES.get(kind, kind)
    if kind not in DETECTOR_KINDS:
        raise ValueError("spotter.detector must be one of %s, not %r" % (", ".join(DETECTOR_KINDS), kind))
    return kind


def make_spotter(cfg, rtcfg, roi=None, mode=None, det=None, input_wh=None, log=print):
    """The spotter from the config. `det` / `input_wh` inject a detector (tests, tools); otherwise spotter.detector
    picks it: "baja" = the deployed rt/ car detector, "coco" = the bench-only COCO detector (zone/bench_detector.py,
    with its own confidence threshold spotter.coco_conf), "blobs" = white boxes (synthetic tests)."""
    s = cfg["spotter"]
    kind = detector_kind(s)
    conf = rtcfg["detection"]["conf"]
    if det is None and kind == "blobs":
        from zone.testing import BlobDetector
        det, input_wh = BlobDetector(), (640, 384)
    elif det is None and kind == "coco":
        from zone.bench_detector import make_coco_detector
        det, input_wh = make_coco_detector(s, int(rtcfg["runtime"]["threads_detector"]), log)
    elif det is None:
        det, input_wh = make_detector(rtcfg, log)
    if kind == "coco":
        conf = float(s["coco_conf"])
    rd = RoiDetector(det, mode or s["roi_mode"], roi, input_wh or (640, 384), s["tile_overlap"], s["tile_scale"],
                     conf, rtcfg["detection"]["nms_iou"])
    sp = Spotter(rd, make_tracker(rtcfg))
    sp.detector_info = dict(kind=kind, desc=str(getattr(det, "desc", type(det).__name__)), bench=kind == "coco")
    return sp
