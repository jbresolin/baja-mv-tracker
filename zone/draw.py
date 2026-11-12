"""Overlay drawn on the spotter frame for the page: corridor, path, keyframes, ROI, tracks with s and v, director
state. Only drawn when the page asks for a frame, so it costs nothing when nobody is watching."""
import cv2
import numpy as np

CORRIDOR = (80, 160, 80)
PATH = (0, 220, 255)
KEYFRAME = (0, 140, 255)      # orange (BGR)
ROI = (200, 200, 200)
TRACK = (255, 255, 255)
LOCKED = (0, 0, 255)
OFF = (128, 128, 128)


def draw_zone(img, zone, path=None, roi=None, selected=None):
    pts = [(int(round(p.x)), int(round(p.y))) for p in zone.points]
    if len(pts) >= 2:
        hw = int(round(zone.corridor_half_width_px))
        over = img.copy()
        cv2.polylines(over, [np.array(pts, np.int32)], False, CORRIDOR, max(1, 2 * hw), cv2.LINE_AA)
        cv2.addWeighted(over, 0.25, img, 0.75, 0, dst=img)
        cv2.polylines(img, [np.array(pts, np.int32)], False, PATH, 2, cv2.LINE_AA)
    kf_s = path.keyframe_s() if path is not None else None
    k = 0
    for i, (p, xy) in enumerate(zip(zone.points, pts)):
        if p.is_keyframe:
            cv2.circle(img, xy, 9, KEYFRAME, -1, cv2.LINE_AA)
            label = "#%d P%d" % (i, p.keyframe.preset)
            if kf_s is not None and k < len(kf_s):
                label += " s=%.2f" % kf_s[k]
            if p.keyframe.approximate:
                label += " ~"
            k += 1
            _text(img, label, (xy[0] + 12, xy[1] - 10), KEYFRAME)
        else:
            cv2.circle(img, xy, 5, PATH, -1, cv2.LINE_AA)
            _text(img, "#%d" % i, (xy[0] + 8, xy[1] - 6), PATH, 0.5)
        if selected is not None and i == selected:
            cv2.circle(img, xy, 15, (0, 0, 255), 2, cv2.LINE_AA)
    if roi is not None:
        x0, y0, x1, y1 = roi
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), ROI, 1)


def draw_tracks(img, views, locked_id=None, tracker_out=()):
    for tid, box, _ in tracker_out:
        if tid not in views:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), OFF, 1)
    for v in views.values():
        if not v.box:
            continue
        x1, y1, x2, y2 = [int(c) for c in v.box]
        col = LOCKED if v.track_id == locked_id else (OFF if v.off_corridor_frames else TRACK)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if v.track_id == locked_id else 2)
        if v.ground:
            cv2.circle(img, (int(v.ground[0]), int(v.ground[1])), 5, col, -1)
        _text(img, "%d s=%.2f v=%.2f%s" % (v.track_id, v.s, v.v, " pred" if v.predicted else ""), (x1, y1 - 8), col)


def draw_status(img, lines):
    y = 30
    for ln in lines:
        _text(img, ln, (12, y), (255, 255, 255), 0.8)
        y += 30


def _text(img, s, org, col, scale=0.6):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1 if scale < 0.7 else 2, cv2.LINE_AA)
