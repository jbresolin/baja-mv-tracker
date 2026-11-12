"""Test helpers: a small synthetic world, so the whole pipeline can be tested without cameras, cars or real time.

* `Scene`: a racing line on the ground (metres), a spotter that sees it (a simple mapping to image pixels), and a PTZ
  standing in front of it. For any ground point it knows the *true* PTZ pose that frames it (pan from the bearing,
  tilt from the distance, zoom growing with distance). `make_zone()` "teaches" a zone the way an operator would:
  path points clicked along the line, keyframes with the true pose at a few of them.
* `Car`: moves along the racing line with a speed that depends on where it is (brakes into the bend, accelerates
  out), starting at a given time.
* `boxes_at()`: the boxes a perfect detector would return at time t; `render()`: a black image with white boxes (a
  "video frame"), and `BlobDetector`: a detector that finds those white boxes, with rt's YoloOnnx call signature.

Everything is deterministic.
"""
import math

import cv2
import numpy as np

from zone.zonefile import Keyframe, PathPoint, Zone

UNITS_PER_DEGREE = 14.4   # raw PTZ units per degree in this fake world (VISCA cameras are often ~14-16)


class Scene:
    # the racing line, ground coordinates in metres (x to the right, y away from the PTZ)
    WORLD = np.array([(-45, 22), (-30, 22), (-10, 25), (5, 32), (20, 35), (35, 30), (50, 24)], dtype=np.float64)

    def __init__(self, image_size=(1920, 1080), ptz_xy=(0.0, 0.0), box_wh=(90, 60), upstream_m=15.0,
                 zone_len_m=60.0):
        self.image_size = image_size
        self.ptz_xy = np.array(ptz_xy, dtype=np.float64)
        self.box_wh = box_wh
        seg = np.diff(self.WORLD, axis=0)
        self.cum = np.r_[0.0, np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))]
        self.length = float(self.cum[-1])
        self.arc_entry = upstream_m                  # s = 0
        self.arc_exit = upstream_m + zone_len_m      # s = 1

    # ---- geometry
    def world_at(self, arc):
        a = float(np.clip(arc, 0, self.length))
        i = int(np.clip(np.searchsorted(self.cum, a, side="right") - 1, 0, len(self.WORLD) - 2))
        t = (a - self.cum[i]) / (self.cum[i + 1] - self.cum[i])
        return self.WORLD[i] + t * (self.WORLD[i + 1] - self.WORLD[i])

    def image_of(self, w):
        x, y = w
        return 960.0 + x * 18.0, 900.0 - (y - 20.0) * 25.0

    def true_pose(self, w):
        """(pan, tilt, zoom) raw units that frame ground point w."""
        dx, dy = w - self.ptz_xy
        dist = math.hypot(dx, dy)
        pan = math.degrees(math.atan2(dx, dy)) * UNITS_PER_DEGREE
        tilt = -math.degrees(math.atan2(3.0, dist)) * UNITS_PER_DEGREE   # PTZ 3 m above the car
        zoom = min(16384.0, dist * 250.0)
        return pan, tilt, zoom

    def s_of_arc(self, arc):
        return (arc - self.arc_entry) / (self.arc_exit - self.arc_entry)

    # ---- teaching
    def make_zone(self, keyframe_arcs=None, step_m=5.0, preset_base=10, half_width=60.0):
        """A zone as an operator would teach it: points every step_m along the line; keyframes at keyframe_arcs
        (default: 5 evenly spread from entry to exit) carrying the true pose."""
        if keyframe_arcs is None:
            keyframe_arcs = list(np.linspace(self.arc_entry, self.arc_exit, 5))
        arcs = sorted(set(list(np.arange(0.0, self.length + 1e-6, step_m)) + list(keyframe_arcs)))
        pts = []
        k = 0
        for a in arcs:
            w = self.world_at(a)
            x, y = self.image_of(w)
            kf = None
            if any(abs(a - ka) < 1e-6 for ka in keyframe_arcs):
                p, t, z = self.true_pose(w)
                kf = Keyframe(int(round(p)), int(round(t)), int(round(z)), preset_base + k, False, "inquiry")
                k += 1
            pts.append(PathPoint(x, y, kf))
        return Zone(points=pts, corridor_half_width_px=half_width, image_size=list(self.image_size))

    # ---- drawing
    def box_of(self, arc, lateral_px=0.0):
        x, y = self.image_of(self.world_at(arc))
        w, h = self.box_wh
        return (x - w / 2 + lateral_px, y - h, x + w / 2 + lateral_px, y)


def default_speed(scene: Scene, base=10.0, slow=5.0):
    """Speed (m/s) as a function of arc: brakes to `slow` around the bend in the middle of the zone."""
    mid = (scene.arc_entry + scene.arc_exit) / 2.0

    def v(arc):
        return slow + (base - slow) * min(1.0, abs(arc - mid) / 20.0)
    return v


class Car:
    """A car that starts at arc 0 at time t0 and drives along the line with speed_fn(arc) (m/s) times `factor`.
    `stop_at` (arc) makes it stop there (breakdown); `lateral_px` shifts its box sideways in the image."""

    def __init__(self, scene: Scene, t0, speed_fn=None, factor=1.0, stop_at=None, lateral_px=0.0, dt=0.01):
        self.scene, self.t0 = scene, float(t0)
        self.lateral_px = lateral_px
        speed_fn = speed_fn or default_speed(scene)
        ts, arcs = [0.0], [0.0]
        a, t = 0.0, 0.0
        while a < scene.length and t < 600:
            v = 0.0 if (stop_at is not None and a >= stop_at) else speed_fn(a) * factor
            a += v * dt
            t += dt
            ts.append(t)
            arcs.append(min(a, scene.length))
            if v == 0.0 and t > 120:
                break
        self.ts, self.arcs = np.array(ts), np.array(arcs)

    def arc_at(self, t):
        """Arc length at time t, or None before the start / after the end of the line."""
        if t < self.t0:
            return None
        tt = t - self.t0
        a = float(np.interp(tt, self.ts, self.arcs))
        if a >= self.scene.length - 1e-6:
            return None          # drove out of the picture
        if tt > self.ts[-1]:
            return float(self.arcs[-1])   # stopped for good: stays where it is
        return a


def boxes_at(scene, cars, t, hidden=lambda car, t: False):
    """Boxes (N,4) and scores (N,) a perfect detector would see at time t."""
    out = []
    for c in cars:
        a = c.arc_at(t)
        if a is None or hidden(c, t):
            continue
        out.append(scene.box_of(a, c.lateral_px))
    if not out:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    return np.array(out, np.float32), np.full(len(out), 0.9, np.float32)


def render(scene, boxes):
    """A grey image with a filled white rectangle per box (what BlobDetector finds)."""
    w, h = scene.image_size
    img = np.full((h, w, 3), 40, np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2) - 1, int(y2) - 1), (255, 255, 255), -1)
    return img


class BlobDetector:
    """Finds bright rectangles. Same call signature as rt.models.YoloOnnx: det(img, conf, iou) ->
    (boxes xyxy, scores, classes). Counts its calls and the image sizes it was given (for the ROI tests)."""

    def __init__(self, min_area=50):
        self.min_area = min_area
        self.calls = []

    def __call__(self, img, conf=0.25, iou=0.7):
        self.calls.append(img.shape[:2])
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        _, th = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
        n, _, stats, _ = cv2.connectedComponentsWithStats(th)
        boxes = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= self.min_area:
                boxes.append((x, y, x + w, y + h))
        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        b = np.array(boxes, np.float32)
        return b, np.full(len(b), 0.9, np.float32), np.zeros(len(b), int)


def write_video(path, scene, cars, t_end, fps=10.0, ffmpeg="ffmpeg"):
    """Render the cars to an H.264 file (what a spotter recording looks like to the replay / detection tools)."""
    import subprocess
    w, h = scene.image_size
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h),
           "-r", "%g" % fps, "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "10", "-pix_fmt", "yuv420p",
           str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(int(round(t_end * fps))):
        b, _ = boxes_at(scene, cars, k / fps)
        proc.stdin.write(render(scene, b).tobytes())
    proc.stdin.close()
    proc.wait()
    return path
