"""Live beam <-> vision helpers (WP-L2): the simple pass/break association rule and the LED latency detector.

Association (live, deliberately simple; the real fusion is offline in rt/tools/fuse_beam.py):
  When rt closes a pass (`pass_end`), look for beam breaks whose crossing time lies within `window_s` of the pass's
  vision crossing time `t_cross` (shifted by `offset_s`, the expected t_cross - t_beam: the camera's centre line is
  not exactly at the beam). Exactly one -> "matched": the pass gets that break's t_beam. None -> "none". Several ->
  "ambiguous" (a pack of cars). Only "matched" carries a t_beam; the others are left for the offline step. The pass
  logic is never changed: the result is a separate `pass_beam` event.

LED latency (optional, off by default):
  If the sensor's output (or a spare adapter channel) also lights an LED that the lap camera sees, every crossing
  measures the camera's latency: the first processed frame where the LED is on, minus the beam edge that turned it
  on. LedWatcher looks at a small region of each processed frame (`roi` = x, y, w, h in processed-frame pixels) and
  reports when its mean brightness goes from below `threshold` to above it.
"""
import numpy as np


def associate_pass(t_cross, breaks, window_s=1.0, offset_s=0.0):
    """The live rule. `breaks` = beam_break records with "t_beam" (and optionally "started_blocked").

    Returns (status, the matched break or None, the list of candidate breaks):
      status "matched"   exactly one break with |t_beam - (t_cross - offset_s)| <= window_s
             "none"      no break in the window
             "ambiguous" two or more (cars side by side or nose to tail): left for the offline fusion
    Breaks that started while the beam was already blocked have no real crossing time and are never candidates."""
    expected = float(t_cross) - float(offset_s)
    cands = [b for b in breaks if not b.get("started_blocked") and abs(b["t_beam"] - expected) <= float(window_s)]
    cands.sort(key=lambda b: abs(b["t_beam"] - expected))
    if len(cands) == 1:
        return "matched", cands[0], cands
    return ("none" if not cands else "ambiguous"), None, cands


def parse_roi(roi):
    """[x, y, w, h] (config) -> (x, y, w, h) ints, or None when the LED feature is off (empty list)."""
    if not roi:
        return None
    if len(roi) != 4:
        raise ValueError("beam.led_roi must be [x, y, w, h] (pixels of the processed frame); got %r" % (roi,))
    x, y, w, h = (int(v) for v in roi)
    if w <= 0 or h <= 0:
        raise ValueError("beam.led_roi width and height must be > 0; got %r" % (roi,))
    return x, y, w, h


class LedWatcher:
    """Watches the LED region of every processed frame. update(frame) returns the mean brightness (0-255) when the
    LED just turned ON (off -> on), else None. The first frame only sets the starting state: an LED that is already
    on when rt starts is not a transition."""

    def __init__(self, roi, threshold=128.0):
        self.roi = parse_roi(roi)
        self.threshold = float(threshold)
        self.state = None            # None = no frame seen yet; True = LED on; False = off
        self.last_mean = None

    def mean_brightness(self, img):
        x, y, w, h = self.roi
        patch = img[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
        if patch.size == 0:
            return None              # ROI outside the frame (wrong config): never a transition
        return float(np.mean(patch))

    def update(self, img):
        if self.roi is None:
            return None
        m = self.mean_brightness(img)
        self.last_mean = m
        if m is None:
            return None
        on = m >= self.threshold
        was = self.state
        self.state = on
        if was is False and on:
            return m
        return None
