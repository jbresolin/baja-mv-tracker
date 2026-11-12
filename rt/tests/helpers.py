"""Fakes and scenario builders used by several test files.

The real pipeline is: camera -> car detector -> tracker -> pass logic -> identity (panel detector -> recognizer ->
vote). The tests skip the models: they feed the pass logic hand-made car tracks ("this car is at this box at this
time"), and give the identity worker a fake panel detector and a fake recognizer that "read" a number we choose.
"""
import numpy as np

from rt.identity import IdentityWorker
from rt.models import CHARS
from rt.passlogic import PassLogic

from .conftest import make_frame, textured_frame

T0 = 1_800_000_000.0     # an arbitrary wall-clock start (epoch seconds) for synthetic scenes
DT = 0.1                 # 10 fps, the processing rate
W, H = 320, 180          # synthetic frame size


def t_of(i):
    """Wall time of synthetic frame number i."""
    return T0 + i * DT


# ------------------------------------------------------------------------------------------------ fake models
class FakePanel:
    """Stands in for the number-panel detector: finds one 'number panel' covering the middle of every crop (or none,
    with boxes=False). Returns (boxes, scores, classes) like rt.models.YoloOnnx."""

    def __init__(self, boxes=True):
        self.boxes = boxes

    def __call__(self, img, conf=0.1, iou=0.7):
        h, w = img.shape[:2]
        if not self.boxes:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
        return np.array([[0.2 * w, 0.2 * h, 0.8 * w, 0.8 * h]]), np.array([0.9]), np.array([0])


def ctc_logp(text, T=10):
    """Log-probabilities (T, 11) of a CTC recognizer that is sure it reads `text`: each character on one time step,
    a blank between them, blanks after. Class 0 = blank, class d+1 = digit d (rt.models layout)."""
    seq = []
    for ch in text:
        seq += [CHARS.index(ch) + 1, 0]
    seq += [0] * (T - len(seq))
    lp = np.full((T, len(CHARS) + 1), -20.0)
    for t, c in enumerate(seq[:T]):
        lp[t, c] = 0.0
    return lp


class FakeRecognizer:
    """Stands in for the recognizer: every number box reads as `read(k)` for the k-th box it sees (a string such as
    "14"), or as the fixed string `read`. Returns logp (T, N, 11) like rt.models.Recognizer."""

    def __init__(self, read):
        self.read = read
        self.n = 0

    def __call__(self, crops):
        out = []
        for _ in crops:
            text = self.read(self.n) if callable(self.read) else self.read
            self.n += 1
            out.append(ctc_logp(text))
        return np.stack(out, 1)


def make_identity(cfg, roster, emit, read="14", boxes=True):
    """A real IdentityWorker (the real vote and roster snapping) with fake models. It is not started as a thread:
    SyncIdentity below calls its methods directly, so results are there as soon as the pass logic asks."""
    return IdentityWorker(cfg, roster, {n: "Team %d" % n for n in roster}, emit, log=print, panel=FakePanel(boxes),
                          recognizer=FakeRecognizer(read))


class SyncIdentity:
    """Gives the pass logic the identity interface (submit_provisional / submit_final) but does the work at once in
    the same thread, instead of queueing it for the worker thread. Same results, no waiting, no threads."""

    def __init__(self, worker):
        self.w = worker

    def submit_provisional(self, p, cands):
        self.w._provisional(p, cands)

    def submit_final(self, p, end_event):
        self.w._final(p, p.selected, end_event)


# ------------------------------------------------------------------------------------------------ scenes
def moving_box(i, i0, i1, x_from, x_to, y1=90, bw=64, bh=54):
    """The box (x1, y1, x2, y2 in pixels) of a car moving at constant speed from x_from (frame i0) to x_to (i1)."""
    f = (i - i0) / max(1, i1 - i0)
    x1 = x_from + f * (x_to - x_from)
    return (x1, y1, x1 + bw, y1 + bh)


def run_scene(cfg, tracks, n_frames, identity=None, linger_frames=30, seed=0):
    """Play a synthetic scene through the real online pass logic and return every event it emitted.

    `tracks` = {track id: {frame number: box}}. Like ByteTrack, a track stays "alive" (lost but still matchable) for
    `linger_frames` frames (30 = 3 s, track_buffer) after its last box; the pass logic waits for that before closing a
    pass. Events are collected in a list of dicts; each one also gets `frame` = the frame number it was emitted at."""
    events = []
    frame_no = [0]

    def emit(ev):
        events.append(dict(ev, frame=frame_no[0]))

    pl = PassLogic(cfg, emit, identity, log=lambda m: None)
    if identity is not None:
        identity.w.emit = emit
        identity.w.now_fn = lambda: pl.now
    img = textured_frame(W, H, seed)
    for i in range(n_frames):
        frame_no[0] = i
        outputs = [(tid, np.array(boxes[i], float), 0.9) for tid, boxes in tracks.items() if i in boxes]
        alive = {tid for tid, boxes in tracks.items() if min(boxes) <= i <= max(boxes) + linger_frames}
        pl.step(make_frame(t_of(i), i, img), outputs, alive)
    pl.flush()
    return events


def of_type(events, kind):
    return [e for e in events if e["type"] == kind]
