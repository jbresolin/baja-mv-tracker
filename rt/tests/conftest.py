"""Shared helpers for the rt tests (pytest loads this file automatically before the tests).

How to run all the tests, from the repo folder:
    python -m pytest rt/tests -v
Nothing here needs a camera, a GPU or the 26SE footage. Only test_e2e_clip.py needs the ONNX models
(deploy/models/*.onnx); without them it is skipped, and pytest says so.

A "fixture" in pytest is a function whose name a test lists as an argument; pytest calls it and passes its result
in. For example `def test_x(cfg):` receives the loaded default config below.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]      # the repo folder (the one that contains rt/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rt import config as C  # noqa: E402
from rt.capture import Frame  # noqa: E402


@pytest.fixture
def cfg():
    """The default settings (rt/config.toml), freshly loaded for each test, so a test can change them freely."""
    return C.load()


def textured_frame(w=320, h=180, seed=0):
    """A random-noise image. The pass logic keeps crops of cars and scores them by sharpness; a flat grey image would
    have zero sharpness, so tests use noise to look like a real (sharp) picture."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def make_frame(t, idx, img):
    """A Frame as the capture would produce it: image, wall time t (seconds), frame number."""
    return Frame(img=img, t=t, pts=t, idx=idx, t_cap=t)
