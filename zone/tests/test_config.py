"""Tests for zone/config.py: the example config loads, overrides work, and "auto" values come from the report."""
import json

import pytest

from zone import config as C
from zone.ptz.sim import PtzModel


def test_example_config_loads_and_builds():
    st = C.build(C.load())
    assert st.controller.branch == "B"               # no report: Branch B
    assert st.report.path == ""
    assert st.director.settle_s == 1.5


def test_set_overrides_nested_keys():
    cfg = C.load(None, ["controller.k_pan=0.5", "rt.capture.fps=5", "spotter.source=video.mp4"])
    assert cfg["controller"]["k_pan"] == 0.5
    assert cfg["rt"]["capture"]["fps"] == 5
    assert cfg["spotter"]["source"] == "video.mp4"


def test_user_config_only_changes_its_keys(tmp_path):
    f = tmp_path / "site.toml"
    f.write_text('[ptz]\nhost = "10.0.0.9"\n[rt.capture]\nfps = 8\n')
    cfg = C.load(f)
    assert cfg["ptz"]["host"] == "10.0.0.9" and cfg["ptz"]["port"] == 52381
    assert cfg["rt"]["capture"]["fps"] == 8 and cfg["rt"]["capture"]["camera_fps"] == 30


def write_report(tmp_path, branch, **model):
    m = PtzModel(**model)
    f = tmp_path / "report.json"
    f.write_text(json.dumps(dict(model=m.to_dict(), verdict=dict(branch=branch, reason="bench"))))
    return f


def test_branch_and_auto_values_come_from_the_report(tmp_path):
    f = write_report(tmp_path, "A", inquiry_mode="always", command_latency_s=0.12, ramp_s=0.2, preset_settle_s=2.5,
                     inquiry_poll_hz=50.0, pan_min_smooth_step=3, pan_sign=-1)
    st = C.build(C.load(None, ["ptz.report=%s" % f]))
    assert st.controller.branch == "A"
    assert st.controller.latency_s == pytest.approx(0.12 + 0.1)   # latency + half the ramp
    assert st.controller.inquiry_hz == 20.0                       # capped
    assert st.director.settle_s == 2.5
    assert st.pan_map.min_step == 3
    assert st.controller.pan_sign == -1


def test_config_can_force_a_branch_but_not_a_impossible_one(tmp_path):
    f = write_report(tmp_path, "A", inquiry_mode="always")
    assert C.build(C.load(None, ["ptz.report=%s" % f, "controller.branch=B"])).controller.branch == "B"
    g = write_report(tmp_path, "B", inquiry_mode="at_rest")
    # Branch A needs inquiry while moving: forcing it on an "at rest only" camera falls back to B
    assert C.build(C.load(None, ["ptz.report=%s" % g, "controller.branch=A"])).controller.branch == "B"
