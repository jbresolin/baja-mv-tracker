"""Tests for zone/tools/agent_rig.py, the all-simulated zone node the studio's conformance suite and the end-to-end
check run against. `prepare` writes a short synthetic spotter recording (needs ffmpeg: skipped without it), a taught
zone, a bench report and a config; here we check that the zone program accepts that config as it is."""
import shutil

import pytest

from zone import config as C
from zone import zonefile
from zone.tools import agent_rig

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def test_prepare_writes_a_rig_the_zone_app_accepts(tmp_path):
    rc = agent_rig.main(["prepare", "--dir", str(tmp_path), "--driver", "visca", "--token", "T", "--length", "3",
                         "--gap", "1", "--visca-port", "52400"])
    assert rc == 0
    for name in ("spotter.mp4", "zone.json", "report.json", "site.toml"):
        assert (tmp_path / name).exists(), name
    # the config loads like any site.toml, and points at the rig's files
    cfg = C.load(tmp_path / "site.toml")
    assert cfg["ptz"]["driver"] == "visca" and cfg["ptz"]["port"] == 52400
    assert cfg["spotter"]["detector"] == "blobs" and cfg["recording"]["spotter"] is False
    assert cfg["agent"]["token"] == "T" and cfg["agent"]["host"] == "127.0.0.1"
    settings = C.build(cfg)
    assert settings.controller.branch == "A"            # the report says: inquiry works while moving
    # the zone is taught: 5 keyframes, presets 10..14, so AUTO can run
    zone = zonefile.load(C.resolve_path(cfg["zone"]["file"]))
    assert [k.preset for k in zone.keyframes()] == [10, 11, 12, 13, 14]
