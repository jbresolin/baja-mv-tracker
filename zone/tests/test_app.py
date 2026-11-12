"""Tests for zone/app.py: the web page's API (mode buttons, big STOP, teach), with a simulated PTZ and no camera."""
import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from zone import config as C  # noqa: E402
from zone.app import ZoneApp, create_web  # noqa: E402
from zone.clock import FakeClock  # noqa: E402
from zone.ptz.sim import PtzModel, SimulatedPtz  # noqa: E402
from zone.runtime import CycleResult  # noqa: E402
from zone.director import Decision  # noqa: E402
from zone.source import ZoneFrame  # noqa: E402
from zone.tests.z2_fixtures import straight_zone  # noqa: E402
from zone.zonefile import Zone  # noqa: E402


def make(tmp_path, zone=None):
    cfg = C.load(None, ["ptz.driver=sim"])
    st = C.build(cfg)
    clock = FakeClock(0.0)
    ptz = SimulatedPtz(PtzModel(), clock)
    app = ZoneApp(cfg, st, ptz, clock, zone if zone is not None else straight_zone(), tmp_path / "zone.json",
                  tmp_path / "out", log=lambda m: None)
    return TestClient(create_web(app)), app, ptz, clock


def test_state_and_page(tmp_path):
    client, app, _, _ = make(tmp_path)
    s = client.get("/api/state").json()
    assert s["runtime"]["director"]["mode"] == "STOPPED"
    assert len(s["zone"]["points"]) == 4 and s["zone"]["entry_preset"] == 10
    assert "PARAR" in client.get("/").text            # the page is pt-BR
    assert client.get("/app.js").status_code == 200


def test_mode_buttons(tmp_path):
    client, app, _, _ = make(tmp_path)
    assert client.post("/api/mode", json={"mode": "PASSIVE"}).json()["mode"] == "PASSIVE"
    assert client.post("/api/mode", json={"mode": "AUTO"}).json()["mode"] == "AUTO"
    assert app.runtime.director.mode == "AUTO"


def test_auto_is_refused_without_a_taught_zone(tmp_path):
    client, app, _, _ = make(tmp_path, Zone())
    r = client.post("/api/mode", json={"mode": "AUTO"})
    assert r.status_code == 409
    assert app.runtime.director.mode == "STOPPED"


def test_big_stop(tmp_path):
    client, app, ptz, clock = make(tmp_path)
    client.post("/api/mode", json={"mode": "AUTO"})
    ptz.pan_tilt_velocity(12, 0)
    clock.advance(0.5)
    assert client.post("/api/stop").json()["ok"]
    clock.advance(0.5)
    assert not ptz.is_moving()
    assert app.runtime.director.mode == "STOPPED"


def test_teaching_is_blocked_in_auto(tmp_path):
    client, app, _, _ = make(tmp_path)
    client.post("/api/mode", json={"mode": "AUTO"})
    r = client.post("/api/teach/jog", json={"pan": 6})
    assert r.status_code == 409 and "AUTO" in r.json()["detail"]


def test_teach_through_the_api(tmp_path):
    client, app, ptz, clock = make(tmp_path, Zone())
    assert client.post("/api/teach/point", json={"x": 100, "y": 500}).json()["ok"]
    client.post("/api/teach/jog", json={"pan": 8, "tilt": 0})
    clock.advance(1.0)
    client.post("/api/teach/jog_stop")
    clock.advance(1.0)
    r = client.post("/api/teach/point", json={"x": 300, "y": 500, "keyframe": True}).json()
    assert r["result"]["keyframe"]["preset"] == 10 and r["result"]["keyframe"]["pan"] > 0
    client.post("/api/teach/point", json={"x": 900, "y": 500, "keyframe": True})
    assert app.runtime.path is not None             # 2 keyframes: the runtime has a path now
    assert client.post("/api/teach/corridor", json={"half_width": 3}).status_code == 409
    assert client.post("/api/teach/save").json()["ok"]
    assert (tmp_path / "zone.json").exists()
    assert client.get("/api/state").json()["zone"]["dirty"] is False


def test_frame_endpoint(tmp_path):
    client, app, _, _ = make(tmp_path)
    assert client.get("/frame.jpg").status_code == 204           # no frame yet
    img = np.zeros((1080, 1920, 3), np.uint8)
    app.latest = (ZoneFrame(img, 0.0, 0.0, 0), CycleResult(0.0, 0.0, {}, Decision("none"), None, []))
    r = client.get("/frame.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"  # a JPEG


def test_health_disk_and_ptz(tmp_path):
    client, app, _, _ = make(tmp_path)
    app.check_health()
    h = client.get("/api/state").json()["health"]
    assert h["disk_free_gb"] > 0 and h["ptz_reachable"] is True
