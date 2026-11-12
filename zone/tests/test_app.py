"""Tests for zone/app.py: the web page's API (mode buttons, big STOP, teach), with a simulated PTZ and no camera."""
from pathlib import Path

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
    # moving the PTZ from the page needs "teach mode" on (it starts off)
    assert client.post("/api/teach/mode", json={"on": True}).json()["result"] is True
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


# ================================================================================= "is the PTZ reachable?" on the page
class SilentPtz:
    """Wraps the simulated PTZ. `answers` decides whether inquiries get an answer (False = the camera is unplugged
    or doesn't support inquiries: every inquiry returns None, like the real driver after its timeout). Every call
    is recorded in `calls`, so a test can check what the health check asked."""

    def __init__(self, inner, answers=True):
        self.inner, self.answers, self.calls = inner, answers, []

    def __getattr__(self, name):
        fn = getattr(self.inner, name)

        def wrapped(*a, **kw):
            self.calls.append(name)
            if name.startswith("inquire") and not self.answers:
                return None
            return fn(*a, **kw)
        return wrapped


def make_with(tmp_path, ptz_wrapper):
    cfg = C.load(None, ["ptz.driver=sim"])
    clock = FakeClock(0.0)
    ptz = ptz_wrapper(SimulatedPtz(PtzModel(), clock))
    app = ZoneApp(cfg, C.build(cfg), ptz, clock, straight_zone(), tmp_path / "zone.json", tmp_path / "out",
                  log=lambda m: None)
    return app, ptz, clock


def test_ptz_reachability_is_unknown_until_something_answers(tmp_path):
    # A camera that answers no inquiry, and nothing was sent to it yet: we simply don't know. The page used to say
    # "respondendo" here, because the controller's "last command ok" started as True before any command.
    app, ptz, _ = make_with(tmp_path, lambda sim: SilentPtz(sim, answers=False))
    app.check_health()
    assert app.health["ptz_reachable"] is None
    assert app.runtime.controller.ptz_ok is None
    # once a command was sent and the camera answered it, it is known to be reachable
    app.runtime.big_stop()
    app.check_health()
    assert app.health["ptz_reachable"] is True and app.health["ptz_reachable_how"] == "last_command"


def test_ptz_reachable_from_the_probe_and_unreachable_when_it_stops_answering(tmp_path):
    app, ptz, _ = make_with(tmp_path, lambda sim: SilentPtz(sim, answers=True))
    app.check_health()
    assert app.health["ptz_reachable"] is True and app.health["ptz_reachable_how"] == "probe"
    ptz.answers = False                      # the cable is pulled
    app.check_health()
    # it answered before, so silence now means "unreachable" (not "unknown")
    assert app.health["ptz_reachable"] is False


def test_page_shows_the_three_ptz_states(tmp_path):
    # the page script has a text for each state: unknown, answering, not answering
    js = (Path(__file__).resolve().parent.parent / "static" / "app.js").read_text(encoding="utf-8")
    assert "desconhecido" in js and "respondendo" in js and "SEM RESPOSTA" in js


def test_health_probe_stays_off_the_wire_while_auto_drives_the_camera(tmp_path):
    # While AUTO follows a car, the control loop refreshes the velocity command every cycle, and the driver sends
    # one request at a time. A health inquiry from another thread could hold the wire long enough to push a refresh
    # to the edge of the TTL (the camera would stop). So while tracking or returning, the health check must not
    # ask the camera anything: it reports the result of the controller's last command instead.
    app, ptz, _ = make_with(tmp_path, lambda sim: SilentPtz(sim, answers=True))
    d = app.runtime.director
    for sub in ("tracking", "returning"):
        d.mode, d.sub = "AUTO", sub
        ptz.calls.clear()
        app.runtime.controller.ptz_ok = True
        app.check_health()
        assert not [c for c in ptz.calls if c.startswith("inquire")], sub
        assert app.health["ptz_reachable"] is True and app.health["ptz_reachable_how"] == "last_command"
    # waiting at the entry (camera still): probing is fine again
    d.sub = "waiting"
    ptz.calls.clear()
    app.check_health()
    assert "inquire_zoom" in ptz.calls


# ======================================================================================================= teach mode
def test_jog_needs_teach_mode(tmp_path):
    # In PARADO (STOPPED) nothing should move the camera by accident: a jog is refused (HTTP 409 = "not now")
    # until the operator turns teach mode on.
    client, app, ptz, clock = make(tmp_path)
    r = client.post("/api/teach/jog", json={"pan": 6})
    assert r.status_code == 409 and "Modo ensino" in r.json()["detail"]
    assert not ptz.is_moving()
    client.post("/api/teach/mode", json={"on": True})
    assert client.post("/api/teach/jog", json={"pan": 6}).status_code == 200
    clock.advance(0.2)
    assert ptz.is_moving()


def test_big_stop_cancels_a_held_jog(tmp_path):
    # The page repeats the jog every 150 ms while a button is held. After the big STOP, those repeats must be
    # refused, or the camera would start moving again right after the STOP.
    client, app, ptz, clock = make(tmp_path)
    client.post("/api/teach/mode", json={"on": True})
    client.post("/api/teach/jog", json={"pan": 6})
    clock.advance(0.2)
    assert client.post("/api/stop").json()["ok"]
    assert app.teach_mode is False                          # the STOP turned teach mode off
    assert client.post("/api/teach/jog", json={"pan": 6}).status_code == 409   # the next repeat is refused
    clock.advance(1.0)
    assert not ptz.is_moving()


def test_teach_mode_ends_with_auto_and_parado_and_cannot_start_in_auto(tmp_path):
    client, app, _, _ = make(tmp_path)
    client.post("/api/teach/mode", json={"on": True})
    client.post("/api/mode", json={"mode": "AUTO"})
    assert app.teach_mode is False
    assert client.post("/api/teach/mode", json={"on": True}).status_code == 409
    client.post("/api/mode", json={"mode": "STOPPED"})
    client.post("/api/teach/mode", json={"on": True})
    client.post("/api/mode", json={"mode": "STOPPED"})       # pressing PARADO also ends it
    assert app.teach_mode is False
    assert client.get("/api/state").json()["teach"]["teach_mode"] is False


def test_page_asks_for_the_zoom_when_the_camera_cant_tell_it(tmp_path):
    # a camera without zoom inquiry: the page gets a flag to show its "type the zoom" field, and a keyframe sent
    # without a zoom is refused (409) instead of being stored with a made-up zoom
    cfg = C.load(None, ["ptz.driver=sim"])
    clock = FakeClock(0.0)
    ptz = SimulatedPtz(PtzModel(zoom_inquiry_mode="never"), clock)
    app = ZoneApp(cfg, C.build(cfg), ptz, clock, Zone(), tmp_path / "zone.json", tmp_path / "out", log=lambda m: None)
    client = TestClient(create_web(app))
    r = client.post("/api/teach/point", json={"x": 300, "y": 500, "keyframe": True})
    assert r.status_code == 409 and "zoom" in r.json()["detail"]
    assert client.get("/api/state").json()["teach"]["zoom_unknown"] is True
    r = client.post("/api/teach/point", json={"x": 300, "y": 500, "keyframe": True, "zoom": 2500})
    assert r.json()["result"]["keyframe"]["zoom"] == 2500
    assert client.post("/api/teach/keyframe_zoom", json={"index": 0, "zoom": 2700}).json()["ok"]
    assert app.teach.zone.points[0].keyframe.zoom == 2700
