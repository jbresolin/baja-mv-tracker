"""Tests for the agent API (zone/agent/): the studio (baja-studio) driving the zone's PTZ, end to end inside one
process: HTTP request -> agent -> zone app -> simulated PTZ, on a fake clock.

How these tests work, for a first-time reader:
* `make_rig()` (zone/tests/agent_fixtures.py) builds the whole zone program with a simulated camera. `rig.studio`
  sends requests the way the studio does; `rig.page` sends the laptop page's requests.
* `rig.advance(seconds)` lets fake time pass: the simulated camera moves, and the agent's background work runs.
* HTTP status codes: 200 = done, 400 = bad request, 401 = wrong token, 404 = unknown camera or preset,
  409 = "not now" (e.g. AUTO while the tracker can't run), 503 = the camera doesn't answer.
* Positions the studio sees are in degrees; the simulated camera speaks in "raw units" (about 13.3 per degree).

The same rules are checked for real, over the network, by the studio's own conformance suite (see zone/README.md,
"Agent API"); these tests pin them here, without a network, so CI keeps them.
"""
import json

import pytest

pytest.importorskip("fastapi")

from zone.agent.camera import speed_step  # noqa: E402
from zone.ptz.sim import PtzModel, SimulatedPtz  # noqa: E402
from zone.tests.agent_fixtures import CountingPtz, DeadPtz, RefusingPtz, make_rig  # noqa: E402
from zone.zonefile import Zone  # noqa: E402


# =========================================================================================================== who
def test_info_describes_one_ptz_with_the_zones_keyframe_presets(tmp_path):
    rig = make_rig(tmp_path)
    info = rig.studio.get("/agent/info").json()
    assert info["contract_version"] == "1.0"
    [cam] = info["cameras"]
    assert cam["id"] == "cam1" and cam["kind"] == "ptz" and cam["supports_auto"] is True
    assert cam["image_controls"] == []                   # not in this package
    assert cam["limits"]["preset_max"] == 127
    # straight_zone() has 3 keyframes with presets 10, 11, 12: listed with their keyframe's name
    assert cam["presets"] == [{"n": 10, "label": "Zone entry (keyframe 1)"}, {"n": 11, "label": "Zone keyframe 2"},
                              {"n": 12, "label": "Zone exit (keyframe 3)"}]
    # top speeds in degrees per second, from the report's speed table (the default model: ~80 deg/s pan)
    assert 70 < cam["limits"]["pan_speed_max"] < 90


def test_health_lists_the_camera(tmp_path):
    health = make_rig(tmp_path).studio.get("/agent/health").json()
    assert health["status"] == "ok" and health["cameras"] == {"cam1": "ok"}


def test_a_token_is_needed_everywhere_but_info(tmp_path):
    # With a token configured, a request without it is refused (401), except /agent/info: the studio's network scan
    # knows no token and must still find the agent.
    rig = make_rig(tmp_path, token="secret")
    anonymous = {"X-Agent-Token": ""}
    assert rig.studio.get("/agent/info", headers=anonymous).status_code == 200
    refused = rig.studio.post("/cameras/cam1/stop", headers=anonymous)
    assert refused.status_code == 401 and refused.json()["error"] == "unauthorized"
    assert rig.studio.get("/agent/health", headers={"X-Agent-Token": "wrong"}).status_code == 401
    assert rig.studio.get("/agent/health").status_code == 200            # the right token (in rig.studio's headers)


def test_the_websocket_needs_the_token_too(tmp_path):
    from starlette.testclient import WebSocketDenialResponse
    rig = make_rig(tmp_path, token="secret")
    with pytest.raises(WebSocketDenialResponse) as refused:
        with rig.studio.websocket_connect("/agent/ws", headers={"X-Agent-Token": "wrong"}):
            pass
    assert refused.value.status_code == 401


def test_errors_are_json_with_the_contracts_codes(tmp_path):
    rig = make_rig(tmp_path)
    unknown = rig.studio.post("/cameras/cam9/stop")
    assert unknown.status_code == 404 and unknown.json()["error"] == "unknown_camera" and unknown.json()["detail"]
    malformed = rig.studio.post("/cameras/cam1/move", content=b"{not json",
                                headers={"Content-Type": "application/json"})
    assert malformed.status_code == 400 and malformed.json()["error"] == "bad_request"   # not FastAPI's 422
    for body in ({"pan": 1.5}, {"tilt": -2}, {"pan": 0.1, "ttl_ms": 0}, {"pan": 0.1, "ttl_ms": 2001}):
        assert rig.studio.post("/cameras/cam1/move", json=body).status_code == 400
    paused = rig.studio.put("/cameras/cam1/mode", json={"mode": "auto_paused"})
    assert paused.status_code == 400 and paused.json()["error"] == "bad_request"


def test_image_controls_and_crop_are_not_supported_but_never_crash(tmp_path):
    rig = make_rig(tmp_path)
    assert rig.studio.get("/cameras/cam1/image").json() == {}             # no controls: an empty object
    assert rig.studio.put("/cameras/cam1/image", json={}).json() == {}
    unknown = rig.studio.put("/cameras/cam1/image", json={"exposure": {"value": 1}})
    assert unknown.status_code == 400 and unknown.json()["error"] == "unknown_control"
    for answer in (rig.studio.get("/cameras/cam1/crop"),
                   rig.studio.put("/cameras/cam1/crop", json={"x": 0, "y": 0, "w": 1, "h": 1})):
        assert answer.status_code == 400 and answer.json()["error"] == "not_supported"


def test_every_command_is_logged_with_the_studio_id_refused_ones_too(tmp_path):
    rig = make_rig(tmp_path)
    rig.studio.post("/cameras/cam1/stop")
    rig.studio.post("/cameras/cam1/presets/10/store")          # refused: a keyframe preset
    log = rig.studio.get("/agent/log", params={"limit": 2}).json()
    assert [(e["command"], e["result"]) for e in log] == [("store_preset", "error: bad_request"), ("stop", "ok")]
    assert log[1]["source"] == "studio:test-studio"


# =========================================================================================================== motion
def test_speed_steps_from_velocities():
    # -1..1 becomes a speed step: 1.0 is the top step, the smallest movement is step 1, the sign is the direction.
    assert speed_step(0.0, 24) == 0
    assert speed_step(1.0, 24) == 24 and speed_step(-1.0, 24) == -24
    assert speed_step(0.01, 24) == 1
    assert speed_step(-0.5, 24) == -13          # 1 + round(0.5 * 23) = 1 + 12 (half up)


def test_a_studio_move_turns_the_camera_and_stops_by_itself_after_the_ttl(tmp_path):
    # The dead-man switch: the studio sends ONE move (as if its link died right after). The camera turns right,
    # then stops on its own about ttl_ms later, with no further command.
    rig = make_rig(tmp_path)
    answer = rig.studio.post("/cameras/cam1/move", json={"pan": 0.5, "ttl_ms": 300}).json()
    assert answer == {"accepted": True, "mode": "manual"}
    rig.advance(0.2)
    assert rig.ptz.is_moving()
    assert rig.agent.camera.state()["moving"] is True
    rig.advance(1.0)                         # ttl 0.3 s + the camera's stop latency and braking
    assert not rig.ptz.is_moving()
    assert rig.agent.camera.state()["moving"] is False
    pan_then = rig.ptz.true_pan_tilt()[0]
    assert pan_then > 20                     # it did turn right (raw units; ~0.2 s at ~130 units/s)
    rig.advance(1.0)
    assert rig.ptz.true_pan_tilt()[0] == pytest.approx(pan_then)     # and stays stopped


def test_moves_resent_before_the_ttl_keep_the_camera_moving(tmp_path):
    rig = make_rig(tmp_path)
    for _ in range(20):                      # 2 s of a held joystick: a move every 0.1 s
        rig.studio.post("/cameras/cam1/move", json={"pan": -0.3, "ttl_ms": 300})
        rig.advance(0.1)
    assert rig.ptz.is_moving() and rig.ptz.true_pan_tilt()[0] < -50


def test_an_all_zero_move_is_a_stop(tmp_path):
    rig = make_rig(tmp_path)
    rig.studio.post("/cameras/cam1/move", json={"pan": 0.5, "ttl_ms": 2000})
    rig.advance(0.3)
    rig.studio.post("/cameras/cam1/move", json={"pan": 0, "tilt": 0, "zoom": 0})
    rig.advance(0.5)
    assert not rig.ptz.is_moving()
    assert rig.studio.get("/agent/log", params={"limit": 1}).json()[0]["command"] == "move"


def test_a_zoom_only_move_stops_a_previous_pan(tmp_path):
    # A new move replaces the previous velocity: pan was 0.5, the new one only zooms, so the pan must stop.
    rig = make_rig(tmp_path)
    rig.studio.post("/cameras/cam1/move", json={"pan": 0.5, "ttl_ms": 2000})
    rig.advance(0.5)
    rig.studio.post("/cameras/cam1/move", json={"zoom": 0.5, "ttl_ms": 2000})
    rig.advance(0.5)
    pan = rig.ptz.true_pan_tilt()[0]
    rig.advance(0.5)
    assert rig.ptz.true_pan_tilt()[0] == pytest.approx(pan)       # pan stopped
    assert rig.ptz.true_zoom() > 100                               # zoom moving in


# =========================================================================================================== position
def test_position_is_in_degrees_from_home_with_zoom_0_to_1(tmp_path):
    rig = make_rig(tmp_path)
    assert rig.studio.get("/agent/info").json()["cameras"][0]["has_position_feedback"] is True
    rig.ptz.define_preset(50, 400, -100, 8192)        # 400 raw units = 30 degrees at 0.075 deg/unit
    rig.ptz.preset_recall(50)
    rig.advance(3.0)
    assert rig.studio.get("/cameras/cam1/position").json() == {"pan": 30.0, "tilt": -7.5, "zoom": 0.5}


def test_without_branch_a_there_is_no_position(tmp_path):
    rig = make_rig(tmp_path, branch="B")
    assert rig.studio.get("/agent/info").json()["cameras"][0]["has_position_feedback"] is False
    assert rig.studio.get("/cameras/cam1/position").json() is None


# =========================================================================================================== presets
def test_the_studio_cannot_store_over_a_keyframe_preset(tmp_path):
    # Storing over preset 10 (the zone's entry keyframe) would move a keyframe's framing without updating its pose,
    # and break the teach: refused with a reason the operator can act on. Recalling it is fine.
    rig = make_rig(tmp_path)
    refused = rig.studio.post("/cameras/cam1/presets/10/store")
    assert refused.status_code == 400 and refused.json()["error"] == "bad_request"
    assert "Zone entry (keyframe 1)" in refused.json()["detail"] and "0-9, 13-127" in refused.json()["detail"]
    assert all(cmd[1] != "preset_store" for cmd in rig.ptz.commands)      # the camera was never told to store
    assert rig.studio.put("/cameras/cam1/presets/10", json={"label": "Mine"}).status_code == 400
    assert rig.studio.post("/cameras/cam1/presets/10/recall").json()["accepted"] is True


def test_the_studio_stores_names_and_recalls_its_own_presets(tmp_path):
    rig = make_rig(tmp_path)
    assert rig.studio.post("/cameras/cam1/presets/127/recall").json()["error"] == "unknown_preset"
    assert rig.studio.post("/cameras/cam1/presets/127/store").json() == {"n": 127, "label": "Preset 127"}
    assert rig.studio.put("/cameras/cam1/presets/127", json={"label": "Jump"}).json() == {"n": 127, "label": "Jump"}
    assert rig.studio.post("/cameras/cam1/presets/127/store").json() == {"n": 127, "label": "Jump"}   # name kept
    assert rig.studio.post("/cameras/cam1/presets/127/recall").status_code == 200
    assert {"n": 127, "label": "Jump"} in rig.studio.get("/agent/info").json()["cameras"][0]["presets"]
    assert [e for e, _ in rig.events()].count("presets_changed") == 3
    # above preset_max (127) is a bad request, not "unknown"
    assert rig.studio.post("/cameras/cam1/presets/128/store").json()["error"] == "bad_request"


def test_studio_presets_survive_a_restart(tmp_path):
    # They are kept in zone.agent.json next to zone.json: a new zone app (a restart) lists them again.
    make_rig(tmp_path).studio.post("/cameras/cam1/presets/5/store")
    saved = json.loads((tmp_path / "zone.agent.json").read_text(encoding="utf-8"))
    assert saved["presets"] == {"5": "Preset 5"}
    again = make_rig(tmp_path)
    assert {"n": 5, "label": "Preset 5"} in again.studio.get("/agent/info").json()["cameras"][0]["presets"]


def test_a_corrupt_state_file_is_not_fatal(tmp_path):
    (tmp_path / "zone.agent.json").write_text("{not json", encoding="utf-8")
    rig = make_rig(tmp_path)
    assert rig.mode() == "manual"


def test_re_teaching_the_zone_tells_the_studio(tmp_path):
    # A keyframe is deleted on the laptop page: the preset list changes, and the studio gets presets_changed (so it
    # reads /agent/info again).
    rig = make_rig(tmp_path)
    rig.events()
    rig.zapp.teach.delete_point(len(rig.zapp.teach.zone.points) - 1)      # the exit keyframe (preset 12)
    rig.advance(0.1)
    assert [e for e, _ in rig.events()] == ["presets_changed"]
    presets = rig.studio.get("/agent/info").json()["cameras"][0]["presets"]
    assert [p["n"] for p in presets] == [10, 11]
    assert presets[1]["label"] == "Zone exit (keyframe 2)"


# =========================================================================================================== modes
def test_studio_auto_puts_the_director_in_auto_and_manual_in_passive(tmp_path):
    rig = make_rig(tmp_path)
    assert rig.director() == "STOPPED"                  # the zone always starts in STOPPED
    assert rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"}).json()["mode"] == "auto"
    assert rig.director() == "AUTO"
    assert rig.studio.put("/cameras/cam1/mode", json={"mode": "manual"}).json()["mode"] == "manual"
    assert rig.director() == "PASSIVE"                  # watching (frames come), driving nothing
    assert rig.studio.put("/cameras/cam1/mode", json={"mode": "hold"}).json()["mode"] == "hold"
    assert rig.director() == "PASSIVE"


def test_auto_is_refused_without_a_taught_zone(tmp_path):
    rig = make_rig(tmp_path, zone=Zone())
    refused = rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"})
    assert refused.status_code == 409 and refused.json()["error"] == "tracker_not_ready"
    assert "teach at least 2 keyframes" in refused.json()["detail"]
    assert rig.director() == "STOPPED" and rig.mode() == "manual"


def test_auto_is_refused_without_spotter_frames(tmp_path):
    rig = make_rig(tmp_path, frames=False)
    refused = rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"})
    assert refused.status_code == 409 and "no spotter frames yet" in refused.json()["detail"]
    rig.frame()
    rig.advance(5.0, frames=False)                       # the webcam stops delivering: 5 s without a frame
    refused = rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"})
    assert refused.status_code == 409 and "no spotter frame for 5 s" in refused.json()["detail"]
    rig.frame()
    assert rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"}).status_code == 200


def test_auto_is_refused_while_the_ptz_does_not_answer(tmp_path):
    # The PTZ is unplugged. Each studio command that gets no answer is 503 device_offline; after 3 in a row the
    # camera counts as offline (state, health, event), and AUTO is refused. One answer brings it back.
    model = PtzModel()
    rig0 = make_rig(tmp_path)
    dead = DeadPtz(SimulatedPtz(model, rig0.clock))
    rig = make_rig(tmp_path / "b", ptz=dead, clock=rig0.clock)
    for _ in range(3):
        answer = rig.studio.post("/cameras/cam1/move", json={"pan": 0.2})
        assert answer.status_code == 503 and answer.json()["error"] == "device_offline"
    assert rig.studio.get("/agent/health").json()["cameras"] == {"cam1": "offline"}
    assert rig.agent.camera.state()["status"] == "offline"
    assert ("device_offline", {}) in rig.events()
    refused = rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"})
    assert refused.status_code == 409 and "not answering" in refused.json()["detail"]
    assert rig.studio.post("/cameras/cam1/presets/10/recall").status_code == 503
    dead.alive = True
    rig.connect_studio()                                 # (the agent polls the position only for a studio)
    rig.advance(0.6)                                     # the agent's position poll gets an answer again
    assert rig.agent.camera.state()["status"] == "ok" and ("device_online", {}) in rig.events()
    assert rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"}).status_code == 200


def test_a_studio_move_pauses_auto_and_auto_resumes_from_a_clean_state(tmp_path):
    # In AUTO, the studio moves the camera: the zone goes quiet (director PASSIVE, it sends the camera nothing) and
    # the camera is auto_paused. 1 s (resume_after_ms) after the move's ttl runs out, AUTO is back, and the director
    # starts clean: back to the entry preset ("returning"), then waits for the next car.
    rig = make_rig(tmp_path)
    rig.studio.put("/cameras/cam1/mode", json={"mode": "auto", "profile": {"resume_after_ms": 1000}})
    rig.advance(2.0)
    rig.events()
    answer = rig.studio.post("/cameras/cam1/move", json={"pan": 0.3, "ttl_ms": 300}).json()
    assert answer == {"accepted": True, "mode": "auto_paused"}
    assert rig.director() == "PASSIVE"
    sent_before = len(rig.ptz.commands)
    rig.advance(1.2)                                     # 0.3 s of motion + 1.0 s to wait = 1.3 s: not yet
    assert rig.mode() == "auto_paused"
    # nothing came from the zone while paused, only the PTZ's own dead-man stop (not in `commands`)
    assert rig.ptz.commands[sent_before:] == []
    rig.advance(0.2)
    assert rig.mode() == "auto" and rig.director() == "AUTO"
    assert rig.zapp.runtime.director.sub == "returning"
    events = [e for e, _ in rig.events()]
    assert events == ["mode_changed", "manual_override_started", "mode_changed", "manual_override_ended"]


def test_stop_and_preset_recall_pause_auto_too(tmp_path):
    rig = make_rig(tmp_path)
    rig.studio.put("/cameras/cam1/mode", json={"mode": "auto", "profile": {"resume_after_ms": 60_000}})
    assert rig.studio.post("/cameras/cam1/stop").json()["mode"] == "auto_paused"
    rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"})
    assert rig.studio.post("/cameras/cam1/presets/11/recall").json()["mode"] == "auto_paused"
    assert rig.ptz.commands[-1][1:] == ("preset_recall", (11,))


def test_manual_and_hold_never_resume_auto(tmp_path):
    rig = make_rig(tmp_path)
    for mode in ("manual", "hold"):
        rig.studio.put("/cameras/cam1/mode", json={"mode": "auto", "profile": {"resume_after_ms": 100}})
        rig.studio.put("/cameras/cam1/mode", json={"mode": mode})
        assert rig.studio.post("/cameras/cam1/move", json={"pan": 0.1, "ttl_ms": 100}).json()["mode"] == mode
        rig.advance(2.0)
        assert rig.mode() == mode and rig.director() == "PASSIVE"


def test_the_tracker_state_follows_the_director(tmp_path):
    rig = make_rig(tmp_path)
    assert rig.agent.camera.state()["tracker"]["state"] == "idle"          # not in an auto mode
    rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"})
    assert rig.agent.camera.state()["tracker"]["state"] == "searching"     # waiting for a car at the entry
    director = rig.zapp.runtime.director
    director.sub, director.locked_id = "tracking", 7                        # as if car track 7 were locked
    assert rig.agent.camera.state()["tracker"] == {"state": "tracking", "target_id": "track-7", "confidence": None}
    director.sub, director.locked_id, director.last_drop_reason = "returning", None, "lost"
    assert rig.agent.camera.state()["tracker"]["state"] == "lost"          # dropped: going back to the entry
    director.last_drop_reason = "exit"
    assert rig.agent.camera.state()["tracker"]["state"] == "searching"     # the car left the zone normally


def test_a_locked_car_is_tracker_acquired_and_its_release_tracker_lost(tmp_path):
    rig = make_rig(tmp_path)
    rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"})
    rig.events()
    rig.zapp._on_event({"type": "director", "event": "lock", "track_id": 4})
    rig.zapp._on_event({"type": "director", "event": "release", "track_id": 4, "reason": "exit"})
    assert rig.events() == [("tracker_acquired", {"target_id": "track-4"}), ("tracker_lost", {})]


# =========================================================================================================== the laptop page
def test_the_page_and_the_studio_change_modes_in_turn(tmp_path):
    # The person at the laptop and the studio take turns; each change is a mode_changed for the studio, and the
    # page shows what the studio did and who did it.
    rig = make_rig(tmp_path)
    rig.page.post("/api/mode", json={"mode": "AUTO"})
    assert rig.mode() == "auto" and rig.director() == "AUTO"
    rig.studio.put("/cameras/cam1/mode", json={"mode": "manual"})
    assert rig.page.get("/api/state").json()["runtime"]["director"]["mode"] == "PASSIVE"
    rig.page.post("/api/mode", json={"mode": "STOPPED"})
    assert rig.mode() == "manual" and rig.director() == "STOPPED"
    rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"})
    rig.studio.post("/cameras/cam1/move", json={"pan": 0.2, "ttl_ms": 300})
    page_state = rig.page.get("/api/state").json()["agent"]
    assert page_state["mode"] == "auto_paused" and page_state["holder"] == "studio:test-studio"
    assert page_state["resume_in_s"] == pytest.approx(3.3)                 # 0.3 s ttl + the default 3 s
    rig.page.post("/api/mode", json={"mode": "PASSIVE"})                   # the page ends the pause: manual
    assert rig.mode() == "manual"
    reasons = [d["reason"] for e, d in rig.events() if e == "mode_changed"]
    assert reasons == ["laptop page", "requested by the studio", "requested by the studio",
                       "studio motion command", "laptop page"]


def test_in_hold_the_page_auto_button_is_refused(tmp_path):
    rig = make_rig(tmp_path)
    rig.studio.put("/cameras/cam1/mode", json={"mode": "hold"})
    refused = rig.page.post("/api/mode", json={"mode": "AUTO"})
    assert refused.status_code == 409 and "HOLD" in refused.json()["detail"]
    assert rig.mode() == "hold" and rig.director() == "PASSIVE"
    assert rig.page.post("/api/mode", json={"mode": "STOPPED"}).status_code == 200
    assert rig.mode() == "hold" and rig.director() == "STOPPED"


def test_parar_on_the_page_stops_the_studios_move_and_ends_auto(tmp_path):
    rig = make_rig(tmp_path)
    rig.studio.put("/cameras/cam1/mode", json={"mode": "auto", "profile": {"resume_after_ms": 60_000}})
    rig.studio.post("/cameras/cam1/move", json={"pan": 0.5, "ttl_ms": 2000})
    rig.advance(0.3)
    rig.page.post("/api/stop")
    rig.advance(0.5)
    assert not rig.ptz.is_moving()
    assert rig.mode() == "manual" and rig.director() == "STOPPED"
    assert rig.agent.camera.state()["last_command_source"] == "local:zone-page"
    assert rig.studio.get("/agent/log", params={"limit": 1}).json()[0]["source"] == "local:zone-page"


def test_teach_mode_ends_when_the_studio_starts_auto(tmp_path):
    rig = make_rig(tmp_path)
    rig.page.post("/api/teach/mode", json={"on": True})
    assert rig.zapp.teach_mode
    rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"})
    assert not rig.zapp.teach_mode


# =========================================================================================================== restart
def test_after_a_restart_in_auto_the_camera_is_manual_and_every_studio_is_told(tmp_path):
    # The zone app crashes in AUTO and the launcher restarts it. The zone always comes back in STOPPED, so for the
    # studio the camera is manual, and each studio that connects gets the contract's notice (from auto to manual,
    # reason tracker_not_ready), until a studio sets a mode.
    rig = make_rig(tmp_path)
    rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"})
    restarted = make_rig(tmp_path)
    assert restarted.mode() == "manual" and restarted.director() == "STOPPED"
    for _ in range(2):                                   # two studios connect, one after the other
        with restarted.studio.websocket_connect("/agent/ws") as ws:
            state = ws.receive_json()
            notice = ws.receive_json()
        assert state["type"] == "state" and state["cameras"]["cam1"]["mode"] == "manual"
        assert notice["event"] == "mode_changed" and notice["camera_id"] == "cam1"
        assert notice["detail"] == {"from": "auto", "to": "manual", "reason": "tracker_not_ready"}
    restarted.studio.put("/cameras/cam1/mode", json={"mode": "manual"})
    with restarted.studio.websocket_connect("/agent/ws") as ws:
        assert ws.receive_json()["type"] == "state"
        assert restarted.agent.greeting()[1:] == []      # no notice any more


def test_hold_survives_a_restart(tmp_path):
    make_rig(tmp_path).studio.put("/cameras/cam1/mode", json={"mode": "hold"})
    assert make_rig(tmp_path).mode() == "hold"


# =========================================================================================================== websocket
def test_the_websocket_sends_state_at_once_answers_pings_and_sends_events(tmp_path):
    rig = make_rig(tmp_path)
    with rig.studio as client:                           # "with": the agent's push loop runs
        with client.websocket_connect("/agent/ws") as ws:
            first = ws.receive_json()
            assert first["type"] == "state" and set(first["cameras"]) == {"cam1"}
            camera = first["cameras"]["cam1"]
            assert set(camera) == {"status", "mode", "tracker", "position", "moving", "last_command_source",
                                   "last_command_ts"}
            ws.send_json({"type": "ping"})
            assert _receive_until(ws, lambda m: m == {"type": "pong"})
            client.put("/cameras/cam1/mode", json={"mode": "hold"})
            event = _receive_until(ws, lambda m: m.get("event") == "mode_changed")
            assert event["detail"] == {"from": "manual", "to": "hold", "reason": "requested by the studio"}
            state = _receive_until(ws, lambda m: m.get("type") == "state")
            assert state["cameras"]["cam1"]["mode"] == "hold"


def _receive_until(ws, matches, tries=50):
    for _ in range(tries):
        message = ws.receive_json()
        if matches(message):
            return message
    raise AssertionError("no matching message")


# =========================================================================================================== review fixes
def test_a_studio_stop_reaches_the_camera_first_and_includes_the_zoom(tmp_path):
    # /stop goes to the PTZ driver before anything else (the mode rules may have to wait for the zone's control
    # loop), for pan/tilt AND zoom, and once more after the mode change.
    rig = make_rig(tmp_path)
    before = len(rig.ptz.commands)
    rig.studio.post("/cameras/cam1/stop")
    sent = [name for _, name, _ in rig.ptz.commands[before:]]
    assert sent == ["stop", "zoom_stop", "stop", "zoom_stop"]


def test_a_zoom_move_stops_by_itself_after_its_ttl(tmp_path):
    # The dead-man switch for zoom too: one zoom move, then nothing; the zoom stops about ttl_ms later.
    rig = make_rig(tmp_path)
    rig.studio.post("/cameras/cam1/move", json={"zoom": 0.5, "ttl_ms": 300})
    rig.advance(0.25)
    assert rig.ptz.is_moving()
    rig.advance(1.0)
    zoom = rig.ptz.true_zoom()
    assert zoom > 100                                   # it did zoom in
    rig.advance(1.0)
    assert rig.ptz.true_zoom() == pytest.approx(zoom)   # and stopped by itself


def test_a_preset_the_camera_refuses_to_store_is_a_bad_request(tmp_path):
    # The camera answered, with an error: it is there (not "offline"), it just won't store that one.
    rig = make_rig(tmp_path)
    refusing = make_rig(tmp_path / "b", ptz=RefusingPtz(rig.ptz), clock=rig.clock)
    answer = refusing.studio.post("/cameras/cam1/presets/5/store")
    assert answer.status_code == 400 and answer.json()["error"] == "bad_request"
    assert refusing.agent.camera.online()


def test_a_stop_while_the_control_loop_is_busy_is_not_lost(tmp_path):
    # In AUTO, the zone's control loop holds its lock (a PTZ call that hangs). The studio presses stop:
    # * the stop reaches the camera at once anyway (sent before any lock);
    # * the studio is told the zone is late (503), instead of a quiet "ok" while the zone keeps tracking;
    # * the camera is auto_paused for the studio, and the zone's director goes quiet at its very next cycle.
    rig = make_rig(tmp_path)
    rig.zapp.runtime.handover_lock_timeout_s = 0.05    # don't wait a whole second in a test
    rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"})
    _hold_lock_in_another_thread(rig, lambda: _check_busy_stop(rig))
    assert rig.director() == "AUTO"                     # the loop hasn't run yet...
    rig.cycle()                                         # ...its next cycle obeys the queued hand-over
    assert rig.director() == "PASSIVE" and rig.mode() == "auto_paused"


def _check_busy_stop(rig):
    before = len(rig.ptz.commands)
    answer = rig.studio.post("/cameras/cam1/stop")
    assert answer.status_code == 503 and answer.json()["error"] == "device_offline"
    assert "busy" in answer.json()["detail"]
    assert [name for _, name, _ in rig.ptz.commands[before:]][:2] == ["stop", "zoom_stop"]


def test_manual_or_hold_while_the_control_loop_is_busy_is_not_lost(tmp_path):
    for mode in ("manual", "hold"):
        rig = make_rig(tmp_path / mode)
        rig.zapp.runtime.handover_lock_timeout_s = 0.05
        rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"})
        _hold_lock_in_another_thread(
            rig, lambda: _expect_status(rig.studio.put("/cameras/cam1/mode", json={"mode": mode}), 503))
        assert rig.mode() == mode
        before = len(rig.ptz.commands)
        rig.cycle()
        assert rig.director() == "PASSIVE"
        assert "stop" in [name for _, name, _ in rig.ptz.commands[before:]]     # the zone's own motion ended


def test_a_director_left_in_auto_is_silenced_by_the_tick(tmp_path):
    # Belt and braces: whatever the cause, if the studio has the camera (manual) but the director is in AUTO, the
    # agent's next tick puts the director back in PASSIVE.
    rig = make_rig(tmp_path)
    rig.zapp.runtime.director.set_mode("AUTO", rig.clock.monotonic(), "test: a lost hand-over")
    rig.advance(0.1)
    assert rig.director() == "PASSIVE" and rig.mode() == "manual"


def test_hand_over_drops_the_directors_queued_stop(tmp_path):
    # Leaving AUTO normally queues a stop for the next cycle. When the studio takes over (a pause), that stop would
    # cut the studio's own move ~0.1 s later, so the hand-over drops it: the next cycle sends nothing.
    rig = make_rig(tmp_path)
    rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"})
    rig.cycle()                                         # the entry preset recall that starts AUTO
    ok, _ = rig.zapp.runtime.hand_over("PASSIVE", "test", stop_camera=False)
    assert ok
    before = len(rig.ptz.commands)
    rig.cycle()
    assert rig.ptz.commands[before:] == []


def test_the_page_releases_a_hold_only_when_confirmed(tmp_path):
    rig = make_rig(tmp_path)
    rig.studio.put("/cameras/cam1/mode", json={"mode": "hold"})
    assert rig.page.post("/api/agent/release_hold", json={}).status_code == 400        # no confirmation
    assert rig.mode() == "hold"
    assert rig.page.post("/api/agent/release_hold", json={"confirm": True}).json() == {"ok": True, "mode": "manual"}
    assert rig.mode() == "manual"
    log = rig.studio.get("/agent/log", params={"limit": 1}).json()[0]
    assert log["command"] == "release_hold" and log["source"] == "local:zone-page"
    assert log["args"] == {"holder": "studio:test-studio"}
    assert rig.page.post("/api/agent/release_hold", json={"confirm": True}).status_code == 409   # not in hold now


def test_a_hold_remembered_across_a_restart_can_be_released_on_the_page(tmp_path):
    make_rig(tmp_path).studio.put("/cameras/cam1/mode", json={"mode": "hold"})
    restarted = make_rig(tmp_path)
    assert restarted.page.post("/api/mode", json={"mode": "AUTO"}).status_code == 409
    restarted.page.post("/api/agent/release_hold", json={"confirm": True})
    assert restarted.page.post("/api/mode", json={"mode": "AUTO"}).status_code == 200
    assert make_rig(tmp_path).mode() == "manual"      # and the release itself is remembered


def test_the_position_is_polled_only_for_a_studio_and_never_in_auto(tmp_path):
    # A zone running without a studio must not send the camera anything new; in AUTO the controller owns the wire.
    rig0 = make_rig(tmp_path)
    counting = CountingPtz(rig0.ptz)
    rig = make_rig(tmp_path / "b", ptz=counting, clock=rig0.clock)
    rig.advance(2.0)
    assert counting.inquiries == 0                     # nobody connected: no inquiry at all
    rig.connect_studio()
    rig.advance(2.0)
    assert 6 <= counting.inquiries <= 10               # 2 Hz, pan/tilt + zoom each time
    rig.studio.put("/cameras/cam1/mode", json={"mode": "auto"})
    counted = counting.inquiries
    rig.advance(2.0)
    assert counting.inquiries == counted               # AUTO: none from the agent


def test_the_page_shows_when_the_studio_api_is_down(tmp_path):
    from zone.app import agent_down
    rig = make_rig(tmp_path)
    agent_down(rig.zapp, "porta 8890: SystemExit (outro programa usa a porta?)")
    assert rig.page.get("/api/state").json()["agent"]["error"].startswith("porta 8890")


def _expect_status(answer, status):
    assert answer.status_code == status, answer.text


def _hold_lock_in_another_thread(rig, action):
    """Make the zone's control loop "busy": another thread holds its lock while `action()` runs."""
    import threading
    taken, done = threading.Event(), threading.Event()

    def busy_loop():
        with rig.zapp.runtime.lock:
            taken.set()
            done.wait(5)
    thread = threading.Thread(target=busy_loop)
    thread.start()
    taken.wait(5)
    try:
        action()
    finally:
        done.set()
        thread.join(5)
