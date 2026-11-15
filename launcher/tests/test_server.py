"""Tests for the launcher's web API (launcher/server.py), through FastAPI's TestClient: it calls the app directly,
in this process, as a browser would over HTTP (no port, no network).

The roles, the camera and the roster site are fakes, so the tests check the page's rules: validation answers,
the PIN, the camera-in-use refusal, the export download, and that /api/shutdown only listens to this PC.
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from launcher import settings as S  # noqa: E402
from launcher.procs import ManagedProcess  # noqa: E402
from launcher.roles import RoleManager  # noqa: E402
from launcher.server import Launcher, create_app  # noqa: E402


class FakeCharacterization:
    def __init__(self):
        self.started = []

    def start(self, ip, folder, quick=False, zone_running=False):
        self.started.append((ip, folder, quick, zone_running))
        return (not zone_running), "ok" if not zone_running else "Pare a Zona PTZ primeiro"

    def status(self, lines=60):
        return dict(running=False, log=[], out_dir=None, exit_code=None)


@pytest.fixture
def setup(paths, fake_system, fake_clock):
    def make_process(name, make_cmd):
        return ManagedProcess(name, make_cmd, paths.logs_dir, spawn=fake_system.spawn, kill=fake_system.kill,
                              clock=fake_clock, create_time=lambda pid: None)
    roles = RoleManager(paths, make_process=make_process, zone_stop=lambda port: None,
                        ptz_stop=lambda host, port: None, log=lambda m: None)
    snapshots = []

    def fake_snapshot(device, ffmpeg, width, height):
        snapshots.append((device, width, height))
        return b"\xff\xd8fakejpeg", (width, height)
    launcher = Launcher(paths, 8780, roles=roles, characterization=FakeCharacterization(), snapshot=fake_snapshot,
                        cameras=lambda ffmpeg: ([dict(index=0, name="Cam", device="usb#vid_1")], None),
                        roster_fetch=lambda event, p: dict(ok=True, message="fake", cars=3, path=None),
                        open_folder=lambda folder: None, log=lambda m: None)
    client = TestClient(create_app(launcher))
    return launcher, client, snapshots


def test_state_describes_both_roles(setup):
    _, client, _ = setup
    state = client.get("/api/state").json()
    assert state["processes"]["lapline"]["running"] is False
    assert state["lapline"]["config_exists"] is False
    assert "bench" in state["choices"]["bench_profiles"]


def test_the_page_itself_is_served(setup):
    _, client, _ = setup
    assert "Baja MV" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/../server.py").status_code == 404


def test_saving_bad_settings_answers_400_with_messages_per_field(setup):
    _, client, _ = setup
    response = client.post("/api/settings/lapline", json=dict(event="x", camera_device="0"))
    assert response.status_code == 400
    assert "event" in response.json()["errors"]


def test_saving_good_settings_writes_the_config(setup, paths):
    _, client, _ = setup
    response = client.post("/api/settings/lapline", json=dict(event="27SE", camera_device="0"))
    assert response.status_code == 200, response.text
    assert S.read_lapline(paths).event == "27SE"


def test_both_roles_on_one_laptop_cannot_share_a_port(setup, paths):
    _, client, _ = setup
    client.post("/api/settings/launcher", json=dict(roles=["lapline", "zone"]))
    client.post("/api/settings/zone", json=dict(camera_device="1", ptz_host="10.0.0.5", web_port=8090))
    response = client.post("/api/settings/lapline", json=dict(event="27SE", camera_device="0", status_port=8090))
    assert response.status_code == 400
    assert "status_port" in response.json()["errors"]


def test_start_and_stop_a_role(setup, fake_system):
    _, client, _ = setup
    client.post("/api/settings/lapline", json=dict(event="27SE", camera_device="0"))
    assert client.post("/api/roles/lapline/start").json()["ok"] is True
    assert client.get("/api/state").json()["processes"]["lapline"]["running"] is True
    assert client.post("/api/roles/lapline/stop").json()["ok"] is True
    assert fake_system.killed


def test_unknown_actions_are_refused(setup):
    _, client, _ = setup
    assert client.post("/api/roles/lapline/explode").status_code == 409


def test_pin_protects_every_change_once_set(setup):
    _, client, _ = setup
    assert client.post("/api/settings/launcher", json=dict(roles=["lapline"], pin="4321")).status_code == 200
    # without the PIN: refused
    assert client.post("/api/settings/lapline", json=dict(event="27SE", camera_device="0")).status_code == 401
    # with it: accepted
    ok = client.post("/api/settings/lapline", json=dict(event="27SE", camera_device="0"),
                     headers={"X-Launcher-Pin": "4321"})
    assert ok.status_code == 200
    # reading stays open (a phone can always look)
    assert client.get("/api/state").status_code == 200


def test_saving_without_a_pin_field_keeps_the_pin(setup, paths):
    # The page leaves "pin" out when its PIN box is empty: that means "keep it", not "remove it".
    _, client, _ = setup
    client.post("/api/settings/launcher", json=dict(roles=["lapline"], pin="4321"))
    client.post("/api/settings/launcher", json=dict(roles=["zone"]), headers={"X-Launcher-Pin": "4321"})
    assert S.read_launcher(paths).pin == "4321"
    assert S.read_launcher(paths).roles == ["zone"]


def test_snapshot_returns_a_jpeg(setup):
    _, client, snapshots = setup
    response = client.get("/api/cameras/snapshot", params=dict(device="usb#vid_1", resolution="1280x720"))
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert snapshots == [("usb#vid_1", 1280, 720)]


def test_snapshot_is_refused_for_the_camera_of_a_running_role(setup):
    # One program per camera: while the lap line uses camera 0, no preview of camera 0.
    _, client, snapshots = setup
    client.post("/api/settings/lapline", json=dict(event="27SE", camera_device="0"))
    client.post("/api/roles/lapline/start")
    response = client.get("/api/cameras/snapshot", params=dict(device="0"))
    assert response.status_code == 409
    assert snapshots == []


def test_camera_list(setup):
    _, client, _ = setup
    assert client.get("/api/cameras").json()["cameras"][0]["device"] == "usb#vid_1"


def test_roster_button(setup):
    _, client, _ = setup
    result = client.post("/api/roster/fetch", json=dict(event="26SE")).json()
    assert result["ok"] is True
    assert result["roster"]["where"] == "embutida"          # 26SE is bundled with the code


def test_characterization_is_refused_while_the_zone_runs(setup):
    launcher, client, _ = setup
    client.post("/api/settings/zone", json=dict(camera_device="1", ptz_host="10.0.0.5"))
    client.post("/api/roles/zone/start")
    response = client.post("/api/characterize", json=dict(ip="10.0.0.5"))
    assert response.status_code == 409
    assert launcher.characterization.started[-1][3] is True   # the runner was told the zone runs


def test_export_then_download(setup, paths):
    _, client, _ = setup
    (paths.logs_dir / "lapline.log").write_text("hello", encoding="utf-8")
    result = client.post("/api/export").json()
    assert result["ok"] is True
    download = client.get("/api/export/" + result["name"])
    assert download.status_code == 200
    assert download.content[:2] == b"PK"                       # a zip file
    assert client.get("/api/export/..%2Fconfig%2Flauncher.json").status_code == 404


def test_shutdown_only_from_this_pc(setup):
    launcher, client, _ = setup
    # TestClient's requests come from the host "testclient", which is not this PC's loopback address
    assert client.post("/api/shutdown").status_code == 403
    assert not launcher.exit_requested.is_set()


# ---------------------------------------------------------------------------- the studio token is a secret
def test_state_never_shows_the_studio_token_only_whether_one_is_set(setup, paths):
    # Reading /api/state needs no PIN (anyone on the LAN can open the page), so the zone's X-Agent-Token must not
    # be in it: only "token_set" (true / false), like the launcher's own PIN.
    _, client, _ = setup
    S.write_zone(paths, S.validate_zone(dict(camera_device="1", ptz_host="192.168.0.100",
                                             agent_token="segredo-do-estudio"))[0])
    response = client.get("/api/state")
    assert "segredo-do-estudio" not in response.text
    state = response.json()
    assert state["zone"]["token_set"] is True
    assert "agent_token" not in state["zone"]["settings"]


def test_saving_the_zone_with_an_empty_token_field_keeps_the_token(setup, paths):
    # The page never receives the token, so its token field is empty: saving must keep the saved token. A single
    # space (stripped to "") is how the operator removes it on purpose.
    _, client, _ = setup
    S.write_zone(paths, S.validate_zone(dict(camera_device="1", ptz_host="192.168.0.100",
                                             agent_token="segredo"))[0])
    form = dict(camera_device="1", ptz_host="192.168.0.100", agent_token="")
    assert client.post("/api/settings/zone", json=form).json()["ok"] is True
    assert S.read_zone(paths).agent_token == "segredo"
    form["agent_token"] = " "
    assert client.post("/api/settings/zone", json=form).json()["ok"] is True
    assert S.read_zone(paths).agent_token == ""


def test_the_page_gets_the_public_network_warning(paths):
    launcher = Launcher(paths, 8780, network_warning=lambda: "Rede marcada como Pública", log=lambda m: None)
    assert launcher.state()["network_warning"] == "Rede marcada como Pública"


# ---------------------------------------------------------------------------- camera in use: index vs device path
def test_a_camera_is_in_use_whether_it_is_named_by_index_or_by_device_path(setup, paths):
    # The lap line's config names its camera "0" (an index); the page asks for a preview of the same camera by its
    # device path "usb#vid_1" (or the other way round). Both are camera 0 of the list, so the preview is refused.
    launcher, client, snapshots = setup
    S.write_lapline(paths, S.validate_lapline(dict(event="27SE", camera_device="0"))[0])
    assert launcher.roles.start("lapline")[0]
    assert launcher.camera_in_use("USB#VID_1") == "lapline"            # device path, any letter case
    assert client.get("/api/cameras/snapshot", params=dict(device="usb#vid_1")).status_code == 409
    assert snapshots == []
    # another camera is free
    assert launcher.camera_in_use("usb#vid_2") is None
