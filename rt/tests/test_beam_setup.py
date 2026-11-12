"""Tests of the on-site Beam setup (rt/beam/control.py, rt/beam/monitor.py, the /beam endpoints of rt/status.py).

The story these tests follow, in plain words:
  * The owner gets the light-beam sensor and its USB adapter only AT the event. rt is already running on the
    lap-line laptop. He opens the status page on his phone and goes to "Beam setup".
  * He picks the COM port and presses "Start monitor": rt pauses the beam driver (only one program part can hold a
    COM port) and shows, live, the adapter's four modem input lines (CTS, DSR, DCD, RI) and any bytes it sends.
    That is the BeamMonitor: a little "probe" built into rt.
  * He waves a hand through the beam a few times and presses "Suggest": rt guesses the settings from what it saw
    (which line flips, or which bytes arrive) and fills the form.
  * He presses "Apply": rt checks every value (on the laptop, not only in the phone's browser), saves the settings
    to a small file next to the config (`<config>.beam.toml`, loaded again at every start) and restarts ONLY the
    beam part. Detection, camera capture and recording are never touched.
  * Anything wrong (a typo, a value out of range, an adapter that is not there) must never crash rt: the request
    is refused with a message, or the beam stays off with the error shown.

No real serial port, no real waiting: fake serial ports and fake clocks (rt/tests/beam_fakes.py), and fake beam
services that only record what was done to them. The HTTP tests start the real status web server on a free port of
this machine and talk to it like the phone would.
"""
import json
import urllib.error
import urllib.request

import pytest

from rt import config as C
from rt.beam.control import (BeamController, ControllerSink, list_serial_ports, load_override, override_path,
                             validate_settings, write_override)
from rt.beam.monitor import BeamMonitor
from rt.status import StatusBoard, StatusServer

from .beam_fakes import SerialPortWorld, TickingClock

GOOD = dict(driver="modem-line", port="COM5", line="dsr", active="low", baud=9600, byte_pattern="",
            byte_release_pattern="", group_gap_s=0.3, stuck_s=6.0, set_dtr=True, set_rts=False)


# ------------------------------------------------------------------------------------------------ fakes
class FakeService:
    """Pretends to be a BeamService: records start / stop and what it was built with. Nothing real runs."""

    def __init__(self, cfg, emit, log):
        self.cfg = cfg
        self.log = log
        self.source = type("Src", (), dict(driver=cfg["beam"]["driver"], port=cfg["beam"]["port"]))()
        self.events = []

    def start(self):
        self.log.append(("start", self.cfg["beam"]["driver"], self.cfg["beam"]["port"]))

    def stop(self):
        self.log.append(("stop", self.cfg["beam"]["driver"]))

    def status(self):
        return dict(enabled=True, driver=self.cfg["beam"]["driver"], port=self.cfg["beam"]["port"], state="open")

    def on_frame(self, fr):
        pass

    def on_event(self, ev):
        self.events.append(ev)


class FakeMonitor:
    """Pretends to be a BeamMonitor: records start / stop / clear, returns a fixed suggestion."""

    def __init__(self, port, baud, set_dtr, set_rts, log):
        self.port, self.baud, self.log = port, baud, log

    def start(self):
        self.log.append(("monitor start", self.port, self.baud))
        return self

    def stop(self):
        self.log.append(("monitor stop", self.port))

    def clear(self):
        self.log.append(("monitor clear",))

    def snapshot(self):
        return dict(port=self.port, state="open", lines=dict(CTS=False, DSR=True, DCD=False, RI=False))

    def suggest(self):
        return dict(driver="modem-line", settings=dict(driver="modem-line", line="dsr", active="low", port=self.port),
                    config=['driver = "modem-line"'], notes=[])


def make_controller(cfg, tmp_path, clock=None, failing_service=False, ports=(), use_watchdog=False):
    """A BeamController wired to fakes; its override file goes to the test's temporary folder (never next to the
    real config). Returns (controller, the list of everything done, the list of emitted events)."""
    log, events = [], []
    cfg["beam_override"] = dict(path=str(tmp_path / "lapline.beam.toml"), config=str(tmp_path / "lapline.toml"))

    def make_service(service_cfg, emit):
        if failing_service:
            raise ValueError("beam.line must be one of cts, dsr, cd (dcd), ri; got 'txd'")
        return FakeService(service_cfg, emit, log)

    controller = BeamController(cfg, events.append, make_service=make_service,
                                make_monitor=lambda p, b, d, r: FakeMonitor(p, b, d, r, log),
                                comports=lambda: list(ports), clock=clock or TickingClock(),
                                use_watchdog=use_watchdog)
    return controller, log, events


# ------------------------------------------------------------------------------------------------ validation
def test_good_settings_pass_validation():
    settings, errors = validate_settings(GOOD)
    assert errors == []
    assert settings["line"] == "dsr" and settings["active"] == "low" and settings["set_rts"] is False


@pytest.mark.parametrize("change, message", [
    (dict(driver="laser"), "driver must be one of"),
    (dict(port="COM5; del *"), "port must look like COM5"),     # nothing odd can reach the serial library
    (dict(port=""), "choose a port"),
    (dict(line="txd"), "line must be CTS, DSR, DCD or RI"),
    (dict(active="sideways"), "active must be high or low"),
    (dict(baud=12345), "baud must be one of"),
    (dict(baud="fast"), "baud must be one of"),
    (dict(byte_pattern="zz"), "must be hex bytes"),
    (dict(byte_pattern="00" * 40), "at most 32 bytes"),
    (dict(byte_release_pattern="30"), "needs a byte pattern"),
    (dict(byte_pattern="31", byte_release_pattern="31"), "must differ"),
    (dict(group_gap_s="nan"), "between"),                       # NaN is not "between" anything
    (dict(group_gap_s=5), "between"),
    (dict(stuck_s=0.2, group_gap_s=0.25), "between"),
    (dict(stuck_s=0.6, group_gap_s=1.0), "longer than the group gap"),
    (dict(set_dtr="maybe"), "set_dtr must be true or false"),
    (dict(evil_key=1), "unknown setting"),                      # only known keys are ever stored
])
def test_bad_settings_are_refused_with_a_clear_message(change, message):
    _, errors = validate_settings(dict(GOOD, **change))
    assert any(message in e for e in errors), errors


def test_dcd_is_accepted_and_stored_as_cd_and_missing_keys_keep_current_values():
    settings, errors = validate_settings(dict(line="DCD"), current=GOOD)
    assert errors == [] and settings["line"] == "cd" and settings["port"] == "COM5"


def test_driver_off_needs_no_port():
    assert validate_settings(dict(driver="off"))[1] == []


# ------------------------------------------------------------------------------------------------ override file
def test_override_file_round_trip(tmp_path):
    path = override_path(tmp_path / "lapline.toml")
    assert path.name == "lapline.beam.toml"                    # next to the config it overrides
    settings, _ = validate_settings(GOOD)
    write_override(path, settings, tmp_path / "lapline.toml")
    loaded, error = load_override(path)
    assert error is None and loaded == settings
    assert not (tmp_path / "lapline.beam.toml.tmp").exists()   # written through a temporary file, then renamed


def test_config_load_order_config_then_override_then_set(tmp_path):
    # rt/config.toml -> the user config -> <config>.beam.toml -> --set options (the command line wins).
    config = tmp_path / "lapline.toml"
    config.write_text('[beam]\ndriver = "off"\nport = "COM3"\ngroup_gap_s = 0.4\n', encoding="utf-8")
    settings, _ = validate_settings(GOOD)
    write_override(override_path(config), settings, config)
    cfg = C.load(config, ["beam.stuck_s=9.0"])
    assert cfg["beam"]["driver"] == "modem-line" and cfg["beam"]["port"] == "COM5"   # from the override
    assert cfg["beam"]["group_gap_s"] == 0.3                                        # override beats the config
    assert cfg["beam"]["stuck_s"] == 9.0                                            # --set beats the override
    assert cfg["beam_override"]["loaded"] is True and cfg["beam_override"]["error"] is None


def test_no_override_file_changes_nothing(tmp_path):
    config = tmp_path / "lapline.toml"
    config.write_text('[beam]\nport = "COM3"\n', encoding="utf-8")
    cfg = C.load(config)
    assert cfg["beam"]["port"] == "COM3" and cfg["beam_override"]["loaded"] is False


@pytest.mark.parametrize("content", ["this is [not toml", '[beam]\ndriver = "modem-line"\nport = "COM5"\nbaud = 7\n',
                                     'driver = "modem-line"\n'])
def test_broken_or_invalid_override_turns_the_beam_off_with_an_error_and_rt_still_starts(tmp_path, content):
    config = tmp_path / "lapline.toml"
    config.write_text('[beam]\ndriver = "modem-line"\nport = "COM3"\n', encoding="utf-8")
    override_path(config).write_text(content, encoding="utf-8")
    cfg = C.load(config)                                       # must not raise
    assert cfg["beam"]["driver"] == "off"
    assert "beam off until it is fixed or deleted" in cfg["beam_override"]["error"]


# ------------------------------------------------------------------------------------------------ controller
def test_start_from_config_runs_the_service_only_when_the_driver_is_on(cfg, tmp_path):
    controller, log, _ = make_controller(cfg, tmp_path)
    controller.start_from_config()                              # default config: driver = "off"
    assert controller.service is None and log == []
    cfg["beam"].update(driver="modem-line", port="COM5")
    controller, log, _ = make_controller(cfg, tmp_path)
    controller.start_from_config()
    assert log == [("start", "modem-line", "COM5")]


def test_apply_saves_and_restarts_only_the_beam(cfg, tmp_path):
    cfg["beam"].update(driver="modem-line", port="COM3")
    controller, log, events = make_controller(cfg, tmp_path)
    controller.start_from_config()
    ok, errors = controller.apply(GOOD)
    assert ok and errors == []
    # The old service was stopped, a new one started with the new settings.
    assert log == [("start", "modem-line", "COM3"), ("stop", "modem-line"), ("start", "modem-line", "COM5")]
    assert controller.service.cfg["beam"]["line"] == "dsr"
    # ... and they are saved for the next start of rt.
    saved, _ = load_override(tmp_path / "lapline.beam.toml")
    assert saved["port"] == "COM5" and saved["active"] == "low"
    assert any("applied from the status page" in e.get("msg", "") for e in events)


def test_bad_apply_changes_nothing(cfg, tmp_path):
    cfg["beam"].update(driver="modem-line", port="COM3")
    controller, log, _ = make_controller(cfg, tmp_path)
    controller.start_from_config()
    ok, errors = controller.apply(dict(GOOD, baud=12345))
    assert not ok and "baud" in errors[0]
    assert log == [("start", "modem-line", "COM3")]             # still the old beam, not restarted
    assert not (tmp_path / "lapline.beam.toml").exists()        # nothing saved


def test_a_service_that_fails_to_start_leaves_the_beam_off_with_the_error_shown(cfg, tmp_path):
    controller, log, events = make_controller(cfg, tmp_path, failing_service=True)
    ok, _ = controller.apply(GOOD)                              # valid settings, but the start fails anyway
    assert ok                                                   # saved; rt goes on
    assert controller.service is None
    status = controller.status()
    assert status["enabled"] is False and "beam input NOT started" in status["error"]


def test_turn_off_stops_the_beam_and_saves_driver_off(cfg, tmp_path):
    cfg["beam"].update(driver="modem-line", port="COM3")
    controller, log, _ = make_controller(cfg, tmp_path)
    controller.start_from_config()
    ok, _ = controller.turn_off()
    assert ok and controller.service is None and log[-1] == ("stop", "modem-line")
    assert load_override(tmp_path / "lapline.beam.toml")[0]["driver"] == "off"


def test_setup_mode_pauses_the_beam_and_leaving_restarts_it(cfg, tmp_path):
    cfg["beam"].update(driver="modem-line", port="COM3")
    controller, log, _ = make_controller(cfg, tmp_path)
    controller.start_from_config()
    ok, _ = controller.start_setup(dict(port="COM5", baud=19200))
    assert ok
    assert log[-2:] == [("stop", "modem-line"), ("monitor start", "COM5", 19200)]   # driver paused, port monitored
    assert controller.status()["setup_mode"] is True
    assert controller.suggest()["settings"]["line"] == "dsr"
    controller.leave_setup()
    assert log[-2:] == [("monitor stop", "COM5"), ("start", "modem-line", "COM3")]  # back as it was saved


def test_setup_mode_ends_by_itself_when_the_page_is_gone(cfg, tmp_path):
    # The phone is locked mid-setup: no page request for setup_idle_s (300 s). The beam must not stay paused for
    # the race: setup mode ends by itself and the beam restarts with the saved settings.
    cfg["beam"].update(driver="modem-line", port="COM3")
    clock = TickingClock()
    controller, log, _ = make_controller(cfg, tmp_path, clock=clock)
    controller.start_from_config()
    controller.start_setup(dict(port="COM3"))
    clock.advance(200)
    controller.setup_state()                                    # the page asked for data: still open
    clock.advance(299)
    assert controller.check_setup_idle() is False
    clock.advance(2)
    assert controller.check_setup_idle() is True
    assert controller.monitor is None and log[-1] == ("start", "modem-line", "COM3")


def test_start_setup_refuses_a_bad_port(cfg, tmp_path):
    controller, log, _ = make_controller(cfg, tmp_path)
    ok, errors = controller.start_setup(dict(port="../../etc"))
    assert not ok and "port must look like" in errors[0] and log == []


def test_events_reach_whichever_service_is_running(cfg, tmp_path):
    # rt's event emitter knows only the ControllerSink; after Apply the NEW service gets the pass events.
    cfg["beam"].update(driver="modem-line", port="COM3")
    controller, _, _ = make_controller(cfg, tmp_path)
    controller.start_from_config()
    sink = ControllerSink(controller)
    controller.apply(GOOD)
    sink.event(dict(type="pass_end", pass_id=1))
    assert controller.service.events == [dict(type="pass_end", pass_id=1)]


def test_ports_are_listed_with_the_cp2102_first_and_flagged():
    class Port:
        def __init__(self, device, vid, pid, description):
            self.device, self.vid, self.pid, self.description = device, vid, pid, description

    ports = list_serial_ports(lambda: [Port("COM1", None, None, "Communications Port"),
                                       Port("COM7", 0x10C4, 0xEA60, "Silicon Labs CP210x USB to UART Bridge")])
    assert ports[0]["device"] == "COM7" and ports[0]["cp210x"] and ports[0]["usb_id"] == "10C4:EA60"
    assert ports[1]["cp210x"] is False and ports[1]["usb_id"] == ""
    assert list_serial_ports(lambda: 1 / 0) == []              # a failing listing never breaks the page


# ------------------------------------------------------------------------------------------------ monitor
def make_monitor(world, clock):
    return BeamMonitor("COM5", 9600, clock=clock, sleep=clock.sleep, serial_factory=world.factory,
                       modem_status_check=lambda ser: None)


def test_monitor_shows_the_lines_and_the_messages_live():
    clock = TickingClock()
    world = SerialPortWorld(clock)
    mon = make_monitor(world, clock)
    mon.step()                                                  # opens the port
    assert mon.snapshot()["state"] == "open"
    world.serial.lines["dsr"] = True
    world.serial.incoming += b"B1\r\n"
    for _ in range(10):                                         # 10 rounds of 2 ms: the message is complete
        mon.step()
    snap = mon.snapshot()
    assert snap["lines"] == dict(CTS=False, DSR=True, DCD=False, RI=False)
    assert snap["changes"][0]["line"] == "DSR" and snap["changes"][0]["asserted"] is True
    assert snap["messages"][0]["hex"] == "42 31 0d 0a" and snap["messages"][0]["text"] == "B1\\r\\n"


def test_monitor_survives_unplug_and_reopens():
    clock = TickingClock()
    world = SerialPortWorld(clock)
    mon = make_monitor(world, clock)
    mon.step()
    world.unplug()                                              # like Windows: the lines just read False ...
    mon.step()                                                  # ... but in_waiting fails: noticed at once
    assert mon.snapshot()["state"] == "lost"
    mon.step()                                                  # waits (fake time) before trying again
    world.plug_in()
    for _ in range(20):
        mon.step()
    assert mon.snapshot()["state"] == "open"


def test_monitor_suggest_after_waving_through_the_beam():
    # DSR is asserted while the beam is clear and drops for each of 4 waves: the suggestion is DSR, active low.
    clock = TickingClock()
    world = SerialPortWorld(clock, lines={"dsr": True})
    mon = make_monitor(world, clock)
    mon.step()
    for _ in range(4):
        for _ in range(100):                                    # beam clear for ~0.2 s (2 ms per round)
            mon.step()
        world.serial.lines["dsr"] = False                       # a hand in the beam
        for _ in range(25):
            mon.step()
        world.serial.lines["dsr"] = True
    suggestion = mon.suggest()
    assert suggestion["driver"] == "modem-line"
    assert suggestion["settings"] == dict(driver="modem-line", line="dsr", active="low", port="COM5")
    mon.clear()                                                 # "Clear": a fresh start
    assert mon.suggest()["driver"] is None


def test_monitor_suggest_for_byte_messages_split_in_pieces():
    clock = TickingClock()
    world = SerialPortWorld(clock)
    mon = make_monitor(world, clock)
    mon.step()
    for _ in range(3):
        world.serial.incoming += b"T"                           # one message in two pieces, 2 ms apart
        mon.step()
        world.serial.incoming += b"1\r\n"
        for _ in range(50):
            mon.step()
    suggestion = mon.suggest()
    assert suggestion["settings"]["byte_pattern"] == "54310d0a" and suggestion["settings"]["baud"] == 9600


# ------------------------------------------------------------------------------------------------ HTTP endpoints
@pytest.fixture
def page(cfg, tmp_path):
    """The real status web server on a free port, with a controller on fakes. Yields a small client."""
    cfg["beam"].update(driver="modem-line", port="COM3")
    controller, log, _ = make_controller(cfg, tmp_path)
    controller.start_from_config()
    board = StatusBoard(cfg, TickingClock())
    board.beam = controller

    class FakeCapture:
        """Detection's camera: Apply must never touch it."""
        touched = False

        def status(self):
            return dict(state="streaming")

        def stop(self):
            FakeCapture.touched = True

    board.capture = FakeCapture()
    server = StatusServer(board, "127.0.0.1", 0).start()
    base = "http://127.0.0.1:%d" % server.port

    class Client:
        def get(self, path):
            with urllib.request.urlopen(base + path, timeout=5) as r:
                return r.status, r.read()

        def post(self, path, body=None, content_type="application/json", token=None, raw=None):
            data = raw if raw is not None else json.dumps(body or {}).encode()
            req = urllib.request.Request(base + path, data=data, method="POST",
                                         headers={"Content-Type": content_type, **({"X-Setup-Token": token}
                                                                                    if token else {})})
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

    client = Client()
    client.controller, client.log, client.board, client.capture = controller, log, board, FakeCapture
    yield client
    server.stop()


def test_setup_page_and_its_data_are_served(page):
    status, body = page.get("/beam")
    assert status == 200 and b"Beam setup" in body and b"Suggest" in body
    status, body = page.get("/beam/setup.json")
    data = json.loads(body)
    assert data["settings"]["port"] == "COM3" and data["setup_mode"] is False and data["beam"]["enabled"]
    _, body = page.get("/")
    assert b'href="/beam"' in body                             # the main page links to it


def test_apply_over_http_validates_saves_and_restarts_only_the_beam(page, tmp_path):
    sinks_before = list(page.board.__dict__)                    # nothing else on the board changes
    code, answer = page.post("/beam/apply", GOOD)
    assert code == 200 and answer["ok"]
    assert page.log[-2:] == [("stop", "modem-line"), ("start", "modem-line", "COM5")]
    assert load_override(tmp_path / "lapline.beam.toml")[0]["line"] == "dsr"
    assert page.capture.touched is False                       # detection's camera untouched
    assert list(page.board.__dict__) == sinks_before


def test_bad_values_over_http_are_refused_and_change_nothing(page, tmp_path):
    code, answer = page.post("/beam/apply", dict(GOOD, stuck_s="soon"))
    assert code == 400 and not answer["ok"] and "stuck after" in answer["errors"][0]
    assert page.log == [("start", "modem-line", "COM3")]
    assert not (tmp_path / "lapline.beam.toml").exists()


def test_only_json_posts_to_the_beam_actions_are_accepted(page):
    assert page.post("/status.json", {})[0] == 405              # the rest of the page stays read-only
    assert page.post("/", {})[0] == 405
    assert page.post("/beam/apply", raw=b"driver=off", content_type="application/x-www-form-urlencoded")[0] == 415
    assert page.post("/beam/apply", raw=b"{not json")[0] == 400
    assert page.post("/beam/apply", raw=b"[1, 2]")[0] == 400
    assert page.post("/beam/apply", raw=b"{" + b" " * 9000 + b"}")[0] == 413
    assert page.log == [("start", "modem-line", "COM3")]       # none of them did anything


def test_setup_token_is_required_when_configured(page):
    page.board.setup_token = "4321"
    assert page.post("/beam/off", {})[0] == 403
    assert page.post("/beam/off", {}, token="1234")[0] == 403
    code, answer = page.post("/beam/off", {}, token="4321")
    assert code == 200 and page.controller.service is None


def test_monitor_flow_over_http(page):
    assert page.post("/beam/suggest", {})[0] == 409             # no monitor yet
    code, _ = page.post("/beam/setup/start", dict(port="COM5", baud=9600))
    assert code == 200 and page.log[-1] == ("monitor start", "COM5", 9600)
    code, answer = page.post("/beam/suggest", {})
    assert code == 200 and answer["suggestion"]["settings"]["line"] == "dsr"
    assert page.post("/beam/monitor/clear", {})[0] == 200
    code, _ = page.post("/beam/setup/stop", {})
    assert code == 200 and page.log[-1] == ("start", "modem-line", "COM3")


def test_a_crashing_action_is_an_error_answer_not_a_dead_page(page):
    def boom(data):
        raise RuntimeError("bug")

    page.controller.apply = boom
    code, answer = page.post("/beam/apply", GOOD)
    assert code == 500 and "RuntimeError" in answer["errors"][0]
    assert page.get("/status.json")[0] == 200                  # the page still works


def test_status_page_shows_a_setup_error_as_a_warning(page):
    page.controller.error = "beam input NOT started: ValueError: bad"
    payload = json.loads(page.get("/status.json")[1])
    assert "beam input NOT started: ValueError: bad" in payload["warnings"]


def test_controller_threads_do_not_leak(cfg, tmp_path):
    # The real watchdog thread (it ends a forgotten setup mode) starts with setup mode and ends with stop().
    controller, _, _ = make_controller(cfg, tmp_path, use_watchdog=True)
    controller.start_setup(dict(port="COM5"))
    assert controller.watchdog.is_alive()
    controller.stop()
    controller.watchdog.join(1.0)
    assert not controller.watchdog.is_alive()
