"""Tests for launcher/roles.py: the commands each role runs, and the zone's safe stop.

The roles must run the existing entry points (rt/supervise.py, zone.supervisor + zone.app) with the config in the
data folder. The zone must be stopped in three steps: ask the zone page for PARAR, kill, then send the PTZ a VISCA
stop directly. Fake processes and fake network calls record what happened and in which order.
"""
from launcher import settings as S
from launcher.procs import ManagedProcess
from launcher.roles import RoleManager, child_env, lapline_command, zone_command


def make_manager(paths, fake_system, fake_clock, calls):
    def make_process(name, make_cmd):
        return ManagedProcess(name, make_cmd, paths.logs_dir, spawn=fake_system.spawn, kill=fake_system.kill,
                              clock=fake_clock, create_time=lambda pid: None)

    def zone_stop(port):
        calls.append(("zone PARAR", port))
        fake_system.events.append(("zone PARAR", port))

    def ptz_stop(host, port):
        calls.append(("VISCA stop", host, port))
        fake_system.events.append(("VISCA stop", host))

    return RoleManager(paths, make_process=make_process, zone_stop=zone_stop, ptz_stop=ptz_stop, log=lambda m: None)


def save_both(paths):
    S.write_lapline(paths, S.validate_lapline(dict(event="27SE", camera_device="0"))[0])
    S.write_zone(paths, S.validate_zone(dict(camera_device="1", ptz_host="192.168.0.100", web_port=8091))[0])


def test_commands_are_the_existing_supervisors_with_the_data_folder_configs(paths):
    assert lapline_command(paths)[1:] == [str(paths.code_root / "rt" / "supervise.py"), "--config",
                                          str(paths.lapline_config)]
    zone = zone_command(paths)
    assert zone[1:3] == ["-m", "zone.supervisor"]
    assert zone[zone.index("--") + 1:][1:] == ["-m", "zone.app", "--config", str(paths.zone_config)]


def test_children_get_the_data_root_and_no_nvidia_gpu(paths):
    env = child_env(paths, base={"PATH": "x"})
    assert env["BAJAMV_DATA"] == str(paths.data_root)
    assert env["CUDA_VISIBLE_DEVICES"] == "-1"      # as the .bat files: CPU / Intel iGPU only
    assert env["PATH"] == "x"


def test_a_role_without_saved_settings_does_not_start(paths, fake_system, fake_clock):
    manager = make_manager(paths, fake_system, fake_clock, [])
    ok, message = manager.start("lapline")
    assert not ok and "Salve" in message
    assert fake_system.started == []


def test_start_and_stop_the_lap_line(paths, fake_system, fake_clock):
    save_both(paths)
    manager = make_manager(paths, fake_system, fake_clock, [])
    assert manager.start("lapline")[0]
    assert manager.running_roles() == ["lapline"]
    assert manager.stop("lapline")[0]
    assert manager.running_roles() == []


def test_zone_stop_order_is_parar_then_kill_then_visca_stop(paths, fake_system, fake_clock):
    save_both(paths)
    calls = []
    manager = make_manager(paths, fake_system, fake_clock, calls)
    manager.start("zone")
    zone_pid = fake_system.started[0].pid
    manager.stop("zone")
    order = [event[0] for event in fake_system.events if event[0] != "spawn"]
    assert order == ["zone PARAR", "kill", "VISCA stop"]
    assert ("zone PARAR", 8091) in calls                      # the zone page's port from the config
    assert ("VISCA stop", "192.168.0.100", 52381) in calls    # the PTZ's address from the config
    assert fake_system.killed == [zone_pid]


def test_zone_is_still_killed_and_the_ptz_stopped_when_the_page_does_not_answer(paths, fake_system, fake_clock):
    save_both(paths)
    calls = []
    manager = make_manager(paths, fake_system, fake_clock, calls)

    def page_down(port):
        raise ConnectionRefusedError("zone page down")
    manager.zone_stop = page_down
    manager.start("zone")
    ok, _ = manager.stop("zone")
    assert ok
    assert fake_system.killed                                 # killed anyway
    assert ("VISCA stop", "192.168.0.100", 52381) in calls    # and the PTZ still got its stop


def test_ptz_recorder_runs_only_when_the_config_asks_for_it(paths, fake_system, fake_clock):
    save_both(paths)
    manager = make_manager(paths, fake_system, fake_clock, [])
    manager.start("zone")
    assert [p.cmd[p.cmd.index("--name") + 1] for p in fake_system.started] == ["zone"]
    manager.stop("zone")
    cfg = S.read_toml(paths.zone_config)
    cfg.setdefault("recording", {})["ptz_usb"] = True
    S.write_toml_file(paths.zone_config, cfg)
    manager.start("zone")
    assert [p.cmd[p.cmd.index("--name") + 1] for p in fake_system.started[1:]] == ["zone", "ptz_recorder"]
    manager.stop("zone")
    assert manager.status()["ptz_recorder"]["running"] is False


def test_restart_is_stop_then_start(paths, fake_system, fake_clock):
    save_both(paths)
    manager = make_manager(paths, fake_system, fake_clock, [])
    manager.start("lapline")
    manager.restart("lapline")
    assert len(fake_system.started) == 2
    assert fake_system.killed == [fake_system.started[0].pid]
    assert manager.running_roles() == ["lapline"]
