"""Tests for launcher/settings.py: checking what the operator typed, and writing the TOML config files.

Two kinds of tests:
  * validation: a bad value gets a clear message in Portuguese, a good value passes;
  * files: what the launcher writes is loaded by the real rt / zone config loaders (rt/config.py, zone/config.py)
    with the expected values, and keys the launcher does not manage are kept.
"""
from dataclasses import asdict

from launcher import settings as S


# ============================================================================================ validation
def good_lapline(**changes):
    data = dict(event="27se", camera_device="0", resolution="1920x1080", ntp_server="192.168.88.1",
                status_port="8765", car_detector="openvino_gpu")
    data.update(changes)
    return data


def test_a_good_lap_line_form_passes_and_is_cleaned_up():
    settings, errors = S.validate_lapline(good_lapline())
    assert errors == {}
    assert settings.event == "27SE"            # upper-cased
    assert settings.status_port == 8765        # the page sends text; it becomes a number


def test_event_code_must_look_like_27SE():
    for bad in ("", "27", "SE27", "27 SE", "2027SE"):
        _, errors = S.validate_lapline(good_lapline(event=bad))
        assert "event" in errors, bad


def test_ports_must_be_numbers_in_range():
    for bad in ("abc", "80", "70000", ""):
        _, errors = S.validate_lapline(good_lapline(status_port=bad))
        assert "status_port" in errors, bad


def test_ntp_server_may_be_empty_or_an_address_but_not_garbage():
    assert "ntp_server" not in S.validate_lapline(good_lapline(ntp_server=""))[1]
    assert "ntp_server" not in S.validate_lapline(good_lapline(ntp_server="router.local"))[1]
    assert "ntp_server" in S.validate_lapline(good_lapline(ntp_server="no spaces please"))[1]


def test_camera_is_required_and_cannot_hold_quotes():
    assert "camera_device" in S.validate_lapline(good_lapline(camera_device=""))[1]
    assert "camera_device" in S.validate_lapline(good_lapline(camera_device='a"b'))[1]
    # a DirectShow device path is fine
    ok = good_lapline(camera_device="usb#vid_046d&pid_085e&mi_00#7&2f0c1a3b&0&0000")
    assert "camera_device" not in S.validate_lapline(ok)[1]


def good_zone(**changes):
    data = dict(camera_device="1", resolution="1280x720", ntp_server="", ptz_host="192.168.0.100", ptz_report="",
                profile="normal", web_port=8090, agent_port=8890, agent_token="segredo")
    data.update(changes)
    return data


def test_a_good_zone_form_passes():
    settings, errors = S.validate_zone(good_zone())
    assert errors == {}
    assert settings.ptz_host == "192.168.0.100"


def test_zone_needs_a_ptz_address():
    assert "ptz_host" in S.validate_zone(good_zone(ptz_host=""))[1]
    assert "ptz_host" in S.validate_zone(good_zone(ptz_host="192.168.0.1 2"))[1]


def test_zone_report_must_exist_when_given():
    exists = {"C:/BajaMV/reports/a/report.json"}
    _, errors = S.validate_zone(good_zone(ptz_report="C:/nowhere.json"), reports_exist=lambda p: p in exists)
    assert "ptz_report" in errors
    _, errors = S.validate_zone(good_zone(ptz_report="C:/BajaMV/reports/a/report.json"),
                                reports_exist=lambda p: p in exists)
    assert "ptz_report" not in errors


def test_zone_profile_must_be_a_known_one_and_bench_ones_are_marked():
    assert "profile" in S.validate_zone(good_zone(profile="fast"))[1]
    assert S.validate_zone(good_zone(profile="bench"))[1] == {}
    assert "bench" in S.BENCH_PROFILES and "bench_strict" in S.BENCH_PROFILES
    assert "normal" not in S.BENCH_PROFILES and "strict" not in S.BENCH_PROFILES


def test_zone_page_and_studio_api_need_different_ports():
    assert "agent_port" in S.validate_zone(good_zone(web_port=8090, agent_port=8090))[1]


def test_port_conflicts_between_roles_on_one_bench_laptop():
    # Both roles on one machine: the lap line's status page and the zone page may not share a port, and nobody
    # may take the launcher's own port.
    lapline = S.validate_lapline(good_lapline(status_port=8090))[0]
    zone = S.validate_zone(good_zone())[0]
    conflicts = S.port_conflicts(lapline, zone, launcher_port=8780)
    assert "zone.web_port" in conflicts
    assert S.port_conflicts(S.validate_lapline(good_lapline())[0], zone, 8780) == {}
    assert "lapline.status_port" in S.port_conflicts(S.validate_lapline(good_lapline(status_port=8780))[0], None,
                                                     8780)


def test_launcher_settings_keep_only_known_roles_in_a_fixed_order():
    settings, errors = S.validate_launcher(dict(roles=["zone", "lapline", "coffee"], autostart=1, pin=" 1234 "))
    assert errors == {}
    assert settings.roles == ["lapline", "zone"]
    assert settings.autostart is True
    assert settings.pin == "1234"


# ============================================================================================ files
def test_a_new_lap_line_config_starts_from_the_race_ready_example(paths):
    from rt import config as rt_config
    settings = S.validate_lapline(good_lapline())[0]
    S.write_lapline(paths, settings)
    # The real rt loader reads it, on top of rt/config.toml, exactly like rt/supervise.py will
    cfg = rt_config.load(paths.lapline_config)
    assert cfg["launch"]["event"] == "27SE"
    assert cfg["camera"]["device"] == 0
    assert cfg["clock"]["ntp_server"] == "192.168.88.1"
    assert cfg["runtime"]["car_backend"] == "openvino" and cfg["runtime"]["car_device"] == "GPU"
    # race-ready settings from lapline.example.toml came along (exposure, high priority)
    assert cfg["camera"]["settings"]["exposure"] == -10
    assert cfg["runtime"]["priority"] == "high"
    # every written path points into the data folder
    assert cfg["launch"]["out_dir"] == str(paths.out_dir / "lapline")
    assert cfg["identity"]["rosters_dir"] == str(paths.rosters_dir)
    # and the models are found through the manifest's absolute path
    assert cfg["models"]["manifest"] == str(paths.manifest)


def test_reading_back_gives_what_was_saved(paths):
    settings = S.validate_lapline(good_lapline(camera_device="usb#vid_1&pid_2", resolution="1280x720",
                                               car_detector="onnxruntime_cpu"))[0]
    S.write_lapline(paths, settings)
    assert S.read_lapline(paths) == settings


def test_saving_keeps_keys_the_page_does_not_show(paths):
    # The operator (or the status page's beam panel) may have changed other keys by hand: they must survive.
    S.write_lapline(paths, S.validate_lapline(good_lapline())[0])
    text = paths.lapline_config.read_text(encoding="utf-8")
    paths.lapline_config.write_text(text.replace("live_latency_s = 0.0", "live_latency_s = 0.123"),
                                    encoding="utf-8")
    S.write_lapline(paths, S.validate_lapline(good_lapline(event="28SE"))[0])
    cfg = S.read_toml(paths.lapline_config)
    assert cfg["capture"]["live_latency_s"] == 0.123
    assert cfg["launch"]["event"] == "28SE"


def test_keep_the_file_detector_choice_writes_nothing(paths):
    # car_detector "" = "manter o que está no arquivo": the runtime keys stay as they were
    S.write_lapline(paths, S.validate_lapline(good_lapline(car_detector="openvino_cpu"))[0])
    S.write_lapline(paths, S.validate_lapline(good_lapline(car_detector=""))[0])
    cfg = S.read_toml(paths.lapline_config)
    assert (cfg["runtime"]["car_backend"], cfg["runtime"]["car_device"]) == ("openvino", "CPU")


def test_zone_config_is_loaded_by_the_zone_loader(paths):
    from zone import config as zone_config
    from zone.spotter import rt_config
    S.write_zone(paths, S.validate_zone(good_zone(profile="bench"))[0])
    cfg = zone_config.load(paths.zone_config)
    assert cfg["ptz"]["host"] == "192.168.0.100"
    assert cfg["spotter"]["detector"] == "coco"          # the bench profile was applied by the zone loader
    assert cfg["zone"]["file"] == str(paths.zones_dir / "zone.json")
    assert cfg["zone"]["out_dir"] == str(paths.out_dir / "zone")
    assert cfg["agent"]["token"] == "segredo"
    # the spotter camera goes through rt's [camera] section (index 1, 1280x720)
    assert cfg["spotter"]["source"] == "camera"
    rt_cfg = rt_config(cfg)
    assert rt_cfg["camera"]["device"] == 1
    assert (rt_cfg["camera"]["width"], rt_cfg["camera"]["height"]) == (1280, 720)


def test_normal_profile_is_written_as_no_profile(paths):
    S.write_zone(paths, S.validate_zone(good_zone(profile="normal"))[0])
    assert S.read_toml(paths.zone_config)["profile"] == ""
    assert S.read_zone(paths).profile == "normal"


def test_zone_keeps_a_recording_set_by_hand_as_its_source(paths):
    # Someone set spotter.source to a video file by hand (teaching on a recording). Saving the page without
    # touching the camera field must not replace it with the webcam.
    S.write_zone(paths, S.validate_zone(good_zone())[0])
    cfg = S.read_toml(paths.zone_config)
    cfg["spotter"]["source"] = "C:/videos/wide.mp4"
    S.write_toml_file(paths.zone_config, cfg)
    shown = S.read_zone(paths)
    assert shown.camera_device == "C:/videos/wide.mp4"
    S.write_zone(paths, S.validate_zone(asdict(shown))[0])
    assert S.read_toml(paths.zone_config)["spotter"]["source"] == "C:/videos/wide.mp4"
    # choosing a camera again switches back
    S.write_zone(paths, S.validate_zone(good_zone(camera_device="0"))[0])
    assert S.read_toml(paths.zone_config)["spotter"]["source"] == "camera"


def test_launcher_json_round_trip(paths):
    S.write_launcher(paths, S.LauncherSettings(roles=["zone"], autostart=True, pin="9"))
    assert S.read_launcher(paths) == S.LauncherSettings(roles=["zone"], autostart=True, pin="9")


def test_a_broken_launcher_json_gives_defaults_instead_of_a_crash(paths):
    paths.launcher_settings.write_text("{not json", encoding="utf-8")
    assert S.read_launcher(paths) == S.LauncherSettings()
