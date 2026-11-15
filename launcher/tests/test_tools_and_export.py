"""Tests for three small helpers of the page:
  * launcher/exportlogs.py: the "Exportar logs" zip (what goes in, secrets blanked, videos left out);
  * launcher/tools.py: starting the PTZ characterization (refusals, the command it runs, the report list);
  * launcher/teerun.py: running a module while copying its output to a log file.
"""
import sys
import zipfile

from launcher.exportlogs import export_logs, redact
from launcher.teerun import main as teerun_main
from launcher.tools import CharacterizationRunner, list_reports


# ============================================================================================ export
def test_export_contains_logs_configs_and_the_newest_run_but_no_video(paths):
    (paths.logs_dir / "lapline.log").write_text("lap line log", encoding="utf-8")
    (paths.config_dir / "zone.toml").write_text('[agent]\ntoken = "segredo"\nport = 8890\n', encoding="utf-8")
    day = paths.out_dir / "lapline" / "2026-11-21"
    day.mkdir(parents=True)
    (day / "lapline_09-00-00.jsonl").write_text("{}\n", encoding="utf-8")
    (day / "lapline_09-00-00.csv").write_text("a\n", encoding="utf-8")
    (day / "rec").mkdir()
    (day / "rec" / "rec_1.mkv").write_bytes(b"video")
    import datetime as dt
    zip_path = export_logs(paths, now=dt.datetime(2026, 11, 21, 10, 0, 0), hostname="T14 lap")
    assert zip_path.name == "BajaMV-logs-T14_lap-2026-11-21_10-00-00.zip"
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "logs/lapline.log" in names
        assert "out/lapline/2026-11-21/lapline_09-00-00.jsonl" in names
        assert "out/lapline/2026-11-21/lapline_09-00-00.csv" in names
        assert not any(name.endswith(".mkv") for name in names)        # never the recordings
        zone_toml = zf.read("config/zone.toml").decode("utf-8")
        assert "segredo" not in zone_toml and 'token = "***"' in zone_toml
        assert "port = 8890" in zone_toml


def test_redact_blanks_tokens_and_pins_in_toml_and_json():
    assert redact('setup_token = "1234"\nport = 1') == 'setup_token = "***"\nport = 1'
    assert redact('{"pin": "9999", "roles": []}') == '{"pin": "***", "roles": []}'


# ============================================================================================ characterization
class RecordingSpawn:
    def __init__(self):
        self.commands = []

    def __call__(self, cmd):
        from launcher.tests.conftest import FakeProcess
        self.commands.append(cmd)
        return FakeProcess(cmd)


def test_characterization_runs_the_bench_tool_through_teerun(paths):
    spawn = RecordingSpawn()
    runner = CharacterizationRunner(paths, spawn=spawn)
    ok, message = runner.start("192.168.0.100", "smtav-a", quick=True)
    assert ok, message
    cmd = spawn.commands[0]
    assert cmd[1:3] == ["-m", "launcher.teerun"]
    assert "zone.tools.characterize" in cmd
    assert cmd[cmd.index("--ip") + 1] == "192.168.0.100"
    assert cmd[cmd.index("--out") + 1] == str(paths.reports_dir / "smtav-a")
    assert "--quick" in cmd
    assert "--yes" not in cmd                # the confirmation before moving is never skipped
    assert runner.status()["running"] is True


def test_characterization_refusals(paths):
    runner = CharacterizationRunner(paths, spawn=RecordingSpawn())
    assert runner.start("192.168.0.100", "x", zone_running=True)[0] is False     # two VISCA clients
    assert runner.start("not an ip", "x")[0] is False
    assert runner.start("192.168.0.100", "../escape")[0] is False                # folder name stays in reports/
    assert runner.start("192.168.0.100", "ok")[0] is True
    assert runner.start("192.168.0.100", "again")[0] is False                    # one at a time


def test_report_list_finds_report_json_files(paths):
    (paths.reports_dir / "ptz1").mkdir()
    (paths.reports_dir / "ptz1" / "report.json").write_text("{}", encoding="utf-8")
    (paths.reports_dir / "empty").mkdir()
    reports = list_reports(paths.reports_dir)
    assert [r["name"] for r in reports] == ["ptz1"]


# ============================================================================================ teerun
def test_teerun_copies_the_output_to_the_log_and_returns_the_exit_code(tmp_path, monkeypatch):
    # A tiny module that prints and exits with code 3, run through teerun like the characterization is.
    (tmp_path / "tiny_tool.py").write_text("import sys\nprint('measuring...')\nsys.exit(3)\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")      # the "press Enter to close" at the end
    log = tmp_path / "tool.log"
    stdout, stderr = sys.stdout, sys.stderr
    try:
        code = teerun_main([str(log), "tiny_tool"])
    finally:
        sys.stdout, sys.stderr = stdout, stderr
    assert code == 3
    text = log.read_text(encoding="utf-8")
    assert "measuring..." in text
    assert "[fim: código 3]" in text
