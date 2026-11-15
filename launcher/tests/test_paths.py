"""Tests for launcher/paths.py: where the launcher finds the code, the data folder and ffmpeg.

The launcher runs in two modes:
  * "repo mode": a developer's git checkout. Nothing about the repo workflow may change.
  * "installed mode": the Windows installer put the code in Program Files (read-only) and wrote build_info.json.
Each test builds a tiny fake code folder in a temporary directory and checks what resolve_paths() decides.
"""
import json

from launcher.paths import DATA_ROOT_ENV, console_python, resolve_paths


def make_installed_code_root(tmp_path, data_root=None):
    """A fake install: INSTALL/app/build_info.json (+ install.ini) and INSTALL/ffmpeg/bin/ffmpeg(.exe)."""
    install = tmp_path / "BajaMV"
    app = install / "app"
    app.mkdir(parents=True)
    (app / "build_info.json").write_text(json.dumps({"version": "2026.11.16-abc1234"}), encoding="utf-8")
    if data_root is not None:
        (app / "install.ini").write_text("[paths]\ndata_root = %s\n" % data_root, encoding="utf-8")
    return app


def test_repo_mode_uses_a_folder_inside_the_repo_by_default(tmp_path):
    # A checkout has no build_info.json: repo mode, data under <repo>/out/bajamv (out/ is gitignored).
    paths = resolve_paths(code_root=tmp_path, env={})
    assert paths.installed is False
    assert paths.data_root == tmp_path / "out" / "bajamv"
    assert paths.ffmpeg == "ffmpeg"            # repo mode: whatever ffmpeg is on PATH, as today


def test_installed_mode_reads_the_data_root_the_installer_wrote(tmp_path):
    # The installer writes the chosen data folder into install.ini; the launcher must use exactly that.
    app = make_installed_code_root(tmp_path, data_root=tmp_path / "MyData")
    paths = resolve_paths(code_root=app, env={})
    assert paths.installed is True
    assert paths.data_root == tmp_path / "MyData"
    assert paths.version == "2026.11.16-abc1234"


def test_installed_mode_without_install_ini_uses_c_bajamv(tmp_path):
    app = make_installed_code_root(tmp_path)
    paths = resolve_paths(code_root=app, env={})
    assert str(paths.data_root).replace("\\", "/").rstrip("/") == "C:/BajaMV"


def test_command_line_beats_environment_beats_install_ini(tmp_path):
    # Order of precedence: --data-root, then the BAJAMV_DATA variable, then install.ini.
    app = make_installed_code_root(tmp_path, data_root=tmp_path / "FromIni")
    env = {DATA_ROOT_ENV: str(tmp_path / "FromEnv")}
    assert resolve_paths(code_root=app, env=env).data_root == tmp_path / "FromEnv"
    assert resolve_paths(data_root=tmp_path / "FromArg", code_root=app, env=env).data_root == tmp_path / "FromArg"


def test_installed_mode_uses_the_bundled_ffmpeg_when_it_is_there(tmp_path):
    # The installer puts ffmpeg in INSTALL/ffmpeg/bin, so PATH does not matter on the race laptops.
    import sys
    app = make_installed_code_root(tmp_path)
    bundled = app.parent / "ffmpeg" / "bin"
    bundled.mkdir(parents=True)
    exe = ".exe" if sys.platform == "win32" else ""
    (bundled / ("ffmpeg" + exe)).write_text("fake")
    (bundled / ("ffprobe" + exe)).write_text("fake")
    paths = resolve_paths(code_root=app, env={})
    assert paths.ffmpeg == str(bundled / ("ffmpeg" + exe))
    assert paths.ffprobe == str(bundled / ("ffprobe" + exe))


def test_data_folders_are_inside_the_data_root(tmp_path):
    paths = resolve_paths(data_root=tmp_path / "d", code_root=tmp_path, env={})
    paths.make_folders()
    for folder in ("config", "out", "logs", "rosters", "zones", "reports", "exports"):
        assert (tmp_path / "d" / folder).is_dir()
    assert paths.lapline_config == tmp_path / "d" / "config" / "lapline.toml"
    assert paths.zone_config == tmp_path / "d" / "config" / "zone.toml"


def test_children_are_started_with_python_exe_not_pythonw(tmp_path):
    # The tray app runs under pythonw.exe (no console). Its children must use python.exe, whose output can be
    # written to a log file. If there is no python.exe next to it, the same executable is used.
    (tmp_path / "python.exe").write_text("")
    assert console_python(str(tmp_path / "pythonw.exe")) == str(tmp_path / "python.exe")
    assert console_python(str(tmp_path / "python3")) == str(tmp_path / "python3")


# ---------------------------------------------------------------------------- install.ini written by Inno Setup
def test_install_ini_in_the_windows_ansi_code_page_is_read(tmp_path):
    # Inno Setup writes install.ini through a Windows function that uses the PC's "ANSI" code page (cp1252 on a
    # Brazilian Windows), not UTF-8. A data folder with an accent ("Gravações") then arrives as bytes that are not
    # valid UTF-8. Reading it must neither crash nor mangle the accent.
    app = make_installed_code_root(tmp_path)
    (app / "install.ini").write_bytes("[paths]\r\ndata_root=D:\\Gravações\\BajaMV\r\n".encode("cp1252"))
    paths = resolve_paths(code_root=app, env={})
    assert "Gravações" in str(paths.data_root)


def test_install_ini_in_utf8_with_or_without_bom_is_read(tmp_path):
    from launcher.paths import read_install_ini
    app = make_installed_code_root(tmp_path)
    (app / "install.ini").write_bytes("\ufeff[paths]\ndata_root = E:\\Ação\n".encode("utf-8"))
    assert read_install_ini(app) == "E:\\Ação"
    (app / "install.ini").write_bytes("[paths]\ndata_root = E:\\Ação\n".encode("utf-8"))
    assert read_install_ini(app) == "E:\\Ação"


# ---------------------------------------------------------------------------- which code an installed Python runs
def test_an_installed_python_knows_its_own_app_folder(tmp_path):
    # INSTALL/python/python.exe belongs to INSTALL/app (it has install.ini / build_info.json). The self-test uses
    # this to notice when the installed Python imported some other copy of the code.
    from launcher.paths import installed_app_of_python, same_folder
    app = make_installed_code_root(tmp_path, data_root=tmp_path / "Data")
    (app.parent / "python").mkdir()
    (app.parent / "python" / "python.exe").write_text("")
    found = installed_app_of_python(str(app.parent / "python" / "python.exe"))
    assert found is not None and same_folder(found, app)
    # a Python somewhere else (a developer's venv) is not an installed one
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    assert installed_app_of_python(str(tmp_path / "venv" / "bin" / "python")) is None
