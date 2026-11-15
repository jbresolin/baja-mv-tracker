"""Tests for the parts of the self-test and of the installer build that do not need Windows:

  * launcher/selftest.py: which modules it imports, how results are counted, how it waits for a page;
  * tools/installer/stage_app.py: which repo files go into the installer, the model check against the manifest's
    sha256, and the build_info.json + .pth files it writes.

The real installer build (Python download, Inno Setup) only runs on Windows, in CI (installer-smoke.yml).
"""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import pytest

from launcher.selftest import Results, module_names, wait_for_http

ROOT = Path(__file__).resolve().parents[2]


def load_stage_app():
    """tools/installer is not a package: load stage_app.py from its file."""
    spec = importlib.util.spec_from_file_location("stage_app", ROOT / "tools" / "installer" / "stage_app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage_app = load_stage_app()


# ============================================================================================ self-test
def test_self_test_imports_the_app_modules_but_not_tests_or_dev_tools():
    names = module_names(ROOT)
    assert "rt.run" in names and "zone.app" in names and "launcher.server" in names
    assert "rt" in names and "zone.agent" in names                   # packages themselves (their __init__)
    assert not any(".tests" in name for name in names)
    assert "rt.eval_26se" not in names                               # needs pandas: not installed


def test_results_count_failures_only():
    lines = []
    results = Results(out=lines.append)
    results.add("ok", "a")
    results.add("warn", "b", "no GPU")
    results.add("fail", "c", "missing")
    assert [item["name"] for item in results.failed] == ["c"]
    assert lines[1] == "[WARN] b: no GPU"


def test_waiting_for_a_page_stops_as_soon_as_the_process_is_gone():
    # If the role died, waiting the full timeout (minutes) would only hide the problem.
    started = time.monotonic()
    assert wait_for_http("http://127.0.0.1:9/nothing", timeout_s=60, is_alive=lambda: False, poll_s=0.01) is None
    assert time.monotonic() - started < 10


# ============================================================================================ staging: files
def test_installer_takes_code_and_examples_but_not_tests_or_local_configs():
    wanted = stage_app.wanted
    assert wanted("rt/run.py") and wanted("zone/app.py") and wanted("launcher/static/app.js")
    assert wanted("rt/config.toml")                                  # rt's defaults
    assert wanted("rt/configs/lapline.example.toml") and wanted("zone/configs/zone.example.toml")
    assert not wanted("rt/configs/lapline.toml")                     # a local config never ships
    assert not wanted("zone/configs/site.toml")
    assert not wanted("rt/tests/test_passlogic.py") and not wanted("launcher/tests/conftest.py")
    assert not wanted("rt/eval_26se.py")
    assert wanted("rosters/26SE.csv") and wanted("teacher/rosters.py") and wanted("lapvision/clock.py")
    assert not wanted("teacher/detect.py") and not wanted("lapvision/vlm.py") and not wanted("students/x.py")
    assert wanted("deploy/models/manifest.json")


# ============================================================================================ staging: models
def write_models(folder, contents):
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"note": "test set", "models": {}}
    for key, data in contents.items():
        (folder / ("%s.onnx" % key)).write_bytes(data)
        manifest["models"][key] = {"file": "%s.onnx" % key,
                                   "sha256_16": hashlib.sha256(data).hexdigest()[:16]}
    return manifest


def test_models_that_match_the_manifest_pass(tmp_path):
    manifest = write_models(tmp_path / "m", {"car_n": b"car", "recog": b"rec"})
    chosen = stage_app.check_models(manifest, tmp_path / "m", allow_dummy=False)
    assert [name for name, _ in chosen] == ["car_n.onnx", "recog.onnx"]


def test_a_wrong_model_file_stops_the_build(tmp_path):
    manifest = write_models(tmp_path / "m", {"car_n": b"car"})
    (tmp_path / "m" / "car_n.onnx").write_bytes(b"an older export")
    with pytest.raises(stage_app.StageError, match="car_n"):
        stage_app.check_models(manifest, tmp_path / "m", allow_dummy=False)


def test_a_missing_model_stops_the_build_even_with_dummy_models_allowed(tmp_path):
    manifest = write_models(tmp_path / "m", {"car_n": b"car"})
    (tmp_path / "m" / "car_n.onnx").unlink()
    with pytest.raises(stage_app.StageError, match="missing"):
        stage_app.check_models(manifest, tmp_path / "m", allow_dummy=True)


def test_dummy_models_pass_only_with_the_flag(tmp_path):
    manifest = write_models(tmp_path / "m", {"car_n": b"car"})
    (tmp_path / "m" / "car_n.onnx").write_bytes(b"dummy")
    assert stage_app.check_models(manifest, tmp_path / "m", allow_dummy=True)


def test_the_bench_coco_model_is_optional(tmp_path):
    manifest = write_models(tmp_path / "m", {"car_n": b"car"})
    assert [n for n, _ in stage_app.check_models(manifest, tmp_path / "m", False)] == ["car_n.onnx"]
    (tmp_path / "m" / "yolo11n_coco.onnx").write_bytes(b"coco")
    assert [n for n, _ in stage_app.check_models(manifest, tmp_path / "m", False)] == ["car_n.onnx",
                                                                                         "yolo11n_coco.onnx"]


# ============================================================================================ staging: end to end
def test_stage_writes_code_models_build_info_and_pth(tmp_path):
    # A miniature repo without git: stage_app walks the folders instead of asking git.
    repo = tmp_path / "repo"
    for name, text in {"rt/run.py": "x", "rt/tests/test_a.py": "x", "rt/configs/lapline.example.toml": "x",
                       "rt/configs/lapline.toml": "local", "zone/app.py": "x", "launcher/__init__.py": "",
                       "rosters/26SE.csv": "numero,equipe\n", "teacher/rosters.py": "x",
                       "teacher/detect.py": "x", "lapvision/clock.py": "x"}.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="utf-8")
    manifest = write_models(repo / "deploy" / "models", {"car_n": b"car"})
    (repo / "deploy" / "models" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    stage = tmp_path / "stage"
    info = stage_app.stage(repo, stage, repo / "deploy" / "models", python_dir=stage / "python", version="t1")
    app = stage / "app"
    assert (app / "rt" / "run.py").exists() and (app / "zone" / "app.py").exists()
    assert not (app / "rt" / "tests").exists()
    assert not (app / "rt" / "configs" / "lapline.toml").exists()
    assert not (app / "teacher" / "detect.py").exists()
    assert (app / "deploy" / "models" / "car_n.onnx").read_bytes() == b"car"
    build_info = json.loads((app / "build_info.json").read_text(encoding="utf-8"))
    assert build_info["version"] == "t1" and build_info["dummy_models"] is False
    assert build_info["models_note"] == "test set"
    # the .pth puts INSTALL\app on the private Python's path: site-packages -> Lib -> python -> INSTALL
    pth = stage / "python" / "Lib" / "site-packages" / "bajamv.pth"
    assert pth.read_text(encoding="utf-8").strip() == "../../../app"
    assert (pth.parent / "../../../app").resolve() == app.resolve()
    # 8 repo files: run.py, the example config, app.py, launcher/__init__.py, the roster, teacher/rosters.py,
    # lapvision/clock.py and the manifest
    assert info["files"] == 8


# ============================================================================================ self-test: new checks
from launcher.selftest import check_code_location, check_network_profile, check_vc_runtime, stop_verdict  # noqa: E402


def make_install(tmp_path):
    """A fake install folder: INSTALL/python/python.exe and INSTALL/app/install.ini."""
    install = tmp_path / "BajaMV"
    (install / "python").mkdir(parents=True)
    (install / "python" / "python.exe").write_text("")
    (install / "app").mkdir()
    (install / "app" / "install.ini").write_text("[paths]\ndata_root = C:\\BajaMV\n")
    return install


def test_code_location_fails_when_an_installed_python_runs_other_code(tmp_path):
    # The first CI run: the installed python.exe, started from the repo checkout, imported the CHECKOUT's launcher
    # (python -m puts the current folder first). Every installed file then looked "missing". Now this is one clear
    # FAIL that names both folders.
    install = make_install(tmp_path)
    results = Results(out=lambda line: None)
    check_code_location(results, tmp_path / "checkout", executable=str(install / "python" / "python.exe"))
    assert results.items[0]["level"] == "fail"
    assert "checkout" in results.items[0]["detail"]


def test_code_location_is_ok_for_the_installed_code_and_for_a_developer_venv(tmp_path):
    install = make_install(tmp_path)
    results = Results(out=lambda line: None)
    check_code_location(results, install / "app", executable=str(install / "python" / "python.exe"))
    (tmp_path / "venv").mkdir()
    check_code_location(results, tmp_path / "repo", executable=str(tmp_path / "venv" / "python"))
    assert [item["level"] for item in results.items] == ["ok", "ok"]


def test_vc_runtime_older_than_14_40_fails(tmp_path):
    # ONNX Runtime / OpenVINO crash with an msvcp140.dll older than 14.40. The DLL versions are faked here.
    python_dir, system_dir = tmp_path / "python", tmp_path / "System32"
    python_dir.mkdir()
    system_dir.mkdir()
    (system_dir / "msvcp140.dll").write_text("")
    versions = {system_dir / "msvcp140.dll": (14, 38, 33135, 0)}
    results = Results(out=lambda line: None)
    check_vc_runtime(results, python_dir, version_of=versions.get, system_dir=system_dir)
    assert results.items[-1]["level"] == "fail"
    versions[system_dir / "msvcp140.dll"] = (14, 44, 35211, 0)
    check_vc_runtime(results, python_dir, version_of=versions.get, system_dir=system_dir)
    assert results.items[-1]["level"] == "ok"


def test_an_old_app_local_vc_runtime_fails_even_if_the_system_one_is_new(tmp_path):
    # A msvcp140.dll next to python.exe is loaded instead of the system one: an old one there is the real problem.
    python_dir, system_dir = tmp_path / "python", tmp_path / "System32"
    python_dir.mkdir()
    system_dir.mkdir()
    (python_dir / "msvcp140.dll").write_text("")
    (system_dir / "msvcp140.dll").write_text("")
    versions = {python_dir / "msvcp140.dll": (14, 29, 0, 0), system_dir / "msvcp140.dll": (14, 44, 0, 0)}
    results = Results(out=lambda line: None)
    check_vc_runtime(results, python_dir, version_of=versions.get, system_dir=system_dir)
    assert results.items[-1]["level"] == "fail" and "python" in results.items[-1]["detail"]


def test_public_network_is_a_warning_not_a_failure():
    results = Results(out=lambda line: None)
    check_network_profile(results, lookup=lambda: [dict(name="RedeBaja", interface="Ethernet", category="Public")])
    assert results.items[-1]["level"] == "warn" and results.failed == []


def test_role_stop_check_needs_every_process_of_the_tree_gone():
    # tree = [(pid, start time)] of the supervisor and its children, listed BEFORE the stop. find() says which of
    # them still run (here: a fake that knows one survivor). The old check could never fail; this one can.
    tree = [(100, 1.0), (101, 2.0), (102, 3.0)]
    survivor = {(102, 3.0)}

    def find(pid, created):
        return object() if (pid, created) in survivor else None
    ok, text = stop_verdict(tree, find=find, timeout_s=0, sleep=lambda s: None)
    assert ok is False and "102" in text
    survivor.clear()
    ok, text = stop_verdict(tree, find=find, timeout_s=0, sleep=lambda s: None)
    assert ok is True
    # a tree that could not even be listed is not proof of a stop
    assert stop_verdict([], find=find, timeout_s=0)[0] is False


# ============================================================================================ staging: git fallback
def test_staging_warns_loudly_when_git_cannot_list_the_files(tmp_path):
    # Without git (or with git refusing a folder owned by another Windows user: "dubious ownership") staging walks
    # the folders, which may pick up untracked local files. That must be impossible to miss in the build log.
    class Failed:                                 # what subprocess.run returns when git refuses
        returncode = 128
        stdout = b""
        stderr = b"fatal: detected dubious ownership in repository at 'C:/x'"
    warnings = []
    assert stage_app.git_files(tmp_path, run=lambda *a, **k: Failed(), warn=warnings.append) is None
    text = " ".join(warnings[0])
    assert "dubious ownership" in text and "safe.directory" in text

    def no_git(*a, **k):
        raise FileNotFoundError("git")
    warnings.clear()
    assert stage_app.git_files(tmp_path, run=no_git, warn=warnings.append) is None
    assert warnings and "git could not run" in warnings[0][0]
