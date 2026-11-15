"""`python -m launcher --self-test`: check an installation end to end, without a camera or a PTZ. Exit code 0 = all
good. The CI smoke test runs it right after a silent install; run it on a laptop after installing, too.

Checks, in order (each prints [ OK ], [WARN] or [FAIL] with a detail):
  1. Python: version and where it runs from; an installed Python must run the installed code (not, say, a repo
     checkout that happens to be the current folder). Windows: the Visual C++ runtime is new enough, and whether
     a network is marked Public (a warning).
  2. Imports: every module of rt/, zone/ and launcher/ (tests excluded).
  3. Models: each model of deploy/models/manifest.json loads in ONNX Runtime; the car detector runs once.
  4. OpenVINO: imports, lists its devices, compiles the car detector on CPU (and says whether a GPU is there).
  5. OpenCV: encodes an image.
  6. ffmpeg / ffprobe: run, and the build has libx264, h264_qsv and h264_amf (rt probes them in that order).
  7. Tray: pystray + Pillow import and the icon can be drawn.
  8. Lap line role: started through the launcher's own RoleManager (rt/supervise.py -> rt/run.py) on a synthetic
     video instead of a camera; its status page must answer; then stopped (the supervisor and every process it
     started must be gone: checked with psutil, not only with the launcher's own bookkeeping).
  9. Zone PTZ role: the same, with the synthetic video as spotter and a fake VISCA camera (zone/ptz/fake_camera.py)
     on 127.0.0.1 as PTZ; its page and the studio API must answer; then the safe stop.
 10. The launcher's own page answers.

Everything is written to a temporary data folder, never to the real one; the roles use free ports, so a launcher
that is already running is not disturbed.
"""
import contextlib
import importlib
import json
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from dataclasses import replace
from pathlib import Path

from launcher import settings as S
from launcher.paths import installed_app_of_python, resolve_paths, same_folder
from launcher.procs import find_process

PACKAGES = ("rt", "zone", "launcher")
DEV_ONLY_MODULES = {"rt.eval_26se"}         # needs pandas + the 26SE data: not installed, not needed at the race


class Results:
    def __init__(self, out=print):
        self.items, self.out = [], out

    def add(self, level, name, detail=""):
        self.items.append(dict(level=level, name=name, detail=str(detail)))
        self.out("[%s] %s%s" % ({"ok": " OK ", "warn": "WARN", "fail": "FAIL"}[level], name,
                                ": " + str(detail) if detail else ""))

    @property
    def failed(self):
        return [item for item in self.items if item["level"] == "fail"]


@contextlib.contextmanager
def step(results, name):
    """Run one check; an exception is a FAIL with its message (the traceback goes to the console)."""
    try:
        yield
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        results.add("fail", name, "%s: %s" % (type(e).__name__, e))


# ============================================================================================== small helpers
def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_http(url, timeout_s, is_alive=lambda: True, poll_s=0.5):
    """GET url until it answers 200 (-> the body) or the time runs out / the process dies (-> None)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.read()
        except Exception:  # noqa: BLE001
            pass
        if not is_alive():
            return None
        time.sleep(poll_s)
    return None


def wait_for_json(url, accept, timeout_s, is_alive=lambda: True, poll_s=0.5, get=None):
    """GET url (JSON) until `accept(data)` is true (-> data) or the time runs out / the process dies (-> None).
    A page that answers is not enough to call a role healthy: in the smoke test rt and the zone both served their
    page for a moment and then crashed on their first video frame, and the check passed or failed by luck."""
    def default_get(address):
        with urllib.request.urlopen(address, timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))
    get = get or default_get
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            data = get(url)
            if accept(data):
                return data
        except Exception:  # noqa: BLE001
            pass
        if not is_alive():
            return None
        time.sleep(poll_s)
    return None


def role_log(process, lines=40):
    """The last `lines` lines of a role's own log (logs/<role>.log: everything the role and the processes under it
    printed, tracebacks included), one per line, for a FAIL detail. Fifteen lines joined on one line were not enough
    to see why a role died in CI: the useful lines had scrolled away behind a long traceback."""
    status = process.status(lines)
    exit_text = "running" if status["running"] else "exit code %s" % status["exit_code"]
    return "\n".join(["%s (%s), last %d lines:" % (status["log_file"], exit_text, lines)] +
                     ["    | " + line for line in status["log"]])


def module_names(code_root, packages=PACKAGES):
    """Every importable module of the packages, tests and dev-only modules excluded."""
    names = []
    for package in packages:
        for path in sorted((Path(code_root) / package).rglob("*.py")):
            relative = path.relative_to(code_root).with_suffix("")
            parts = list(relative.parts)
            if "tests" in parts or "__pycache__" in parts:
                continue
            if parts[-1] == "__init__":
                parts = parts[:-1]
            name = ".".join(parts)
            if name not in DEV_ONLY_MODULES:
                names.append(name)
    return names


def ffmpeg_encoders(ffmpeg):
    result = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=60)
    return result.stdout


def make_test_video(ffmpeg, path, seconds=60):
    """A synthetic 640x360, 10 fps video (ffmpeg's test pattern): the camera stand-in for both roles."""
    for codec in (["-c:v", "libx264", "-pix_fmt", "yuv420p"], ["-c:v", "mpeg4"]):
        cmd = [ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10",
               "-t", str(seconds)] + codec + [str(path)]
        if subprocess.run(cmd, capture_output=True, timeout=120).returncode == 0 and Path(path).exists():
            return path
    raise RuntimeError("ffmpeg could not write a test video")


def process_tree(pid):
    """[(pid, create_time)] of a process and all of its children, grandchildren... ([] when psutil cannot see it).
    The create time tells a later check apart from a new process that got the same pid."""
    try:
        import psutil
        parent = psutil.Process(int(pid))
        return [(p.pid, p.create_time()) for p in [parent] + parent.children(recursive=True)]
    except Exception:  # noqa: BLE001
        return []


def still_alive(tree, find=find_process, timeout_s=15.0, poll_s=0.5, clock=time.monotonic, sleep=time.sleep):
    """The pids of `tree` (from process_tree) that still run after up to timeout_s seconds ([] = all gone)."""
    deadline = clock() + timeout_s
    while True:
        alive = [pid for pid, created in tree if find(pid, created) is not None]
        if not alive or clock() >= deadline:
            return alive
        sleep(poll_s)


def stop_verdict(tree, find=find_process, **kwargs):
    """(ok, text) for a role's stop: ok only when the supervisor and every process under it are gone."""
    if not tree:
        return False, "DID NOT STOP? (its processes could not be listed before stopping)"
    alive = still_alive(tree, find=find, **kwargs)
    if alive:
        return False, "DID NOT STOP: pid %s still running" % ", ".join(str(pid) for pid in alive)
    return True, "stopped (%d processes gone)" % len(tree)


# ============================================================================================== the checks
def check_code_location(results, code_root, executable=None):
    """An installed Python must import the installed code. `python -m launcher` puts the current folder first on
    the import path, so started from a repo checkout it would test the checkout instead (the first CI run did:
    every installed file "missing"). BajaMV.cmd and the shortcuts use `python -P`, which this check guards."""
    app = installed_app_of_python(executable)
    if app is None:
        results.add("ok", "code location", "%s (not an installed Python)" % code_root)
    elif same_folder(app, code_root):
        results.add("ok", "code location", "installed code %s" % code_root)
    else:
        results.add("fail", "code location", "this installed Python runs the code in %s, not the installed %s "
                    "(started from a folder with its own launcher\\? use BajaMV.cmd, or python -P -m launcher)" % (
                        code_root, app))


def file_version(path):
    """(major, minor, build, revision) of a Windows DLL / EXE, or None (needs Windows' version.dll)."""
    import ctypes
    from ctypes import wintypes
    version = ctypes.WinDLL("version")
    size = version.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return None
    buffer = ctypes.create_string_buffer(size)
    if not version.GetFileVersionInfoW(str(path), 0, size, buffer):
        return None
    pointer, length = ctypes.c_void_p(), wintypes.UINT()
    if not version.VerQueryValueW(buffer, "\\", ctypes.byref(pointer), ctypes.byref(length)):
        return None
    # VS_FIXEDFILEINFO: signature, struct version, then FileVersionMS and FileVersionLS
    words = ctypes.cast(pointer, ctypes.POINTER(wintypes.DWORD * 4)).contents
    ms, ls = words[2], words[3]
    return (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)


MIN_VC_RUNTIME = (14, 40)       # older msvcp140.dll + newer ONNX Runtime / OpenVINO = crash in std::mutex


def check_vc_runtime(results, python_dir, version_of=None, system_dir=None):
    """Windows: the Visual C++ runtime that ONNX Runtime / OpenVINO load (msvcp140.dll) must be 14.40 or newer. The
    installer runs Microsoft's vc_redist when the PC's copy is older; a msvcp140.dll next to python.exe would win
    over the system one, so an old one there is a failure too."""
    import os
    version_of = version_of or file_version
    system_dir = Path(system_dir or Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32")
    problems, seen = [], []
    for where in (Path(python_dir), system_dir):
        dll = where / "msvcp140.dll"
        if not dll.exists():
            continue
        found = version_of(dll)
        seen.append("%s %s" % (dll, ".".join(map(str, found)) if found else "?"))
        if found is not None and tuple(found[:2]) < MIN_VC_RUNTIME:
            problems.append("%s is %s (needs %d.%d or newer)" % (dll, ".".join(map(str, found)), *MIN_VC_RUNTIME))
    if not seen:
        results.add("fail", "vc runtime", "msvcp140.dll not found (install the VC++ 2015-2022 x64 redistributable)")
    elif problems:
        results.add("fail", "vc runtime", "; ".join(problems))
    else:
        results.add("ok", "vc runtime", "; ".join(seen))


def check_network_profile(results, lookup=None):
    """Windows: warn when a connection is marked Public (see launcher/netprofile.py)."""
    from launcher.netprofile import public_network_warning, public_networks
    warning = public_network_warning((lookup or public_networks)())
    if warning:
        results.add("warn", "network profile", warning)
    else:
        results.add("ok", "network profile", "no network marked Public")


def check_imports(results, code_root):
    bad = []
    names = module_names(code_root)
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            bad.append("%s (%s: %s)" % (name, type(e).__name__, e))
    if bad:
        results.add("fail", "imports", "; ".join(bad))
    else:
        results.add("ok", "imports", "%d modules" % len(names))


def check_models(results, paths):
    import numpy as np
    import onnxruntime as ort
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    models = manifest["models"]
    for key, info in models.items():
        model_path = paths.manifest.parent / info["file"]
        with step(results, "model %s" % key):
            if not model_path.exists():
                results.add("fail", "model %s" % key, "missing: %s" % model_path)
                continue
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            detail = "loads (%d KB)" % (model_path.stat().st_size // 1024)
            shape = (info.get("input") or {}).get("shape") or []
            if shape and all(isinstance(x, int) for x in shape):
                feed = {session.get_inputs()[0].name: np.zeros(shape, dtype=np.float32)}
                outputs = session.run(None, feed)
                detail += ", runs: output %s" % (tuple(outputs[0].shape),)
            results.add("ok", "model %s" % key, detail)
    results.add("ok", "onnxruntime", "%s, providers %s" % (ort.__version__, ort.get_available_providers()))
    return models


def check_openvino(results, paths, models):
    import openvino as ov
    core = ov.Core()
    devices = core.available_devices
    car = paths.manifest.parent / models["car_n"]["file"]
    core.compile_model(core.read_model(str(car)), "CPU")
    level = "ok" if "GPU" in devices or sys.platform != "win32" else "warn"
    results.add(level, "openvino", "%s, devices %s, car_n compiled on CPU%s" % (
        ov.__version__ if hasattr(ov, "__version__") else "?", devices,
        "" if "GPU" in devices else " (no Intel GPU here: the lap line falls back to CPU)"))


def check_opencv(results):
    import cv2
    import numpy as np
    ok, _ = cv2.imencode(".jpg", np.zeros((32, 32, 3), np.uint8))
    results.add("ok" if ok else "fail", "opencv", cv2.__version__)


def check_ffmpeg(results, paths):
    encoders = ffmpeg_encoders(paths.ffmpeg)
    missing = [name for name in ("libx264", "h264_qsv", "h264_amf") if (" %s " % name) not in encoders]
    probe = subprocess.run([paths.ffprobe, "-version"], capture_output=True, text=True, timeout=30)
    if probe.returncode != 0:
        results.add("fail", "ffprobe", "did not run: %s" % paths.ffprobe)
    if missing:
        # the installer bundles a build with all three; a developer's own ffmpeg (repo mode) only gets a warning
        results.add("fail" if paths.installed else "warn", "ffmpeg", "%s lacks encoders %s" % (paths.ffmpeg, missing))
    else:
        results.add("ok", "ffmpeg", "%s: libx264, h264_qsv, h264_amf present" % paths.ffmpeg)


def check_tray(results):
    try:
        from launcher.tray import make_icon_image
        make_icon_image()
        import pystray  # noqa: F401
        results.add("ok", "tray", "pystray + Pillow")
    except Exception as e:  # noqa: BLE001
        # a Linux CI box has no display for pystray: only a failure on Windows
        results.add("fail" if sys.platform == "win32" else "warn", "tray", "%s: %s" % (type(e).__name__, e))


def check_lapline_role(results, paths, video, timeout_s=180):
    from launcher.roles import RoleManager
    port = free_port()
    settings = S.LaplineSettings(event="26SE", camera_device="0", status_port=port, car_detector="onnxruntime_cpu")
    S.write_lapline(paths, settings)
    cfg = S.read_toml(paths.lapline_config)
    cfg["launch"]["source"] = str(video)               # a file instead of the camera
    cfg["launch"]["extra_args"] = ["--realtime"]        # paced at real speed, so the run lasts the video's length
    S.write_toml_file(paths.lapline_config, cfg)
    roles = RoleManager(paths)
    ok, message = roles.start("lapline")
    if not ok:
        results.add("fail", "lap line role", message)
        return
    process = roles.processes["lapline"]
    url = "http://127.0.0.1:%d/status.json" % port
    body = wait_for_http(url, timeout_s, is_alive=process.running)
    # the page alone is not enough: rt must also read and process frames of the video (it serves the page before it
    # opens its source, so a crash on opening the source would otherwise go unnoticed)
    status = wait_for_json(url, lambda data: (data.get("processing") or {}).get("frames", 0) > 0, timeout_s,
                           is_alive=process.running) if body is not None else None
    log_text = role_log(process)                       # read before the stop (which adds its own lines)
    tree = process_tree(process.proc.pid) if process.running() else []
    roles.stop("lapline")
    stopped, stop_text = stop_verdict(tree)
    if body is None or status is None:
        results.add("fail", "lap line role", "%s; log %s" % (
            "status page never answered" if body is None else "status page answered but no frame was processed",
            log_text))
        return
    results.add("ok" if stopped else "fail", "lap line role", "status page answered (event %s, %s frames); %s" % (
        (status.get("info") or {}).get("event"), (status.get("processing") or {}).get("frames"), stop_text))


def check_zone_role(results, paths, video, timeout_s=180):
    from launcher.roles import RoleManager
    from zone.clock import SystemClock
    from zone.ptz.fake_camera import FakeViscaCamera, FakeViscaUdpServer
    from zone.ptz.sim import PtzModel, SimulatedPtz
    camera = FakeViscaUdpServer(FakeViscaCamera(SimulatedPtz(PtzModel(), SystemClock()))).start()
    try:
        web_port, agent_port = free_port(), free_port()
        settings = S.ZoneSettings(camera_device="0", ptz_host="127.0.0.1", web_port=web_port, agent_port=agent_port)
        S.write_zone(paths, settings)
        cfg = S.read_toml(paths.zone_config)
        cfg["spotter"]["source"] = str(video)            # the synthetic video, looped, instead of the webcam
        cfg["ptz"]["port"] = camera.address[1]           # the fake VISCA camera
        S.write_toml_file(paths.zone_config, cfg)
        roles = RoleManager(paths)
        ok, message = roles.start("zone")
        if not ok:
            results.add("fail", "zone role", message)
            return
        process = roles.processes["zone"]
        state_url = "http://127.0.0.1:%d/api/state" % web_port
        page = wait_for_http(state_url, timeout_s, is_alive=process.running)
        # the zone must also be processing spotter frames (its frame loop runs in a thread: the page answers even
        # when the loop crashed, until the process exits a moment later)
        state = wait_for_json(state_url, lambda data: (data.get("frame") or {}).get("size") is not None,
                              timeout_s, is_alive=process.running) if page is not None else None
        agent = wait_for_http("http://127.0.0.1:%d/agent/info" % agent_port, 30, is_alive=process.running) \
            if state is not None else None
        log_text = role_log(process)                   # read before the stop (which adds its own lines)
        tree = process_tree(process.proc.pid) if process.running() else []
        ok, message = roles.stop("zone")
        stopped, stop_text = stop_verdict(tree)
        if page is None or state is None or agent is None:
            what = ("page did not answer" if page is None else
                    "page answered but no spotter frame was processed" if state is None else
                    "studio API did not answer")
            results.add("fail", "zone role", "%s; log %s" % (what, log_text))
            return
        results.add("ok" if stopped else "fail", "zone role", "page and studio API answered, frame %s processed; "
                    "%s (%s)" % ("x".join(str(v) for v in state["frame"]["size"]), stop_text, message))
    finally:
        camera.stop()


def check_launcher_page(results, paths):
    from launcher.__main__ import serve
    from launcher.server import Launcher
    port = free_port()
    launcher = Launcher(paths, port, log=lambda m: None)
    server, thread = serve(launcher, port)
    try:
        body = wait_for_http("http://127.0.0.1:%d/api/state" % port, 30)
        page = wait_for_http("http://127.0.0.1:%d/" % port, 5)
        if body is None or page is None or b"Baja MV" not in page:
            results.add("fail", "launcher page", "did not answer")
        else:
            results.add("ok", "launcher page", "version %s" % json.loads(body)["version"])
    finally:
        server.should_exit = True
        thread.join(timeout=5)


# ============================================================================================== main
def run_self_test(keep=False, out=print):
    results = Results(out)
    real = resolve_paths()
    temp_root = Path(tempfile.mkdtemp(prefix="bajamv-selftest-"))
    paths = replace(real, data_root=temp_root)          # same code and tools, throw-away data folder
    paths.make_folders()
    out("Baja MV self-test: version %s, code %s, temporary data %s" % (real.version, real.code_root, temp_root))
    results.add("ok", "python", "%s %s (%s)" % (platform.python_version(), sys.executable, platform.platform()))
    check_code_location(results, real.code_root)
    if sys.platform == "win32":
        with step(results, "vc runtime"):
            check_vc_runtime(results, Path(sys.executable).parent)
        with step(results, "network profile"):
            check_network_profile(results)
    with step(results, "imports"):
        check_imports(results, real.code_root)
    models = {}
    with step(results, "models"):
        models = check_models(results, paths)
    with step(results, "openvino"):
        check_openvino(results, paths, models)
    with step(results, "opencv"):
        check_opencv(results)
    with step(results, "ffmpeg"):
        check_ffmpeg(results, paths)
    check_tray(results)
    video = None
    with step(results, "test video"):
        video = make_test_video(paths.ffmpeg, temp_root / "selftest.mkv")
    if video is not None:
        with step(results, "lap line role"):
            check_lapline_role(results, paths, video)
        with step(results, "zone role"):
            check_zone_role(results, paths, video)
    with step(results, "launcher page"):
        check_launcher_page(results, paths)
    (temp_root / "selftest.json").write_text(json.dumps(results.items, indent=1), encoding="utf-8")
    if results.failed:
        out("SELF-TEST FAILED: %d problem(s): %s" % (len(results.failed), ", ".join(i["name"] for i in results.failed)))
        out("details kept in %s" % temp_root)
        return 1
    out("SELF-TEST PASSED (%d checks)" % len(results.items))
    if not keep:
        shutil.rmtree(temp_root, ignore_errors=True)
    else:
        out("kept: %s" % temp_root)
    return 0
