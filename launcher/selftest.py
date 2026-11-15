"""`python -m launcher --self-test`: check an installation end to end, without a camera or a PTZ. Exit code 0 = all
good. The CI smoke test runs it right after a silent install; run it on a laptop after installing, too.

Checks, in order (each prints [ OK ], [WARN] or [FAIL] with a detail):
  1. Python: version and where it runs from.
  2. Imports: every module of rt/, zone/ and launcher/ (tests excluded).
  3. Models: each model of deploy/models/manifest.json loads in ONNX Runtime; the car detector runs once.
  4. OpenVINO: imports, lists its devices, compiles the car detector on CPU (and says whether a GPU is there).
  5. OpenCV: encodes an image.
  6. ffmpeg / ffprobe: run, and the build has libx264, h264_qsv and h264_amf (rt probes them in that order).
  7. Tray: pystray + Pillow import and the icon can be drawn.
  8. Lap line role: started through the launcher's own RoleManager (rt/supervise.py -> rt/run.py) on a synthetic
     video instead of a camera; its status page must answer; then stopped (the process tree must be gone).
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
from launcher.paths import resolve_paths

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


# ============================================================================================== the checks
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
    body = wait_for_http("http://127.0.0.1:%d/status.json" % port, timeout_s, is_alive=process.running)
    roles.stop("lapline")
    stopped = not process.running()
    if body is None:
        results.add("fail", "lap line role", "status page never answered; log: %s" % " | ".join(
            process.status(15)["log"]))
        return
    status = json.loads(body.decode("utf-8"))
    results.add("ok" if stopped else "fail", "lap line role", "status page answered (event %s, fps %s); %s" % (
        (status.get("info") or {}).get("event"), (status.get("processing") or {}).get("fps"),
        "stopped" if stopped else "DID NOT STOP"))


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
        page = wait_for_http("http://127.0.0.1:%d/api/state" % web_port, timeout_s, is_alive=process.running)
        agent = wait_for_http("http://127.0.0.1:%d/agent/info" % agent_port, 30, is_alive=process.running) \
            if page is not None else None
        ok, message = roles.stop("zone")
        stopped = not process.running()
        if page is None or agent is None:
            results.add("fail", "zone role", "%s did not answer; log: %s" % (
                "page" if page is None else "studio API", " | ".join(process.status(15)["log"])))
            return
        results.add("ok" if stopped else "fail", "zone role", "page and studio API answered; %s (%s)" % (
            "stopped" if stopped else "DID NOT STOP", message))
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
