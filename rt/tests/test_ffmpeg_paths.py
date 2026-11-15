"""ffmpeg and ffprobe must always be run from their configured path, never as a bare "ffmpeg" found on PATH.

Why this matters: the installed app (Program Files\\BajaMV) bundles its own ffmpeg in INSTALL\\ffmpeg\\bin, and that
folder is NOT on the Windows PATH. The launcher writes its full path into the configs (capture.ffmpeg,
capture.ffprobe). One place, the OBS clock overlay reader (lapvision/clock.py, called by rt/capture.py for every
video FILE source), still ran a bare "ffmpeg". On a PC without ffmpeg on PATH that raised FileNotFoundError, and both
rt (lap line) and the zone app crashed a second after starting on a file. The installer smoke test caught it; the
ordinary test runs did not, because there ffmpeg happened to be on PATH.

So the tests here:
  1. check that the overlay reader is handed the configured program,
  2. open a real video file with ffmpeg removed from PATH (only the configured full paths work), and
  3. scan the runtime code for any command that starts with a bare "ffmpeg" / "ffprobe".

Words used below, for a beginner:
  * PATH: the list of folders where the operating system looks for a program given only by its name.
  * monkeypatch: a pytest helper that changes something (an environment variable, a function) for one test only and
    puts it back afterwards.
"""
import ast
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]          # the repo folder (the one that contains rt/)

FFMPEG = shutil.which("ffmpeg")                     # full paths, looked up BEFORE any test hides PATH
FFPROBE = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="ffmpeg/ffprobe not on PATH")


def hide_path(monkeypatch, tmp_path):
    """Make PATH point at one empty folder: a bare "ffmpeg" can no longer be found, only a full path works."""
    empty = tmp_path / "empty_path_folder"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))


def make_video(path, seconds=3):
    """A small synthetic video (ffmpeg's colour test pattern, 10 frames per second). It has no clock overlay, so the
    overlay reader reads nothing and rt falls back to the file's creation/modification time, as on the self-test."""
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10",
                    "-t", str(seconds), "-c:v", "mpeg4", str(path)], check=True)
    return path


# ------------------------------------------------------------------------------------------ 1. the overlay reader
def test_overlay_reader_runs_the_configured_ffmpeg():
    from rt.capture import overlay_start
    programs_used = []

    def fake_grab(video, t, ffmpeg):
        # stands in for lapvision/clock.grab: remember which program it was told to run, return "no frame"
        programs_used.append(ffmpeg)
        return None

    result = overlay_start("some_video.mkv", r"C:\Program Files\BajaMV\ffmpeg\bin\ffmpeg.exe", log=lambda m: None,
                           grab=fake_grab)
    assert result is None                            # no frame -> no overlay time
    assert programs_used == [r"C:\Program Files\BajaMV\ffmpeg\bin\ffmpeg.exe"]


def test_file_start_epoch_hands_the_configured_ffmpeg_to_the_overlay_reader(tmp_path):
    from rt.capture import file_start_epoch
    video = tmp_path / "clip.mkv"
    video.write_bytes(b"not really a video")         # only its modification time is used in the end
    programs_used = []

    def fake_grab(video, t, ffmpeg):
        programs_used.append(ffmpeg)
        return None

    epoch, how = file_start_epoch(video, dict(duration=10.0), "auto", "/opt/ff/ffmpeg", log=lambda m: None,
                                  grab=fake_grab)
    assert programs_used == ["/opt/ff/ffmpeg"]
    assert how == "mtime-duration"                   # no overlay, no creation tag -> file time minus its length


def test_clock_grab_runs_the_program_it_is_given(monkeypatch):
    # lapvision/clock.grab builds an ffmpeg command line; the first word must be the program passed in.
    sys.path.insert(0, str(ROOT / "lapvision"))
    import clock
    commands = []

    class NoOutput:
        stdout = b""                                 # ffmpeg "wrote nothing": grab returns None

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return NoOutput()

    monkeypatch.setattr(clock.subprocess, "run", fake_run)
    assert clock.grab("v.mkv", 0.5, ffmpeg="/opt/ff/ffmpeg") is None
    assert commands[0][0] == "/opt/ff/ffmpeg"


# ------------------------------------------------------------------------------------------ 2. a real file, no PATH
@needs_ffmpeg
def test_file_source_opens_and_decodes_with_ffmpeg_off_path(cfg, tmp_path, monkeypatch):
    # This is what the lap line role does in the self-test: open a video FILE (here with the default clock mode
    # "auto", which tries the overlay reader first) and decode frames, with ffmpeg reachable only by full path.
    from rt.capture import FileCapture
    video = make_video(tmp_path / "clip.mkv")
    cfg["capture"]["ffmpeg"], cfg["capture"]["ffprobe"] = FFMPEG, FFPROBE
    cfg["capture"]["fps"] = 10.0
    assert cfg["clock"]["mode"] == "auto"
    hide_path(monkeypatch, tmp_path)
    assert shutil.which("ffmpeg") is None            # really hidden: a bare "ffmpeg" would fail now

    capture = FileCapture(cfg, video, log=lambda m: None)
    capture.start()
    frames = list(capture.frames())
    assert len(frames) >= 20                         # 3 s at 10 fps = 30 frames (allow a few lost at the end)
    assert isinstance(frames[0].img, np.ndarray)


# ------------------------------------------------------------------------------------------ 3. scan the code
RUNTIME_FOLDERS = ("rt", "zone", "launcher")
RUNTIME_LAPVISION_MODULES = ("clock.py",)            # the only lapvision module the runtime imports (rt/capture.py)


def runtime_files():
    """Every .py file the installed app may run: rt/, zone/, launcher/ (tests and dev tools excluded) and the
    lapvision modules the runtime imports."""
    files = []
    for folder in RUNTIME_FOLDERS:
        for path in sorted((ROOT / folder).rglob("*.py")):
            parts = path.relative_to(ROOT).parts
            if "tests" in parts or "__pycache__" in parts:
                continue
            files.append(path)
    files += [ROOT / "lapvision" / name for name in RUNTIME_LAPVISION_MODULES]
    return files


def bare_tool_commands(path):
    """The line numbers of command lists that START with the literal text "ffmpeg" or "ffprobe" (for example
    `subprocess.run(["ffmpeg", "-i", ...])`). A command that starts with a variable (the configured path) is fine."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.List) and node.elts:
            first = node.elts[0]
            if isinstance(first, ast.Constant) and first.value in ("ffmpeg", "ffprobe"):
                # a list such as ["ffmpeg", "-v", ...] is a command line; ["ffmpeg"] alone (a name) is not flagged
                if len(node.elts) > 1:
                    found.append(node.lineno)
    return found


def test_runtime_code_never_runs_a_bare_ffmpeg_or_ffprobe():
    offenders = {}
    for path in runtime_files():
        lines = bare_tool_commands(path)
        if lines:
            offenders[str(path.relative_to(ROOT))] = lines
    assert offenders == {}, ("these commands run ffmpeg/ffprobe by bare name (found on PATH only); use the "
                             "configured path (capture.ffmpeg / capture.ffprobe / recording.ffmpeg): %s" % offenders)


def test_the_scan_does_catch_a_bare_command(tmp_path):
    # The scan above must not pass simply because it is broken: show it finds a bare command in a sample file.
    sample = tmp_path / "sample.py"
    sample.write_text('import subprocess\nsubprocess.run(["ffmpeg", "-version"])\n'
                      'subprocess.run([ffmpeg_path, "-version"])\n', encoding="utf-8")
    assert bare_tool_commands(sample) == [2]
