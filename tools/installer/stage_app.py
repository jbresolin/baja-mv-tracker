"""Stage the app's own files for the installer (called by build_installer.ps1; stdlib only, runs on any OS).

usage: python tools/installer/stage_app.py --repo <repo> --stage <stage dir> --models <folder with the .onnx files>
           [--allow-dummy-models] [--ffmpeg-version TEXT] [--python-dir <stage>/python]

What it does:
  1. Copies the runtime code into <stage>/app: rt/, zone/, launcher/ (tests left out), rt/config.toml and the
     *.example.toml configs, lapvision/clock.py + its templates (rt reads the OBS clock overlay of 26SE-style
     recordings with it), teacher/rosters.py (the roster button), rosters/*.csv, deploy/models/manifest.json.
     Only files git tracks are copied (a local site.toml, outputs or caches never end up in the installer).
  2. Copies the ONNX models listed in the manifest from --models, and checks each one against the manifest's
     sha256 prefix (sha256_16). A missing or different model stops the build, unless --allow-dummy-models (CI's
     stand-in models: then only their presence is checked, and build_info.json says so). yolo11n_coco.onnx (the
     zone's bench detector) is optional: copied when present.
  3. Writes <stage>/app/build_info.json (version, commit, models) - its presence is what tells the launcher it runs
     installed (launcher/paths.py) - and, with --python-dir, the .pth file that puts <stage>/app on the private
     Python's import path.
"""
import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

CODE_FOLDERS = ("rt", "zone", "launcher")
EXTRA_FILES = ("lapvision/clock.py", "teacher/rosters.py", "deploy/models/manifest.json")
EXTRA_FOLDERS = ("lapvision/templates", "rosters")
LEFT_OUT_PARTS = {"tests", "__pycache__"}
LEFT_OUT_FILES = {"rt/eval_26se.py"}               # development-only (pandas, the 26SE footage)
OPTIONAL_MODELS = ("yolo11n_coco.onnx",)


class StageError(Exception):
    """Something that must stop the build, with a message saying what to fix."""


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_files(repo):
    """Files git tracks (relative posix paths), or None when git is not available / not a checkout."""
    try:
        result = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return {name for name in result.stdout.decode("utf-8").split("\0") if name}


def wanted(relative):
    """Should this repo file (posix path) be in the installer?"""
    parts = relative.split("/")
    if LEFT_OUT_PARTS & set(parts) or relative in LEFT_OUT_FILES:
        return False
    if parts[0] in CODE_FOLDERS:
        if parts[-1].endswith(".toml") and "/configs/" in "/" + relative:
            return parts[-1].endswith(".example.toml")       # never a local site.toml / lapline.toml
        return True
    if relative in EXTRA_FILES:
        return True
    return any(relative.startswith(folder + "/") for folder in EXTRA_FOLDERS)


def list_app_files(repo, tracked=None):
    """The repo files to stage (relative posix paths, sorted)."""
    repo = Path(repo)
    if tracked is None:                                   # no git: walk the folders instead
        tracked = {p.relative_to(repo).as_posix() for folder in CODE_FOLDERS + EXTRA_FOLDERS
                   for p in (repo / folder).rglob("*") if p.is_file()}
        tracked |= {name for name in EXTRA_FILES if (repo / name).exists()}
    return sorted(name for name in tracked if wanted(name) and (repo / name).is_file())


def check_models(manifest, models_dir, allow_dummy):
    """-> [(file name, sha256)] of the models to ship. Raises StageError when one is missing or different."""
    models_dir = Path(models_dir)
    problems, chosen = [], []
    for key, info in manifest["models"].items():
        path = models_dir / info["file"]
        if not path.exists():
            problems.append("%s: %s is missing" % (key, path))
            continue
        digest = sha256_of(path)
        expected = str(info.get("sha256_16", "")).lower()
        if not allow_dummy and expected and not digest.startswith(expected):
            problems.append("%s: %s has sha256 %s..., the manifest says %s (wrong or old model file)" % (
                key, path, digest[:16], expected))
            continue
        chosen.append((info["file"], digest))
    for name in OPTIONAL_MODELS:
        if (models_dir / name).exists():
            chosen.append((name, sha256_of(models_dir / name)))
    if problems:
        raise StageError("model check failed:\n  " + "\n  ".join(problems))
    return chosen


def git_version(repo):
    """'<commit date YYYY.MM.DD>-<short hash>[-dirty]' and the full hash, or ('dev', '')."""
    def git(*args):
        result = subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True, text=True, timeout=30)
        return result.stdout.strip() if result.returncode == 0 else ""
    try:
        full = git("rev-parse", "HEAD")
        date = git("show", "-s", "--format=%cs", "HEAD").replace("-", ".")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return "dev", ""
    if not full:
        return "dev", ""
    return "%s-%s%s" % (date, full[:7], "-dirty" if dirty else ""), full


def write_pth(python_dir):
    """<python>/Lib/site-packages/bajamv.pth -> ../../../app, i.e. INSTALL\\app (a relative line is resolved
    against site-packages), so `python -m launcher` works from any folder."""
    site_packages = Path(python_dir) / "Lib" / "site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)
    (site_packages / "bajamv.pth").write_text("../../../app\n", encoding="utf-8")


def write_icon(app_dir):
    """bajamv.ico for the Start menu (the tray icon's drawing); skipped when Pillow is not there."""
    try:
        sys.path.insert(0, str(app_dir))
        from launcher.tray import make_icon_image
        make_icon_image(256).save(str(Path(app_dir) / "bajamv.ico"), sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
        return True
    except Exception as e:  # noqa: BLE001
        print("icon not written (%s: %s)" % (type(e).__name__, e))
        return False
    finally:
        sys.path.remove(str(app_dir))


def stage(repo, stage_dir, models_dir, allow_dummy=False, ffmpeg_version="", python_dir=None, version=None,
          now=None):
    repo, stage_dir = Path(repo), Path(stage_dir)
    app = stage_dir / "app"
    if app.exists():
        shutil.rmtree(app)
    files = list_app_files(repo, git_files(repo))
    if not files:
        raise StageError("no files to stage under %s" % repo)
    for name in files:
        target = app / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)
    manifest = json.loads((repo / "deploy" / "models" / "manifest.json").read_text(encoding="utf-8"))
    models = check_models(manifest, models_dir, allow_dummy)
    for name, _ in models:
        shutil.copy2(Path(models_dir) / name, app / "deploy" / "models" / name)
    git_ver, commit = git_version(repo)
    info = dict(version=version or git_ver, commit=commit,
                built=(now or dt.datetime.now()).astimezone().isoformat(timespec="seconds"),
                models_note=manifest.get("note", ""), dummy_models=bool(allow_dummy),
                models={name: digest for name, digest in models}, ffmpeg=ffmpeg_version,
                python="%d.%d.%d" % sys.version_info[:3], files=len(files))
    (app / "build_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if python_dir:
        write_pth(python_dir)
    write_icon(app)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--allow-dummy-models", action="store_true")
    ap.add_argument("--ffmpeg-version", default="")
    ap.add_argument("--python-dir", default=None)
    ap.add_argument("--version", default=None, help="override the version (default: git commit date + hash)")
    args = ap.parse_args(argv)
    try:
        info = stage(args.repo, args.stage, args.models, args.allow_dummy_models, args.ffmpeg_version,
                     args.python_dir, args.version)
    except StageError as e:
        print("STAGING FAILED: %s" % e, file=sys.stderr)
        return 1
    print("staged %d files, version %s%s" % (info["files"], info["version"],
                                            " (DUMMY MODELS)" if info["dummy_models"] else ""))
    print("VERSION=%s" % info["version"])          # build_installer.ps1 reads this line
    return 0


if __name__ == "__main__":
    sys.exit(main())
