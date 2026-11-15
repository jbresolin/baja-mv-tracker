# Windows installer (AUTOPTZ-26)

Builds `dist\BajaMV-Setup-<version>.exe`: one installer for both laptop roles (lap line, zone PTZ). How to install
and use it on a laptop: the main README, "Installing on a laptop".

## Building (on the Windows PC that has the models)

Once:

```
winget install JRSoftware.InnoSetup
```

Each build, from the repo folder, with the real models in `deploy\models\` (car_n, car_s, panel, recog; plus
yolo11n_coco.onnx if the bench detector should be included):

```
powershell -ExecutionPolicy Bypass -File tools\installer\build_installer.ps1
```

Output: `dist\BajaMV-Setup-<commit date>-<commit>.exe` (~250-350 MB). `-dirty` in the name = the working tree had
uncommitted changes to tracked files; **new files that are not committed yet are not included** (only files git
tracks are staged, so local configs and outputs never leak into the installer).

Options: `-ModelsDir <folder>` (models elsewhere), `-FfmpegZip <zip or folder>` (a local ffmpeg instead of the
pinned download), `-Iscc <path>` (ISCC.exe not in its usual place), `-SkipCompile` (stage only, into `build\stage`),
`-AllowDummyModels` (CI only: accept models that don't match the manifest; the page then says "MODELOS DE TESTE").

What the script does (each step stops the build on error):

1. Downloads the pinned **Python 3.12** (python-build-standalone, a normal relocatable CPython: pip, `Lib\site-packages`,
   `pythonw.exe`; no `._pth` quirks like the embeddable zip) and **ffmpeg 8.0.1 essentials** (gyan.dev GPL build:
   libx264, h264_qsv, h264_amf) into `build\cache`, verifying the sha256 in `pins.json`. A cached file is reused.
2. Unpacks Python into `build\stage\python` and installs `requirements-runtime.txt` into it (every package pinned
   with hashes, wheels only), then copies the Visual C++ runtime DLLs from this PC's System32 next to `python.exe`
   (ONNX Runtime / OpenVINO need `msvcp140*.dll`; a fresh Windows may lack them; Microsoft allows this app-local
   copy).
3. Puts `ffmpeg.exe` and `ffprobe.exe` into `build\stage\ffmpeg\bin` (the configs the launcher writes point there, so
   PATH does not matter).
4. `stage_app.py`: copies the runtime code (rt, zone, launcher, rt/config.toml and the example configs, lapvision's
   clock reader, teacher/rosters.py, rosters, the manifest) into `build\stage\app`, checks every model against the
   manifest's `sha256_16` (a missing or different model fails the build), writes `build_info.json` (version, commit,
   model hashes) and the `.pth` that puts `app\` on the private Python's path; then the code is byte-compiled.
5. Compiles `BajaMV.iss` with Inno Setup.

Re-running is safe: `build\stage` is rebuilt from scratch every time.

## What the installer does

* Per machine, admin: `C:\Program Files\BajaMV\{python, ffmpeg, app}` + `BajaMV.cmd`.
* Asks for the data folder (default `C:\BajaMV`), creates it writable for all users, writes it to
  `app\install.ini`. Kept on upgrade (the question is skipped) and on uninstall.
* Start menu: Baja MV (`pythonw.exe -m launcher`, no console), Baja MV (autoteste) (`BajaMV.cmd --self-test`),
  Pasta de dados. Optional: Baja MV in the all-users Startup folder ("open at login").
* Optional (on by default): Windows Firewall rules for private networks: TCP 8765, 8090, 8890, 8780, and the bundled
  `python.exe` / `pythonw.exe` (so Windows does not ask). Removed on uninstall.
* Upgrade = run a newer installer: it first asks a running Baja MV to stop both roles and exit
  (`python -m launcher --shutdown`), then replaces `app`, `python` and `ffmpeg` completely.
* Silent: `BajaMV-Setup-x.exe /VERYSILENT /SUPPRESSMSGBOXES /DATADIR=D:\BajaMV /TASKS=firewall,startatlogin`.

## CI smoke test

`.github/workflows/installer-smoke.yml` (windows-latest; manual or on `v*` / `installer-*` tags, not on every push):
builds with stand-in models (`make_dummy_models.py`: same inputs and output shapes, detect nothing), installs
silently, runs `BajaMV.cmd --self-test` (imports, models in ONNX Runtime and OpenVINO, ffmpeg encoders, then each role
started and stopped through the launcher on a synthetic video, the zone against a fake VISCA camera), runs the rt /
zone / launcher test suites on the installed Python, re-installs on top (data kept) and uninstalls (data kept,
firewall rules gone).

```
gh api -X POST repos/jbresolin/baja-mv-tracker/actions/workflows/installer-smoke.yml/dispatches -f ref=<branch>
```

## Updating pins

* Python: pick a newer `cpython-3.12.x+<date>-x86_64-pc-windows-msvc-install_only_stripped.tar.gz` from the
  python-build-standalone releases; its sha256 is in the release's `SHA256SUMS` (or in uv's download metadata).
* ffmpeg: a newer gyan.dev release from github.com/GyanD/codexffmpeg/releases (`-essentials_build.zip`); the sha256 is
  in the winget manifest `manifests/g/Gyan/FFmpeg/Essentials/<version>/` of microsoft/winget-pkgs.
* Packages: edit `requirements-runtime.in`, then regenerate the locked file (on any OS, with uv):
  `uv pip compile tools/installer/requirements-runtime.in --python-platform x86_64-pc-windows-msvc
  --python-version 3.12 --only-binary :all: --generate-hashes -o tools/installer/requirements-runtime.txt`
