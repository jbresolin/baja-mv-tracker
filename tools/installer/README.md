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

Runs in Windows PowerShell 5.1 (`powershell`, what CI uses too) and PowerShell 7.

Options: `-ModelsDir <folder>` (models elsewhere), `-FfmpegZip <zip or folder>` (a local ffmpeg instead of the
pinned download), `-VcRedist <vc_redist.x64.exe>` (a local copy instead of the download), `-Iscc <path>` (ISCC.exe not in its usual place), `-SkipCompile` (stage only, into `build\stage`),
`-AllowDummyModels` (CI only: accept models that don't match the manifest; the page then says "MODELOS DE TESTE").

What the script does (each step stops the build on error):

1. Downloads the pinned **Python 3.12** (python-build-standalone, a normal relocatable CPython: pip, `Lib\site-packages`,
   `pythonw.exe`; no `._pth` quirks like the embeddable zip) and **ffmpeg 8.0.1 essentials** (gyan.dev GPL build:
   libx264, h264_qsv, h264_amf) into `build\cache`, verifying the sha256 in `pins.json`. A cached file is reused.
2. Unpacks Python into `build\stage\python` and installs `requirements-runtime.txt` into it (every package pinned
   with hashes, wheels only). Downloads Microsoft's `vc_redist.x64.exe` (Visual C++ 2015-2022 x64 runtime) from
   `https://aka.ms/vs/17/release/vc_redist.x64.exe`. That link always serves the newest version, so it cannot be
   sha256-pinned: the file must carry a valid Microsoft Authenticode signature and be version 14.40 or newer
   (ONNX Runtime / OpenVINO crash in `std::mutex` with an older `msvcp140.dll`); its version goes into
   `build_info.json`. No runtime DLLs are copied next to `python.exe` (an old app-local copy would shadow a newer
   system one).
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
* Runs the bundled `vc_redist.x64.exe /install /quiet /norestart` when the PC's VC++ x64 runtime (registry
  `HKLM\SOFTWARE\[WOW6432Node\]Microsoft\VisualStudio\14.0\VC\Runtimes\x64`) is older than the bundled one.
* Optional (on by default): Windows Firewall rules for every profile (domain, private, public: a race LAN without
  internet is usually Public): TCP 8765, 8090, 8890, 8780, and the bundled `python.exe` / `pythonw.exe` (so Windows
  does not ask). Removed on uninstall. The self-test and the page warn when a network is marked Public anyway.
* Every Python start (BajaMV.cmd, shortcuts, the installer's own calls) uses `python -P`, and the roles get
  `PYTHONSAFEPATH=1` and the code folder as working folder: the installed code runs even when started from a folder
  that has its own `launcher\` (a repo checkout). The self-test fails ("code location") if it does not.
* Upgrade and uninstall first run `python -P -m launcher --shutdown`: a running Baja MV is asked to stop both roles
  and exit; roles running with no launcher (closed with "Fechar o painel") are found through their pid files and
  stopped by `--shutdown` itself (the zone with its safe PTZ stop). Then `app`, `python` and `ffmpeg` are replaced
  completely.
* Silent: `BajaMV-Setup-x.exe /VERYSILENT /SUPPRESSMSGBOXES /DATADIR=D:\BajaMV /TASKS=firewall,startatlogin`.

## CI smoke test

`.github/workflows/installer-smoke.yml` (windows-latest; manual or on `v*` / `installer-*` tags, not on every push):
builds with stand-in models (`make_dummy_models.py`: same inputs and output shapes, detect nothing), installs
silently, runs `BajaMV.cmd --self-test` from the checkout folder (code location, models in ONNX Runtime and OpenVINO, ffmpeg encoders, then each role
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
