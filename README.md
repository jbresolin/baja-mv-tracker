# Baja SAE lap vision

Machine-vision assistant for Baja SAE BRASIL endurance races: it detects each car passing the start line, reads its race
number, and emits a timestamped pass with a confidence, as a third counting source next to the human lap counters. The
same events can feed a PTZ live-stream camera ("car N just entered"). Everything that runs at the track is small ONNX
models on a laptop CPU. No GPU, no cloud.

## Results (26SE start line, 3 h of endurance, 252 human-counted laps)

| pipeline | laps found (recall / precision) | right number, always answering | auto-confirmed | runs on |
|---|---|---|---|---|
| teacher: YOLO-World v2-x + Qwen3-VL-4B | 0.933 / 0.975 | 67.9% | 98.7% correct, 65% of laps | GPU |
| **deployed: YOLO11n + panel detector + PARSeq-tiny** | **0.925 / 0.967** | **77.4%** | **142/142 offline, ~99% online, 56% of laps** | **laptop CPU, ~2 cores** |

- Real-time replay of the whole race through `rt/`: 0 dropped frames on 2 pinned cores through the finish, ~260 MB RAM.
  The number arrives ~1 s after the car leaves the frame; a provisional number arrives ~0.7 s after it enters (98.5% correct).
- Weak spots: a car fully caked in mud (car 1) is read but never auto-confirmed; heavily stylised digits; end-of-race clutter.
- Caveat: 26SE was used to compare many variants, so it is partly a development set. The next event is the real test.

## How it was built

```
YouTube live streams (9 videos, 5 events, 42 h)                    rosters scraped from resultados.bajasaebrasil.net
        |
  TEACHER (offline, RTX 2080 Ti)  teacher/
    shots + CLIP shot tags + overlay masks -> YOLO-World "off-road buggy" @4 fps -> per-shot tracker + flags
    -> Qwen3-VL-4B number reads on the best crops -> VLM yes/no car check -> grounded number boxes
    -> broadcast lower thirds read as free identity labels -> consensus labels (~97% precise, checked twice)
        |
  DATASETS  teacher/export.py -> datasets/ (not in git)
    cars 46k frames | number panels 5k (+30k proposed) | digits 5.5k | ids 52k crops / 133 identities
    split by whole event: train 19SU 25BR 26BR, val 25SU, test 24SU; final models retrained on all 5
        |
  STUDENTS  students/
    car detector YOLO11n/s @640 (exported 384x640) -> number-panel detector YOLO11n @320 on each car crop
    -> PARSeq-tiny digit recognizer (scene-text pretrained + digit splicing; 84% on numbers never seen in training)
    -> crop-count-independent vote per pass, snapped to the event roster, abstain below a threshold picked on val
        |
  REAL TIME  rt/   capture -> detect @10 fps -> ByteTrack -> online pass logic -> async identity -> JSONL / CSV / sinks
```

Key lessons (details in `LESSONS.md`):
- COCO detectors see muddy baja cars as "bear" or "cow". The open-vocabulary YOLO-World prompted "off-road buggy" works.
- The recording PC's clock was 80 s off the race server. NTP-sync it, always.
- A recognizer trained only on the ~50 numbers in the footage memorised whole numbers: 55% on unseen numbers. Splicing real
  digits into new numbers plus a scene-text-pretrained model fixed it (84%).
- Broadcast lower thirds ("#35 Carpoeira Baja") are free, independent identity labels.

## Repository layout

| path | what |
|---|---|
| `rt/` | **the track-side program** (see `rt/README.md`). `run.py`, `bench_laptop.py`, `config.toml`, `sinks.py` (lap-app stub) |
| `deploy/models/manifest.json` | the deployed model set: files, preprocessing, every threshold, provenance (`*.onnx` not in git) |
| `teacher/` | offline labelling pipeline: `manifest` `shots` `shot_tags` `overlay` `detect` `track` `overlay_ids` `crops` `read` `verify` `labels` `export`, plus `rosters.py` |
| `students/` | detector training (`train_det.py`), 26SE harnesses (`eval_26se.py`, `e2e_26se.py`), CPU bench, ONNX checks |
| `students/recog/` | number recognizer: CRNN and PARSeq training, digit splicing, held-out-number protocol, vote and threshold picking |
| `tools/qa/` | blind review page for teacher labels (keyboard or touch, LAN mode with an access token) and its precision report |
| `launcher/` | **Baja MV**: the tray app + local page (pt-BR) that configures and runs a laptop's role(s); `--self-test` |
| `tools/installer/` | the Windows installer build (`build_installer.ps1`, Inno Setup script, pins); see its README |
| `lapvision/` | session-1 start-line pipeline (OBS clock reader, coarse timeline, tracking, pass rules, VLM reads, GT evaluation) |
| `rosters/` | car number to team for 60 events (2003–2026) |
| `data/` | stream manifest, event splits, per-video summaries |
| `LESSONS.md` | full findings log with numbers, including corrections. **Read this before changing anything.** |
| `TRAINING_PLAN.md` | the plan, decisions and status |

Not in git (regenerable or large): videos, `youtube/`, `work/` (teacher outputs), `datasets/`, `out/`, training runs,
model weights, `deploy/models/*.onnx` (~84 MB).

## Running

```
# environment: Python 3.13, torch cu128 for training only, see requirements.txt; ffmpeg on PATH
python rt/run.py --source <camera index | rtsp url | video file> --event 26SE --out passes.jsonl
python rt/bench_laptop.py --video <recording> --sustain 120 --hwaccel d3d11va    # go / no-go for a candidate laptop
python tools/qa/serve.py out/qa/<EV>_teacher.jsonl [--host 0.0.0.0]              # review teacher labels
```

The track-side runtime needs only `onnxruntime`, `numpy`, `opencv-python` and `ffmpeg`. OpenVINO is an option for the Intel
iGPU. The worst-case target is a 4-core 8th-gen Intel laptop with 8 GB of RAM. YOLO11n fits it; YOLO11s does not.

## Installing on a laptop (Windows installer, AUTOPTZ-26)

The race laptops don't need git, a venv, pip, winget or hand-copied models: one installer has everything (a private
Python 3.12 with every package, the code, the models, ffmpeg with x264 / QuickSync / AMF).

**Build it** (on the PC that has the real models in `deploy\models\`): see `tools/installer/README.md`. In short,
`winget install JRSoftware.InnoSetup` once, then
`powershell -ExecutionPolicy Bypass -File tools\installer\build_installer.ps1` -> `dist\BajaMV-Setup-<version>.exe`
(the build refuses models that don't match `deploy/models/manifest.json`).

**Install**: copy the .exe to the laptop and run it (admin). It asks for the **data folder** (default `C:\BajaMV`:
configs, recordings, logs; put it on the disk with the most space) and offers two options: firewall rules for the
app's ports (8765, 8090, 8890, 8780) on every network type, private and public (a race LAN without internet is
usually "Public" for Windows; leave it on), and "open Baja MV at login". It installs to `C:\Program Files\BajaMV`,
installs Microsoft's Visual C++ runtime when the laptop's is older than the bundled one, and adds Start-menu entries:
**Baja MV**, **Baja MV (autoteste)**, **Pasta de dados**.

**First run, on each laptop**: run "Baja MV (autoteste)" once (all lines OK; a WARN about no Intel GPU is fine on the
XPS; a WARN "network profile" means the race network is marked Public: if a phone or the other laptop cannot open
the pages, run the `Set-NetConnectionProfile ... -NetworkCategory Private` command it prints in an admin PowerShell;
the page shows the same warning). Then start **Baja MV**: a red "MV" icon appears in the tray and the page opens (`http://localhost:8780/`; from
a phone on the same network: `http://<laptop IP>:8780/`). On the page:

* *Este computador*: tick the role (Linha de chegada on the T14, Zona PTZ on the XPS; both only on a bench laptop),
  optionally "iniciar esses papéis quando o Baja MV abrir" and a PIN for the page. Save.
* **Lap line (T14)**: event code (e.g. `27SE`) and **Baixar lista de equipes** (the results site lists the teams only
  shortly before the event; until then rt reads any number, without auto-confirm); camera (pick it from the list:
  the device path is saved, so identical webcams don't swap; **Prévia** shows a frame); resolution; NTP server = the
  event router's IP; car detector (Intel iGPU by default). Save, then **Iniciar**. The status page link (:8765) and
  the last log lines are on the card. The beam is still set up from the status page (Beam setup), as before.
* **Zone PTZ (XPS)**: spotter camera + preview, resolution, NTP, PTZ IP, characterization report, profile (a bench
  profile shows a loud warning: never at the race), page / studio API ports and the **studio token** (set one at the
  event; the page never shows it back: leave the field empty to keep it, type a single space to remove it). Save, then **Iniciar**; teach and operate on the zone page (:8090) as before. **Parar** first asks the zone
  for PARAR, then stops the process, then sends the PTZ a VISCA stop itself, so the camera never keeps moving.
* **Caracterização do PTZ** (zone laptop, zone stopped): IP + folder name -> the bench tool opens in a console window
  on the laptop; confirm the move and answer its questions **there**; the page shows its output. The report lands in
  `<data>\reports\<folder>\report.json` and appears in the zone's report list.
* **Exportar logs** makes a zip (logs, configs with tokens blanked, the newest lap-line run and zone log; no video)
  to send back; **Abrir pasta de dados** opens the data folder.

The roles still run through `rt/supervise.py` and `zone/supervisor.py` (restart with backoff, stall checks, the zone
always back in STOPPED). Closing the page or the tray icon ("Fechar o painel") leaves them running; the next Baja MV
finds them again. "Parar tudo e sair" stops both.

**Where the data lives** (`C:\BajaMV` by default):

```
config\    lapline.toml, zone.toml (written by the page; other keys you add by hand are kept, comments are not),
           lapline.beam.toml (the status page's Beam setup), launcher.json
out\       lapline\<date>\lapline_<time>.jsonl/.csv/.stats.json + rec\ (video), lapline\launcher.log;
           zone\ (zone_*.jsonl control logs, spotter / PTZ recordings)
logs\      launcher.log, lapline.log, zone.log, ptz_recorder.log, characterize_*.log
rosters\   rosters fetched with the button (used before the ones bundled with the code)
zones\     zone.json (the taught zone), zone.agent.json, zone.speed.json, zone_backups\
reports\   PTZ characterization reports
exports\   zips from "Exportar logs"
```

**Upgrade**: run a newer installer on top. It stops a running Baja MV and both roles (also roles left running after
"Fechar o painel": the zone still gets its safe PTZ stop), replaces the program, and keeps
the data folder and everything in it (the data-folder question is skipped). Start Baja MV again.

**Uninstall**: Settings > Apps > Baja MV. The program and the firewall rules go; the data folder stays (delete it by
hand if you really want to).

**Repo mode** (development) is unchanged: `rt\run_lapline.bat`, `zone\run_zone.bat` and the configs under
`rt/configs` / `zone/configs` work as before. `python -m launcher` also runs from a checkout (data under
`out/bajamv/`, ffmpeg from PATH).

**Race-day runbook, conceptually** (replaces "clone, venv, pip, copy models, winget ffmpeg, copy TOML, run .bat"):
install -> autoteste -> Baja MV -> pick the role -> fill the card -> (lap line: Baixar lista de equipes; zone:
characterization report if it changed) -> Salvar -> Iniciar -> open the role's own page from the card for everything
else (beam setup, teach, operation). For hands-off restarts after a reboot: install with "open at login" and tick
"iniciar esses papéis quando o Baja MV abrir".

## Status and open items

Done: teacher, datasets, students, the 26SE evaluation, the real-time prototype, final models on all events.
Open (need hardware or decisions): benchmark on the real laptop, the lap-app integration (API vs DB insert;
`rt/sinks.py` is a stub), the PTZ interface and scope, the camera connection and fps, re-ID for mud cars, a pass-length
minimum in seconds.

Third-party components: Ultralytics YOLO (AGPL-3.0), YOLO-World, Qwen3-VL (Apache-2.0), PARSeq (Apache-2.0), CLIP.
