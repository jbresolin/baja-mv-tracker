# zone/: spotter-driven PTZ tracking (AUTOPTZ-10)

This package holds WP-Z1 (AUTOPTZ-15): the PTZ driver, a simulated PTZ and the bench script that characterizes a
real camera (Python standard library only), and WP-Z2 (AUTOPTZ-16, -17, -18): the zone tracker, click-teach and
controller (numpy, OpenCV, ONNX Runtime and FastAPI; see "WP-Z2" below).

## WP-Z1: PTZ driver, simulator and bench

| Module | What it is |
| --- | --- |
| `ptz/interface.py` | The `Ptz` interface (raw device units, signed speed steps) and `CommandResult` |
| `ptz/visca.py` | `ViscaPtz`: VISCA over IP (UDP 52381), dead-man TTL, JSONL datagram log |
| `ptz/visca_packets.py` | Packet bytes and reply parsing (pure functions) |
| `ptz/sim.py` | `SimulatedPtz` + `PtzModel`: a camera model driven by an injected clock |
| `ptz/fake_camera.py` | A fake VISCA camera on top of the simulator (in-memory, UDP, or standalone) |
| `tools/characterize.py` | The bench script |
| `clock.py` | `SystemClock` and `FakeClock` |

## Running the bench on a camera

From the repo root, with the camera on the bench and free to move:

```
python -m zone.tools.characterize --ip 192.168.1.50 --out reports/smtav-a/
```

- It asks before the first move and starts from the VISCA home position. Use `--home-preset N` (0-based) to start
  from a preset instead; `--repeat-preset N` picks the preset for the repeatability test.
- A full run takes roughly 15-25 minutes. `--quick` measures ~10 speed steps per axis instead of all of them.
- During the zoom section it stops at 9 zoom positions for FOV screenshots (`--no-fov-pauses` to skip). At the end it
  asks for two marks a known angle apart to get units per degree. Type nothing to skip, or pass
  `--pan-units-per-degree` if the camera documents it.
- `--skip zoom,units` skips sections. Ctrl+C stops the camera and keeps what was measured.
- Output: `report.json` (everything, plus a `model` section the simulator loads with `PtzModel.from_report`),
  `summary.txt` (ends with the Branch A/B verdict), `visca_log.jsonl` (every datagram, monotonic and wall clock).

Dry run without a camera, in two terminals:

```
python -m zone.ptz.fake_camera --port 52399 [--inquiry-mode at_rest]
python -m zone.tools.characterize --ip 127.0.0.1 --port 52399 --out reports/fake --quick --yes
```

## How the bench measures

- **Feedback.** It pans at a fixed speed and polls position at 5/10/20/50 Hz. A rate is "ok" when ≥ 90% of the
  inquiries are answered, the answers arrive at ≥ 80% of the asked rate, and the value changes. While moving, the
  inquiry is classified as `yes`, `frozen` (answers, but the value doesn't change until the camera stops) or `no`.
- **Speed table.** With live feedback, it moves at each step and fits the slope of the position over the second
  half of the move. Without it (position readable only at rest), it makes two timed moves of T and 2T per direction:
  (travel(2T) − travel(T)) / T is the steady rate with the latency and ramp cancelled out.
- **Latency.** With live feedback at ≥ 10 Hz, it polls around the command: time to the first change, time to 90%
  of the steady rate, then the same for stop. A camera that answers only at rest goes silent when it starts moving
  and answers again once it has stopped; those two edges time the start latency and the stop. The resolution of the
  stop edge is the driver timeout (`--timeout`, 0.2 s by default).
- **Verdict.** Branch A if pan/tilt inquiry works while moving at ≥ 10 Hz, otherwise Branch B.

## Tests

```
python -m pytest zone
```

Everything runs on a fake clock (no real sleeps) except a few UDP round trips on 127.0.0.1. Full bench runs execute
against the simulator in well under a second. The WP-Z2 tests need `numpy opencv-python fastapi httpx` (and ffmpeg
for the replay / detection-check tests, which are skipped without it).

## WP-Z2: zone tracker, click-teach and controller (AUTOPTZ-16, -17, -18)

A fixed spotter webcam watches one zone of the track. The PTZ follows one car at a time through it, then goes back
to the entry and takes the next car. Tracking is 1D: each car becomes its progress `s` along a taught path (0 = entry
keyframe, 1 = exit keyframe) and its speed `v`. The PTZ pose is interpolated from a few taught keyframes as a function
of `s`, so there is no camera calibration. The PTZ driver and simulator (`zone/ptz/`) are WP-Z1's.

```
spotter frame -> car detector on the corridor ROI (rt/) -> ByteTrack (rt/) -> s, v per car (Kalman)
  -> director (which car, when) -> controller (pan/tilt speed steps, zoom) -> PTZ (VISCA, or SimulatedPtz)
```

| module | what |
|---|---|
| `zonefile.py` | `zone.json`: path points, keyframes (pose + preset), corridor; save with backup |
| `path.py` | `s` along the path, projection with a monotonic window, pose(s) (pchip / linear, with ease-in) |
| `spotter.py` | corridor ROI crop (`downscale` / `tile` / `full`) in front of the rt detector, then rt's ByteTrack |
| `bench_detector.py` | bench only: a stock COCO YOLO11 in place of the Baja detector (`spotter.detector = "coco"`) |
| `progress.py` | s/v Kalman filter per track, speed profile learned from finished passes (`zone.speed.json`) |
| `director.py` | STOPPED / PASSIVE / AUTO (waiting, tracking, returning) and the lock / drop rules |
| `speedmap.py` | wanted rate -> speed step: inverse table, hysteresis, ramp, minimum smooth step |
| `controller.py` | the control law, Branch A (inquiry feedback) / B (dead reckoning), zoom, TTL on every command |
| `teach.py` | click-teach edits and the three pose sources (inquiry / commanded absolute / dead reckoning) |
| `runtime.py` | one control cycle per frame + the JSONL control log (used live and by the replay) |
| `app.py`, `static/` | the process and its page (FastAPI, pt-BR): teach, operation, health |
| `recorder.py`, `tools/record_ptz.py` | spotter recording (frames already grabbed) and PTZ USB `-c copy` |
| `supervisor.py`, `run_zone.bat` | launcher: restart with backoff; always back in STOPPED |
| `tools/replay.py` | offline: real pipeline on a recording -> SimulatedPtz on a fake clock -> framing error per pass |
| `tools/detect_check.py` | AUTOPTZ-16: one continuous track per car? per ROI mode, with timing |

### Install (laptop, Windows)

Python 3.11+, ffmpeg/ffprobe on PATH, the deployed models in `deploy/models/` (see `rt/README.md`). Then:

```
pip install onnxruntime numpy opencv-python psutil fastapi uvicorn
```

Copy `zone/configs/zone.example.toml` to `zone/configs/site.toml` and keep only the keys you change (camera index,
PTZ IP, report path, recording). `zone\run_zone.bat` picks `site.toml` up by itself.

### Running

* `zone\run_zone.bat [config.toml]` starts the tracker (this window) and the PTZ recorder (a second window, only if
  `recording.ptz_usb = true`). Each restarts with backoff when it dies. The tracker always comes back in **STOPPED**.
* Page: `http://<laptop>:8090/`. Buttons: PARADO / PASSIVO / AUTO and the big **PARAR** (also the Esc key).
* Without the launcher: `python -m zone.app --config zone/configs/site.toml`.

### Teaching a zone (~15 min for 5-8 keyframes)

1. Mode PARADO or PASSIVO (teaching is refused in AUTO), then "Modo ensino" on: the page moves the PTZ (jog, nudge,
   home, "testar") only in teach mode. PARAR, PARADO and AUTO turn it off (a held jog is cancelled with it).
2. "Adicionar ponto", click the racing line upstream of the zone (the approach, where `v` is measured).
3. Jog the PTZ (buttons or arrow keys, PageUp/PageDown for zoom) to frame a car at the entry. Click "Adicionar
   keyframe", then click that spot in the spotter image. The pose is read and a preset is stored (10, 11, ...).
4. Repeat along the zone, with path points in between to follow curves, until the exit keyframe.
5. Check: "testar" recalls a keyframe's preset; "re-ensinar" stores the current framing in it. Points can be moved
   (select in the list, "Mover ponto", click), deleted and reordered. Set the corridor half-width.
6. "Salvar zona". The previous file goes to `zone_backups/`.

The first keyframe is s = 0, the last is s = 1. Teaching on a recording with the PTZ simulated also works:
`--set spotter.source="rec.mp4" --set ptz.driver=sim`.

### Spotter capture latency (measure it on the bench)

The webcam delivers each frame some time after the picture was taken (exposure, MJPG, USB, driver): typically
~0.1 s. `rt.capture.live_latency_s` (default 0.1) is subtracted from every live frame's time, so the controller's
lookahead covers it (lookahead = capture latency + processing time + `controller.latency_s`, the PTZ command latency).
Measure it once per webcam, in its real mode (1080p30 MJPG), and put the value in `site.toml`:

* **LED method:** point the spotter at an LED that a script on the laptop toggles (or the laptop's caps-lock LED).
  Log the toggle time and the time of the first frame that shows the change (`zone_<start>.jsonl` has every frame's
  grab time). Average over ~20 toggles.
* **Stopwatch method:** point the spotter at a millisecond stopwatch on the laptop's own screen and take a phone
  photo of both the screen and the page's live frame. The difference between the stopwatch and its image in the frame
  is the latency (plus the page's display delay; take a few photos and the smallest difference).

### Choosing the control branch

`controller.branch = "auto"` reads the WP-Z1 report (`ptz.report`): **A** if pan/tilt inquiry works while moving,
otherwise **B**. Forcing A on a camera that can't answer while moving falls back to B. All the timing values marked
"auto" in the example config (latency, settle time, inquiry rate, minimum smooth step, pan/tilt sign) come from the
report too.

### Divergence guard and the `strict` profile

* **Divergence guard** (`controller.divergence_deg = 15`, `divergence_cycles = 5`): every cycle the target is
  compared with where the controller believes the camera is (dead reckoning in Branch B, inquiry in A). When the gap
  stays over the limit for more than N cycles in a row, the car is dropped (reason `diverged`, plus a `divergence`
  event in the control log with the gap) and the PTZ goes back to the entry. `divergence_deg = 0` turns it off.
* **`strict` profile**: Branch B as pure feed-forward: `dead_reckoning_trust_s = 0` (no feedback on the dead-reckoned
  pose) and `k_quant = 0` (no quantization compensation). For Sunday's comparison, run the same cars twice, once with
  the defaults and once with `--set profile=strict` (or `profile = "strict"` in `site.toml`), and compare the PTZ
  recordings and the replay scores. Profiles live in `[profiles.<name>]` at the end of `zone.example.toml`; `--set`
  still wins over them.

### Bench detector (COCO, home tests only)

The deployed car detector was trained on Baja buggies and may barely see ordinary street cars. For a bench test with
the spotter pointed at a public street, the spotter can use a stock COCO-trained YOLO11n instead:
`spotter.detector = "coco"` (default `"baja"`). The COCO classes in `spotter.coco_classes` (default car, truck, bus,
motorcycle; add `"person"` for a walk test) all become the tracker's single "car" class, with their own threshold
`spotter.coco_conf` (default 0.25). Everything after the detector (ROI crop, ByteTrack, progress, director,
controller) is unchanged.

* Each box is scored by its best class **among the selected ones**, so a van whose top COCO class is "train" or
  "boat" still counts (rt's Baja detector keeps the usual best-of-all-classes rule).
* YOLO removes duplicate boxes per class only, so one pickup can come out as both "car" and "truck". Two boxes of
  any selected classes that overlap by more than `spotter.coco_agnostic_iou` (default 0.5, IoU) are one vehicle:
  the weaker one is dropped. Raise it if two cars side by side get merged; lower it if one vehicle gets two tracks.
* `spotter.detector` must be `"baja"`, `"coco"` or `"blobs"` (`"rt"` = old name of `"baja"`). Any other value, e.g.
  a typo, now stops the program at startup with an error listing the valid ones (before, any unknown value
  silently ran the Baja detector).

**Bench only.** COCO models are unreliable on muddy Baja cars (LESSONS.md), and the Ultralytics YOLO11 weights are
**AGPL-3.0**: don't ship it in the race setup. While it is on, the page shows an amber "DETECTOR DE BANCADA (COCO)"
banner in the header and a red "detector" line in the health panel, and the console prints a warning at startup.

Export the model once on your PC, in a separate venv (ultralytics pulls in torch; keep it out of the tracker's env).
In a `cmd` window:

```
python -m venv %USERPROFILE%\venv-yolo-export
%USERPROFILE%\venv-yolo-export\Scripts\activate
pip install ultralytics
yolo export model=yolo11n.pt format=onnx "imgsz=[384,640]" opset=17 simplify=True dynamic=False
copy yolo11n.onnx <repo>\deploy\models\yolo11n_coco.onnx
```

* `yolo11n.pt` is downloaded by the first command that needs it; `yolo11n.onnx` is written next to it (the current
  folder).
* `imgsz=[384,640]` (height, width) is the input the spotter feeds the Baja detector: a `downscale` ROI crop is
  letterboxed to it and `tile` mode cuts tiles of that size. Keep it in quotes (PowerShell would split the comma).
* `opset=17` loads in the repo's onnxruntime (1.31; any opset up to 21 is fine). `dynamic=False` = fixed input
  shape, which `rt.models.YoloOnnx` reads from the file.
* Check: `python -c "import onnxruntime as o; s = o.InferenceSession('deploy/models/yolo11n_coco.onnx'); print(s.get_inputs()[0].shape, s.get_outputs()[0].shape)"`
  prints `[1, 3, 384, 640] [1, 84, 5040]`. The tracker refuses at startup a missing file or a model whose output is
  not `(1, 84, N)` (80 classes).

Run it with the `bench` profile (detector only) or `bench_strict` (bench detector + the `strict` controller, since only
one profile can be active at a time). The profiles only switch `spotter.detector`; a `coco_model` (or any other
`coco_*` key) set in `site.toml` is kept:

```
python -m zone.app --config zone/configs/site.toml --set profile=bench
python -m zone.app --config zone/configs/site.toml --set profile=bench_strict
python -m zone.app --config zone/configs/site.toml --set profile=bench --set "spotter.coco_classes=['person']"
```

Through the launcher (restarts on crash): copy `site.toml` to `site_bench.toml`, add `profile = "bench"` (or
`"bench_strict"`) as its first line (before any `[section]`), and run `zone\run_zone.bat zone\configs\site_bench.toml`.
`site.toml` itself stays race-ready.

### Offline tools

```
python -m zone.tools.detect_check --video wide.mp4 --zone zones/oct.json --passes 23      # AUTOPTZ-16
python -m zone.tools.replay --video out/zone/spotter_<start>.mkv --zone zones/zone.json --report reports/ptz1/report.json
python -m zone.tools.replay --tracks out/zone/replay_x/tracks.json --zone ... --set controller.k_pan=1.0
```

Our own spotter recordings are encoded at a fixed 10 fps, so their pts are only nominal: the replay takes each
frame's real grab time from the `.frames.csv` sidecar next to the video (found by itself; `--frames x.frames.csv`
to point at another one, `--no-frames` to use the pts). Videos without a sidecar use their pts.

The replay writes `cycles.csv`, `passes.csv` (time locked, drop reason, mean / max error), a `pass_NN.png` per pass,
and `tracks.json` (re-run the control part alone with other settings).

### Outputs (`zone.out_dir`, default `out/zone/`)

* `zone_<start>.jsonl`: one line per control cycle (grab time, tracks with s / v / lateral offset, director state,
  target pose, commanded steps, pan_now and its source), plus mode changes and events. `mono` + `wall` on every line.
* `spotter_<start>.mkv` (Matroska: readable after a crash) + `.frames.csv` (wall and monotonic grab time of every frame) + `.json`.
* `ptz_<start>.mkv` + `.json` (the PTZ's own H.264, copied).

Tests: no camera, no real time (fake clock, `SimulatedPtz`, synthetic cars in `zone/testing.py`).
