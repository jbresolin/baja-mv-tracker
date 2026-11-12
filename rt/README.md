# rt: real-time lap-pass prototype (WP11)

Runs track-side on a modest laptop, CPU only (or an Intel iGPU through OpenVINO):

```
capture (file / USB camera / rtsp)  ->  car detector @ 10 fps  ->  ByteTrack (numpy)  ->  online pass logic
   -> pass_start / pass_end events  ->  identity worker (async): number-panel detector -> recognizer -> boxed vote
   -> roster number  ->  pass_provisional / pass_identity events  ->  JSONL + CSV (+ LapAppSink stub)
```

Everything is a faithful streaming port of what was evaluated offline on 26SE. Each part was checked against the
offline code. The car detector and the panel detector are the ultralytics models exported to ONNX. The tracker is
ultralytics' ByteTrack in numpy. The pass rules are `lapvision/passes.py`, the crop choice is `passes.pick_crops`,
and the vote is WP10b's `vote_boxed` with the val-picked tau / k. Results and caveats are in LESSONS.md, section
"Session 2: real-time prototype (WP11)".

## Install on a laptop

Python 3.11+ (3.13 tested) and ffmpeg/ffprobe on PATH (Windows: `winget install Gyan.FFmpeg`). Then:

```
pip install onnxruntime numpy opencv-python psutil          # psutil: CPU/RAM stats, CPU affinity, launcher kill
pip install openvino                                        # optional: Intel CPU / iGPU backend
```

No torch, no ultralytics. Copy the repo folder, or at least `rt/`, `rosters/` and the model files. `lapvision/clock.py`
plus `lapvision/templates/` are needed only to read the 26SE OBS clock overlay. The model files are the four
`[models]` paths in `rt/config.toml`. Once the final models are published, use `deploy/models/` and set one line:
`manifest = "deploy/models/manifest.json"`.

**First run `python rt/bench_laptop.py` on the laptop** (2-4 min; add `--video <a recording>` for a real frame and the
decode cost, and `--sustain 120` to check thermal throttling). It prints ms per frame / crop / box for every model on
ONNX Runtime 1/2/4 threads and OpenVINO CPU / GPU, the CPU share at 10 fps, and a GO / MARGINAL / NO-GO verdict.
It writes `bench_laptop.json`. Send that file back.

## The worst-case laptop (4-core 8th-gen U-series, 8 GB, UHD 620)

Measured on the 3900X and extrapolated x2 (throttled U-series). Details: LESSONS.md WP11.

* **cars_n640_v2 at 10 fps fits; cars_s640 does not.** s640 drops 37% of frames on one 3900X thread and still drops
  frames on 4.
* Budget with 2 detector threads + 1 identity thread: ~1.9 cores (47% of 4) before decoding. Software decode of a
  1080p30 camera adds ~1 core. So, in order: use hardware decode (`capture.hwaccel = "d3d11va"`) or set the camera to
  10-15 fps; try the iGPU for the car detector (`runtime.car_backend = "openvino"`, `runtime.car_device = "GPU"`);
  keep identity on 1 thread (it is async and only bursts at packs of cars).
* RAM: ~260 MB per process (bounded crop buffers: JPEG, 48 candidates per track; frames are dropped, never queued).
* Race day: `--set runtime.priority=high`. If OBS runs too, pin rt to 2 cores (`--affinity 2,4`): that config
  kept up with 0 dropped frames through the 26SE finish on this PC.

## Lap-line node: live USB camera, unattended (WP-L1)

One command on the lap-line laptop, for a whole race day:

```
copy rt\configs\lapline.example.toml rt\configs\lapline.toml     (then set launch.event and camera.device)
python rt\run.py --list-cameras                                    (prints each camera's device path)
rt\run_lapline.bat                                                 (starts rt, restarts it if it dies or stalls)
```

Then open `http://<laptop IP>:8765/` on a phone on the same network (allow Python through the Windows firewall for
private networks when Windows asks).

* **Camera** (`rt/livecam.py`, `rt/camsource.py`, `[camera]`): DirectShow on Windows, MJPG FOURCC set explicitly,
  1920x1080 @ 30 fps; what the driver gives is read back and logged, and a camera that will not do MJPG fails loudly
  (and is retried). Select the camera by device path (`camera.device`), never by name: identical webcams share a
  name. Image settings (`[camera.settings]`: manual exposure, gain, white balance, focus) are applied at every open
  and read back; a setting the camera does not support is a warning.
* **Read loop per backend** (`camera.read_mode = "auto"`). OpenCV's DirectShow backend (cap_dshow.cpp) does not do
  what its API suggests: `grab()` returns True at once, and `retrieve()` is what waits (up to ~1 s) for the next
  frame and decodes it; it also ignores `CONVERT_RGB = 0` for MJPG, so every frame is decoded anyway. A "grab every
  frame, retrieve only 1 in 3" loop therefore spins grab() at full speed on DirectShow (a whole core and the GIL)
  and saves nothing. So:
  * `retrieve_every_frame` (DirectShow, and every backend not verified otherwise): one grab() + one (waiting)
    retrieve() per camera frame, stamped as soon as retrieve() returns, then 2 of 3 frames are dropped in
    software. The decode of all 30 fps is paid (~0.6 core on the 3900X, ~1.2 on a U-series laptop: LESSONS WP11);
    measure it on the T14.
  * `grab_then_retrieve` (V4L2 on Linux, where grab() really waits for the next buffer): every frame is grabbed and
    stamped at grab(), only the processed ones are retrieved (decoded).
  * Future options to avoid decoding 20 of 30 fps on Windows (not implemented): the MSMF backend, if its grab()
    is confirmed to block per frame; or an ffmpeg `-f dshow -vcodec mjpeg` source that copies the camera's JPEGs
    through a pipe (decode only the processed ones, and record without re-encoding).
* **Camera lost**: DirectShow reports an unplug once (one grab() returns False) and then retrieve() just times out,
  so loss is detected by time: no good frame for `camera.lost_after_s` (3 s, monotonic clock) or
  `camera.lost_after_failures` (10) failed reads in a row. The status page then shows "camera lost", the recording
  file is closed, and the camera is reopened with backoff (1 s, doubling, max 30 s) until it is back, in the same
  process (the launcher is only the last resort).
* **Timestamps**: each frame is stamped with the wall clock (`t`, what every event uses) and the monotonic clock
  (`t_cap`) as soon as it is in hand: after retrieve() on DirectShow (this includes the MJPEG decode, ~10-20 ms;
  fold it into `capture.live_latency_s` once measured), at grab() on V4L2. A `clock_map` event once a minute
  relates the two clocks (for the beam input, WP-L2).
* **Recording** (`rt/recorder.py`, `[recording]`): the processed frames (exactly what rt saw) go to
  `<folder of --out>/rec/rec_<date>_<time>.mkv`, one file per 30 min, with a `.json` sidecar (start time on both
  clocks, camera settings read back, software version, encoder) and a `.frames.csv` (true time of every frame).
  Default H.264 (QuickSync, AMF, else x264 veryfast): ~4 GB/h at 10 fps. `mode = "mjpeg"` copies the camera JPEGs
  without re-encoding but needs ~20-30 GB/h at 10 fps (66-89 GB/h at 30 fps), measured on 26SE frames. It never
  slows detection (a full queue drops frames from the recording, counted). Under 20 GB free the status page warns;
  under 5 GB recording stops (detection goes on) until space is back above 20 GB. The ffmpeg process is the only
  extra consumer; it reads frames from rt through a pipe, so the webcam has one handle.
* **Status page** (`rt/status.py`, `[status]`): processing fps and dropped frames, camera state and settings read
  back, clock offset (red above 0.5 s; `clock.ntp_server` = the event router for a measured SNTP offset, else the
  Windows time service's phase offset), recording state and free disk, last 20 passes with thumbnail, number,
  confidence and t_cross. `/status.json` has the same data. Read-only.
* **Launcher** (`rt/supervise.py`, `rt\run_lapline.bat`, `[launch]`): restarts rt when it exits (5 s, doubling, max
  60 s; reset after a 10-min healthy run) or when the status page reports 0 fps (or does not answer) for 30 s
  (after a 60 s start-up grace). It kills the whole process tree. Each run writes
  `out/lapline/<date>/lapline_<time>.jsonl` (+ csv, stats, rec/); restarts are logged in `out/lapline/launcher.log`.

Replay a recording through rt like any file (`--source out/lapline/<date>/rec/rec_....mkv`): its `creation_time` tag
is the segment's wall-clock start.

## Tests

```
pip install -r rt/requirements-test.txt
python -m pytest rt/tests -v
```

No camera, GPU or footage needed: the capture runs on a fake camera and a fake clock, the launcher on a fake process.
`rt/tests/test_e2e_clip.py` also runs the whole pipeline on a 7.5 s 26SE clip (`rt/tests/fixtures/`, car 14 at
11:45:17) when the ONNX models are present (it must find one pass and read 14); otherwise it is skipped. Every test
explains what it checks and why in plain words.

## Run

```
python rt/run.py --source <file | camera | camera index | rtsp/http url> --event <EV> [--config my.toml] [--realtime]
                 [--out passes.jsonl] [--set section.key=value ...] [--threads N] [--affinity 2,4] [--stats]
```

* `--event 26SE` selects `rosters/26SE.csv` (numbers that the reader snaps to).
* Files are processed as fast as possible. With `--realtime` they are paced at real speed and frames are dropped when
  detection falls behind, exactly like a live source. `--start 11:38` (wall clock, or seconds into the file) and
  `--duration 600` select a segment. `--as-live` sends a file through the live-stream code path.
* Outputs: `passes.jsonl` (all events), `passes.csv` (one row per pass, written when its number is known),
  `passes.stats.json` (frames, drops, ms, lag, identity latency; CPU/RAM with `--stats`). With `[lapapp] enabled`
  there is also `passes.lapapp.jsonl`.
* Validation tools (they need pandas, and are not needed at the race): `rt/eval_26se.py` (replay metrics vs the
  human laps, and an online-vs-offline pass-logic check), `rt/loadtest.py` (real-time CPU/RAM matrix).

Live sources use the system clock. At start, `rt/timesync.py` checks Windows time sync (`w32tm /query /status`).
**It warns loudly when the PC is not synced.** The session-1 PC was 80 s ahead of the lap server. Fix it with
Settings > Time > Sync now, or set `clock.offset_s`. `clock.ntp_server` adds an SNTP check. It is off by default,
so the program makes no network calls unless you set it.

## Config reference (`rt/config.toml`; a user config only needs the keys it changes)

| section | key | default | meaning |
|---|---|---|---|
| models | manifest / manifest_car | "" / car_n | path to `deploy/models/manifest.json`: its models override the paths (one-line switch); `car_n` or `car_s` as the car detector |
| models | car_detector / car_imgsz | cars_n640_v2 ONNX / [384, 640] | YOLO11n + washout, 16:9 export |
| models | car_detector_alt | cars_s640 ONNX | only benchmarked by bench_laptop.py |
| models | panel_detector / panel_imgsz | num_n320_r2 ONNX (dynamic shape) / 320 | number-panel detector |
| models | recognizer | pq_spl_full `recog.onnx` | PARSeq-tiny + splicing (WP10b) |
| models | vote | pq_spl_full `eval/panel_tau.json` | tau 0.9 / min_support 3 / share score (25SU val) |
| runtime | car_backend / car_device | onnxruntime / CPU | or `openvino` + `CPU` / `GPU` (iGPU) / `AUTO`; falls back to ORT CPU with a warning |
| runtime | id_backend / id_device | onnxruntime / CPU | same for the panel detector + recognizer |
| runtime | threads_detector / threads_identity | 2 / 1 | ONNX Runtime intra-op threads per model (`--threads N` sets both); 2 detector threads because one thread of a throttled U-series laptop needs ~110 ms per 100 ms frame |
| runtime | ort_spinning | false | idle ORT threads sleep (do not burn a core between frames) |
| runtime | affinity / affinity_decoder | [] / true | pin the process (and ffmpeg) to these CPUs (psutil) |
| runtime | priority | normal | `high` = above-normal priority on race day |
| capture | fps | 10 | detection / tracking rate (as evaluated) |
| capture | max_width | 1920 | downscale wider sources at decode (4K -> 1080p); 0 = native |
| capture | decode_threads / queue_size | 0 / 4 | ffmpeg decoder threads; frames buffered before detection |
| capture | hwaccel | none | ffmpeg `-hwaccel` (`d3d11va` / `qsv` on an Intel laptop): software decode of a 1080p30 camera costs ~0.6 core here, ~1 core on the laptop |
| capture | live_latency_s | 0 | known capture latency subtracted from live frame times |
| camera | backend / device / width / height / fps / fourcc | auto / 0 / 1920 / 1080 / 30 / MJPG | USB camera (rt/livecam.py); `device` = index or device path (`--list-cameras`) |
| camera | require_fourcc / raw_mjpeg | true / false | fail if not MJPG; ask for JPEG bytes (V4L2 only; DirectShow always decodes) |
| camera | read_mode | auto | `retrieve_every_frame` (DirectShow) or `grab_then_retrieve` (V4L2); see "Read loop per backend" |
| camera | lost_after_s / lost_after_failures / reopen_first_s / reopen_max_s / clock_map_every_s | 3 / 10 / 1 / 30 / 60 | camera lost after 3 s without a good frame or 10 failed reads; reopen backoff; clock_map event period |
| camera.settings | exposure, gain, auto_exposure, auto_wb, wb_temperature, autofocus, focus, ... | (none) | OpenCV CAP_PROP_* names, backend units; applied at each open and read back |
| recording | mode / all_frames / segment_min | h264 / false / 30 | h264, mjpeg (copy) or off; record every camera frame (mjpeg only); file length |
| recording | warn_free_gb / stop_free_gb / h264_encoders | 20 / 5 / qsv, amf, x264 | disk guard; H.264 encoders tried in order |
| status | enabled / bind / port / passes / time_check_s | true / 0.0.0.0 / 8765 / 20 / 60 | status page |
| launch | event / source / out_dir / extra_args | "" / camera / out/lapline / [] | read by rt/supervise.py |
| launch | restart_first_s / restart_max_s / healthy_reset_min / stall_s / startup_grace_s | 5 / 60 / 10 / 30 / 60 | restart rules |
| clock | mode | auto | files: overlay (OBS clock) > creation_time > mtime-duration; `pts`, `system` also accepted |
| clock | offset_s / video_minus_server_s | 0 / 0 | correction added to frame times / this PC minus the lap server (LapAppSink) |
| detection | conf / nms_iou | 0.15 / 0.7 | as the offline tracking stage |
| tracker | ... | bytetrack.yaml | 0.25 / 0.1 / 0.25, buffer 30 frames, match 0.8, fuse_score |
| passes | fg_h, fg_y2, min_frames, edge, merge_gap_s, static_gap_s, nested_ratio, recent_passes | 0.2, 0.4, 6, 0.02, 3, 10, 0.6, 4 | lapvision/passes.py |
| passes | pending_max_s | 10 | a pass waits at most this long for a younger track that is not foreground yet |
| identity | max_crops / crop_pad / crop_min_h | 30 / 0.05 / 0.10 | crops per pass (pick_crops), padding, min box height for a candidate |
| identity | candidates_per_track / crop_jpeg | 48 / 92 | bounded candidate buffer (thinned in time), stored as JPEG q92 |
| identity | panel_conf / panel_pad / panel_rect | 0.1 / 0.05 / true | as picked on 25SU val |
| identity | provisional* | on, every 0.5 s, 4 crops, max 16 | early reads while the car is in view (PTZ) |
| identity | queue_max | 64 | identity backlog; provisional jobs are skipped first |
| output | csv / crops_dir | "" | CSV path (default `<out>.csv`); save each pass's selected crops (review / training data) |
| lapapp | enabled, fonte_id, dispositivo_id, only_numbered, outbox | false, 5, 0, true, "" | LapAppSink stub |

Manifest format accepted (`deploy/models/manifest.json`; paths relative to the manifest):
`{"car_detector": {"path": "cars_n640.onnx", "imgsz": [384, 640]}, "panel_detector": {"path": "num_n320.onnx",
"imgsz": 320}, "recognizer": "recog.onnx", "vote": {"tau": 0.9, "min_support": 3, "score": "share"}}`. A role can
be a plain path. A `"roles": {...}` or `"models": {...}` wrapper is also accepted, and `vote` may point to a
`panel_tau.json`.

## Event schema (JSONL, one object per line)

Every event has `type`, `wall` (system epoch seconds at emission) and `t_video` / `t_video_iso` (the video clock of
the frame being processed). Times are epoch seconds of the video PC's clock, each with an `_iso` twin in local time.

| type | when | fields |
|---|---|---|
| `info`, `warning` | start / end / clock problems | `msg` |
| `camera` | live camera state change (opening, streaming, lost, retrying, stopped) | `state`, `detail` |
| `clock_map` | live camera, once a minute | `frame_wall` (epoch), `frame_mono` (time.monotonic) of one frame, `wall_minus_mono`, `camera_frame_no` (frames received from the camera since start), `stamped_at` (`after retrieve` or `grab`) |
| `pass_start` | a track becomes foreground (box > 0.2 H, bottom > 0.4 H, 6 frames: ~0.6 s after the car appears) and joins no open pass | `pass_id`, `t_enter`, `box` (first box, normalised x1 y1 x2 y2), `hmax` |
| `pass_provisional` | first confident read while the pass is open (same rule as auto-confirm) | `pass_id`, `number`, `confidence`, `support`, `n_read`, `top3` |
| `pass_end` | the pass closed: all its tracks gone and 10 s without a box (the re-attach windows expired) | `pass_id`, `t_enter`, `t_cross`, `t_exit`, `dwell_s`, `n_frames`, `n_tracks`, `tracks`, `exited`, `hmax`, `n_crops` |
| `pass_identity` | identity of the closed pass is done (async) | everything in `pass_end` plus `number` (null = no panel box in any crop), `team`, `confidence` (vote share), `auto_confirm`, `support`, `n_read`, `n_boxed`, `n_boxes`, `top3` [{number, share}], `provisional_number`, `identity_latency_s` |

* `t_enter` = the first frame of the pass's first track. This is the human-click equivalent: the counters stand at the
  entry side (LESSONS: median click -> enter +0.35 s), and it is the time the evaluation matches on.
  `t_cross` = the box centre crosses the frame centre line. `t_exit` = the last box.
* `number` is always the roster number with the most votes ("no abstain"). On the 26SE replay it was right on 83%
  of passes. `auto_confirm` (score >= tau and >= k supporting crops; tau / k from 25SU val, never tuned on 26SE) was
  right on 134 / 135 passes (it answered 58% of matched passes = 53% of laps). The miss was car 6's stylised 6 read as 5, so treat it as ~99%:
  accept it without a human, but keep the reviewer able to correct it. Anything else goes to the reviewer, with
  `top3` as suggestions.
* A pass is final only at `pass_end`. Later tracks can still join a pass before that. `pass_start` is a fast signal
  for the PTZ, not a count.

## What is stubbed / not done

* **LapAppSink** (`rt/sinks.py`): it builds the `contagem_passagem` row the camera would insert: `fonte_id` (a new
  `contagem_fonte` row, e.g. 5 'Câmera'), `uuid_cliente` (a stable uuid5, so retries are idempotent), `equipe_numero`,
  `ts_ms` = t_enter - `video_minus_server_s`, `ts_dispositivo_ms`, `qualidade_tempo` 'exato', `modo_entrada`
  'camera'. It writes the row to an outbox JSONL. `deliver()` (API POST or DB INSERT) is not implemented, and
  nothing connects to any database or network. Decisions needed: API vs DB; what to do with unread passes
  (`equipe_numero` is NOT NULL in all 586 rows of 26SE, so they are skipped by default); how `auto_confirm` maps to
  the reviewer flow (`contagem_decisao`); and server-clock anchoring (`ancora_*` columns).
* **PTZ interface**: only the events above (`pass_start` with the entry box, `pass_provisional` with the number).
  The transport (UDP / websocket / HTTP) is still to be decided. Add a `Sink` subclass for it.
* **USB camera capture** (WP-L1) is tested with fake cameras (rt/tests; `FakeDshowCamera` copies DirectShow's
  grab/retrieve behaviour as read in OpenCV 4.x cap_dshow.cpp) and a paced fake camera through run.py; it has not
  met a real webcam yet. **RTSP** is written but
  untested here; it was exercised with `--as-live` on a file.
* **OpenVINO GPU (Intel iGPU)**: implemented with static shapes and a fallback to ORT CPU, but untested here
  (Ryzen, no Intel GPU). OpenVINO CPU was tested. Run `rt/bench_laptop.py` on the laptop.
* Re-ID for mud cars (car 1), a pass minimum in seconds instead of frames, and the exposure of time sync to the lap
  app are open items (TRAINING_PLAN).
