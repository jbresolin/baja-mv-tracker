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
pip install onnxruntime numpy opencv-python psutil pyserial # psutil: CPU/RAM stats, affinity, launcher kill; pyserial: beam
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
* **Recording** (`rt/recorder.py`, `[recording]`): the processed frames (exactly what rt saw, 10 fps) go to
  `<folder of --out>/rec/rec_<date>_<time>.mkv` as H.264. The encoder is probed once at start and logged: QuickSync
  (`h264_qsv`), AMF (`h264_amf`), else x264 `veryfast`. ~4 GB/h at 10 fps (x264 CRF 23 on 26SE frames; QSV / AMF
  sizes differ: check the real files on the T14). DirectShow hands rt decoded images, so there are no camera JPEGs
  to copy: `mode = "mjpeg"` and `all_frames` are refused at start with an explanation (`h264` or `off`).
  * **Files**: a new file every 30 min, and also after a camera loss or a gap of more than 2 frame periods: the
    file's frame rate is nominal, so a file never spans a gap. Next to each file: a `.json` sidecar (start time on
    both clocks, camera settings read back at the last open, software version, encoder; written provisionally at
    open, finalised at close), a `.frames.csv` (the authoritative time of every frame; flushed every 30 rows) and a
    `.ffmpeg.log` (ffmpeg's messages: a file, because an unread pipe fills up and freezes ffmpeg).
  * **Never slows detection**: offer() never blocks; the queue holds ~1 s (10 frames, ~60 MB at 1080p) and drops
    the oldest frame when full (counted). A watchdog kills ffmpeg if it does not accept a frame for 10 s or exits,
    marks recording failed on the status page and starts a new file 10 s later; after 5 failures in a row
    recording is off for the run (detection goes on).
  * **Disk guard**: under 20 GB free the status page warns; under 5 GB recording stops (detection goes on) until
    space is back above 20 GB. ffmpeg reads frames from rt through a pipe, so the webcam has one handle.
* **Status page** (`rt/status.py`, `[status]`): processing fps and dropped frames, camera state and settings read
  back, clock offset (red above 0.5 s; `clock.ntp_server` = the event router for a measured SNTP offset, else the
  Windows time service's phase offset), recording state and free disk, last 20 passes with thumbnail, number,
  confidence and t_cross. `/status.json` has the same data. Read-only.
* **Launcher** (`rt/supervise.py`, `rt\run_lapline.bat`, `[launch]`): restarts rt when it exits (5 s, doubling, max
  60 s; reset after a 10-min healthy run) or when the status page does not answer, or reports 0 fps while the camera
  says "streaming", for 30 s (after a 60 s start-up grace). 0 fps with the camera "lost" / "retrying" / "opening" is
  left to rt, which is already reopening the camera itself, within two limits: rt is restarted anyway when the
  camera has been recovering for `camera_recover_max_s` (600 s) in one go, or when rt's camera open-attempt counter
  (`open_attempts` on the status page) has not gone up for `camera_attempt_stall_s` (60 s: rt's reopen loop is
  stuck, e.g. an open that never returns; keep `camera.reopen_max_s` well below it). It kills the whole process tree. It starts rt with
  `--require-status`: if the status port is busy (another rt, or another program), rt exits with code 3 and says
  why, and launcher.log names the reason, instead of rt running blind and being killed as "stalled" every ~90 s. On
  Windows the page binds its port exclusively (Python's default SO_REUSEADDR would let two programs share it
  silently). Each run writes
  `out/lapline/<date>/lapline_<time>.jsonl` (+ csv, stats, rec/); restarts are logged in `out/lapline/launcher.log`.

Replay a recording through rt like any file (`--source out/lapline/<date>/rec/rec_....mkv`): when the file's
`.frames.csv` is next to it, each frame gets its recorded time from it (the authoritative clock); otherwise the
file's `creation_time` tag (the segment's start) + the nominal frame rate.

## Beam input: timing light-beam sensor at the line (WP-L2)

The event's timing sensors (a light beam across the start/finish line) give a digital pulse when a car crosses. A USB
serial adapter (Silicon Labs CP2102) brings it to the lap-line laptop, and `rt/beam/` adds **precise crossing times**
to the same JSONL. Vision stays responsible for identity and counting; the beam only gives times. Shadow mode:
nothing goes to the lap app. Off by default (`[beam] driver = "off"`).

**On site (the usual way, from a phone, rt running, nothing to edit or restart):**

1. Plug the CP2102 adapter into the lap-line laptop and wire the sensor (see "Wiring" below).
2. On the phone (same network) open the status page `http://<laptop IP>:8765/` and tap **Beam setup**
   (`/beam`). The CP2102 is listed first, marked "CP2102 (10C4:EA60)".
3. Choose the port (baud 9600 unless known otherwise), tick DTR / RTS if they power a pull-up, tap
   **Start monitor**. The beam driver pauses; the four lines (CTS / DSR / DCD / RI) and any bytes now show live.
4. With the beam clear, tap **Clear**, then walk / wave a hand through the beam 5-10 times: one line should flip
   (green = asserted) or byte messages should appear.
5. Tap **Suggest**: the probe's guess (driver, line, polarity or byte pattern) fills the settings, with notes.
   Check them (group gap 0.25 s and stuck after 5 s are fine to start with).
6. Tap **Apply**: rt validates the settings, saves them to `<config>.beam.toml` next to the config
   (`rt/configs/lapline.beam.toml`, loaded at every start on top of `[beam]`; delete it to go back) and restarts
   ONLY the beam (never detection, capture or recording).
7. Check "Beam now" on the same page (or the Beam section of the status page): state `open`, the level flips
   when the beam is broken, **edges/min** goes up, and each pass through the beam adds one line to **last
   breaks** (2-10 edges for a car). **Turn off** sets driver = off (saved) if anything misbehaves.

Setup mode ends by itself (beam restarted with the saved settings) when the page has not been refreshed for
`beam.setup_idle_s` (300 s), so a locked phone never leaves the race without its beam. Only the beam setup
actions accept POST (JSON only, at most 8 KB, every value checked on the laptop; a bad value is refused and
changes nothing, a setting that still fails to start leaves the beam off with the error in red). The rest of the
page stays read-only. Anyone on the LAN can use the panel unless `[status] setup_token` is set: then the page asks
for it once (kept in the phone's browser) and sends it in the `X-Setup-Token` header.

The command-line probe does the same with rt closed (only one program can open a COM port):

```
python -m rt.tools.beam_probe --list               (which COM port is the adapter: USB id 10C4:EA60 = CP210x)
python -m rt.tools.beam_probe --port COM5          (break the beam 5-10 times, then Ctrl+C: it prints a guess)
python -m rt.tools.beam_probe --port COM5 --baud 115200 --seconds 60      (if bytes arrive and look like garbage)
```

Close rt first (only one program can open a COM port). The probe prints every change of the four modem input
lines (CTS, DSR, DCD, RI, read 1000 times a second) and every message of bytes, with times (bytes less than 3
character times apart are one message, as in rt). At the end it guesses the driver, the line and the polarity (the
line's state at the start = beam clear, so start with the beam clear; a warning if the line then spent most of the
time in the other state), or the byte pattern and prints the lines to paste into the `[beam]` section
of `rt/configs/lapline.toml` (example and every key: `rt/configs/lapline.example.toml`).

* **Two ways the pulse can arrive**, because the adapter is not known yet:
  * `modem-line`: the sensor switches one adapter input line. A dedicated thread polls it at `poll_hz` (1 kHz) and
    stamps each change. `active = "high"` means line asserted = beam blocked; `"low"` the opposite.
  * `byte-stream`: the adapter sends bytes per pulse. Bytes less than `byte_merge_chars` (3) character times apart
    (3.1 ms at 9600 baud) are joined into one message even if they arrive in several reads; each message (or each
    `byte_pattern`, hex) is an edge, stamped at the receipt of the message's first byte; the raw bytes are kept. With `byte_release_pattern` the "clear" message is an edge too.
  * `fake`: one synthetic tube-frame car every `fake_every_s` (check the chain and the status page without hardware).
* **Same clocks as the frames.** Each edge is stamped right after it is seen with `time.time()` and
  `time.monotonic()` (the clock object rt/livecam.py stamps frames with). `t_edge` / `t_beam` = wall time +
  `clock.offset_s`, like frame times. The camera's `capture.live_latency_s` is subtracted from frames only, so
  `t_cross - t_beam` contains the camera latency and where the camera's centre line is relative to the beam. On
  DirectShow frames are stamped after `retrieve()` (decode included), so frame times run late by the decode time
  too; `assoc_offset_s` and the LED latency absorb it.
  At 1 kHz an edge is stamped within ~1 ms (the `poll_gap_ms` field says how long the gap before it was). Windows:
  Python 3.11+ sleeps with a high-resolution timer, and rt also calls `timeBeginPeriod(1)` at start (1 ms timer
  tick, undone at exit; rt/wintimer.py), so 1 kHz is reachable; rt logs `sys.version` and warns below 3.11; the `beam_poll` event (every 60 s) and
  the status page show the achieved rate and the largest gap.
* **One break per car, in software.** A tube-frame car blocks the beam several times (nose, driver, firewall,
  wheels). Edges less than `group_gap_s` (0.25 s) apart form one `beam_break`; its first edge is the crossing time.
  Cars nose to tail closer than that become one break (the offline fusion flags it). No hardware hold-off.
* **Stuck beam.** A break that lasts more than `stuck_s` (5 s: a person standing in the beam, a car stopped on the
  line, a misaligned sensor, or the polarity set the wrong way round) gives one `beam_stuck` "stuck" event, no breaks,
  and one "cleared" event when the beam has been quiet again.
* **Never in the way.** The reader thread only reads and stamps; a second thread writes the events. A missing or
  unplugged adapter is logged, shown in red on the status page, and reopened with backoff (1 s, doubling, max 30 s;
  on Windows pyserial reads the modem lines of an unplugged adapter as "not asserted" without an error, so the
  modem-line driver also checks the port every 100 ms and before reporting any change: `in_waiting`, which does
  fail, plus GetCommModemStatus' own result);
  detection never waits for the beam. A wrong `[beam]` setting is a warning at start and rt runs without the beam.
* **Live association** (`pass_beam`, one per `pass_end`): exactly one break within `assoc_window_s` of
  `t_cross - assoc_offset_s` -> `matched` with `t_beam`; none -> `none`; several -> `ambiguous`. The pass events
  themselves are unchanged. The real fusion is offline (below).
* **LED latency (optional).** If the sensor output can also light an LED in the camera's view, set `led_roi`
  (x, y, w, h in processed-frame pixels) and `led_threshold`: every LED off -> on in a processed frame gives a
  `camera_latency` event = frame grab time - the beam edge before it. Frames are processed at 10 fps, so a single
  value is up to 100 ms too high: use the minimum of many `clean` ones (one beam edge in the window).

**Wiring a CP2102 RS232 adapter** (DB9 male on the adapter, PC side = DTE):

* Inputs (what the probe watches): **DCD pin 1, DSR pin 6, CTS pin 8, RI pin 9**; ground **pin 5**. Outputs:
  **DTR pin 4, RTS pin 7** (rt asserts both: `set_dtr`, `set_rts`), TXD pin 3. RXD pin 2 is the byte input.
* **RS232 levels**: an input reads "asserted" above about +3 V and "not asserted" below about -3 V; real receivers
  (MAX3232 type) switch around +1.3 to +1.5 V and read an open input as not asserted (they have a ~5 kOhm resistor to
  ground). A 0 / 5 V signal therefore usually works on an RS232 input even though it is out of spec; 0 / 3.3 V is
  marginal. Up to +-25 V is allowed, so a 12 V PNP sensor output is fine on an RS232 input.
* **3.3 V TTL adapters are different**: a bare CP2102 module with CTS / DSR / DCD / RI header pins is 3.3 V logic,
  **active low** (0 V = asserted) and must **never** see RS232 voltages or a 12 V sensor output. Keep the signal
  within 0-3.3 V (5 V only if the module says its inputs tolerate it). The probe and `active` handle the inverted
  polarity.
* **NPN open-collector sensor outputs only pull down**: with the beam in the other state the output floats, and the
  line reads whatever it likes. It needs a **pull-up**. On RS232: a resistor from **DTR (pin 4) or RTS (pin 7)**
  (asserted = about +5 to +10 V) to the input pin, the sensor output to the same input pin, and the sensor's 0 V to
  pin 5. Because the receiver has its own ~5 kOhm to ground, use a small pull-up (1-2.2 kOhm) and check with a
  multimeter that the input is above +3 V when the transistor is off. On a 3.3 V TTL module: 10 kOhm to 3.3 V
  (unless the module already has one). PNP (sourcing) outputs need a pull-down instead (the RS232 receiver's own
  resistor usually does it).
* Ask the adapter's designer: which pin the input lands on (or what it sends), voltage range and polarity, whether it
  is opto-isolated (then the sensor ground is separate), and the sensor's response time. The probe answers the first
  two in practice.

**Reading the status page (Beam section)**: driver, port and state (`open` in green; `retrying` / `lost` in red with
the error and the next try; a **Beam setup** link next to it), the line and its **current level** (wave a hand in the beam: it must flip to ACTIVE and
back), edges per minute, breaks, poll rate and largest gap (red below 200 Hz), stuck episodes (STUCK in red while it
lasts), live association counts, LED numbers, and the **last 10 breaks** (time, edges, duration, matched pass and its
number). On race day: a car crossing should add one break with 2-10 edges and, ~10 s later, the matching pass.

**After the event**: `python rt/tools/fuse_beam.py out/lapline/<date>/*.jsonl [--dump <lap app dump.sql>
--video-minus-server <s>]` matches breaks to passes one-to-one (nearest first within `--window`, after the
camera - beam offset it estimates from clean single matches), classifies the rest (`shared_break` = two cars side by
side on one break, `break_no_pass` = marshals / unseen cars, `pass_no_break`), and with a dump compares the crossings
with the human laps and the vision numbers with the human car numbers. CSV + `.summary.json` + a printed summary. Set
`assoc_offset_s` to its "camera - beam offset" for the next run.

## Tests

```
pip install -r rt/requirements-test.txt
python -m pytest rt/tests -v
```

No camera, GPU or footage needed: the capture runs on a fake camera and a fake clock, the launcher on a fake process.
`rt/tests/test_e2e_clip.py` also runs the whole pipeline on a 7.5 s 26SE clip (`rt/tests/fixtures/`, car 14 at
11:45:17) when the ONNX models are present (it must find one pass and read 14); otherwise it is skipped. Every test
explains what it checks and why in plain words. The beam tests (`test_beam_*.py`, `test_fuse_beam.py`) use a fake
serial port (`rt/tests/beam_fakes.py`) and a fake clock: no adapter, no sensor, no waiting.

## Run

```
python rt/run.py --source <file | camera | camera index | rtsp/http url> --event <EV> [--config my.toml] [--realtime]
                 [--out passes.jsonl] [--set section.key=value ...] [--threads N] [--affinity 2,4] [--stats]
```

* `--event 26SE` selects `rosters/26SE.csv` (numbers that the reader snaps to). If that file is missing or empty,
  rt still runs, in open-set mode (see Troubleshooting).
* Files are processed as fast as possible. With `--realtime` they are paced at real speed and frames are dropped when
  detection falls behind, exactly like a live source. `--start 11:38` (wall clock, or seconds into the file) and
  `--duration 600` select a segment. `--as-live` sends a file through the live-stream code path.
* Outputs: `passes.jsonl` (all events), `passes.csv` (one row per pass, written when its number is known),
  `passes.stats.json` (frames, drops, ms, lag, identity latency; CPU/RAM with `--stats`). With `[lapapp] enabled`
  there is also `passes.lapapp.jsonl`.
* Validation tools (they need pandas, and are not needed at the race): `rt/eval_26se.py` (replay metrics vs the
  human laps, and an online-vs-offline pass-logic check), `rt/loadtest.py` (real-time CPU/RAM matrix).

Live sources use the system clock. At start and every minute, `rt/timesync.py` checks it. **It warns loudly when
the PC is not synced.** The session-1 PC was 80 s ahead of the lap server. Fix it with Settings > Time > Sync now, or
set `clock.offset_s`. The primary check is SNTP against `clock.ntp_server` (the lap-line example config sets it to
a placeholder for the event router: set the real IP); it is empty in `rt/config.toml`, so a plain run makes no
network calls. Windows' own status (`w32tm /query /status [/verbose]`) is read by its layout (field order and
value shapes), so a pt-BR Windows with Portuguese labels and a cp850 console works too.

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
| recording | mode / segment_min / split_gap_periods | h264 / 30 / 2 | h264 or off (mjpeg copy is refused); file length; a gap of more than this many frame periods starts a new file |
| recording | warn_free_gb / stop_free_gb / h264_encoders | 20 / 5 / qsv, amf, x264 | disk guard; H.264 encoders probed in order at start |
| recording | queue_frames / csv_flush_rows | 10 / 30 | frames waiting for ffmpeg (oldest dropped when full); frames.csv flush period |
| recording | write_timeout_s / retry_wait_s / max_failures | 10 / 10 / 5 | watchdog: ffmpeg stuck or dead -> kill, new file after the wait; give up after this many failures in a row |
| status | enabled / bind / port / passes / time_check_s / required | true / 0.0.0.0 / 8765 / 20 / 60 / false | status page; `required` (or `--require-status`, which the launcher passes): exit with code 3 if the port is busy |
| status | setup_token | "" | Beam setup POSTs need this token in `X-Setup-Token` when set; "" = no token (LAN only) |
| launch | event / source / out_dir / extra_args | "" / camera / out/lapline / [] | read by rt/supervise.py |
| launch | restart_first_s / restart_max_s / healthy_reset_min / stall_s / startup_grace_s | 5 / 60 / 10 / 30 / 60 | restart rules |
| launch | camera_recover_max_s / camera_attempt_stall_s | 600 / 60 | camera recovering this long, or rt's open-attempt counter not moving this long -> restart rt |
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
| identity | rosters_dir | "" | extra roster folder searched first (the installed app's data folder, where the launcher's "Baixar lista de equipes" saves); "" = only `rosters/` |
| output | csv / crops_dir | "" | CSV path (default `<out>.csv`); save each pass's selected crops (review / training data) |
| lapapp | enabled, fonte_id, dispositivo_id, only_numbered, outbox | false, 5, 0, true, "" | LapAppSink stub |
| beam | driver / port | off / COM5 | off, modem-line, byte-stream, fake (section "Beam input") |
| beam | line / active / poll_hz / set_dtr / set_rts | cts / high / 1000 / true / true | modem-line: input line, which state = blocked, polls per second, adapter outputs |
| beam | baud / byte_pattern / byte_release_pattern / read_timeout_s | 9600 / "" / "" / 0.002 | byte-stream: hex patterns ("" = any bytes = a pulse) |
| beam | group_gap_s / stuck_s | 0.25 / 5 | one break per car; longer = beam_stuck |
| beam | byte_merge_chars / advance_guard_s / queue_max | 3 / 0.02 / 10000 | byte-stream: bytes closer than this many character times = one message; breaks close this far behind now; reader -> dispatcher queue bound (oldest dropped, counted) |
| beam | setup_idle_s | 300 | Beam setup page not refreshed this long in setup mode -> setup ends, beam restarts as saved |
| (file) | `<config>.beam.toml` | absent | beam settings saved by the setup page; loaded after the config's [beam], before --set; broken / invalid = beam off with the error shown |
| beam | reopen_first_s / reopen_max_s / max_edge_events_per_s / poll_report_s | 1 / 30 / 100 / 60 | port retry backoff; beam_edge lines per second; beam_poll period |
| beam | assoc_window_s / assoc_offset_s | 1.0 / 0 | live pass_beam rule: window around t_cross - offset |
| beam | led_roi / led_threshold / led_max_latency_s | [] / 128 / 0.5 | optional LED latency (x, y, w, h in processed-frame pixels) |
| beam | fake_every_s / fake_pulses | 20 / 3 | fake driver: synthetic cars |

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
| `pass_identity` | identity of the closed pass is done (async) | everything in `pass_end` plus `number` (null = no panel box in any crop), `team`, `confidence` (vote share), `auto_confirm`, `support`, `n_read`, `n_boxed`, `n_boxes`, `top3` [{number, share}], `provisional_number`, `identity_latency_s`, `roster_checked` (false = no roster, open-set mode) |
| `beam` | beam port state change (open, retrying, lost, stopped) | `state`, `detail`, `driver`, `port` |
| `beam_edge` | every raw beam edge (at most `max_edge_events_per_s`) | `edge_no`, `line`, `level` (1 blocked, 0 clear), `raw` (line state / bytes hex), `t_edge` (wall + offset_s), `t_wall`, `t_mono`, `poll_gap_ms` |
| `beam_break` | one car: edges grouped by `group_gap_s` | `break_id`, `t_beam` (first edge = crossing), `t_beam_mono`, `t_last`, `t_last_mono`, `n_edges`, `n_pulses`, `duration_s`, `started_blocked`, `interrupted` |
| `beam_stuck` | beam blocked / flickering longer than `stuck_s`; again when it clears | `state` (stuck / cleared), `t_since`, `since_mono`, `n_edges`, `duration_s`, `reason` |
| `beam_poll` | every `poll_report_s` | `rate_hz`, `polls`, `mean_gap_ms`, `max_gap_ms`, `late_polls` (gap > 5 ms), `window_s` |
| `pass_beam` | after each `pass_end` (live association) | `pass_id`, `status` (matched / none / ambiguous), `t_cross`, `t_beam`, `beam_minus_cross_s`, `break_id`, `n_candidates`, `candidates`, `break_also_matched_to` |
| `camera_latency` | LED ROI off -> on (only with `led_roi`) | `latency_s` (frame grab mono - edge mono), `frame_t`, `t_edge`, `residual_s` (frame t - t_edge), `clean`, `roi_mean` |

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

## Troubleshooting

Both problems below leave rt running; each shows a yellow notice at the top of the status page and a `warning` line in
the log. Both are read once at start: after fixing them, **restart rt** (close it; the launcher starts it again).

* **"no roster for 26SU ... numbers are not checked against the entry list"**: `rosters/26SU.csv` is missing or empty
  (or unreadable). The results site lists the teams only shortly before the event, so the roster cannot be fetched
  earlier. rt then runs in *open-set mode*: it reads any number from 1 to `identity.open_set_max` (999) instead of
  snapping to the entry list, `team` is empty, `roster_checked` is false, `auto_confirm` is always false (its ~99%
  precision was measured with a real roster) and no `pass_provisional` is sent. Every pass goes to a human.
  Fix: once the event's results page lists the teams, run `python teacher/rosters.py 26SU` (writes
  `rosters/26SU.csv`; check it has rows), then restart rt. Open-set snapping costs ~6x more CPU per crop than a
  100-car roster; on a slow laptop `--set identity.open_set_max=99` (every event so far used 1-99) makes it cheaper.
* **"ffmpeg not found at 'ffmpeg': recording is off"**: the ffmpeg program is not installed or not on PATH (this used
  to show up as "no H.264 encoder passed the probe (h264_qsv failed, h264_amf failed, libx264 failed)"). Detection
  and identity run normally; only the recording is off. Fix: `winget install Gyan.FFmpeg`, then open a **new**
  terminal (PATH is read when it opens) and check `ffmpeg -version`; or set `capture.ffmpeg` in your config to the
  full path of `ffmpeg.exe`. Then restart rt. Note: a `camera.device` given as a DirectShow device path (not an
  index) also needs ffmpeg to find the camera.
* **"no H.264 encoder passed the probe (h264_qsv failed (...), ...)"** with ffmpeg installed: each encoder's
  reason is in brackets (the first line of ffmpeg's error output). rt still tries libx264; if that fails too,
  recording errors show on the status page. Usual causes: no Intel/AMD GPU driver for QSV/AMF (normal, x264 is used),
  or an ffmpeg build without libx264 (install the Gyan "full" or "essentials" build).

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
* **Beam input** (WP-L2) is tested with a fake serial port and a fake clock, and the byte-stream driver once with
  real pyserial on a Linux pseudo-terminal. It has not met the real CP2102 adapter or the sensors: run the probe
  first. Not done: event-driven line waits (Windows `WaitCommEvent`) instead of 1 kHz polling; the FTDI latency
  timer setting (irrelevant for the CP2102).
* **OpenVINO GPU (Intel iGPU)**: implemented with static shapes and a fallback to ORT CPU, but untested here
  (Ryzen, no Intel GPU). OpenVINO CPU was tested. Run `rt/bench_laptop.py` on the laptop.
* Re-ID for mud cars (car 1), a pass minimum in seconds instead of frames, and the exposure of time sync to the lap
  app are open items (TRAINING_PLAN).
