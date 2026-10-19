# Training plan: distil a laptop-sized car detector + number reader

**Status (2026-10-19): M3/M4 done. The students alone match the teacher on 26SE.**

| pipeline (26SE, 252 laps) | pass recall / precision | right number, no abstain | auto-confirm (val rule) | CPU |
|---|---|---|---|---|
| teacher: YOLO-World + Qwen3-VL-4B | 0.933 / 0.975 | 67.9% | 98.7% precision at 65% (share ≥ 0.5) | GPU |
| cars_s640 + num_n320_r2 + pq_spl_full | 0.929 / 0.967 | **75.8%** | **144/144 = 100%**, 57.1% of laps | 45 ms/frame (4 threads) |
| cars_n640_v2 (same chain) | 0.925 / 0.932 (0.971 tracking at 10 fps) | 75.8% | 139/139, 55.2% | 21 ms/frame (4 thr), 45 (1 thr) |

Identity costs ~0.25–0.75 s per pass on 4 threads (~3% of a core at one pass per 40 s); RAM < 0.4 GB.
**Caveat:** 26SE has now been used to compare many variants. Its ranking was never contradicted by val or test, but it is
becoming a development set. The next event's start line, with human counts, is the real test.
Next: WP11 (real-time prototype + laptop benchmark), then a final retrain on all 5 events. Open: pass-logic minimum in
seconds (decide on non-26SE footage), and re-ID for mud cars (car 1 is never auto-confirmed; 7 of its no-abstain errors read "11").

Earlier status (2026-10-18): **teacher done.** WP0–WP6 and WP8 are done (each agent's section in LESSONS.md has the numbers).
- Labels: 7.4k consensus tracks. VLM vs overlay identity agree 96.9% (DIN/SPP close-ups).
- Datasets: being re-exported from the final labels. Car set ≈ 45k frames.
- Waiting on: the user's QA review of `out/qa/<EV>_teacher.jsonl`, and WP9 training on the 2080 Ti.
- WP10 recognizer is done: a 1.3M-param CRNN, 1–2.6 ms/crop on CPU, 96% track precision at 43% coverage on 24SU. Next on the
  GPU: the number-panel detector, then the 26SE identity chain end to end, then cars_s640.
- cars_n640 on the full 26SE: recall 0.913, precision 0.935, borrowed end-to-end 65.1% (teacher 0.933 / 0.975 / 67.9%);
  21 ms on 4 CPU threads. Misses are the overexposed 10:24–10:51 stretch → exposure augmentation in cars_n640_v2.
- 26SE identity chain (num_n320 + recognizer), no abstain: 62–67% top-1, end-to-end 62% (teacher 67.9%). Abstain
  scoring is being fixed (WP10b). Round 2: 32k proposed panel boxes → numbers_crop_r2 → num_n320_r2, plus more digits.
- GPU queue: cars_s640 (running) → num_n320_r2 → cars_n640_v2 → student-only 26SE end to end.
- **WP10b is done. Deploy `pq_spl_full`** (PARSeq-tiny + digit splicing + uniform synthetic numbers): 84% on unseen
  numbers vs 87% on seen ones (the WP10 CRNN gets 55%: it memorised numbers), 21 ms/box on 1 thread. 26SE on the teacher's
  passes: 72.6% of laps right with no abstain (VLM 67.9%), or 99.2% precision at 54.5% coverage. More events are now worth
  only ~+1–2 points per doubling. New liveries (future start-line footage) matter more than more numbers.
- Number boxes exist for only 2.6k of the 7.4k consensus tracks. Plan: train the number-panel student on them, then let it
  propose boxes for the rest (cheaper than ~5 GPU-h of grounded reads), keeping only boxes whose read matches the consensus.

Read `LESSONS.md` first.
The previous version of this file (session 1) is merged in here, so nothing from it was dropped.

This file is the brief for every agent. Each work package (WP) states its inputs, outputs and done criteria. Agents
work only on their WP's paths, write findings to `LESSONS.md`, and report back to the manager thread. The manager
decides at the gates (G1–G3) and serialises GPU use.

## Why

The session-1 pipeline works but is too heavy for track-side use: YOLO-World v2-x at 1280 px as the detector, and a 4B
VLM (Qwen3-VL-4B) reading ~10 crops per pass. A 16 GB laptop without a big GPU cannot run it comfortably. Measured:
Qwen3-VL-2B on CPU takes 0.8–2.3 s per crop and 5+ GB of RAM. Plan: run the heavy pipeline as a **teacher** offline
over years of the team's YouTube live streams, and train small **students** on its labels.

## Machine and ground rules

- This PC (`D:\video_baja`) **is the teacher machine**: Ryzen 9 3900X (12C/24T), 64 GB, RTX 2080 Ti 11 GB
  (Turing: fp16 only, no bf16 or FlashAttention-2; ~1.8 GB of VRAM is taken by the desktop). Bare-metal Windows 11.
  D: has ~1.85 TB free.
- **One GPU job at a time.** VRAM spills to shared memory on Windows and everything gets ~10× slower. The
  manager hands out the GPU. Long GPU jobs check for a `PAUSE` file at the repo root between videos and stop
  cleanly when it is there, so the PC can still be used for streaming.
- NVDEC on the 2080 Ti decodes H.264, HEVC and VP9, **not AV1**. AV1 files fall back to CPU decoding (libdav1d).
- Before a multi-day run: disable sleep (`powercfg`), and pause Windows Update or set active hours.
- Code layout, so agents don't edit each other's files: `lapvision/` (session 1, keep working), `teacher/` (WP1–WP7),
  `tools/qa/` (WP6), `students/` (WP9–WP11). Big data lives outside the code folders (see Layout below).
- 26SE start-line footage is the **benchmark. Never train or tune thresholds on it.**

## Layout (new)

```
youtube/                    raw downloads + <id>.info.json (user)
rosters/<event>.csv         numero,equipe  (one per event; user or WP5 auto-roster)
data/videos.csv             manifest, one row per video (WP1)
data/events.csv             event code, name, date, location, split (train / val / test)
work/<video_id>/            teacher outputs per video, each stage writes <stage>.done
   shots.csv                shot boundaries + shot tags (WP2)
   overlay_mask.png         static overlay regions (WP2)
   dets.parquet             frame, pts, shot, track, cls, conf, box (WP3)
   crops/                   selected crops only (WP4)
   reads.parquet            per-crop plain read + grounded numbers/boxes (WP4)
   labels.parquet           per-track consensus id, number boxes, quality flags (WP5)
   errors.log
datasets/{cars,numbers,digits,ids}/   exported training sets (WP8)
students/runs/              training runs (WP9, WP10)
```

Detections only, **never frames**, are stored during the run: at 4 fps, frames would take ~3 GB per stream-hour. WP8 re-decodes
the few frames it exports.

## Inputs (user)

- **Streams** in `youtube/`, named `<EV>_<PROVA>.mp4` (`END`, `DIN`, `SPP`). Current set: 9 files, 5 events, **42.4 h, all
  720p30 H.264 at ~1–1.8 Mbps**. See LESSONS.md "YouTube streams: first look" for content and hazards. Contact sheets are in
  the session-2 scratchpad and can be regenerated with ffmpeg. For future downloads: `-f "bv*[height<=1080][vcodec!^=av01]+ba/b"`.
  **Higher-quality 26BR originals are coming via Google Takeout.** When they land, WP1 marks the 720p 26BR rows as `superseded`
  and the teacher re-runs 26BR from scratch. Timelines may differ (trim), so don't map old labels across by timestamp.
- **Rosters per event** (car number ↔ team): **done.** `teacher/rosters.py` scrapes them from
  `https://resultados.bajasaebrasil.net/<EV>/prova.php?id=<EV>_GER` into `rosters/<EV>.csv`. There are 60 events, from 03SU to 26SE.
  11BR has no table online. The roster belongs to the **event, not the prova**: every video from 26BR uses `26BR.csv`,
  whether it shows the enduro, dynamics, pits or interviews. Use the Geral page, not the enduro page, because cars that skipped
  the enduro are missing from it (26BR: 51 vs 38). Numbers change between events (Tuffão: #1 at 26SE, #14 at 26BR).
  The WP5 auto-roster is only a fallback for 11BR and for videos with no event code.
- **Event per video:** from the filename (`26BR_END.mp4` → event 26BR, prova END). Every video belongs to an event.

## Work packages

### WP0 · Environment and regression (M0). Blocks all GPU work.

1. Install Python 3.13 (`py install 3.13`, or python.org), then recreate `.venv`. torch cu128 first, then
   `requirements.txt` + `pyarrow`, `scenedetect`. Put ffmpeg (a full build with `-hwaccel cuda`, e.g. gyan.dev) and yt-dlp on PATH.
   `git init` with a `.gitignore` for `*.mkv *.mp4 *.webm *.pt out/ work/ datasets/ youtube/ .venv/`.
2. CPU regression: `analyze.py` on the existing `out/` must reproduce the LESSONS.md numbers.
3. **fp16 check (GPU):** re-read ~500 crops from `out/world_1129` with Qwen3-VL-4B in fp16 and compare with the stored bf16
   `reads.csv`. Pass: ≥ 97% identical reads, no empty or garbage outputs. If it fails, try sdpa in fp32 for the vision tower,
   or fp32 on the whole model at lower batch size. Record s/crop and peak VRAM at batch 1/4/8 (11 GB card).
4. Throughput: YOLO-World v2-x fp16 at 960 and 1280 px, and decode speed for each codec found in `youtube/`.

Done when: the numbers are in LESSONS.md, and `lapvision/vlm.py` takes the dtype from a flag (default fp16 on this machine).

### WP1 · Ingest and manifest (CPU, can start now)

`teacher/manifest.py`: scan `youtube/`, parse `<EV>_<PROVA>` from the filename, ffprobe → `data/videos.csv` (id = stem,
path, event, prova, duration, w, h, fps, vcodec, bitrate, roster present). Also `data/events.csv`. Idempotent: re-running it
picks up new downloads.

### WP2 · Shots, shot tags, overlay mask (CPU + light GPU)

- Shot cuts with PySceneDetect (or ffmpeg `scdet`), plus fades. Tracking never crosses a cut.
- Shot tags with zero-shot CLIP (`lapvision/ViT-B-32.pt` is already there) on 1–3 frames per shot: track / pit / interview /
  studio / graphics / crowd. They are used for QA stratification and dataset balancing, not to drop data (cars in the pit are good
  number data).
- Replay detection where possible (replay wipes and logos), so replays don't count twice as data.
- **Overlay mask** per video: screen regions that persist **across cuts** (leaderboard tower, tickers, bracket boxes,
  lower thirds, the corner logo, and rotating sponsor logos whose position is fixed while the content changes). Overlay text
  contains numbers, and the Baja logo contains a buggy drawing. Masked regions are dropped from detections and number labels and
  become hard negatives. 9 videos is few enough for the user to check the masks by eye.
- **Screen-fixed artifacts:** a "car" track whose box stays fixed in screen coordinates for a whole shot while the scene moves is
  the onboard ego car or a graphic. Drop it as a car label, but keep the other cars seen from onboard (rear view, firewall numbers).

### WP2b · Overlay identity reader (GPU, light). New: free labels independent of the car's number

Per shot, read the lower-third / bracket graphics every ~2 s (VLM on the overlay crop, or the full frame with a
"transcribe the on-screen graphics" prompt). Parse `#N team` / `N - TEAM` and accept it only if the number **and** the
team name match the roster. Output `work/<id>/overlay_ids.parquet` (shot, t, numbers announced, test name).
Uses: (1) identity labels for single-car shots (DIN tests: one car on the course), including mud cars; (2) an automatic
precision check of the VLM consensus wherever both exist; (3) dataset D (re-ID).

### WP3 · Car detection + tracking (GPU, teacher stage 1)

- **Done** (4.85 GPU-h, 50k car tracks). YOLO-World v2-x, prompts `["off-road buggy", "person"]`. The absorbing classes
  were tested and rejected: they relabel real buggies as car/pickup/tractor. 1280 px was kept, because 960 loses far cars.
  Precision at conf ≥ 0.25 is only ~71–83% real cars, so **every track gets a VLM yes/no car check in WP4/WP5**, and the
  detector dataset is built from verified tracks only.
- Settings from WP0: fp16, imgsz 1280, batch 8, 4 fps, CPU decode through an ffmpeg `fps=4` pipe.
- Two steps, so the GPU step doesn't depend on shots: (1) GPU: raw per-frame detections for every video → `dets_raw.parquet`;
  (2) CPU: tracking per shot on the stored detections (ByteTrack or an IoU tracker, `track_buffer` adapted to 4 fps) →
  `dets.parquet` with track ids. Keep tracks with ≥ 3 detections. Store every detection down to conf 0.05; WP5 decides.
- The detector was only validated on the side-on start line. Close-ups where the car fills or overflows the frame are
  new, and the G1 QA must look at them specifically.

### WP4 · Crops + number reading with location (GPU, teacher stage 2)

- Per track, K = 6–8 crops (large, sharp, spread in time: `lapvision/passes.pick_crops`), only from boxes ≥ ~100 px high
  (smaller cars are still detector data but have no readable number). Crops are cut at native resolution with 5% padding.
- VLM pass (**decided in WP0**): (a) the plain read from `lapvision/vlm.py` (fp16, batch 4–6) on all K crops → identity;
  (b) the **grounded** read only on 2–3 crops of tracks that already have a consensus, purely to get number boxes.
  Keep a box only if its label equals the consensus. The grounded read cannot replace (a): its precision is 0.78 vs 0.93,
  it reads numbers on crops where (a) says 'none', and it is ~9× slower. Prompt and format in LESSONS.md (WP0):
  `bbox_2d` in **0–1000 relative** crop coordinates.
- Stage-batched: WP3 runs over a batch of videos, then WP4 over the same batch, so the GPU never holds both models.

### WP5 · Consensus labels + auto-roster (CPU)

- Identity sources: the VLM consensus and the overlay identity (WP2b). When they agree, the label is gold. When they
  disagree, flag it for QA. When the overlay identity is the only source, use it only if exactly one car is tracked in the shot.
- A track's number = majority over its crops, snapped to the event roster. Keep it only if vote share ≥ 0.5 and ≥ 3 votes
  (~98% precise in session 1). Number boxes are kept only where their read equals the consensus.
- **Auto-roster** only where `rosters/<EV>.csv` is missing (11BR, videos without an event): numbers that reach consensus on ≥ N
  tracks across ≥ 2 shots, sent to the user to confirm.
- Negative frames: no detection in the frame **and** in its ±1 s neighbours, outside fades. Cap their share per shot tag.
- Flags per track: shot tag, overlay overlap, replay, box size, blur. They feed WP6 and WP8.

### WP6 · QA review page (CPU, can start now on session-1 data)

A local HTML page under `tools/qa/` (served by `python -m http.server`, images stay on disk): random ~150 tracks
per event, stratified by shot tag. It shows the crops, the consensus number and the number boxes, takes a keypress
verdict (right / wrong number / not a car / unsure), and exports JSON. It outputs label precision per event and shot type.
Build and test it on `out/world_*` first. Target: ~15 min of user time per event.

### WP7 · Queue runner (M2)

`teacher/run.py`: round-robin by event (so a partial run is already diverse), stage-batched, `.done` markers, a failure
on one video is logged and skipped, honours `PAUSE`, and logs throughput and an ETA. Runs for days unattended.

### WP8 · Dataset export (CPU + decode)

- Re-decode the selected frames only. Per track, ≤ 1 frame per second, perceptual-hash dedupe of static shots, interpolated
  track boxes to fill the teacher's single-frame misses (missed cars in frames used for training poison it).
- Balance by event and shot tag. Split by **whole event** (`data/events.csv`), with 1–2 test events chosen before any training.
- Outputs: A `cars` (YOLO format, plus negatives), B `numbers` (boxes, on frames and on car crops), C `digits` (number crops +
  string), D `ids` (car crops grouped by event + number, for re-ID later).

### WP9 · Student detector (GPU, M3)

- Default candidate: YOLO11n/s at 640, class `car`. For numbers, compare two designs at G2:
  (i) one 2-class net (`car`, `number`) on the full frame; (ii) a car net on the full frame plus a small number-panel net
  run on each car crop. (ii) is the recommended starting point, because numbers are tiny in a 640 px wide shot and cars per
  frame are few.
- Evaluation: plug it into `lapvision/det.py` and run the 26SE pass metrics (`analyze.py`) against both baselines. Also run
  mAP on the held-out events versus teacher labels and the QA-checked tracks.

### WP10 · Number recognizer (GPU, M4)

PARSeq-tiny or CRNN class (a few M params) on dataset C, digits charset, augmentation for mud, blur and perspective. Read
every frame of the pass and vote per track, snapped to the roster. End-to-end evaluation on 26SE with `analyze.py`.

### WP11 · Laptop benchmark + real-time prototype (M5)

Export at **384×640**, not a padded 640 square (WP9: half the cost, no retraining). On this 3900X, YOLO11n runs at
22 ms with 4 threads (50 ms with 1) and YOLO11s at 49 ms with 4, both < 0.3 GB of RAM. ONNX/OpenVINO export, measured on the real target laptop (≤ 50 ms/frame detector, a few ms per number crop, < 1 GB RAM).
Real-time loop: capture → detect → track → `passes.py` rules → read → emit a pass (timestamp + confidence) as a third
`fonte` to the lap app, plus a number feed for the PTZ tracker.

### WP12 · Re-ID for mud cars (later, optional)

Metric-learning head on dataset D, a rolling gallery per race. Session 1: DINOv2 kNN alone 50%, fused with the VLM 80% → 91%.

## Order, parallelism, gates

```
now      WP0 (GPU: fp16 check)   ‖ WP1 manifest   ‖ WP6 QA page          (CPU work in parallel)
then     WP2–WP5 code, tested on 26SE + the first streams
M1       pilot on 1 h slices of 26BR_END (onboard, PTZ, tower), 25SU_DIN (lower thirds, pits), 19SU_END (fixed start line) → QA
G1       decide fps, imgsz, prompts, K, grounded-only read, thresholds; measure label precision + throughput
M2       WP7 full run (days, GPU busy)  ‖ WP8 export + WP9/WP10 code on the pilot data (CPU)
G2       first export at ~30% of the run → first student; choose detector design (i) vs (ii)
M3/M4    train students; evaluate on 26SE + held-out events
G3       student vs teacher on 26SE good enough? → WP11
```

**Decided:** all training also runs on this PC's 2080 Ti (the user's preference; the RTX 5080 PC is only a fallback if
something proves impossible here). Training runs after the teacher. If more streams arrive later, use `PAUSE` to hand the GPU
over. Turing: train with AMP fp16 (no bf16); YOLO11n/s at 640 with batch ~32 fits in 11 GB.

**Throughput (measured in WP0, LESSONS.md):**
- Detection: YOLO-World v2-x fp16 @1280 batch 8 runs at 35 img/s end to end (8.9× real time at 4 fps). **42.4 h of streams =
  4.7 GPU-h at 4 fps** (2.4 h at 2 fps). CPU decode beats NVDEC here and is never the bottleneck.
- VLM: the plain read is ~0.16 s/crop at batch 4–6, ≈ 22k crops per GPU-hour. The grounded read is ~1.4 s/crop at batch 4,
  ≈ 2.5k per GPU-hour. Decode is bound by kernel-launch overhead on Windows, so cost ≈ output tokens × 125 ms ÷ batch.
- So the whole teacher run fits in one night, and the GPU conflict below only matters if many more streams arrive.

**Final models:** once the design and thresholds are chosen on val/test, retrain the deployed students on all 5 stream
events (train + val + test). That adds the SU teams' liveries; it adds no new numbers. 26SE stays the external benchmark.
Future events' start-line recordings, which come with human lap counts and so carry per-pass identity labels, become
training data. Keep recording the start line at every event, NTP-synced.

**Held-out split (proposal, decided at G2):** test = 24SU (END + SPP); train/val = 19SU, 25BR, 25SU, 26BR. 19SU stays in
training because its fixed start-line camera is the closest footage to the target.

## Evaluation (never train on it)

- **Benchmark:** the 26SE start-line footage + `lap_count_dump_26se.sql`, with the existing metrics (`lapvision/analyze.py`:
  pass recall and precision, identity accuracy, end-to-end).
- Baselines: COCO YOLO11x + VLM (whole race: recall 0.87, precision 0.94, end-to-end 62%), and YOLO-World + VLM (last
  file: 1.00 / 0.98 / 79%).
- Domain gap: the streams are broadcast PTZ, the target is a fixed side-on camera. If the student is weak on 26SE, the fix
  is start-line footage from **future** events as training data. Keep recording the start line at every event, NTP-synced.

## Open questions for the user

1. ~~Rosters~~: answered, see Inputs. Still open: which code do videos without an event get (off-season, team videos)?
2. ~~RTX 5080 for training~~: answered, train on the 2080 Ti here.
3. Roughly how many stream-hours and events, and which codec are the downloads coming in as?
4. Later (WP11): which laptop is the target, and how does the lap app accept a machine `fonte` (API, or direct DB insert into
   `contagem_passagem`)? What interface does the PTZ tracker expect?
