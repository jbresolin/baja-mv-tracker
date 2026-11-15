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

## Status and open items

Done: teacher, datasets, students, the 26SE evaluation, the real-time prototype, final models on all events.
Open (need hardware or decisions): benchmark on the real laptop, the lap-app integration (API vs DB insert;
`rt/sinks.py` is a stub), the PTZ interface and scope, the camera connection and fps, re-ID for mud cars, a pass-length
minimum in seconds.

Third-party components: Ultralytics YOLO (AGPL-3.0), YOLO-World, Qwen3-VL (Apache-2.0), PARSeq (Apache-2.0), CLIP.

## Licence

This project is licensed under the **GNU Affero General Public License v3.0** (see `LICENSE`). It has to be: the
detectors are trained with Ultralytics YOLO (AGPL-3.0), the ONNX models derive from those weights, and the tracker
reuses AGPL-3.0 code. In practice: anyone who receives a copy (including the Windows installer) or uses a modified
version over a network is entitled to its source.

`lap_count_dump_26se.sql` is the 26SE race's human lap count, kept as the benchmark's ground truth. Team and school
names in it are public on the results site; the counting devices' token hashes were removed before publishing.
