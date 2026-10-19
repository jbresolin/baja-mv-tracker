# Baja SAE machine-vision lap counting / car number reading

Exploratory project. Read these first. They hold all the history of the first session (2026-10-17), and chat history does
not travel with the folder:
- `LESSONS.md`: everything learned on the start-line footage: findings, numbers, and corrections. **Read it fully.**
- `TRAINING_PLAN.md`: the next phase: distilling a small detector + number reader from a big "teacher" pipeline run
  over years of YouTube live streams.

## Goals
1. Assist (not replace) human lap counting at the start line: detect each car pass, read its number, emit a pass with a
   timestamp + confidence, as a third "fonte" next to the human counters.
2. "Entry camera" for the live-stream PTZ tracker: read car numbers at the start of the PTZ section so tracking can
   prioritise cars of interest.
The final system must run on a **modest laptop**. The laptop isn't chosen yet; the worst-case budget the user gave
(2026-10-19) is a **4-core 8th-gen Intel, 8 GB RAM, Intel iGPU**. Heavy models are only for offline
labelling (teacher). Offline teacher hardware: dedicated streaming PC, Ryzen 3900X + RTX 2080 Ti (11 GB, Turing, fp16 only,
no bf16) + 64 GB RAM, available for weeks.

## Layout
- `2026-10-17 *.mkv`: start-line recordings (OBS clock overlay burned in, top-left). Main benchmark footage.
- `lap_count_dump_26se.sql`: MySQL dump of the race's human counts. Use `contagem_volta` (consolidated laps) and
  `contagem_passagem` (raw entries; `ts_ms` = pass time in server clock). Parser: `lapvision/gt.py`.
- `lapvision/`: pipeline code (stages listed at the end of LESSONS.md). `.venv/`: local venv (not portable; recreate
  from `requirements.txt`).
- `rosters/<EV>.csv`: car number ↔ team for 60 events (03SU–26SE), scraped from the official results site by
  `teacher/rosters.py` (Geral page). One roster per event, valid for every prova of it.
- `out/`: generated: timelines `tl_*` (COCO) / `tlw_*` (YOLO-World), crops + tracks + reads per file
  (`crops_*` = COCO run, `world_*` = YOLO-World run), `passes_all.csv`, review videos. Can be regenerated.

## Hard-won facts (details in LESSONS.md)
- The video clock is linear in pts in every file: wall = file offset + pts. **The video PC clock ≈ server clock + 80 s.**
  Always apply that before comparing with the GT. One wrong conclusion in session 1 came from skipping it.
- Evaluate only after 09:18:30 (the camera was being set up before that).
- COCO detectors are unreliable on baja cars (muddy cars come out as "bear"/"cow"). Use YOLO-World v2-x prompted
  "off-road buggy", or the trained student. Whole race after 09:18:30: recall 0.933, precision 0.975, end-to-end 67.9%
  (COCO: 0.87 / 0.95 / 62.7%).
- On the 2080 Ti (this PC) the VLM runs in **fp16**: 99.2% identical to bf16. `torch.cuda.is_bf16_supported()` wrongly says True
  (emulation), and `vlm.py` checks native support. Don't batch the VLM above 6 here (11 GB, desktop uses ~1.8 GB).
- **Laptop students (session 2) match the teacher on 26SE:** `cars_s640` / `cars_n640_v2` (car) → `num_n320_r2` (number
  panel) → `pq_spl_full` (PARSeq-tiny + digit splicing) → boxed vote, all ONNX on CPU. Recall 0.93, precision 0.97, right
  number on 75.8% of laps, auto-confirm at 100% precision on 57%. Details: TRAINING_PLAN.md status and LESSONS.md (WP9/WP10b).
- Number reading: Qwen3-VL-4B-Instruct on ~10 crops per pass, majority vote snapped to the event roster
  (≈94–96% accurate when it answers; abstains on mud). Car #1 is unreadable (mud), which is a re-ID problem.
- Pass rules: one car = one count. Merge tracks that re-start where the last box ended (<10 s), and merge coexisting
  tracks whose boxes nest (same car, two boxes). The lap app also drops duplicates within 5 s (`duplicata_ms`).
- To keep a process off the GPU on Windows use `CUDA_VISIBLE_DEVICES=-1`. An empty value is dropped by Windows, so torch
  still sees CUDA. `.venv\Scripts\python.exe` is a launcher: killing it leaves the real interpreter running, so stop long
  jobs with the `PAUSE` file instead.
- On Windows with a 16 GB GPU, don't run two GPU stages at once (VRAM spills to shared memory and gets ~10× slower).

## Conventions
- Times internally are local seconds-of-day (`tod`, UTC-3). Use `clock.fmt()` to print.
- Write findings to `LESSONS.md` as you go (including corrections of earlier claims, marked as such).
