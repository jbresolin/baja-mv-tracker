# Lap counting / car number reading from the start-line camera: lessons log

Exploratory experiment, 2026-10-17 footage (Baja SAE Sudeste 2026, Enduro).
Code lives in `lapvision/`, outputs in `out/`.

## Footage facts

| file | res / fps | wall clock span |
|---|---|---|
| 08-46-43 | 4K30 | 8 s (test) |
| 09-09-26 | 4K30 HEVC | 09:09:26 – 09:56:08 |
| 09-56-32 | 1080p60 HEVC | 09:56:32 – 11:09:02 |
| 11-09-03 | 1080p60 HEVC | 11:09:03 – 11:29:47 |
| 11-29-49 | 1080p60 HEVC (stream says 60, not 30) | 11:29:49 – 12:10:10 |

* There are no gaps between files that matter: race laps run 09:02 to 12:07, and the video covers 09:09 to 12:10.
* A framing survey (one thumbnail every 150 s; `lapvision/scenes.py`) shows:
  * 09:09–09:18: camera still being set up and pointed at other parts of the track. The GT laps in that window
    (e.g. the 09:15 bunch of cars 1, 3, 4, 5, 10, 13, 20, 22) are **not in the footage at all**, so evaluation starts at 09:18:30.
  * 10:24–10:51: heavily overexposed (washed-out white), but the cars are still visible.
  * People walk or stand in front of the lens all the time (marshals, the counters, visitors).
* The camera framing changes between files and within them (someone touched or moved the camera). Any fixed
  "finish line" ROI must be per-segment, so prefer framing-agnostic logic: use the frame centre line and
  frame edges.

## Clock overlay

* The OBS overlay `YYYY-MM-DD HH:MM:SS.mmm UTC-03:00` is a monospace font at fixed positions. At 1920 width,
  cell i sits at x = 14 + 24.2·i, y = 16..48. Template matching on a binarised cell (min(BGR) > 200) reads it.
* A first template set built from 4 hand-labelled frames was **not enough**: it confused 5 with 3, because
  the translucent bar lets the sky bleed through. Fix: auto-label ~100 frames using the (later proven)
  linear pts→clock relation and average the templates. That gave zero errors and a 5× larger decision margin.
* **Wall clock is perfectly linear in pts in every file** (drift < 30 ms/hour, no freezes or jumps found).
  Reading the clock once per file and using `offset + pts` is enough. Reading it per frame is only
  a sanity check.

## Ground-truth alignment

* `contagem_passagem.ts_ms` is the server-anchored pass time. `ts_dispositivo_ms` (column 7) is ~7.3–7.8 s
  off and must not be used. `contagem_volta.ts` is the consolidated time of 2 human sources.
* **The video PC clock is ~80 s ahead of the race server clock** (video_tod ≈ server_tod + 80 s). This was
  found by cross-correlating foreground vehicle events against GT lap times. It is consistent across files:
  the median is 80.0–80.7 s and the p10–p90 spread is ±1.5 s, which is human reaction plus the definition
  of "pass time". Lesson: **always NTP-sync the recording PC**, or film a server clock at the start.
* The two human counters disagree with each other by 0.1–3.3 s per pass (`disp_ms`). Matching tolerance
  must be about ±5 s, so timing alone cannot separate cars that follow each other closely.

## Detection

* COCO YOLO11x detects baja cars well, mostly as **"truck"** (sometimes "car", rarely "motorcycle").
  With yolo11m the confidence is noticeably lower. No custom training was needed to *detect* them.
* The vehicle box heights are clearly bimodal: background traffic has h < 0.1·H, cars passing the line have
  h > 0.25·H. A first filter (h > 0.25 and bottom > 0.7·H) was too strict. Cars take different lines through the
  section, so some pass mid-frame with h ≈ 0.35 and bottom ≈ 0.55. Settled on h > 0.2·H and bottom > 0.4·H.
* Lots of **background vehicles** (behind the fence, a parking area, other track sections) are detected too,
  and the height filter above removes them.
* People are everywhere (marshals, the counters, people standing in front of the lens). ~~In the last file a person
  stood in front of the camera for several passes, and those passes are invisible.~~ **Correction:** that observation came
  from contact sheets made *before* the +80 s clock offset was known, so I was looking at the wrong moment. Those cars
  pass fully visible 80 s later. With a proper detector (YOLO-World, below) **all 63 laps of the last file are detected**.
  So occlusion by people is real but rare; it mostly causes double counts (the car "leaves" behind a person and "re-enters"),
  not misses. Mounting the camera high is still good practice.
* **Lesson: fix the time alignment before drawing conclusions from what you see at a timestamp.**
* **COCO detectors are brittle on these cars.** Car 4 at 11:40:31 (clear, number visible) is detected in ~1 frame of 30.
  With all COCO classes enabled, YOLO11x calls it **"bear" (0.46) / "cow" (0.28)**: a brown, chunky shape on mud. Baja cars
  are not a COCO class and only *sometimes* look like a "truck". The open-vocabulary **YOLO-World v2-x with the prompt "off-road
  buggy"** detects the same car in 49/56 frames (conf ≈0.7). A detector trained or prompted for this class is needed.
* The detector sometimes puts two boxes on one car (whole car + partial), and ByteTrack gives them two ids → double count
  (car 14 at 11:45:19). Fix: union tracks that coexist in time where the smaller box lies mostly inside the larger one
  (median intersection/smaller-area > 0.6). Precision: 1109 0.95 → 1.00, 1129 0.83 → 0.85.
* Throughput: 5 fps coarse scan with yolo11x @1280 on an RTX 5080 runs about 6–7× real time with NVDEC decoding.

## Reading numbers

* Each car shows its number in several places: the rear firewall panel (big, coloured, often the best
  one), white side plates, and sometimes the nose. All are aggregated per pass. They are votes, never
  extra passes.
* Readability ranges from trivial (car 11: red "11" on the firewall plus a white side plate) to **impossible**
  (car 1 around 11:17: fully caked in mud, no digit visible at any frame). Humans still recognise
  car 1, presumably by shape, colour or driver, which suggests appearance re-ID as a complement to OCR.
* EasyOCR (digits allow-list) over ~20 crops per pass, with confidence-weighted voting, got 11 ✓, 4 ✓,
  20 → "21/2/20" (partial), 10 → weak "1", 1 → nothing. Stylised fonts (car 10's "10") and the "0"
  shape hurt it.
* **A small local VLM (Qwen3-VL-4B-Instruct, bf16, ~8 GB VRAM) is far better than generic OCR** on the same crops,
  with a majority vote per pass: 11 (16/17 crops), 20 (15/17), 4 (8/10), 10 (14/18), all correct, and car 1 → "none" on all
  45 crops (it abstains instead of hallucinating). Single-crop errors look like '94' for 4 or '50'/'0' for 10, so voting
  over several crops is essential.
* Running the VLM and YOLO at the same time on a 16 GB card saturates VRAM. On Windows the driver then spills into shared
  memory silently and everything gets ~10× slower (4 s/crop). Run GPU stages sequentially.

## Pass logic (one car = one count; re-count only after it left the frame)

* Tracker fragmentation is the main source of over-counting, not misdetection. Typical cases:
  * a car **stopped at the frame edge** (stuck or being helped, 11:18–11:19) for 75 s, with ByteTrack losing and re-acquiring
    it ~9 times. Because the box touched the edge, "ended at edge = left the frame" was wrongly true each time;
  * **end of race (≈12:04, chequered flag)**: cars stop right in front of the camera, surrounded by marshals and team members.
* Fix: a new track that **starts where the previous pass's box ended** (IoU > 0.2, gap < 10 s) belongs to the same pass,
  whether or not it touched the edge. A genuinely new car enters from the entry side, so the boxes don't overlap.
  Effect on precision (recall unchanged): 0909 0.97 → 1.00, 0956 0.94 → 0.97, 1109 0.67 → 0.95, 1129 0.75 → 0.83.
* Pass-detection recall vs. the human laps with COCO YOLO11x: 0909 **0.99**, 0956 0.83, 1109 0.83, 1129 0.79.
  The misses were mostly the **COCO detector not recognising the car** (see "bear/cow" above), especially mud-coloured cars
  like #1, #4, #6 and #13. With YOLO-World the last file reaches 1.00 recall (see below).
* The time of "car enters frame" is a better match to the humans' click than the time the box centre crosses the frame centre
  (median +0.4 s vs +1.3 s after the click). The counters stand roughly at the camera's entry side.

## Detector swap: COCO YOLO11x → YOLO-World v2-x ("off-road buggy"), last file (11:29–12:10, 63 human laps)

| | COCO YOLO11x | YOLO-World v2-x |
|---|---|---|
| passes detected | 59 | 64 |
| pass recall / precision | 0.79 / 0.85 | **1.00 / 0.98** |
| VLM read when matched, accuracy | 88%, 100% | 83%, 96% |
| end-to-end (lap found + right number) | 70% | **79%** |

* The only false pass is a 40 s stationary detection at the very start of the file. All of car #1's 10 laps are now
  detected, but none is read (mud).
* The two VLM errors are both car 6 read as "8" (share 0.5 and 0.2). Car 6's livery has a stylised 6.

## Identity results with COCO YOLO11x (whole race after 09:18:30; 252 human laps; `lapvision/analyze.py`)

| | value |
|---|---|
| passes detected / human laps | 232 / 252 |
| pass recall / precision | **0.87 / 0.94** |
| VLM (Qwen3-VL-4B, 10 crops/pass, roster-snapped majority vote): passes with a number | 76% of matched passes |
| ... accuracy when it gives a number | **94%** |
| ... with vote share ≥ 0.5 | 98% accuracy at 67% coverage |
| EasyOCR, same crops | 46% coverage, 88% accuracy |
| end-to-end (human lap has a machine pass with the right number) | 62% (VLM) |

* Accuracy per car mirrors "how readable is the livery". 3, 4, 5, 9, 11, 13, 14, 22 are near perfect. **Car 1 is 1/26.**
  It is caked in mud and its number is never legible. Car 20 is 25/34, car 6 15/23, car 10 7/13 (stylised digits).
* VLM errors are low-consensus votes (share 0.2–0.4), digit fragments ("2" or "7" for 20, "17" for 10), or several cars in one pass.
  Snapping to the roster removes most nonsense reads. A vote-share threshold turns the VLM into a ~98% precise
  "auto-confirm" signal, and everything else goes to a human.
* Reading speed (measured on an RTX 5080, Qwen3-VL-4B bf16, crops ≤640 px, 9.9 GB VRAM peak): **0.20 s/crop** unbatched,
  **0.085 s/crop** in batches of 8. 10 crops/pass is ≈1–2 s per pass, so the number is available ~1–2 s after the car
  leaves the frame. That is real-time capable for lap counting and borderline-OK for the PTZ entry camera (read the first
  3–4 good crops while the car is still in view).

### Appearance re-ID (DINOv2-base CLS+mean-patch embedding, kNN over passes; `lapvision/reid.py`)

* Gallery = passes before 11:09 labelled by GT time matching; test = last hour (70 passes).
  Re-ID alone: **50% top-1** (49% with the split at 10:30). Mud accumulation and lighting change the appearance
  over the race, and generic embeddings focus on the background and the pose.
* But **as a fallback when the VLM abstains it is useful**: fused identity accuracy in the last hour goes from 80% → 91%.
  Car 1 goes 0/10 → 5/10.
* Next step if pursued: fine-tune a metric-learning head on car crops (the time-matched passes give free labels),
  crop tighter (masked car, no background), and keep the gallery rolling (update the embeddings with recent confident passes),
  because appearance drifts with mud.

## Recommendations for the two use cases

**Lap counting (assist, not replace, the humans):**
* With an open-vocabulary detector (YOLO-World "off-road buggy") the machine found every lap of the last file
  (63/63, 1 false pass). Remaining risks are physical: the camera moved or pointed elsewhere, and people in front.
  Mount it high and behind a barrier, lock its framing, and NTP-sync the PC.
* Use the machine as a **third counting source** (`fonte`) with identity confidence: auto-confirm when the VLM's vote
  share is ≥ 0.5 and it agrees with a human source. Flag disagreements and "car seen but unread" for the reviewer.
  It also gives an exact pass timestamp. Human pairs disagree by up to 3 s, while the camera gives ±0.1 s.
* Keep the "same box re-appearing = same pass" rule. Most over-counting came from stopped or occluded cars, especially at the
  finish (chequered flag), when cars stop in front of the line.

**Entry camera for the PTZ tracker:**
* Detection + tracking (YOLO11x COCO "truck"/"car") works out of the box. The number read is available
  about 1–2 s after the car enters the frame if the VLM runs on the first good crops.
* Expect "unknown" for ~25–35% of passes (mud), so the PTZ priority logic must handle "unknown car". Re-ID against the
  car's recent appearance (from its last confident read on a previous lap) is the natural complement.
* Readable number locations, in order: the rear firewall panel (large, high contrast), the white side plates, the nose. A
  camera placed to see the firewall as the car moves away (or a corner where cars slow down) would read much better than one
  side-on.

## Review videos

`lapvision/render_review.py` overlays, per frame: tracked boxes coloured by pass, `P<id> #<voted car> [car:votes ...]`,
a HUMAN panel (GT laps within ±6 s, after the +80 s clock fix; green = now), a MACHINE panel (machine passes with their
matched GT car), and running counts. The `--condensed` variant keeps only ±4 s around every machine pass or human lap.

* `out/review_1129_condensed.mp4`: last file, 9 min 12 s (from 40 min of footage).
* `out/review_1129_full.mp4`: last file, full length.

## Pipeline / how to re-run (`lapvision/`, venv in `.venv`)

1. `clock.py survey <video> 240`: checks the clock overlay (templates in `lapvision/templates/`).
2. `detect_timeline.py <video> out/tl_XXXX.csv 5`: coarse YOLO scan at 5 fps (≈6–7× real time).
3. `align.py out/tl_XXXX.csv`: fits the video ↔ server clock offset against the GT.
4. `extract.py <video> out/tl_XXXX.csv out/crops_XXXX`: YOLO + ByteTrack at ~30 fps on the foreground events, saving crops.
5. `read_passes.py out/crops_XXXX 10`: builds passes, picks 10 crops per pass, reads them with the VLM + EasyOCR.
6. `analyze.py`: votes, GT matching, metrics → `out/passes_all.csv`, `out/per_car.csv`.
7. `reid.py embed` / `reid.py eval 11:09`: appearance re-ID experiment.
8. `render_review.py <video> out/crops_XXXX out/review.mp4 [--condensed]`.

## Rosters (session 2)

* The official results site has a roster for every event: `https://resultados.bajasaebrasil.net/<EV>/prova.php?id=<EV>_<PROVA>`.
  Event codes are `YY` + `BR`/`SE`/`NE`/`SU`. `teacher/rosters.py` writes `rosters/<EV>.csv` for 60 events (03SU–26SE).
  11BR has no table online.
* Use the **Geral** page (`_GER`). The enduro page only lists cars that ran it (26BR: 38 of 51 cars).
* Car numbers are per event, not per team: Tuffão is #1 at 26SE and #14 at 26BR. `26SE_GER` matches the SQL `equipe` table
  (cars 1–24; the SQL also has 99 and 100, which look like test entries).

## YouTube streams: first look (session 2, contact sheets of 30 frames per video)

* 9 files in `youtube/`, named `<EV>_<PROVA>.mp4` (`END` endurance, `DIN` other dynamics, `SPP` Super Prime). Events
  19SU, 24SU, 25BR, 25SU, 26BR. **42.4 h in total, all 1280×720 H.264 30 fps at a low 0.9–1.8 Mbps.** NVDEC handles them.
  At 720p, cars in wide shots are ~30–80 px tall and their numbers are unreadable. Readable numbers come from PTZ close-ups,
  pits and parc fermé.
* Content mix:
  * END: wide trackside plus PTZ shots. **19SU_END is almost all one fixed camera at the "LARGADA" (start) banner**, the closest
    footage to the 26SE target. 26BR_END has long **onboard** segments: the car's own roll cage and the driver's helmet fill the
    frame, and other cars are seen from behind (firewall numbers). It also has a split screen "CARRO 7 | CARRO 9".
  * DIN: up to ~half is studio talk and interviews. Studio TV screens show team cards with big car numbers and car photos.
    There are also drone overhead shots, cone courses, and the pits / parc fermé with many cars and big, readable number panels.
  * SPP: head-to-head heats on one wide camera.
* **Free identity labels from the overlays.** DIN lower-thirds name the car in shot ("#7 Imperador", "#14 Baja Univates Team",
  "#4 Sinuelo", with the test name "SUSPENSÃO 1" / "TRAÇÃO"). SPP bracket graphics name both cars of the heat
  ("9 - PATO BAJA" vs "2 - UNIVATES"). Number + team name are checkable against the roster. This identity does not depend on the
  car's number being readable, so it covers mud cars, and it estimates VLM label precision automatically.
* Hazards for the teacher:
  * graphics with numbers: leaderboard towers (END, right side), bottom tickers, bracket boxes, studio team cards;
  * graphics that look like cars: the Baja SAE logos contain a buggy drawing;
  * rotating sponsor logos in a fixed corner;
  * the onboard ego car, which stays in the same screen position for the whole shot;
  * people's shirts printed with numbers (marshals and team members, e.g. "29", "2");
  * a road car in one DIN test (TRAÇÃO tows it).

## Session 2: QA review tool (WP6)

* `tools/qa/` (stdlib server + vanilla-JS page, see `tools/qa/README.md` for the item format WP5 must emit):
  `make_batch.py` (session-1 test batch), `serve.py` (127.0.0.1, verdicts appended + fsynced to `<batch>.verdicts.jsonl`
  per keypress, resume on reload), `report.py` (precision / coverage / not-a-car / missed reads with Wilson 95% CIs per event,
  shot tag or any `meta.<key>`; GT comparison when `meta.hidden.gt_car` exists), `simulate_gt.py` (test harness).
* Session-1 test batch: 150 of the 241 YOLO-World passes after 09:18:30, stratified by file × read/no-read
  (35 no-read). Votes reproduce `passes_all.csv` `vlm_car` except one pass: `analyze.py` snaps to the SQL `equipe` table,
  which includes the test entries 99/100, while the batch snaps to `rosters/26SE.csv` (cars 1–24). Pass 73 of `world_0909`
  voted "100" (2/10) and becomes "no read" with the roster file.
* With answers simulated from the GT (and 6 deliberate reviewer mistakes), GT precision of proposed numbers on the sample is
  96.4% [91–99] (n=112), the report shows the reviewer's 92.0% and lists the 5 false alarms. With n≈110 judged items the
  Wilson interval is about ±5 points, so ~150 items per event separates "~95%" from "~85%" but not 95% from 98%. Per shot
  tag the intervals are much wider: stratify with a floor per tag and read the tag rows as rough.
* Layout checked with headless Edge at 1366×768 and 1100×700 (`msedge --headless=new --screenshot`); keyboard flow tested by
  dispatching key events into a scratch copy of the page. 10 crops fit a 4×3 / 5×2 grid at laptop size.

## Session 2: environment, fp16 and throughput on the 2080 Ti (WP0)

Teacher PC (Ryzen 3900X, RTX 2080 Ti 11 GB, Windows 11 / WDDM), torch 2.11 cu128, transformers 5.20.1, ultralytics 8.4.181.
Bench scripts: `tools/bench/` (`decode.py`, `yolo.py`, `vlm_fp16.py`, `grounding_probe.py`). Logs in `out/bench_*.txt`,
`out/fp16_check*`, `out/grounding_probe*`. **Caveat for all timings:** the WP2 agent's `shots.py` (4 workers + ffmpeg) kept
the CPU at 100% the whole time, so CPU-side numbers are pessimistic.

### CPU regression (`analyze.py`)

* COCO dirs (`crops_*`): recall **0.869**, VLM coverage **76%** (167/219), accuracy **94.6%**, share ≥ 0.5 → 98.6% at 67%
  coverage, EasyOCR 46% / 89%, end-to-end **62.7%**: all as recorded. Two small differences: **230 passes, not 232, and
  precision 0.952, not 0.94** (219 matched in both). The 2 extra session-1 passes were false ones that the current pass rules
  (probably the later "nested boxes" merge) now merge. Per car the same, except car 10 now 8/13 (was 7/13).
* The session-1 `out/passes_all.csv` / `per_car.csv` were already the YOLO-World run. The re-run reproduces them byte for
  byte. Backups: `out/passes_all.session1.csv`, `out/per_car.session1.csv`; COCO re-run: `out/passes_all.coco_s2.csv`.
* **New: whole-race YOLO-World v2-x numbers** (after 09:18:30, 252 human laps; `world_*`):

| | COCO YOLO11x | YOLO-World v2-x |
|---|---|---|
| passes / matched | 230 / 219 | 241 / 235 |
| pass recall / precision | 0.869 / 0.952 | **0.933 / 0.975** |
| VLM coverage / accuracy | 76% / 94.6% | 77% (182/235) / **94.0%** |
| VLM share ≥ 0.5 | 98.6% at 67% | **98.7% at 65%** |
| EasyOCR coverage / accuracy | 46% / 89% | 45% / 91.5% |
| end-to-end (lap with right number) | 62.7% | **67.9%** |
| median click → enter frame | +0.47 s | +0.35 s |

  * Per car: car 1 is now detected 32/34 times (COCO 26), but read 4/32. Car 20 29/37 (4 wrong: "7" ×3, "2"), car 6 17/25
    (wrong: "8" ×2, "11", "100"), car 10 8/13, car 15 5/10, car 16 6/9. 4, 5, 9, 11, 14, 22 are perfect, 3 is 16/18, 13 is 19/21.
  * 6 unmatched passes. 3 are long stationary tracks (11:30:09 at the start of the last file, 09:37:10, 10:20:56; 400–1200
    frames each), and 1 is the 11:05:46 pass that COCO also has (read 7).
  * `analyze.py` snaps to the SQL `equipe` table, which includes the test entries 99/100. One wrong read ("100" for car 6)
    exists only because of that. The roster file `rosters/26SE.csv` (1–24) is the better snap set.

### VLM in fp16 (Qwen3-VL-4B-Instruct, the plain read)

* **`torch.cuda.is_bf16_supported()` returns True on the 2080 Ti** because it counts emulation. Use
  `is_bf16_supported(including_emulation=False)` (False here). `lapvision/vlm.py` now takes `Reader(dtype=..., attn=...)`, env
  `LAPVISION_VLM_DTYPE=fp16|bf16|fp32|auto`, and `read_passes.py --dtype=`. Default `auto` = bf16 only with native support
  (Ampere+), fp16 on Turing, fp32 on CPU.
* **fp16 passes.** 500 crops stratified from `out/world_*/reads.csv` (300 stored numbers, 200 'none'), batch 8, compared
  with the stored bf16 reads from the RTX 5080: **99.2% identical parsed reads** (496/500), 99.0% identical raw text, 100% on the
  'none' crops. No empty, NaN or garbage outputs (the 2 non-numeric-looking outputs, "1256420" and "17001", match the stored
  bf16 ones). The 4 differences: 111→11, 08→89, 220→none, 30→38. After roster snapping, only 2 votes move (one more 11, one
  less 8). No fix (sdpa, fp32 vision tower, fp32) was needed. Attention is sdpa (the default).
* **The 4B model barely fits.** Weights are 8.88 GB in fp16 and the desktop takes 0.6–1.8 GB of the 11 GB. Plain read, crops
  ≤ 640 px:

| batch | s/crop | torch peak | nvidia-smi total (incl. desktop) |
|---|---|---|---|
| 1 | 0.28–0.29 | 9.08 GB | 9.8 GB |
| 2 | 0.19 | 9.15 GB | 9.9 GB |
| 4 | 0.16–0.165 | 9.37–9.40 GB | 10.2 GB |
| 6 | 0.15 | 9.65 GB | 10.6 GB |
| 8 | 0.16–0.18 (once 0.39) | 9.85–9.89 GB | 10.6–10.7 GB |

  Batch 8 sits at the edge, and one of three batch-8 runs was 2.4× slower (probably a spill into shared memory). **Use batch
  4–6.** That is ~2× slower than the RTX 5080 (0.085 s/crop at batch 8).
* **Why it is slow: Python/launch-bound, not GPU-bound.** One decode step (1 token) takes **120–130 ms**, but the GPU is busy
  only ~21 ms of it. The rest is ~2,600 kernel launches per step at ~30–45 µs each (WDDM plus the busy CPU). Raising the process
  priority did not help. The cost per step is the same at batch 1 and batch 4, so **cost ≈ generated tokens × 125 ms / batch**,
  plus prefill. The plain read generates ~3 tokens, so it is cheap. Long outputs (grounding) are expensive.
* Tried a static KV cache plus `torch.compile(backend="cudagraphs")` through `generate(cache_implementation="static",
  compile_config=...)`. It allocated 7.2 GB of **shared** (system) memory, crawled, and was killed after 12 min. Inductor
  needs triton, which is not installed (`triton-windows` would be needed). **Not solved.** A hand-captured CUDA graph of the
  decode step at fixed batch and length is the remaining option if the grounded read is ever needed at scale.
* Tip for the user: in the NVIDIA Control Panel, set "CUDA – Sysmem Fallback Policy" to "Prefer No Sysmem Fallback" for
  python.exe. A VRAM overflow then fails loudly (OOM) instead of silently running 10× slower.

### Grounding probe (Qwen3-VL-4B, fp16): boxes are 0–1000 relative

* Prompt (works first time, 20/20 valid JSON):
  `Locate every race number painted on the off-road race car(s) in this image (firewall panel, side plates, nose). Output a
  JSON list, one entry per number: [{"bbox_2d": [x1, y1, x2, y2], "label": "<digits>"}]. Output [] if no number is legible.`
  Output: `[{"bbox_2d": [158, 335, 266, 438], "label": "22"}]`, sometimes wrapped in a ```json fence.
* **Coordinate convention: relative 0–1000 of the image** (x / 1000 · W, y / 1000 · H), confirmed by eye
  (`out/grounding_probe/sheet_p0.jpg`, `p0_*.jpg`: green = 0–1000, red / blue = absolute-pixel readings, which are clearly wrong).
  It is **not** Qwen2.5-VL's absolute pixels.
* On 20 clearly readable crops (high-agreement passes, one per pass, 15 cars): **25/25 boxes sit on a real number and are
  tight.** The label equals the car on 20/20 crops ("08" → 8), with no other number. About 5 secondary instances were missed
  (a side plate or the firewall of the same car). So it is precise but not exhaustive per crop.
* **The grounded read cannot replace the plain read.** On a mixed sample of 120 crops (60% numeric / 40% 'none' plain reads;
  108 with a GT pass), crop-level precision against the GT car is **plain 0.93 (55/59) vs grounded 0.78 (52/67)**. Where the
  plain read says 'none', the grounded prompt still answers 11 times, and only 1 is right ("1", "11", "12", "100": hallucinated on
  mud). Agreement between the two is 85%.
* **Cost:** ~36 generated tokens, so **4.5–5.1 s/crop at batch 1 and 1.3–1.46 s/crop at batch 4**, ~9× the plain read.

### Decode (720p30 H.264, `youtube/25SU_END.mp4`, 300 s segment)

| decoder | rate | to `-f null` | piped to numpy (bgr24) |
|---|---|---|---|
| CPU (libavcodec, auto threads) | full 30 fps | 3,070 fps (102× RT) | 200 fps (6.7× RT) |
| CPU | fps=4 | 100× RT | **190 fps, 47× RT** |
| `-hwaccel cuda` | full | 1,000 fps (33× RT) | 237 fps (7.9× RT) |
| `-hwaccel cuda` | fps=4 | 34× RT | 132 fps, 33× RT |

* At these bitrates the 12-core CPU decodes faster than NVDEC with copy-back. Decoding is never the bottleneck at 2–4 fps,
  and the full-rate pipe is limited by the 2.7 MB/frame copy, not by decoding. CUDA decode is still useful to keep CPU free
  for other jobs.

### Detector throughput (fp16, 720p frames from RAM, conf 0.15)

| model | imgsz | batch 1 | batch 8 | batch 16 | torch peak (b16) |
|---|---|---|---|---|---|
| YOLO-World v2-x ["off-road buggy","person"] | 960 | 27 img/s | 54 img/s | 55 img/s | 1.4 GB |
| YOLO-World v2-x | 1280 | 29 img/s | 36 img/s | 36 img/s | 2.4 GB |
| YOLO11x COCO | 960 | | 63 img/s | 61 img/s | 1.6 GB |
| YOLO11x COCO | 1280 | | 41 img/s | 43 img/s | 2.8 GB |

* End to end (ffmpeg fps=4 pipe → reader thread → batch 8 YOLO-World @1280): **35 img/s = 8.9× real time** with CPU decode,
  8.3× with CUDA decode. This matches the from-RAM number, so decode overlaps completely.
* VRAM is not an issue for YOLO (≤ 4 GB as nvidia-smi sees it at 1280 / batch 16). ultralytics 8.4 warns that `half=` is
  deprecated in favour of `quantize`. It still works.
* **Detection time over the 42.4 h of streams** (152,650 s): YOLO-World @1280: **4.7 h at 4 fps, 2.4 h at 2 fps**. @960:
  3.1 h / 1.6 h. YOLO11x @1280: 4.0 h / 2.0 h. Tracking adds little. This is well under the TRAINING_PLAN guess (10–15 h).
* VLM budget for WP4: plain read ≈ 0.16 s/crop at batch 4–6, so **~22,000 crops per GPU-hour**. The grounded read
  manages ~2,500 crops per hour.
* ultralytics' `weights_dir` setting points to `D:\video_baja\weights`, so the CLIP `ViT-B-32.pt` now also exists there
  (downloaded by `tools/bench/yolo.py`). It is the same file as `lapvision/weights/clip/ViT-B-32.pt`.

### Recommendations for WP3 / WP4

* WP3: YOLO-World v2-x fp16, **imgsz 1280, batch 8, 4 fps** (4.7 h for the current streams, so the 2 vs 4 fps question can be
  decided on label quality alone). 960 is 1.5× faster, but small far cars are the reason to keep 1280. Check at G1.
  Use CPU decode (ffmpeg `fps=4` pipe).
* WP4: plain read in **fp16, batch 4** (6 if the desktop is idle). Never 8 on this card while the desktop is in use.
* Grounded read: **only for boxes, never for identity.** Run it on 2–3 crops of tracks that already have a plain-read
  consensus, and keep a box only if its label equals the consensus (as WP5 planned). Coordinates are 0–1000 relative to the crop.

## Session 2: shots, shot tags and overlays (WP1/WP2)

Code: `teacher/manifest.py`, `shots.py`, `shot_tags.py`, `overlay.py` (+ `shots_qa.py`, `overlay_qa.py`, `wp2_summary.py`,
`common.py`). All CPU, resumable (`<stage>.done`, `errors.log`). Outputs: `data/videos.csv`, `data/events.csv`,
`data/wp2_summary.csv`, `work/<id>/{shots.csv, overlays.csv, overlay_mask.png, overlay_preview.jpg}`.

* **Manifest.** Event/prova are parsed from the first `<YY><BR|SE|SU|NE>_<PROVA>` in the file stem, so a later
  `26BR_END_takeout.mkv` (any folder under `youtube/`) groups with `26BR_END.mp4`; the higher (height, bitrate) file is
  `active`, the other `superseded` (tested with synthetic files). Stages only run on active rows. Ids are sanitised stems.
  If Takeout delivers one prova split into several files, they would supersede each other: rename them `_p1`, `_p2`
  with distinct provas or extend the rule.
* **The streams have a fixed 5 s GOP.** Decoding I-frames only (`-skip_frame nokey`) is ~200× real time and gives a
  frame every 5 s for free. The overlay stage uses only keyframes. Full decode at 10 fps / 128×72 is ~30× real time per
  process (3 threads).
* **Shot detection** (own detector on 10 fps signals; PySceneDetect not needed): hard cut = HSV content jump that is
  large, ≥ 2.6× its neighbours and changes the colour histogram. Gradual transitions = the middle frame is a linear blend
  of the frames k = 3/6/10 steps before and after (relative residual < 0.2) **and** dense optical flow cannot explain the
  change (motion-compensated residual > 0.7). Without the flow test, PTZ zooms and pans produced ~3 false dissolves per
  minute; the blend residual alone does not separate a slow zoom over grass from a dissolve. Transitions become their
  own `is_transition=1` shots (tracking should skip them). Result: 5,720 shots in 42.4 h; 39/40 random boundaries in
  25SU_DIN + 26BR_END were real (1 false: same scene on both sides); 5 spot checks inside the longest shots (up to 19 min)
  were continuous PTZ / handheld zooms, no missed cut. 19SU_END's median shot of 1.6 s is real: sponsor commercials with
  ~1 s cuts. Most transitions in these streams are 0.3–1 s dissolves, not hard cuts.
* **Shot tags** (CLIP ViT-B/32, CPU, 1–3 keyframes per shot, max over several prompts per tag): **79% on the 150 shots
  used to tune the prompts, 70% on 50 held-out shots** (`teacher/qa_labels/`). Main confusions: `track_wide` vs
  `track_closeup` (CLIP has no notion of how big the car is: 9/27 held-out wide shots called close-up) and
  `studio_interview` vs `graphics_fullscreen` (studio desks with a big logo and a TV). Onboard, pits, studio and drone are
  recognised well. Mean-of-prompts pooling was worse than max (62% vs 75%). A generic "broadcast graphic" prompt pulled
  studio shots into graphics; sponsor commercials are tagged graphics_fullscreen. Use the tags for stratification only,
  as planned. Embeddings are cached (`shot_clip.npz`), so re-scoring with new prompts takes ~1 s per video.
* **Overlays: what worked.** Per keyframe i and 24 random partners j ≥ 2 min away: an edge pixel counts when (a) the
  local edge structure matches almost exactly (≥ 70% of the 9×9 window's edges coincide), (b) the surroundings differ
  between i and j (≥ 30% of thumbnail cells on a 60–180 px annulus differ), and (c) i and j show different views
  (< 30% of thumbnail cells look the same outside pass-1 regions). P(x) = share of frames with such a match. Pass 1 picks
  the least similar partners (robust for static cameras), pass 2 random different-view partners (finds graphics that
  only appear over one kind of shot, like DIN lower thirds). Regions: connected components of P ≥ 0.004 after removing
  long straight lines (encoder grid streaks and ticker edges chain everything together otherwise), guillotine splits of
  L-shapes, closed line rectangles (sponsor slots), then merges of regions that switch on and off together.
  Visibility per keyframe = persistent edges present at ≥ 3× chance rate, or the region's colours match its typical "on"
  look (scrolling tickers have no persistent text edges).
* **Overlays: what failed.** Plain per-pixel stability (value or edge frequency over time) is dominated by static cameras
  (19SU's start-line camera, SPP's single camera, studio), whose background is as stable as any graphic. "Partner must be
  a different scene" by global similarity alone failed for studio shots (people move, layout changes) and, with random
  partners, static views re-matched themselves. A whole-frame bounding box per component merged tower + ticker + logo
  into one 1280×720 box until lines were split off.
* **Overlay results** (previews `work/<id>/overlay_preview.jpg`, mask = union of boxes, `overlays.csv` has intervals):
  END towers found in all END streams (19SU 54%, 24SU 7%, 25BR 39%, 25SU 86%, 26BR 27% of the time), corner logos
  (19SU, 24SU, 24SU_SPP, 26BR_DIN), tickers (19SU, 24SU), 19SU's rotating sponsor slot, 26BR_DIN's corner labels and
  lower third, 24SU_SPP's heat lower third and parts of the bracket, 26BR_END's onboard telemetry bar. Text-only lower
  thirds whose text changes (25BR_DIN, 25SU_DIN) are only partly persistent, so they were added by hand in
  `teacher/overlay_overrides.csv` (box + a reference time; visibility = colour match to that keyframe). One false
  region (19SU, static scene) is dropped there. Big sparse graphics (SPP / DIN brackets) get loose boxes that cover live
  video; WP3's `track.py` already ignores regions > 6% of the frame.
* Runtimes (CPU, 3–4 videos in parallel): manifest 10 s; shots 3–17 min per video (26 min wall for all 42 h);
  tags 1–9 min per video; overlays 2–16 min per video (two passes, ~1 h wall for all).

## Session 2: detection + tracking on the streams (WP3)

Code: `teacher/detect.py` (GPU), `teacher/track.py` (CPU), tools in `tools/wp3/` (`calib_sheets.py`, `track_slice.py`,
`track_qa.py`, `qa_sample.py`, `frag_stats.py`, `summary.py`, `track_loop.sh`). Outputs per video in `work/<id>/`:
`dets_raw.parquet` + `detect_settings.json` + `detect.done`; `dets.parquet`, `tracks.parquet`, `track_feats.npz`,
`track_summary.json`, `track.done`; QA sheets in `work/<id>/qa/track_*.jpg`. Calibration data + sheets: `work/calib/`.

### Calibration (3 slices of 20 min = 4,800 frames each, judged by eye on contact sheets)

Slices: 26BR_END 2:00–2:20 (onboard, split screen, PIP, leaderboard), 25SU_DIN 1:00–1:20 (pits, bracket graphics,
lower thirds), 19SU_END 1:00–1:20 (fixed start-line camera). Counts are car ("off-road buggy") detections ≥ 0.25.

| slice | A dets | B dets (+ absorbing classes) | A-only (lost in B): sample by eye | B-only |
|---|---|---|---|---|
| 26BR_END | 3,812 | 4,065 (+car 443, tractor 63, moto 36, pickup 35) | 43: ~25 real buggies relabelled car/pickup/tractor, ~18 FP | 70: mostly onboard ego car + FPs |
| 25SU_DIN | 5,935 | 5,920 (+car 1,577, tractor 114, moto 90, ATV 25, pickup 9) | 126; 60 seen: ~11 real buggies, ~49 FP (white road sedan, crop sprayer, people) | 57 |
| 19SU_END | 10,373 | 10,356 (+car 377, pickup 255, moto 50, tractor 12) | 341; 60 seen: ~4 real buggies, ~40 parked pickup, ~15 static tyre | 51 |

* **Prompt set B (A + "pickup truck", "car", "ATV", "tractor", "motorcycle") is not a pure absorber.** It removes real
  FPs (road sedan, crop sprayer, parked pickup: ~1–3% of car detections) but also relabels real buggies as "car" /
  "pickup" / "tractor" (visible loss in all 3 slices, worst on 26BR: ~25 of 43) and adds detections of its own (the ego
  car). Changing the prompt list changes the buggy scores too (YOLO-World text embeddings interact via the per-anchor
  argmax). Decision rule "any visible recall loss → keep A": **A = ["off-road buggy", "person"]**.
* Precision of A (100 random car detections ≥ 0.25 per slice): 26BR ~74 real other cars + ~15 onboard ego car + ~11 FP;
  25SU ~83 real, 17 FP (9 road sedan / crop sprayer, 3 bracket-graphic boxes, 5 other); 19SU ~71 real, ~29 FP (a big
  tyre lying in the field ~13, a parked pickup ~8, a dark vehicle under a tent ~5, people ~3). Static props and parked
  road vehicles under a fixed camera are the main FP source; `long_static` marks them (see below), WP5 / VLM must reject.
* Missed cars (40 random frames per slice, half of them where A and B disagree): cars surrounded by marshals during
  a recovery (25SU #4 stuck: not detected ≥ 0.25 in 4 of 5 frames even when large and clear), cars behind fences /
  people in the pits, cars ahead in onboard views at the top of the frame. Far small cars are found well at 1280.
* **imgsz 960 vs 1280 (19SU, prompt A):** 1280 gives 10,373 car dets vs 9,017; boxes < 60 px 3,931 vs 2,746, < 40 px
  467 vs 192. 1280-only (1,377): of 60 seen ~22 real (mostly far/small cars), ~35 the static tyre; 960-only (490): of
  60 seen ~40 real. Net ≈ +500 vs +330 real detections → **keep 1280** (960 is not equal on far cars).
* Throughput as WP0 measured: 35.2–35.8 img/s at 1280 (8.8–8.9× real time at 4 fps), 55.5 img/s at 960.

### Detection run (`teacher/detect.py`)

* Settings: YOLO-World v2-x fp16, prompts A, imgsz 1280, batch 8, 4 fps (ffmpeg CPU `fps=4` bgr24 pipe), conf ≥ 0.05,
  NMS IoU 0.7, max_det 300. `frame_idx` = frame of the 4 fps stream, `pts = frame_idx / 4`.
* Parts of 600 s of video (`dets_raw_parts/part_NNNNN.parquet`), atomic rename; restart resumes after the last
  contiguous part (tested by killing the process mid-video: the resumed part starts at the exact next frame). `PAUSE`
  at the repo root → finishes the current part, exit code 3 (tested). Run order = order of ids on the command line.
  Gotcha: `.venv/Scripts/python.exe` is a launcher that spawns the real interpreter; killing the launcher (e.g.
  `timeout`) leaves the child running. Kill both PIDs (or use PAUSE).
* The ultralytics `half` deprecation warning is printed on every predict call; silence via `ultralytics.utils.LOGGER`.

### Tracking (`teacher/track.py`)

* ByteTrack-style two-stage Hungarian matcher per shot and class (car, person; never across a shot boundary):
  stage 1 vs dets ≥ 0.25 with similarity max(IoU(predicted, det), 0.6 × centre-distance score), the distance score
  gated at 1.2 × box size and a height ratio ≤ 1.8 (cars move a lot between 4 fps frames in close-ups); stage 2 vs dets
  0.08–0.25 with IoU ≥ 0.3; new tracks from dets ≥ 0.25; track buffer 12 frames (3 s); constant-velocity prediction.
  Tracks with ≥ 3 detections are kept (`track_id` -1 otherwise).
* Fragmentation proxy (`frag_stats.py`: a track that starts ≤ 3 s after another ended, near its end box): with an 8-frame
  buffer and low threshold 0.10: 11% / 14% / 11% of car tracks (26BR / 25SU / 19SU slices); with 12 frames and 0.08:
  **8% / 12% / 5%**. The rest are mostly fast PTZ pans (the whole picture jumps) and crowded parked groups where boxes
  split / merge. Camera-motion compensation would be the next step; WP5 can also merge per identity.
* `nested_in`: id of a larger coexisting car track containing this one (session-1 double-box rule). 25SU_DIN has many
  (parked groups produce boxes on groups of cars).
* Per track: shot, t0/t1, n, conf stats, box size in px and / H, median box, centre MAD, speed, path, size ratio,
  edge contact, `bg_motion` / `inbox_change` (median mean |ΔI| outside / inside the box between frames 0.5 s apart,
  from a second CPU decode at 2 fps 192×108 gray), `pos_persist` (share of 4 fps frames with a car detection at the same
  place), recurrence counts, flags.

### Flags: what worked and what did not

* **`screen_fixed`** = OR of three rules:
  1. `fixed_persistent` (onboard ego car): bottom-anchored box (y2 ≥ 0.97 H), scene moving (inbox or bg ≥ 8), at the
     same place in ≥ 20% of the frames of the shot or of ±10 s around it, shot ≥ 10 s, **and** the same place (median
     IoU ≥ 0.85) with the same look (12×12 patch NCC ≥ 0.3) in ≥ 2 other shots. Then propagated to fragments of the same shot (IoU ≥ 0.6 with a flagged ego box) and to
     big bottom-anchored boxes elsewhere at a well-established ego place (IoU ≥ 0.7 with ego boxes of ≥ 5 shots).
  2. `fixed_bg_moving` (graphic over moving video): static box (centre MAD ≤ 1.2% H, height MAD ≤ 4%) with bg_motion ≥
     10 and ≥ 1.6× inbox_change, seen at the same place in ≥ 2 shots or for ≥ 10 s.
  3. `fixed_recurrent`: same static place + look (12×12 patch NCC ≥ 0.6, rest of frame NCC < 0.5) in ≥ 4 shots, counting
     only instances whose surroundings move.
* Things that failed on the way (kept here so nobody retries them):
  * The onboard ego car is **not** pixel-stable: per-pixel temporal stability inside its box (share of pixels with
    median |ΔI| < 2.5 over 0.5 s) was 0.05 on ego tracks, same as on normal cars (vibration, helmet moves, the box
    covers the view through the cage). Its box is also not very stable (centre MAD 1–3% H) and it is detected only
    intermittently at conf 0.15–0.35, so the tracker fragments it into many short tracks. What identifies it is
    geometry: a big bottom-anchored box at the same place shot after shot (26BR: x 290–905 single onboard, 650–1280 right
    half of the split screen, plus helmet-only variants y1 ≈ 330).
  * Persistence alone flagged zoomed PTZ close-ups of real cars (25SU_DIN: pits, crop sprayer; 27 tracks, 12/12 sampled
    were real cars) because they also hold a big bottom-anchored box. Requiring the same place within IoU 0.85 in ≥ 3 shots
    (the mount is rigid) fixed it there: 25SU_DIN 27 → 1, 26BR_END 219 → 215. **But** on the full run 25BR_END's
    handheld pit walk (phone-style close-ups of parked cars, all frame-filling and bottom-anchored) still produced a seed,
    and same-shot propagation then flagged 28 real close-ups (8/8 sampled were real cars with big readable numbers, the
    best number data there is). Fix: the seed must also *look* the same in the other shots (patch NCC ≥ 0.3 in ≥ 2 other
    shots; ego pairs median 0.47 on 26BR_END, the close-ups 0.07). Final: 25BR_END 0, 25SU_DIN 0, 26BR_END 211,
    26BR_DIN 207 (PIP onboard insets, 12/12 sampled correct).
  * Background-motion threshold 4 flagged parked cars under the 19SU fixed camera (people walking give bg 4–8). Static
    cameras sit at 1–8, pans / onboard at 12–35 → 10.
  * A PTZ operator following a small far car keeps it still on screen for 1–3 s with the background moving → rule 2 needs
    a repeat (≥ 2 shots) or ≥ 10 s.
  * Recurrence with a "different surroundings" test fails under a fixed camera: people walking or a large overlay elsewhere
    make the same view look different. Only moving-background instances count now. Consequence: a **graphic over a still
    picture** (e.g. the corner Baja logo over SPP's static wide shot) is caught only when the camera moves; the rest must
    come from `in_overlay`.
* **`in_overlay`**: share of the track's detections with ≥ 60% of the box inside a WP2 overlay region visible at that
  time (intervals ± 2.5 s). Only compact regions count (≤ 6% of the frame; kinds `bracket_or_fullscreen_graphic`,
  `inset_or_logo_box` skipped): WP2 boxes are unions of positions, and a bracket / PIP box is mostly live video with real
  cars in it. With 20% allowed, ~11/12 sampled in_overlay tracks on 24SU_SPP were real cars. Even at 6%, about half of the
  sampled ones are real cars, because the visibility intervals of the small bracket boxes are too generous; the corner
  logo hits are right. Treat it as a soft flag until WP2's per-region intervals are tighter. Null when the video had no
  overlays at tracking time; `python teacher/track.py <id> --reflag` recomputes all flags from the stored features in ~1 min
  (track.py does this by itself when it finds overlay.done for a video tracked without overlays).
* **`small`**: median box height < 60 px. **`long_static`** (info): static box for ≥ 20 s (parked cars in the pits,
  parked road vehicles, a tyre, under a fixed camera): not screen_fixed, but a likely FP / no-number source.

### QA by eye (work/<id>/qa/track_*.jpg, one frame per sampled track)

* 26BR_END `track_ego.jpg`: 12/12 flagged ego are the onboard ego car (single view, split screen, PIP); the cars seen from
  onboard in the same frames stay unflagged. `track_onboard.jpg`: 12/12 unflagged tracks in shots with an ego track are
  real cars (incl. cars ahead seen from onboard). `track_bigbottom*.jpg` (unflagged big bottom-anchored boxes):
  before the ego propagation 7/12 were missed ego; after it ~2–3/12 (the dirty-helmet onboard of the other car, small
  PIP insets), rest real close-ups. `track_towerzone.jpg`: tracks under the leaderboard tower area are real cars; the
  tower itself rarely produces car boxes.
* 25SU_DIN `track_static.jpg`: parked pit cars in static PTZ shots are `long_static` and **not** `screen_fixed`.
  `track_ego.jpg` shows the (now fixed) close-up false positives.
* 19SU_END `track_static.jpg`: stopped / parked cars and props near the tent, unflagged. `track_bgfix.jpg`: 5 low-conf
  junk boxes.
* 26BR_DIN `track_bgfix.jpg`: the "BAJA" logo with the buggy drawing (top centre, during the DIN tests) is flagged by
  rules 2+3 and in_overlay: 12/12 sampled bg-moving tracks are that logo. `track_ego.jpg`: 12/12 onboard PIP insets.
* 24SU_END `track_recur.jpg`: the corner logo (top left) is what `fixed_recurrent` catches.
* 25BR_END `track_bigbottom_final.jpg`: frame-filling real close-ups, unflagged (also a TV commercial with a road car
  and a full-screen track map, i.e. FPs WP5 must reject).
* 24SU_SPP `track_overlay.jpg`: see in_overlay above; the corner logo [106,10,170,71] gives ~18 car tracks, 4 flagged by
  rule 2 (camera moving), the others only via in_overlay.

### Full run (all 9 active videos, 42.4 h; detect 00:22 → 05:13 = 4.85 h, 35.0–35.3 img/s; tracking ran alongside on CPU)

| video | h | dets (all ≥ 0.05) | car dets ≥ 0.25 | car tracks | tracks / h | person tracks | screen_fixed | in_overlay | small | long_static | detect min | track min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 19SU_END | 4.68 | 1,994,355 | 171,861 | 6,193 | 1,324 | 13,335 | 0.1% | 0.1% | 52.6% | 5.6% | 31.9 | 10.7 |
| 24SU_END | 4.58 | 1,279,943 | 150,085 | 8,320 | 1,816 | 20,498 | 0.9% | 1.0% | 35.6% | 1.0% | 31.6 | 10.9 |
| 24SU_SPP | 1.45 | 525,042 | 21,453 | 1,134 | 784 | 8,479 | 0.4% | 1.7% | 35.3% | 2.5% | 10.1 | 3.5 |
| 25BR_DIN | 7.05 | 1,021,788 | 50,574 | 2,811 | 399 | 14,018 | 0.0% | 0.0% | 40.0% | 1.8% | 48.0 | 10.7 |
| 25BR_END | 7.11 | 1,609,091 | 142,482 | 9,997 | 1,407 | 32,010 | 0.0% | 0.0% | 41.5% | 0.2% | 49.2 | 16.4 |
| 25SU_DIN | 3.30 | 890,722 | 72,976 | 3,817 | 1,157 | 14,769 | 0.0% | 0.0% | 11.9% | 0.6% | 22.4 | 7.2 |
| 25SU_END | 4.27 | 1,110,493 | 65,099 | 4,516 | 1,057 | 30,712 | 0.0% | 0.0% | 8.2% | 0.3% | 29.4 | 9.2 |
| 26BR_DIN | 5.68 | 1,434,343 | 132,035 | 5,988 | 1,054 | 24,651 | 9.5% | 6.7% | 23.1% | 2.3% | 38.5 | 9.9 |
| 26BR_END | 4.29 | 1,155,569 | 90,451 | 7,387 | 1,724 | 27,417 | 2.9% | 0.0% | 31.5% | 0.2% | 28.9 | 8.5 |
| **total** | **42.4** | **11.0 M** | **897 k** | **50,163** | **1,183** | **185,889** | | | | | **290** | **87** |

Shares are of car tracks. Track minutes = first full tracking pass (tracking + 2 fps feature decode, run while the GPU
job and WP2 used the CPU); `--reflag` afterwards takes ~1 min per video. Disk: dets_raw + dets ≈ 0.2 GB in total.
About 19,900 car tracks are unflagged (screen_fixed, in_overlay) with a median box ≥ 100 px (the WP4 crop pool; K = 6–8 crops each ≈ 120–160 k
crops ≈ 5.5–7 GPU-h of plain reads at WP0's 22 k crops/h, less if the pool is filtered by conf / size / shot tag).

### Open problems for WP4 / WP5

* FPs that no flag removes: road vehicles (sedan, pickup, crop sprayer, a TV commercial), a big tyre and other props under
  a fixed camera (19SU), people in tight groups, full-screen maps. `long_static` covers part of it; the VLM read ('none'
  / not a car) and a per-track "is this a baja car" check in WP5 must do the rest. Prompt set B would cut ~1–3% of FPs but
  costs recall; a cheap alternative is to run the absorbing prompts only on crops of existing tracks.
* Recall gaps: cars in recovery scenes surrounded by marshals, behind fences in the pits, small cars ahead in onboard
  views. Store-to-0.05 means WP5 can still use low-conf detections to fill gaps inside tracks.
* `small` is 8–53% of car tracks (19SU far cars): detector data only.
* Fragmentation 5–12% of tracks restart near where another ended (fast pans, crowded groups); identity consensus per
  shot can merge them. `nested_in` marks double boxes (many in pit scenes: boxes on groups of cars).
* `in_overlay` depends on WP2's region intervals; treat it as soft. Graphics over a still picture are only caught by
  in_overlay (screen_fixed rules need a moving background or recurrence).
* Onboard ego recall ~75–80% on 26BR_END by eye (the other car's dirty-helmet onboard and small PIP insets are missed).
  Ego boxes are bottom-anchored and huge, so a WP4 filter "h_rel > 0.8 and bottom-anchored in a shot with a flagged
  ego track" would catch most of the rest.
* Rerun cost if 26BR originals (Takeout) arrive: detect ≈ 7 min per stream-hour at 720p, more at 1080p (decode + resize
  is still cheap; YOLO runs at imgsz 1280 either way).

## Session 2: students + 26SE harness (WP9)

Code: `students/train_det.py`, `students/eval_26se.py`, `students/bench_cpu.py`; `lapvision/det.py` got a student mode.
Runs in `students/runs/` (gitignored), harness outputs in `out/eval_<name>/`. No GPU was used (the VLM labeller had it).

### Environment gotchas

* **`CUDA_VISIBLE_DEVICES=` (empty) does not hide the GPU on Windows.** Windows drops an empty env var when a process
  starts, and torch then reports `is_available() == True` with `device_count() == 0`. Use `CUDA_VISIBLE_DEVICES=-1`
  (then `is_available()` is False). The students scripts pick the device with `device_count() > 0`.
* **ultralytics sets `OMP_NUM_THREADS=1` on import if it is unset.** If ultralytics is imported before torch, torch CPU
  inference runs on one thread. Import torch first (the students scripts do) or set the thread count explicitly.
* **ffmpeg raw output duplicates frames after a frame-dropping filter.** `select=not(mod(n,2))` piped as rawvideo came out
  at 60 frames/s again (CFR padding), which doubled the time axis of the tracks. Use `framestep=N` (keeps input frames
  0, N, 2N, ..., checked pixel by pixel) and `-fps_mode passthrough`.
* `onnx`, `onnxslim`, `onnxruntime 1.31` and `openvino 2026.4.1` install cleanly with pip on Python 3.13 (no existing package
  changed; added to `requirements.txt`). `albumentations` was **not** installed: it pulls `opencv-python-headless`
  next to `opencv-python`. The blur / JPEG augmentation is a small cv2 transform in `train_det.py` instead.
* `yolo11n.pt` / `yolo11s.pt` are in `weights/` (ultralytics' `weights_dir`, so `YOLO("yolo11n.pt")` finds them from any cwd).

### train_det.py

* YOLO11n/s, imgsz 640 (or 960), AMP, batch by size (n640 64, s640 32, n960 24, s960 16: estimates for 11 GB minus the
  desktop; check nvidia-smi in epoch 1, or `--batch 0.7` = AutoBatch at 70% of free VRAM), early stopping (patience 25),
  `--hours` budget, `--resume`, `--fraction` subsets, `--cache ram`. After training it evaluates `best.pt` on the
  data.yaml `test` split if there is one (`eval_test.json`; also `--test-only`).
* Augmentation: mosaic (closed for the last 10 epochs), HSV (h 0.015, s 0.6, v 0.45: mud, washed-out exposure), scale 0.5,
  translate 0.1, horizontal flip only (no vertical flip, no rotation / shear / perspective), plus `Degrade` (p 0.3: Gaussian
  blur, horizontal motion blur 5-15 px, JPEG q 25-70, or down-up scaling 0.4-0.7).
* Gotcha: a transform inserted once into `dataset.transforms` disappears when `close_mosaic` rebuilds the pipeline. It must
  live in `build_transforms` (a module-level `YOLODataset` subclass, which also lets Windows spawn workers unpickle the
  dataset). Checked: after pickling and `close_mosaic` the pipeline still ends `..., Degrade, Format`.
* CPU smoke test on a throwaway 40-frame set cut from `out/world_1129` boxes (deleted afterwards with its runs; nothing was
  trained on 26SE for real): 1-2 epochs run, mosaic and degrade visible in `train_batch*.jpg`. Killing a run after epoch 1
  and `--resume` continued at epoch 2 with the same args. The `--test-only` path was checked on 4 stream frames.

### eval_26se.py: the benchmark harness

* Same stages and thresholds as session 1: 5 fps timeline (frames scaled to 1920x1080, conf 0.15) -> foreground events
  (vehicle conf > 0.3, h > 0.2 H, bottom > 0.4 H, gap 2.5 s, +-1.5 s pad) -> ByteTrack at ~30 fps on the events only ->
  `passes.build()` -> time-only greedy matching to `contagem_volta` (+80 s, +-6 s on t0) after 09:18:30. Per-file breakdown,
  false-pass and missed-lap lists. Identity is optional and borrowed: each pass takes the VLM vote of the YOLO-World pass
  that overlaps it most in time (IoU >= 0.3), scored with the id-preferring match of `analyze.py`.
* `lapvision/det.py`: `Detector(name, mode=None)`. Auto mode = 'world' if the name says so, 'coco' for an 80-class COCO
  model, otherwise 'student' (vehicle classes = every class except `person` / `number`, so a 2-class `car`+`number` net
  only feeds `car` boxes into the pass logic). Existing callers behave as before.
* The GT window is fixed by the selected files (09:18:30-12:10:10 for all four), not by the first / last detected pass
  as in `analyze.py`, so a weak detector cannot shrink its own GT. For the YOLO-World and COCO runs it gives the same 252 laps.
* **Validation (CPU, metrics from the existing tracks): exact reproduction.** `--from-tracks out/world_*`: 241 passes,
  235 matched, recall **0.9325**, precision **0.9751**, 6 false passes, 17 missed; last file 63/63 laps, 64 passes
  (1.00 / 0.984). Borrowed identity on itself: 182 read, 94.0% right, end-to-end 67.9%, share >= 0.5 98.7% at 65%, all equal to
  WP0. `--from-tracks out/crops_*` (COCO YOLO11x): 230 / 219, 0.869 / 0.952, per-file recall 0.99 / 0.83 / 0.83 / 0.79
  (= session 1). The time-only match finds the same 235 pairs as the id-preferring one.
* **End-to-end check on CPU, all three stages**, COCO YOLO11n (pretrained, untrained on baja cars) at 1280 on file 11-09-03
  (20.7 min, 24 laps; `out/eval_coco_n1280_cpu/`): 15 passes, **recall 0.58**, precision 0.93 (COCO YOLO11x 0.83 / 1.00,
  YOLO-World 0.875 / 1.00). A tiny COCO model is not usable as is; a trained student is needed. 21 events, 157 s of tracking
  windows. Speed on the loaded CPU (8 threads): timeline 9 img/s (687 s), tracking 7.2 frames/s (652 s).
* **Run time of a full evaluation.** Footage 10,816 s = 54 k timeline frames at 5 fps. Tracking windows (YOLO-World
  events): 476 + 473 + 312 + 466 = **1,727 s, about 52 k frames at 30 fps**. CPU decode measured while the other agents loaded
  the CPU: timeline (`fps=5` + scale) 5.8x real time for the 4K file and 7x for 1080p60; full-rate decode 2.3-3.9x.
  * GPU (estimate, not measured: no GPU in this session): timeline decode-bound, 25-30 min (YOLO11n/s at 640 does hundreds
    of img/s at batch 16); tracking 10-15 min (decode overlaps the ~10 ms per `track()` call). **About 40 min per detector.**
  * CPU only (a student at 640 costs ~25% of the YOLO11n@1280 above): **about 1-1.5 h per detector**, so student
    evaluations can run on the CPU while the GPU trains the next model.
  * Cheaper: `--track-fps 15` halves stage 2; `--files 1129` (40 min, 63 laps, the session-1 YOLO-World 1.00 / 0.98 file)
    for a quick look (~10 min on GPU). Decode is the floor, so `--tl-fps` below 5 saves little and risks missed events.

### bench_cpu.py: CPU latency (proxy for the laptop)

Ryzen 9 3900X (Zen 2, 2019), **while the VLM labeller and WP8's export kept the CPU at 50-95%**: worker processes at high
priority, 3 interleaved rounds of 60 frames, fastest round reported (rounds differed by up to 1.7x). Per frame: letterbox
of a real 1080p frame + forward + NMS. RAM = peak working set of the whole process (the ONNX / OpenVINO workers import only
numpy, cv2 and the runtime). Results: `students/runs/bench/*.json`.

| model | input | ONNX RT 1 thr | ONNX RT 4 thr | OpenVINO 1 thr | OpenVINO 4 thr | torch 4 thr | peak RAM (ORT / OV) |
|---|---|---|---|---|---|---|---|
| YOLO11n | 640x640 | 104 ms | 52 ms | 109 ms | 54 ms | 99 ms | 176 / 173 MB |
| YOLO11n | **384x640** | 50 ms | **22 ms** | 49 ms | **21 ms** | 42 ms | 138 / 157 MB |
| YOLO11s | 640x640 | 287 ms | 114 ms | 290 ms | 112 ms | 191 ms | 278 / 238 MB |
| YOLO11s | **384x640** | 135 ms | **49 ms** | 143 ms | **48 ms** | 86 ms | 207 / 216 MB |

* **Export with a 16:9 input (`--imgsz 384x640`).** A square 640 input is 44% padding for a 1080p frame; the rectangular
  export halves the cost. It is the same object scale as training at 640 (long side 640), so no retraining is needed, and
  ultralytics' own `.pt` predict already letterboxes to 384x640 (which is what `eval_26se.py` measures).
* YOLO11n at 384x640 meets the <= 50 ms target even on one thread of this loaded CPU; YOLO11s needs ~4 threads. RAM is far
  below 1 GB (< 0.3 GB). ONNX Runtime and OpenVINO are on par here; OpenVINO may win on an Intel laptop and offers INT8
  (NNCF calibration, not tried). The torch path is ~2x slower and costs ~0.6 GB RAM (the torch import alone is ~0.5 GB).
* Pre-processing (resize + letterbox of a 1080p frame) is 4-8 ms; less if the capture runs at 720p.

### Training plan for when the GPU is free (not run)

Time estimates are **scaled from published numbers, not measured**: YOLOv5 README (V100, COCO 118 k images, 300 epochs:
v5n 1 day, v5s 2 days, i.e. ~410 / ~205 img/s), scaled by GFLOPs (YOLO11n 6.5 vs v5n 4.5, YOLO11s 21.5 vs v5s 16.5) and
x0.75 for a 2080 Ti vs a V100 with AMP: **YOLO11n@640 ~200-250 img/s, YOLO11s@640 ~110-130 img/s**, @960 x0.44 (n ~100,
s ~55 img/s). The 3900X data loader (8 workers, mosaic) should keep up at 640. Per 10 k training images and epoch:
n640 ~45 s, s640 ~85 s, plus ~10% validation. Run 0 calibrates this.

0. Calibration (~15 min): `train_det.py --data datasets/cars/data.yaml --model yolo11n.pt --name calib --epochs 3
   --fraction 0.25` -> img/s and VRAM (nvidia-smi); then delete the run.
1. `--model yolo11n.pt --imgsz 640 --epochs 100 --patience 20 --name cars_n640`: with 40 k images ~5.5 h (early stopping
   usually ends it sooner).
2. `--model yolo11s.pt --imgsz 640 --epochs 100 --patience 20 --hours 10 --name cars_s640`: with 40 k images ~10 h.
3. Only if 24SU shows a small-car gap that matters for the PTZ use case: `n960` (~2.3x n640). Not needed for the start
   line (passing cars are > 0.2 H, >= 77 px at a 384 px input height).
4. After each run: `eval_26se.py --name <run> --weights students/runs/<run>/weights/best.pt` (GPU ~40 min, or CPU ~1-1.5 h
   in parallel with the next training) and `bench_cpu.py --weights ... --imgsz 384x640`. Never tune on 26SE.

Comparison table to fill:

| detector | input | 24SU held-out: mAP50 / mAP50-95 vs teacher labels | 26SE passes | recall | precision | false passes | end-to-end (borrowed ids) | CPU ms (ORT 4 thr, 384x640) |
|---|---|---|---|---|---|---|---|---|
| YOLO-World v2-x (teacher) | 1280 | reference | 241 | 0.933 | 0.975 | 6 | 67.9% | GPU only |
| COCO YOLO11x | 1280 | - | 230 | 0.869 | 0.952 | 11 | 62.7% (own VLM reads) | - |
| COCO YOLO11n, untrained (11-09-03 only) | 1280 | - | 15 (24 laps) | 0.58 | 0.93 | 1 | - | 22 |
| student YOLO11n | 640 | | | | | | | 22 |
| student YOLO11s | 640 | | | | | | | 49 |

## Session 2: dataset export (WP8)

Code: `teacher/export.py` (`pool` per video: decode + guard + sampling + dedupe -> `datasets/_pool/<id>/`; `build`:
splits, balancing, YOLO files, README, stats; `qa`: contact sheets in `datasets/_qa/`). CPU only, ≤ 3 workers,
resumable (`datasets/_state/<id>.pool.done` holds the input mtimes + `POOL_VERSION`, so a changed `labels.parquet`
re-pools that video by itself; `pool.lock` prevents two runs at once). Outputs and counts: `datasets/README.md`.

* **Frame alignment:** `detect.frame_pipe` with `start = frame_idx / 4` (as crops.py) reproduces the detector's
  frames. Boxes drawn on exported frames sit exactly on the cars (checked on 24SU_SPP / 19SU sheets).
* **Splits:** test = 24SU (all), **val = 25SU (whole event, END + DIN)**, train = 19SU, 25BR, 26BR.
* **Result (verified labels, before grounded boxes):** cars train 26,837 images (25,072 pos, 470 hard neg, 1,295 neg;
  43,345 boxes), val 9,981, test 7,985. Negative share only 6.6% (2.8% test): the teacher negative pools are small
  after pHash dedupe, and I did not widen them (QA found missed cars in teacher negatives). Median box h/H 0.14 (train),
  0.24 (val), 0.13 (test). ids: 52k crops, 7,298 tracks, 133 (event, number) identities. Pool: 8.9 GB of JPEGs.
  numbers / digits empty until `labels.parquet` carries grounded boxes (re-run: `python teacher/export.py all`).
* **Poison guard, what QA forced (each found by looking at sheets, kept here so nobody relaxes it):**
  * Interpolating across fast pans produced boxes between cars; two tracks on one car gave doubled boxes. Fix:
    interpolate only gaps ≤ 2 frames whose end boxes overlap (IoU ≥ 0.3), label-box NMS (IoU ≥ 0.5), and a longer
    gap inside a car track blocks its frames.
  * Low-conf detections of real cars (0.15–0.25, untracked) slipped past a 0.25-only guard. Guard now: untracked
    ≥ 0.15, detections of unresolved tracks ≥ 0.10.
  * Nested boxes: "group box ⊃ car + partial of that car" hid a second car when dropped. Rule now uses direct children
    and coverage; a big outer box with little inside it is ambiguous → frame skipped.
  * **WP3 `screen_fixed` fires on real cars** (fixed_persistent on a car held in frame; fixed_bg_moving on a car at the
    edge), and an ego track can **drift onto another car** for a few frames (26BR_DIN PIP inset → car #32). So only
    bottom-anchored ego boxes (per detection) and graphics inside overlay regions are safe background.
  * **Ego/graphics-only frames are bad hard negatives:** 4 of 9 sampled onboard "hard negatives" showed a car ahead that
    the teacher missed. Hard negatives now come only from VLM-verified not-car tracks (road cars, props).
  * **VLM car check: 'no' is unreliable when the detector is confident.** On the 125 eye-labelled crops of
    `work/verify_val`, all 8 tracks with score ≤ 0.3 and conf_median ≥ 0.6 were real cars (2 of 4 at 0.5–0.6); a
    conf-0.92 car hidden behind a person was 'no' in QA. 'no' counts as background only below conf_median 0.50, above
    it the track is unresolved (~1/3 of the 'no' tracks). Below 0.5 'no' is still wrong on ~4/16 real cars.
  * The fallback rule before verification (identity or conf_median ≥ 0.40) labelled a tyre, crouching people and parked
    road cars in 19SU / 25BR (≈ 6–8 wrong boxes in 50 frames): do not train on fallback exports.
* **Final QA (54 random train frames, 73 boxes, verify rule):** boxes aligned on all; 1 frame with a clearly unlabelled
  car (26BR_END f002630: dusty car, VLM 'no' at conf 0.32), ~3 frames with an unclear small dark object; 2 doubtful
  boxes (one group box over two overlapping pit cars, one probable parked road car far away). Earlier verify-rule sample
  also showed a car hidden behind the leaderboard tower (undetected) and a car far ahead in a teacher negative.
  ids: 50 crops consistent with the visible number where readable, ~3 poor crops (dust / mostly background).
* Grounded number boxes (19SU / 24SU, read-only check of `grounded.parquet` `kept`): tight on the digits, in stored-crop
  pixels as the exporter assumes; ~94% of grounded crops keep a box; many boxes are only 10–15 px tall, so the digits
  set needs a size filter at training time.
* Throughput: pool 2–13 min per video with 3 workers (decode of the 1/s candidates at native size + pHash), ~30 min wall
  for all 9; build ~1 min; disk 8.9 GB (datasets are hardlinks).

## Session 2: teacher labels (WP2b/WP4/WP5)

Code: `teacher/overlay_ids.py` (WP2b), `teacher/crops.py` (WP4 CPU), `teacher/read.py` (WP4 GPU: `plain`, `ground`),
`teacher/verify.py` (per-track "is it a Baja car" check, added on the manager's request), `teacher/labels.py` (WP5:
labels, negatives, `qa` batches, `summary`), shared helpers `teacher/idlib.py`. Outputs per video in `work/<id>/`:
`overlay_ids.parquet` / `overlay_reads.parquet`, `crops/` + `crops.parquet` + `crop_tracks.parquet` + `ctx/`,
`reads.parquet`, `grounded.parquet`, `vcrops*/` + `verify.parquet` + `verify2.parquet`, `labels.parquet`,
`negatives.parquet`, `gpu_time.json`. Tables: `data/wp45_summary.csv`, `wp45_height.csv`, `wp45_agreement.csv`.
QA batches: `out/qa/<EV>_teacher.jsonl`.

### WP2b: identities from broadcast overlays

* Identity overlays exist only in DIN (lower third "#35 Carpoeira Baja", 25BR "16 - FORJA BAJA" + school line) and SPP
  (heat box "1 - BAJA DE GALPÃO 0 / 9 - PATO BAJA 1"). END streams have none (towers, tickers; 19SU's "lower_third"
  region is a sponsor-logo slot; 26BR_END's "CARRO 7 | CARRO 9" labels the onboard view). Rule: kind `lower_third` in
  `overlays.csv` of DIN (one number) / SPP (two numbers) videos.
* Method: region crop at 1 fps during its visible intervals (± 4 s), segments of unchanged content by a 32×8 dHash
  (new segment when > 18% of the bits change), one VLM transcription per segment of ≥ 2 samples ("transcribe all of its
  text"), parse `#N team` / `N - TEAM`, accept only if N is in the roster **and** the team text matches that entry
  (difflib / token containment ≥ 0.6), merge same-number segments ≤ 20 s apart. 1,787 reads, 8 GPU-min for 4 videos.
* Gotchas: Qwen often writes "#7\nImperador" on two lines (the parser now joins a bare number with the next line; before
  that 25SU_DIN had 44/187 accepted reads, after 182/187). Heats must have both numbers accepted. A 6 s merge gap split
  the heat boxes into many 3 s rows; 20 s fixed it.
* Validation: sheets `out/overlay_ids_<id>.jpg` (16 random ids per video) all correct by eye. The rejected reads are
  "none", "COMITÊ TÉCNICO", 26BR_DIN's onboard telemetry ("12.6V / 25Mbps") and logo fragments. Change-detection check
  (`overlay_ids.py check`): the first and last sample of 30 random segments read like the middle one in 180/180.
  Result: 228 identity intervals (26BR_DIN 124, 25BR_DIN 38, 25SU_DIN 35, 24SU_SPP 31).
* **Assigning an overlay to a track is the weak step.** In DIN shots the next cars wait in the queue at the side of the
  picture while the lower third names the car on the course. Two VLM-overlay conflicts with VLM share 1.0 were exactly
  this (#49 / #29 in the queue, box glued to the right edge). Rules now: candidate tracks not screen_fixed / ego /
  verified not-car / nested, h ≥ 60 px, ≥ 1 s inside the interval, edge_touch < 0.8; per shot at most n_numbers
  candidates on screen at once; a track overlapped by several rows takes the largest overlap and is 'ambiguous' if
  another number overlaps it ≥ 30% as long. Excluding verified not-car tracks (the road car towed in TRAÇÃO) unblocked
  several single-car windows. Only 22–72 tracks per DIN video get an overlay identity: few, but independent of mud.

### WP4: crops and reads

* Frame mapping: seeking `detect.frame_pipe` to `frame_idx / 4` reproduces the continuous fps=4 decode bit for bit (any
  multiple of 0.25 s, not only part boundaries). Boxes redrawn on an independent accurate-seek decode coincide
  (`out/crops_verify_24SU_SPP.jpg`). So crops decode only the needed spans (5–9 min per video, CPU).
* Readable tracks: buggy, not screen_fixed, not nested, not ego residue, not in a transition shot, ≥ 3 boxes ≥ 60 px.
  `in_overlay` is soft (WP3), not excluded. K = 8 per track (pick_crops logic: area ≥ 0.4 × max, 8 time chunks, best
  rank(sharpness) + rank(area) of 3 decoded candidates per chunk), 5% padding, JPEG q92 ≤ 640 px: 179k crops at 60 px.
* **Overlay masking:** crops contained ticker / tower / heat-box text with numbers ("18° Gralha Azul" in 19SU's
  ticker), so visible compact overlay regions (not bracket / inset boxes) are painted gray inside the crop. Crops more
  than 50% masked are dropped (25BR_END 2,263 and 26BR_END 841 candidates under the tower).
* **Read rate vs box height** (plain read; calibration sample of 140 tracks per video stratified by height, plus the
  full run):

| box h (px) | crops: roster number read | crops: = track consensus | tracks: consensus (share ≥ 0.5, ≥ 3 votes) |
|---|---|---|---|
| 60–80 | 13% | 6% | 6% (n=160) |
| 80–100 | 20% | 10% | 12% (n=162) |
| 100–130 | 30% | 18% | 17% |
| 130–170 | 44% | 28% | 26% |
| 170–230 | 53% | 36% | 37% |
| 230–330 | 64% | 48% | 48% |
| ≥ 330 | 74% | 61% | 66% |

  Reading becomes useful at ~100 px, and the only overlay disagreement below 100 px was a VLM error on an 82 px car. So
  the full run read only crops ≥ 100 px of tracks with ≥ 3 such crops (`read.py plain --hmin 100`). Reads of 60–100 px
  tracks exist only from the calibration sample. Consensus coverage of readable tracks: 7.5% (19SU, wide fixed
  camera) to 42% (25BR_DIN / 26BR_DIN).
* Speed: plain read 0.08–0.12 s/crop at batch 6 (faster than WP0's 0.16 because the crops are small; slower while other
  agents load the CPU). 115k crops = 3.0 GPU-h, calibration included. 10.3–10.8 GB VRAM in total at batch 6, no spill.
* Grounded read: only on crops whose plain read equals the consensus, 2 per track, largest first, round-robin over
  video × shot tag. 0.64 s/crop at batch 6 with an idle CPU, 1.1–1.8 s/crop while WP8 used the CPU (launch-bound, as
  WP0 found). All 7,417 consensus tracks would need 22k crops ≈ 6–9 h, so it was capped (8,000 planned, cut to ≈ 5.5k
  when CPU contention slowed it): 5,819 kept boxes on 2,607 tracks; 93–97% of grounded crops keep a box with the
  consensus label; JSON valid 100%. Sample sheet `out/grounded_sample.jpg`: boxes on the digits, many tiny side plates.

### Car verification (manager request; `teacher/verify.py`)

* Prompt "Is this an off-road Baja SAE race buggy (a small single-seat off-road car with an exposed tubular roll cage)?
  Answer yes or no."; score = P(yes) / (P(yes) + P(no)) from the first token's logits (bimodal). 0.055–0.115 s/crop.
* Validation: 160 crops (stratified by height and long_static) labelled by eye: 102 cars, 23 not cars, 35 can't tell.
  At 0.5: **precision 97.5% (78/80), recall 76%**. At ≥ 100 px recall is 54/58 with 0/3 FP; at 45–60 px 6/15; below
  45 px 2/5. The two FPs: the Baja logo's buggy drawing (score 0.99!) and a 36 px blob. 'no' is unreliable for small
  crops (labels.py turns a 'no' below 60 px into unsure) and, as WP8 found, for confidently detected cars: of 32 random
  large 'no' crops, ~17 were real cars, almost all surrounded by marshals or partly visible (the true 'no's: crop
  sprayer, police car, motorcycle, people, banners).
* Run: the largest crop of each of 45.7k pool tracks (median box ≥ 30 px, not screen_fixed / ego), plus a 2nd crop for
  unsure first scores. Then (WP8 request) 3 more crops with 15% padding for the 2,618 'no' tracks with
  conf_median ≥ 0.5, with a majority over all crops: **663 flip to yes, 615 to unsure, 1,340 stay no**. The 44
  screen_fixed tracks flagged only by the background-moving rule are decided by the VLM (13 yes, 30 no, 1 unsure).
  Ego / recurring-graphic screen_fixed tracks stay no.
* `is_car` in labels.parquet = no for screen_fixed (unless the VLM decides, as above) and for ego residue; yes when the
  track has a roster-number consensus or an overlay agreement (only Baja cars carry numbers); otherwise the VLM verdict.
* Ego residue (box > 0.8 H, bottom-anchored, ≥ 50% of a track's boxes, in a shot with a screen_fixed track): only 16
  tracks (26BR_DIN 12, 26BR_END 3, 24SU_END 1). Most of WP3's missed ego boxes are not that tall (helmet-only views,
  PIP insets).

### WP5: labels, automatic precision, QA

* **VLM consensus vs overlay identity (the automatic precision estimate): 94/97 = 96.9% (Wilson 95% 91–99%).** Lower
  thirds 90/93 (25BR 19/21, 25SU 23/23, 26BR 48/49), SPP heats 4/4 (consensus within the announced pair). By height:
  63/63 at ≥ 330 px, 15/17 at 230–330, 10/10 at 170–230, 0/1 at 80–100. The 3 conflicts: an 82 px car (VLM 12,
  overlay 16), 25BR #47 vs #33, and 26BR #3 (share 0.62) vs #31, the last probably a partial read of "31". Caveat: the
  overlay tracks are DIN / SPP close-ups and pits. END wide shots (consensus coverage 7–30%) are not covered by this
  check; the QA batches are for that.
* Session-1 rule kept: consensus = roster-snapped majority, share ≥ 0.5 of the crops read and ≥ 3 votes.
* Negatives: frames with no car detection ≥ 0.10 within ±1 s, outside transitions, ≥ 10 s apart per shot, split evenly
  over shot tags. Only 3,906 in total (19SU 81, 24SU_END 72): something car-like is detected in nearly every frame of
  the fixed / race cameras. WP8 found missed cars in teacher negatives, so treat them as weak.
* QA batches (`out/qa/<EV>_teacher.jsonl`, 150 items each): strata = shot tag × (proposal / abstained), 2/3 of the items
  with a proposal (n ≈ 100 for the precision interval), floor 6 per tag (3 for abstained), with weights. Crops = the
  exact files read, number boxes in crop pixels, context frame, per-crop reads. Overlay identity, final label and
  source go in `meta.hidden`. Checked in `tools/qa/serve.py` (index, /api/batch without hidden fields, images 200,
  unknown 404).
* GPU time (2080 Ti): overlay ids 10 min, plain 3.0 h, verify 53 min, grounded ≈ 1.55 h → **≈ 5.6 GPU-h**.

### Open problems for WP8 and later

* Use `is_car` (not `car_verified`) for dataset A. 'no' + high detector confidence after the recheck is still roughly
  half real cars occluded by people, so keep treating it as unresolved, not as background.
* Number boxes exist for 2.6k of the 7.4k consensus tracks (cap). The rest needs ~4–6 more GPU-h of grounded reads,
  best run with the CPU idle (0.64 s/crop vs 1.8).
* Reads exist only for crops ≥ 100 px (plus the calibration sample at 60–100 px); `small` tracks have no identity.
* Fragmented tracks of one car vote separately; merging by identity per shot (a WP3 open problem) would raise coverage.
  Overlay identities are few (≈ 140 tracks). The 26BR originals (Takeout) will need a full re-run (track ids differ).

## Session 2: dataset export (WP8), final run on the final labels

Update to the WP8 section above (which describes the provisional run on `car_verified`; its counts are superseded here).

* Positive rule is now WP5's **`is_car`**: a roster-number consensus or an overlay agreement overrides the VLM car-check
  misses. The remaining VLM 'no' with conf_median ≥ 0.5 stays unresolved (blocks the frame), unsure stays unresolved,
  and `sf_effective` replaces `screen_fixed`. 1,176 of 26BR_END's 5,192 car tracks are cars through an identity.
* **datasets/cars is final** (POOL_VERSION 7). Images train / val / test: 27,455 / 10,331 / 8,167 (previous build
  26,837 / 9,981 / 7,985). Boxes: 45,103 / 14,584 / 14,931. Negative share: 6.4% / 6.0% / 2.7%. QA on 54 random train
  frames (90 boxes): all aligned, no clearly unlabelled car, ~2 unclear small objects, 1 doubtful box (a foreground
  vehicle). `build --only numbers,digits,ids` leaves `datasets/cars` untouched.
* **numbers_crop:** 5,064 crops (train 3,095 / val 1,118 / test 851) with 5,819 boxes on 2,607 tracks.
  **numbers_full:** 1,221 frames (730 / 347 / 144) with 1,505 number boxes. Only ~25% of the frames that hold a grounded
  crop pass both the car guard and the "every other labelled car ≥ 60 px also has number boxes" rule. Label noise seen
  in QA: roughly a third of the crops show a second number of the same car (side plate / nose) without a box. Cars
  shown on a studio TV screen are labelled (car + number) in a few DIN frames.
* **digits:** 5,819 crops, 51 distinct numbers. Number-box height (stored-crop px) is 5/25/50/75/95% = 16 / 26 / 40 /
  57 / 86. **Boxes < 16 px (284, 4.9%) go to `images/tiny/<split>/`** (bucket = tiny); trainable: 3,503 / 1,178 / 861.
  **Leading zeros:** side plates show "06" while the consensus is 6. 11.7% of the grounded reads start with "0", so
  `labels.csv` has `text_read` (the grounded string) next to `text`; they differ beyond leading zeros on only 2 crops.
  QA on 50 trainable crops: none clearly wrong, ~5 too blurred to read. 24% of the digit crops come from crops that
  were downscaled to ≤ 640 px (big cars), so they could be re-cut at native resolution if that ever matters.
* ids: 52.8k crops, 7,440 tracks, 133 identities.
