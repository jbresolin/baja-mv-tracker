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

## Session 2: number recognizer (WP10)

Code: `students/recog/` (`prep.py` cache, `augment.py` augmentation + synthetic plates, `common.py` CRNN / CTC / roster
scoring / vote, `train.py`, `evaluate.py`, `window.py` + `eval_window.py` (interim mode without a panel detector),
`harness_26se.py`, `export_onnx.py`). Runs in `students/runs/recog/<name>/` (gitignored). **All CPU** (6 threads,
2 loader workers) while WP9 trained on the GPU; ~25-35 s per epoch of 6,000 samples, a run is 30-40 min.

### Model and data

* CRNN: 6 conv layers (VGG style, 32-192 ch) -> (2,1) conv collapse -> BiLSTM 128 -> 11 classes (CTC blank + 0-9).
  **1.29 M params**, input 32x96 RGB, aspect kept, padded with the border mean. Labels = `text_read` ("06" stays "06");
  metrics strip leading zeros. 2 rows dropped ("C4" and one text mismatch).
* `prep.py` re-cuts every digits box from the stored WP4 car crop with 50% context, so the augmentation can jitter the
  box without black borders: one perspective warp does the box cut / extend (+-12% per side on top of the 10% pad),
  rotation (gauss sd 7, clipped +-15 deg), perspective and shear; then colour (hue, channel swap, invert, gray, contrast,
  gamma), mud blotches / dust haze, edge or tube occlusion, motion / Gaussian blur, down-up resampling to 8-22 px,
  JPEG q15-75, noise. 0.7 ms/sample.
* **"No number" windows** (random car-crop windows that do not touch a grounded box, empty CTC target) are 20% of each
  batch. They keep false reads on background at ~4% (greedy non-empty) and roster P >= 0.5 at ~1%, and are what makes
  the window mode possible at all.
* CTC needs patience: loss sat at the blank plateau (~2.5) for 7 epochs (~300 steps) before it started reading.
* Synthetic plates (`augment.render`): 1-3 digit strings (25% leading zero for 1 digit) in 24 Windows fonts, plates in
  livery colours or none, outline strokes, 20% **hollow / outlined digits**, stencil gaps, letter spacing, real negative
  windows as background, scaled to 14-60 px digit height, then the same augmentation. Synthetic-only training reaches
  49% on val (zero real data) but fails on 2-digit numbers (13, 15, 16 near 0).
* Roster snapping is done in the CTC: `RosterScorer` computes P("n") + P("0n") from the CTC sequence likelihood for every
  roster number; P(other) = the rest. The track vote is the mean over crops of that posterior (soft vote).

### Results (labels = teacher consensus; val 25SU, test 24SU; selection on val only)

Crop level, 'ok' bucket (box >= 16 px), "acc" = greedy read == number after stripping zeros, "snap" = argmax of the
roster posterior (24SU / 25SU rosters have only 16 cars, 26SE has 24):

| run | val acc | val snap | test acc | test snap | test track top-1 | test track @ tau 0.6 (cov / prec) |
|---|---|---|---|---|---|---|
| real only, seed 0 | 79.7% | 81.6% | 49.7% | 57.3% | 66.2% | 38% / 85.6% |
| real only, seed 1 | 78.0% | 80.5% | 51.9% | 59.3% | 66.4% | 37% / 90.1% |
| synthetic only (25 epochs) | 48.9% | 56.3% | 35.3% | 48.1% | 56.2% | 27% / 86.7% |
| **synth pre-train -> real fine-tune, seed 0 (`syn_ft`, chosen)** | **80.2%** | **82.7%** | **59.9%** | **68.0%** | **76.0%** | **43% / 96.3%** |
| synth pre-train -> real fine-tune, seed 1 | 80.0% | 81.9% | 60.8% | 66.3% | 75.1% | 45% / 96.5% |
| synth pre-train -> real + 30% synth | 78.6% | 81.5% | 64.9% | 71.7% | 79.6% | 51% / 97.4% |

* **Synthetic pre-training helps where it matters: +1 point on val, +9.5 points on test** (2 seeds each). The new
  event's liveries are the hard part and font diversity is what synthetic data adds. Caveat: the hollow-digit option
  was added after looking at the real-only model's test errors (24SU cars 1 and 16 use outlined digits). Car 1 went
  49% -> 89%, car 16 stayed at ~25%, so part of the test gain is test-informed. Keeping 30% synthetic during fine-tuning
  was best on test but worse on val (78.6 vs 80.2), so by the rule `syn_ft` is the model.
* **The val -> test gap (80 -> 60%) is per livery.** Test per car (snap): 1 89%, 7 94%, 15 80%, 11 77%, 13 76%, 2 74%,
  9 73%, 6 70%, 4 55%, 3 53%, **16 24%** (96 crops, outlined digits on a triangular panel tilted ~25 deg). One event has
  only 13-16 cars, so one hard livery moves the event mean by 5-10 points. Training has ~100 (event, livery) pairs
  and 51 distinct numbers: more events help more than more crops per event.
* By box height (test acc/snap): 16-24 px 53/60%, 24-32 58/66%, 32-48 56/65%, 48-64 68/76%, 64-96 70/78%; val is
  44/53% at 16-24 px rising to 90/92% at 64-96 px. Tiny bucket (< 16 px): 52% test (n=21), 39% val (n=18).
* Digit confusions (same-length reads). Val: 1->7 (serif "1" with a flag), 0->1 (narrow side-plate "0"), 4->6, 1->3.
  Test: 6->3 and 6->8 (car 6's stylised 6), 0->1 and 0->3, 3->8 (stencil "03"). Length errors: mostly too short (a
  digit dropped: half-visible "1" at the box edge), 134 short vs 53 long on test.
* Error sheet (`students/runs/recog/syn_ft/eval/errors_test.jpg`, 40 test errors), by eye: ~10 car 16 (outlined, tilted
  > 15 deg), ~6 car 6 read 16/8/28 (stylised tail), 3 stencil "03" -> 38/08, 3 "04" -> 14, 3 "13" -> 3 (the grounded box
  cuts the "1": label box issue), ~8 blurred / dark / too small to read by eye, a few 1 -> 7. On val, several errors are
  grounded boxes that cover only part of the number (13 -> 3) and one crop that clearly shows "7" for a car labelled 8.
* **Track level** (test crops are only the grounded crops: 2.0 per track, 441 tracks). Soft vote, abstain below tau,
  tau picked on val: val 0.6 -> 98.6% at 73% coverage; on test the same tau gives **96.3% [92.6-98.2] at 43%
  coverage**; tau 0.7 98.2% at 38%; no abstention 76.0%. Majority of greedy reads (session-1 style) 90% coverage /
  75% precision. The teacher's own track precision is 96.9% vs overlays (WP5), so at matched precision the student
  covers ~43% of what the teacher labelled, with only 2 crops per track; the deployed system reads ~10-30 frames per pass.
  These are student-vs-teacher numbers on crops the teacher could read (selection bias towards readable crops).

### Interim mode without a panel detector (`window.py`) and the 26SE harness

* ~1,000-1,400 windows per car crop (crop resized to 320 px high; 3 window heights 30/44/60 px x aspects 1.0/1.6, 60%
  stride), crop posterior = per number the max over windows. 0.45-0.53 s per car crop with ONNX Runtime (6 threads).
* Teacher-labelled tracks, 4 car crops each (250 random tracks): val 25SU top-1 70.8%, test 24SU **51.2%**. The max over
  ~1,000 windows saturates the score (most tracks score > 0.9), so **the vote margin (top - runner-up) is the abstain
  rule**: margin >= 0.3 (val 19% coverage / 97.9%) gives test 6% / 100%; margin >= 0.2 val 36% / 95.6%, test 17% / 97.7%.
* Window failures (sheet of best windows on test errors): broadcast lower thirds inside the car crop ("10 - Pelznicke",
  "16 - Baja SATC", P = 1.00), and confident windows on wheels, grilles, people and sponsor text. Training on "no number"
  windows is not enough when a crop is searched with 1,000 windows.
* **26SE (evaluation only, 235 passes with a GT car, the 10 crops the VLM read): top-1 51.9%; margin >= 0.3 (from val)
  19.6% coverage at 93.5%; margin >= 0.4 7% at 100%.** The VLM teacher on the same passes: 77% / 94.0%. Per car: 4 96%,
  5 85%, 13 71%, 3 67%; 6 44%, 20 43%, 10 23%; car 1 31% (mud; "1" is the easiest digit to hallucinate from a vertical
  stroke, so its hits are not real reads). When the VLM abstains the window mode is right on only 17%.
  Output: `out/eval_recog_syn_ft_window/`. **End-to-end needs the number-panel detector**; the window mode is a code
  path check, not a usable reader.
* `harness_26se.py --mode panels --boxes boxes.csv` (any box source) or `--panel-weights <WP9 number net>` runs the real
  chain: panel boxes -> recognizer on box + 10% -> per-crop max -> vote -> roster 26SE. `--all-crops` reads every saved
  frame of the pass's tracks (up to `--max-crops`). Smoke-tested with fake boxes only.

### CPU latency (Ryzen 3900X, while WP9 trained on the GPU and its data loader used the CPU)

| runtime | threads | batch 1 (ms / crop) | batch 8 (ms / crop) |
|---|---|---|---|
| ONNX Runtime | 1 | **2.6** | 3.0 |
| ONNX Runtime | 4 | **1.0** | 0.9 |
| torch | 1 | 8.0 | 5.5 |
| torch | 4 | 4.4 | 2.0 |

* Preprocessing (resize + pad) 0.05 ms. ONNX opset 17, dynamic batch and width, max |onnx - torch| 1e-5:
  `students/runs/recog/syn_ft/recog.onnx`. Reading 10-30 number boxes per pass costs < 100 ms per pass on one thread.

### What limits accuracy, and what WP10 needs from the number-panel detector

* Liveries never seen in training (outlined, stencil, stylised digits) and panels tilted beyond +-15 deg. More events
  (more liveries) and a wider rotation range are the next levers; synthetic fonts already gave +9.5 points.
* Label boxes: grounded boxes sometimes cut a digit ("13" -> "3"). A panel detector trained on them inherits that, so
  **pad its boxes by >= 10-15% before reading** (the recognizer was trained on box + 10% with +-12% jitter) and prefer
  recall of whole panels over tight boxes. The vote copes with an extra box per crop; it cannot fix a cut digit.
* The detector should give **every number instance of the car** (side plates, firewall, nose; the grounded labels miss
  ~1 in 5) and **skip broadcast graphics**: in the window mode, lower thirds were the most confident wrong reads.
* Boxes >= 24 px tall read at 58-90%; 16-24 px at ~50%. Below 16 px reading is not reliable.
* Thresholds: crop / track posteriors are calibrated per mode. Pick tau (panel mode: score; window mode: margin) on val
  again once the detector exists. Never on 26SE.

## Session 2: QA review of the teacher labels, 24SU (user, blind, 150 tracks, 19.5 min)

* **Precision of the proposed number: 96.8% [91–99%] (stratum-weighted 97.2%). Wide END shots: 97.6% (n=41).** This agrees
  with the automatic VLM-vs-overlay check (96.9%, close-ups only), so the teacher's identity labels are ~97% precise on
  stream footage, wide shots included.
* Coverage 73%. Of the tracks where the teacher abstained, the reviewer could read 40% [26–56%], so the VLM vote is
  conservative.
* Not a car / bad box: 13.3%, concentrated in studio (44%: cars on TV screens) and pits (33%: two cars in one box).
  Track shots: 10–12%.
* Number boxes flagged wrong: 13.3% of the items that have boxes.
* By box height: 100–130 px gives only 1/3 correct (n=3, weak evidence); ≥130 px is ~100%. Reads near the 100 px floor
  are borderline.
* Effect: `export.py build --only cars` dropped 14 test frames (8,167 → 8,153). Train and val are unchanged.
* **Number variety (user's concern, confirmed):** digits/ids cover only 51/53 distinct numbers out of the 89 used across 60
  events (1–99). Every val/test number also occurs in train, so the WP10 test scores do not measure unseen numbers. WP10's
  24SU errors were style-driven (outlined, stylised and stencil digits). The existing data cannot fix this: the ids set has
  only 2 numbers more than digits. The fix is a pretrained scene-text recognizer, digit splicing (WP10b) and more events.
* **Number ceiling of the current footage:** the 5 events' rosters together contain only **59 distinct numbers (1–62)**,
  and the labels already cover 53 of them. The 6 never labelled are 40, 45, 46, 52, 56 and 62, all BR cars. Holding out
  25SU (val) and 24SU (test) costs no number variety, because all their numbers also occur in train. 26SE adds none
  either (1–24). Numbers above 62 never occur in any stream we have, so the recognizer **must** generalise to unseen digit
  combinations. This is a property of the events, not of the pipeline (user, 2026-10-18: all live streams are in).

## Session 2: students trained on the GPU (WP9, continued)

Data: `datasets/cars` (final): train 27,455 (6.4% negatives), val 25SU 10,331, test 24SU 8,167 (8,153 after the 24SU QA
pruning). All runs AMP fp16 on the 2080 Ti, `students/runs/<name>/`. ultralytics writes `labels/*.cache` next to the
dataset labels (the only write into `datasets/`).

### Calibration and the real training speed (replaces the published-number estimate in the plan above)

* YOLO11n @640, batch 64: **8.3-8.4 GB** torch (9.3 GB with the desktop), fits. Train **3.4 it/s = ~220 img/s** with
  WP8 using the CPU at the same time (calibration with a lighter CPU load: 4.4 it/s = 280 img/s). GPU utilisation ~75-80%
  during training, so it is partly **data-loader bound** (8 workers, CPU 60-85% busy). Validation of the 10,331 val
  images ~37 s. **One epoch = ~165 s**; the estimate (200-250 img/s) was right.
* YOLO11s @640, batch 32: 7.9 GB, 3.0-4.1 it/s (190-260 img/s), ~260-400 s per epoch depending on the CPU load.
* Calibration (3 epochs on 25% of train) already reached test mAP50 0.81: the teacher labels are easy to fit.

### cars_n640 (YOLO11n, 640, 100 epochs max, patience 20)

* Early-stopped at epoch 71 (best 51), **3.3 h**. Val (25SU) mAP50 0.846 / mAP50-95 0.660 at best; test (24SU) **mAP50
  0.924, mAP50-95 0.742** (P 0.912, R 0.853). After the 24SU QA pruning (14 frames with a not_car track dropped):
  0.924 / 0.742, unchanged (`--test-only`; the pre-pruning numbers are in `eval_test_before_qa_prune.json`). The val
  mAP dipped in epochs 2-5 (warm-up of a fine-tuned head, SGD lr 0.01 chosen by `optimizer=auto`) and recovered by 8.
* **26SE, full footage on the GPU (`out/eval_cars_n640/`): 246 passes, recall 0.913, precision 0.935** (YOLO-World
  0.933 / 0.975). Per file recall 0.97 / 0.81 / 0.875 / 1.00 (YOLO-World 0.97 / 0.87 / 0.875 / 1.00). Borrowed identity:
  175 read, 93.7% right, end-to-end 65.1% (teacher 67.9%).
* **Misses: 22, and every lap YOLO-World missed is also missed by the student** (17 shared). The 5 student-only misses
  (`out/eval_cars_n640/strips/`): 4 are in the **10:24-10:51 overexposed stretch** (nearly white frames; YOLO-World
  still boxes the car at 0.3-0.9, the student gives nothing above 0.15), and 1 (car 4, 10:55:39) is two cars passing
  ~1.5 s apart that end up in one pass (the student detected both; a pass-merge case). Washed-out frames are rare in the
  broadcast training data: an **overexposure augmentation** (gamma / additive brightening + contrast loss) is the
  obvious next lever. Not added to s640, so that n and s stay comparable.
* False passes 16 (YOLO-World 6): the same long stationary tracks as the teacher (start of the last file, 09:37,
  10:20), plus **end-of-race clutter 12:04-12:09** (cars stopped at the line among people: 6) and 0.2-0.4 s blips of
  6-11 frames (5). A minimum pass duration / frame count would remove the blips; not tuned (26SE is the benchmark).
* Early check (epoch-32 snapshot, CPU, last file only): 65 passes, recall 0.984, precision 0.954; so 20 more epochs
  mostly sharpened precision.
* **Full 26SE evaluation on the GPU took 31 min** (estimate was ~40): timeline 53 img/s on the 4K file and 69-71 img/s on
  1080p (NVDEC, decode-bound), tracking 50-54 frames/s, tracking windows 1,803 s. CPU only (4 threads, while training
  ran on the GPU): last file 715 s timeline + 1,050 s tracking, i.e. ~1.9 h for the whole race.
* CPU latency of the trained model (ONNX, 384x640, 1080p input, CPU loaded by other jobs, high priority): **ONNX 45 ms
  (1 thread) / 21 ms (4 threads)**, OpenVINO 45 / 20 ms, peak RAM 137-154 MB. Same as the pretrained YOLO11n.
* Gotcha fixed: `bench_cpu.py` crashed when NMS returned nothing (empty index list became a float array); the bench
  frame has no car in it, which is fine for latency.

### Number-panel detector (`datasets/numbers_crop`, class `number`, car crops <= 640 px)

Train 3,095 / val 1,118 (25SU) / test 851 (24SU). Known label noise: ~1/3 of crops show a second number without a box,
which depresses recall and mAP. Crops median ~370-560 px wide; label boxes median 37 px high (val 53), 6.5% < 16 px.

| run | val mAP50 / 50-95 | test mAP50 / 50-95 | epochs (best) | time |
|---|---|---|---|---|
| `num_n320` (YOLO11n, 320, batch 128) | 0.743 / 0.351 | 0.542 / 0.203 | 94 (64) | 12 min |
| `num_n416` (YOLO11n, 416, batch 96) | 0.742 / 0.363 | 0.557 / 0.211 | 101 (71) | 20 min |

* Big val -> test gap, as for the WP10 recognizer (24SU liveries). 416 is not clearly better (+1.5 points), so
  **`num_n320` is the model** (cheaper per crop).
* Chain on val, `students/panel_val.py` (400 teacher-labelled 25SU tracks, 4 crops each, recognizer `syn_ft`): the
  detector threshold matters more than the input size. At conf 0.25, 31% of crops get no box and track top-1 is 75%;
  **at conf 0.1, 11% of crops get no box and top-1 is 86%** (416: 88%). Extra box padding of 5% per side (on top of the
  recognizer's own 10%) adds ~1 point; 10% is no better. Picked on val: **conf 0.1, pad 0.05**. The val curve never
  reaches 97% precision (it tops out at ~95-96%): tau 0.6 71% coverage / 94.4%, **tau 0.8 43% / 95.4%** (chosen = the
  lowest tau with >= 95% on val), tau 0.9 39% / 95.5%. With the grounded boxes WP10 had 98.6% at 73%, so the detector's
  boxes cost ~4 points of precision at the same coverage.
* What it gets wrong (val sheet `students/runs/num_n320/panel_val_sheet.jpg`): it **fires on numbers printed on people's
  shirts** ("96", "153") inside car crops, and misses some large clear numbers (car 7 side panel, car 14, car 9 rear),
  i.e. the missing-box label noise shows up as missed recall. No broadcast-graphic hits in that sample.
* CPU cost per car crop at 320 (ONNX, loaded CPU): **30 ms (1 thread) / 19 ms (4 threads)**, OpenVINO 35 / 20 ms, RAM
  ~110-140 MB. Ten crops per pass is ~0.3 s on one thread; fine for lap counting.

### 26SE identity chain: car crops -> num_n320 -> recognizer -> vote (evaluation only)

`students/panel_boxes.py` (boxes at conf 0.1, pad 0.05, on exactly the harness's crops) -> `students/recog/harness_26se.py
--mode panels --boxes ... --tau 0.8`. 235 passes with a GT car, 252 laps; outputs `out/eval_recog_syn_ft_panels[_all]/`.

| crops per pass | tau | coverage | precision | end-to-end (of 252 laps) |
|---|---|---|---|---|
| the 10 VLM crops | 0 (no abstain) | 100% | 62.6% | 58.3% |
| the 10 VLM crops | **0.8 (val)** | 28% | **95.5%** | 25.0% |
| all crops (<= 30) | 0 | 100% | 66.8% | 62.3% |
| all crops (<= 30) | 0.8 (val) | 16% | 100% | 15.1% |
| all crops (<= 30) | 0.3 (info only) | 51% | 98.3% | 46.4% |
| VLM teacher (10 crops) | - | 77% | 94.0% | 67.9% |

* With no abstention the chain is close to the teacher's end-to-end (62% vs 68%), but its confidence is weak: at the
  val-picked tau it answers only 16-28% of passes. **tau does not transfer between crop counts**: the track score is
  the mean over crops of the per-crop max posterior, and crops with no box add zeros, so 30 crops per pass score lower
  than the 4 crops per track of the val run (at 30 crops, tau 0.3-0.5 already gives 98-100%). Fix for WP10: vote over
  crops that have a box (or weight by box count), and pick tau on val with the same number of crops per track as
  deployment. Not tuned here on 26SE.

### Second-round box proposals (report only, `students/panel_propose.py`)

On the crops of teacher-labelled tracks of the **train** events (19SU, 25BR, 26BR) that have no grounded box: num_n320
at conf 0.1, boxes >= 16 px, recognizer read == track consensus with P >= 0.5. **29,756 crops -> 47,041 boxes proposed
-> 32,425 accepted on 21,698 crops of 4,318 / 4,715 tracks** (4,888 dropped as < 16 px). That is ~9x the current
3.5 k train boxes. Caveat: it only adds boxes the current detector already finds and the recognizer can read (easy
panels); the hard ones (missed panels, unreadable liveries) stay missing. `out/panel_proposals_num_n320.csv`.

### Comparison table (filled so far)

| detector | input | 24SU test mAP50 / 50-95 vs teacher labels | 26SE passes | recall | precision | false passes | end-to-end (borrowed ids) | CPU ms ORT 1 / 4 thr (384x640) |
|---|---|---|---|---|---|---|---|---|
| YOLO-World v2-x (teacher) | 1280 | reference | 241 | 0.933 | 0.975 | 6 | 67.9% | GPU only |
| COCO YOLO11x | 1280 | - | 230 | 0.869 | 0.952 | 11 | 62.7% (own VLM) | - |
| COCO YOLO11n, untrained (11-09-03 only) | 1280 | - | 15 (24 laps) | 0.58 | 0.93 | 1 | - | 50 / 22 |
| **student YOLO11n (`cars_n640`)** | 640 | **0.924 / 0.742** | 246 | **0.913** | 0.935 | 16 | 65.1% | **45 / 21** |
| student YOLO11s (`cars_s640`) | 640 | (training) | | | | | | 135 / 49 (pretrained) |

## Session 2: recognizer generalisation (WP10b)

Code (all CPU, `CUDA_VISIBLE_DEVICES=-1`, <= 8 threads while WP9 trained on the GPU): `students/recog/` `folds.py` +
`folds/numbers_v1.json` (fixed folds), `heldout_eval.py` (the protocol), `gen.py` (fold / car filters, uniform text),
`splice.py` (digit splicing), `parseq_model.py` + `train_parseq.py` (PARSeq-tiny), `prep_props.py` (WP9 proposals as
training data), `panel_tau.py` (panel-chain abstain rule on val), `wp10b_table.py`; options added to `train.py`
(`--fold --keep-cars --syn-text uniform --p-splice --p-prop`), `export_onnx.py` (`--ckpt --export-only`),
`harness_26se.py` (`--vote boxed`), `common.py` (`vote_boxed`, PARSeq checkpoints in `load_model`), `augment.py`
(`render(text=)`, `synth_sample(text_fn=)`). Defaults reproduce WP10 (syn_ft) exactly. New pip package: `timm==1.0.30`
(requirements.txt). PARSeq weights: `weights/parseq/parseq_tiny-e7a21b54.pt` (github.com/baudm/parseq release v1.0.0,
Apache-2.0, sha256 prefix checked). Runs: `students/runs/recog/{gx,pq}_<recipe>_{full,f0..f3}`.

### The unseen-number protocol (reusable)

* Folds v1 (`folds.py`, deterministic): folds 0-2 each hold out 5-6 of the numbers 1-16 (balanced by val+test crops,
  >= 2 one-digit and >= 2 two-digit) plus 5 of 17-39/40-61; **fold 3 = the 7 remaining numbers >= 40** (manager request:
  the high-number case). Every crop of a held-out number, and the "no number" windows cut from the same car crops, is
  removed from training in all events. Every digit keeps >= 30% of its crop-weighted count.
* Val/test contain only 1-16 and folds 0-2 partition 1-16, so **every val/test crop is "unseen" for exactly one fold
  model: the gap is perfectly paired** (same crops, same box heights): full model (all numbers) vs the fold model that
  never saw the crop's number. Fold 3 has no val/test crops: it is scored on its train-split crops ("unseen number, seen
  event"), next to the folds 0-2 numbers under the same condition. Seen-number control: fold models on val/test numbers
  they did see (~20% less training data; it never mattered: within 1 point of the full model in every family).
* Recipes are short and fixed (no early stopping, `last.pt` evaluated): CRNN = syn_ft recipe (init syn_pre, lr 1e-3,
  30 epochs x 6000, ~18-21 min at 4 threads); PARSeq = 5 epochs x 6000, batch 64, lr 3e-4, ~28-31 min. The short CRNN
  full model equals syn_ft (val/test snap 80.5/68.1 vs 82.7/68.0; 75.3 vs 75.0 on all val+test crops), so ~1 point is
  the run-to-run noise to keep in mind. Usage: train the 5 runs with `--fold v1:K`, then
  `heldout_eval.py --family <prefix>`.
* **Seen number, unseen style proxy:** val/test are already that. Of the 28 val/test cars with crops, all numbers occur in
  training, but only 24SU #6 (Imperador, #6 at 26BR too) with the same team; the others belong to a different
  team. Team seen in training with another number vs team never seen: 74.5 vs 78.2% (height-standardised, CRNN), so
  team liveries do not carry over between events either. 24SU #6 (same team, same number) is still 71-73%: liveries
  change between years.

### Result 1: the unseen-number gap (roster-snapped crop accuracy, val+test 'ok' crops, n = 2,038)

| family (recipe) | number seen | number held out | gap | test: seen -> held out | test tracks, held out: top-1, tau 0.6 cov/prec |
|---|---|---|---|---|---|
| CRNN syn_ft recipe (`gx_base`) | 75.3 | **54.6** [52-57] | -20.7 | 68.1 -> 47.3 | 52.4%, 20% / **82.6%** |
| CRNN + splicing + uniform synth (`gx_spl`) | 76.4 | 69.3 | -7.1 | 68.8 -> 62.0 | 70.1%, 33% / 99.3% |
| CRNN + splicing + synth + proposals (`gx_all`) | 76.1 | 68.1 | -8.0 | 68.4 -> 60.7 | 68.5%, 32% / 96.5% |
| PARSeq-tiny, real only (`pq_base`) | 86.1 | 73.3 | -12.8 | 78.3 -> 63.1 | 68.7%, 42% / 91.9% |
| **PARSeq-tiny + splicing + uniform synth (`pq_spl`)** | **87.4** | **84.4** [83-86] | **-3.0** | 82.1 -> 76.3 | **82.3%, 48% / 98.6%** |
| PARSeq-tiny + splicing + synth + proposals (`pq_all`) | 87.9 | 83.8 | -4.1 | 82.3 -> 74.5 | 80.7%, 54% / 98.7% |

* **The WP10 CRNN loses 21 points on a number it never saw** (greedy 69.3 -> 47.0), at every box height (16-24 px 55 ->
  37, 64-96 px 89 -> 66), and **its confidence does not drop**: at tau 0.6 track precision falls from 97% to 83-85%.
  ~70% of the errors on held-out numbers are reads of another *training* number (val 368 of 529 errors, test 376 of
  551: 7 -> "1", 11 -> "17", 16 -> "13", 14 -> "16"). The BiLSTM memorises whole numbers like words. Deploying it at an
  event with new numbers would silently produce confident wrong numbers.
* Unseen number in a seen event (train split, `gx_base` fold models): 1-digit 69%, 10-16 63%, 17-39 77%, >= 40 78%
  (fold 3: 80%). PARSeq + splicing: 90 / 85 / 95 / 95% (fold 3: 96%). **High numbers are not harder than low ones**
  (the train-split crops of 17-61 are larger and cleaner than the 10-16 ones; not paired, read as "no high-number
  penalty").
* Splicing + uniform synthetic numbers closes most of the gap for both models (CRNN -21 -> -7, PARSeq -13 -> -3) and
  restores calibration (held-out track precision at tau 0.6: 99%). It costs nothing on seen numbers (+1 point).
* Scene-text pretraining (PARSeq) alone gives +11 points on seen and +19 on unseen numbers, but still a 13-point gap and
  overconfident unseen reads (92% at tau 0.6); it needs the splicing too.

### Result 2: PARSeq-tiny (scene-text pretrained) vs the CRNN

* PARSeq-tiny (6.0 M params, ViT encoder 12 x 192, 1-layer decoder, 32x128), pretrained on real scene-text datasets,
  vendored model code (`parseq_model.py`, Apache-2.0 header kept), digits-only head/embedding rows sliced from the
  94-char model, permutation-LM fine-tuning (forward + mirrored perms). Zero-shot (94 chars, tight crops) it already
  reads 12/15 val crops; after **one** epoch of fine-tuning val accuracy is 87.5% (the CRNN tops out at ~80%).
* **Same interface as the CRNN:** input "image" (N,3,32,W) RGB in [0,1] (the `augment.plain` / `to_input` 32x96 crop),
  resized inside the model to 32x128; output "logp" (T=10, N, 11) CTC-shaped log-probs (`positions_to_ctc`: position k
  -> a frame with P(blank) = cumulative EOS, digits renormalised, plus a certain-blank separator frame). So `greedy`,
  `RosterScorer`, the votes, `evaluate.py`, `harness_26se.py`, `panel_val.py` and the ONNX loader work unchanged.
  ONNX gotcha: `aten::cumprod` has no ONNX export at opset 17; written as exp(cumsum(log)). max |onnx - torch| 4e-5.
* PARSeq on the 32x64 input (half the encoder tokens, position grid resampled 8x16 -> 8x8; `--img-w 64`): val/test
  83.1/61.6% vs 89-90/77% at 128, i.e. no better than the CRNN on test. Not worth the ~30% speed-up.
* int8 dynamic quantisation (onnxruntime `quantize_dynamic`) broke it (val 8%) and gained only 12% speed. The encoder
  is 29 of 35 ms (loaded CPU). Not pursued.
* PARSeq overfits the training numbers when fine-tuned longer: fold runs often have their best val epoch at 1-2 of 5
  (val holds their unseen numbers). Keep schedules short.

### Result 3: digit splicing + uniform synthetic numbers (`splice.py`, `--syn-text uniform`)

* Bank: syn_ft reads 3,330 of 3,502 train 'ok' crops exactly; CTC forced alignment (Viterbi on the known label) gives
  each digit's frame centre, the cut is the least-ink column (Otsu) within +-20% of the centre gap: 3,302 crops, 5,942
  glyphs (digit counts '1' 1,282 ... '7' 288). Cuts checked by eye on 64 crops (`splice_cuts.jpg`): right on nearly all.
* Composites (`splice_samples.jpg`): 35% same crop ("ab" -> "ba", "aa", "bb", "a", "b"), 25% same team (glyphs from the
  team's crops across events, a number the team does not carry), 40% cross-crop with a target uniform over 1-99 + "0N"
  and glyphs from crops with a similar plate colour. Glyphs scaled to one box height, 6% linear seam blend, outer
  context from the source where the glyph keeps its side. Then the normal augmentation. Plus synthetic plates with
  `gen.rand_text_uniform` (12% "0N", 23% "N", 65% two digits with uniform digits): **63-99 and single digits in both
  forms are covered by both generators**, although no real crop of 63-99 exists.
* Mix used: 20% "no number" windows, 20% spliced, 10% synthetic, 50% real. Fold runs build the bank without their
  held-out numbers' crops (the alignment model saw all training numbers, but no held-out pixels enter a fold's
  training). Effect: see Result 1 (-21 -> -7 CRNN, -13 -> -3 PARSeq), seen numbers +1, test crops +2.5 greedy / +0.7 snapped
  (CRNN), +5.1 / +3.8 (PARSeq, 72.2 -> 77.3 greedy).

### Result 4: WP9's panel proposals as extra real data (`prep_props.py`, `--p-prop`)

* 32,425 proposals -> capped at 6 boxes per track (round-robin over the track's crops; against near-duplicate views and
  confirmation bias) -> 22,004 -> 21,979 with syn_ft's greedy string equal to the consensus (keeps "06": 1,018 leading
  zeros), 4,318 tracks, 50 numbers. Box + 5% (the deployment pad), cut and augmented like the grounded crops.
  Train events only (asserted); folds drop their numbers.
* **No measurable gain**: CRNN val/test 78.4/59.8 vs 78.4/61.3 without, held-out 68.1 vs 69.3; PARSeq 90.5/76.7 vs
  89.4/77.3, held-out 83.8 vs 84.4, panel chain test top-1 92.2 vs 91.4. All within ~1 point (noise). As feared, these
  are panels the current models already read: more views of the same 100 cars, not new liveries. Not used in the
  recommended model (keeps it independent of another model's picks).

### Result 5: track vote fixed for the panel chain (`common.vote_boxed`, `panel_tau.py`, `harness_26se.py --vote boxed`)

* WP9's diagnosis confirmed on val with crop-count simulation (random subsets of 2/4/6/all <= 8 crops per track,
  num_n320 boxes at conf 0.1 + 5% pad, 2,116 val tracks / 15,007 crops, 604 test tracks): with the old mean vote,
  coverage at a fixed tau falls as crops are added (syn_ft, tau 0.8: val 56% at 2 crops -> 45% at all; test 27% -> 16%).
* `vote_boxed`: ignore crops without a box; score = posterior-weighted vote share (sum of P(winner) / sum of each
  crop's max P), so weak crops (blur, partial panel, a box on a shirt) carry little weight; answer only if >= k boxed
  crops vote for the winner with P >= 0.5. Also a "mean over boxed crops" variant. tau and k are chosen on val (max
  coverage at >= 97% precision with all crops, and >= 96% at 4 crops); written to `<run>/eval/panel_tau.json`, which
  the harness reads. The share rule won for every model except `gx_base` (mean over boxed crops, tau 0.9, k 1);
  k = 3 everywhere except `pq_all` (k = 2).
* 26SE (evaluation only; WP9's boxes, nothing tuned): syn_ft 10 crops 28% / 95.5% (old) -> **35.7% / 96.4%**; all crops
  (<= 30) 16% / 100% -> **29.8% / 100%**. Better, but the val tau is still conservative on 26SE: the start-line crops
  give weaker posteriors than broadcast close-ups (domain), and with 30 crops a rule with tau near 1 needs near-unanimity.
  The real lever was the recognizer (below). Val tracks have at most 8 crops, so 10-30 crops cannot be simulated
  directly; precision rises with the crop count at a fixed tau in every simulation, so the val rule errs on the safe side.

### Comparison and recommendation

Crop columns: grounded boxes ('ok' bucket), greedy accuracy / roster-snapped. Panel chain: num_n320 boxes on WP4 crops
of teacher-labelled tracks (<= 8 crops), top-1 and the val-chosen rule on test. Latency: ONNX Runtime, batch 1, CPU
loaded by WP9's training (`export_onnx.py`), 4 threads at batch 8 in brackets. Held-out: Result 1.

| model | params | ms/crop 1 thr / 4 thr | val crop | test crop | panel chain top-1 val / test | chain rule on test (cov / prec) | held-out number (snap) | 26SE panels, all crops: top-1, val rule cov / prec |
|---|---|---|---|---|---|---|---|---|
| syn_ft (WP10) | 1.29 M | 2.5 / 1.0 (0.6) | 80.2 / 82.7 | 59.9 / 68.0 | 89.0 / 78.5 | 35% / 97.2% | ~55 (as gx_base) | 66.8%, 29.8% / 100% |
| CRNN + splice (`gx_spl_full`) | 1.29 M | 2.6 / 1.2 (0.7) | 78.4 / 81.9 | 61.3 / 68.8 | 87.4 / 81.3 | 37% / 98.7% | 69.3 | 68.1%, 40.0% / 98.9% |
| PARSeq-tiny (`pq_base_full`) | 5.98 M | ~21 / 12 | 90.3 / 91.9 | 72.2 / 78.3 | 90.4 / 86.9 | 48% / 97.6% | 73.3 | - |
| **PARSeq-tiny + splice (`pq_spl_full`)** | 5.98 M | **21.0 / 12.1 (7.0)** | 89.4 / 91.3 | **77.3 / 82.1** | **91.3 / 91.4** | **61% / 98.9%** | **84.4** | **77.9%, 54.5% / 99.2%** |
| PARSeq-tiny + splice + proposals (`pq_all_full`) | 5.98 M | same | 90.5 / 91.9 | 76.7 / 82.3 | 91.3 / 92.2 | 56% / 98.5% | 83.8 | - |

* 26SE details (evaluation only, 235 passes with a GT car, 252 laps; `out/eval_recog_pq_spl_full_panels[_all]_boxed/`):
  10 VLM crops: top-1 71.1% (syn_ft 62.6%), val rule 55.3% / 93.8%; all crops: top-1 77.9% (syn_ft 66.8%), val rule
  54.5% / 99.2% [95.7-99.9]. End-to-end of the 252 laps without abstaining: 183 = 72.6% (VLM teacher 67.9%); with the
  val rule 127 = 50.4% at 99% precision. The 10-crop precision (93.8%) is under the val target: calibration does not
  fully transfer to the start-line domain.
* **Recommendation: `pq_spl_full` (PARSeq-tiny + digit splicing + uniform synthetic numbers).** It is the only model
  whose accuracy on numbers it never saw (84%) is close to its accuracy on seen ones (87%), it is the best on test and
  on 26SE, and it is calibrated on unseen numbers. Cost: ~21 ms per number box on one thread of this 3900X under load
  (12 ms on 4 threads, 7 ms/box batched), 123 MB peak RAM, 23 MB ONNX. At 10-30 boxes per pass that is 0.2-0.6 s per
  pass on one thread; fine for lap counting and the PTZ entry camera, but it is ~8x the CRNN. If the laptop budget
  cannot afford it, use `gx_spl_full` (CRNN + splicing, 2.6 ms): never the WP10 syn_ft, whose unseen-number reads are
  confidently wrong. `pq_all` is equivalent within noise; `pq_spl` is chosen as the simpler one (best held-out score,
  no dependency on proposals picked by another model).
* **Swap-in for WP9's 26SE chain:** `students/runs/recog/pq_spl_full/recog.onnx` has the same input / output names and
  layout as syn_ft's (`image` (N,3,32,W) float RGB 0-1 -> `logp` (T,N,11), T = 10 instead of W/4). `common.load_any`
  picks it up by run name: `harness_26se.py --name pq_spl_full --mode panels --boxes out/panel_boxes_num_n320[_all].csv
  [--all-crops] --vote boxed` (tau 0.9 / k 3 from `pq_spl_full/eval/panel_tau.json`); `panel_val.py --recog
  pq_spl_full`. Torch fallback: `common.load_model(pq_spl_full/last.pt)` (= best.pt) needs `timm`; the ONNX path needs
  only onnxruntime.

### What more data would buy

* Distinct training numbers: with splicing + uniform synthetic numbers, PARSeq reads numbers absent from its real
  training data at 80-84% whether it was trained on 23 numbers / 28 cars (`pq_k25`: unseen 82.2%, seen 89.3%), 36 / 52
  (`pq_k50`: 79.8% on the 361 crops of its unseen numbers) or ~40 / ~80 (folds: 83.8%). **More distinct numbers are no
  longer the bottleneck**; the remaining 3-5 point unseen gap is flat in the number count.
* Cars / liveries: 28 -> 52 -> 100 training cars give test crop accuracy 74.8 -> 74.2 -> 76.7% (snap 80.1 -> 80.6 ->
  82.3), val 89.0 -> 90.0 -> 90.5%: about +1 point per doubling of cars. The test errors are concentrated in a few
  liveries (24SU car 16 outlined digits, car 13, stylised 6), so data helps only when it brings new styles: each new
  event adds ~15-50 liveries. Expect +1-2 points on a new event per doubling of events, i.e. diminishing returns;
  start-line footage from future events (target domain) is worth more than more broadcast close-ups.
* SVHN pre-training (task 4): not done. Licence non-commercial (acceptable for an internal team tool), but the
  scene-text pretrained PARSeq already supplies the real-text prior, and the residual errors are livery styles, not
  house-number-like digits. Lower priority than recording start-line footage.

## Session 2: WP9 final round (s640, panel detector round 2, washout augmentation, fully student 26SE chain)

Everything below ran unattended overnight from `students/queue_wp9.sh` / `queue_wp9b.sh` (one GPU job at a time; CPU-only
stages with `CUDA_VISIBLE_DEVICES=-1`). Nothing was tuned on 26SE.

### cars_s640 (YOLO11s, 640, batch 32, patience 20, `--hours 10`)

* 7.9 GB VRAM, ~260 s per epoch (3.0-4.1 it/s), early-stopped at epoch 75 (best 55), **5.4 h**. With `--hours` set,
  ultralytics re-plans the epoch count to fit the budget (it printed "x/127"); patience still applies.
* Val (25SU) mAP50 / 50-95 **0.867 / 0.692** (n640 0.846 / 0.660); 24SU test **0.935 / 0.772** (n640 0.924 / 0.742).
* **26SE: 242 passes, recall 0.929, precision 0.967, 8 false passes**: practically the teacher (0.933 / 0.975, 6). Misses
  18 = the 17 teacher misses + car 11 at 12:04:49 (end-of-race crowd). It finds the 4 washed-out laps that n640 missed.
* CPU (ONNX, 384x640, loaded CPU, high priority): **126 ms (1 thread) / 45 ms (4 threads)**, OpenVINO 138 / 44 ms, 205 MB.
  It meets <= 50 ms only with 4 threads.

### cars_n640_v2 = cars_n640 + `--washout 0.15` (everything else identical)

* `Washout` (train_det.py): gamma 0.5-1.0, blend towards white a = 0.4-0.85, desaturate to 30-100%, on 15% of training
  images. Justification from the training data, not only 26SE: in 3,000 random `datasets/cars` train frames the
  5th-percentile luminance is <= 91 in 99% of frames and never > 120 (broadcast cameras keep exposure under control),
  while a washed-out fixed camera (26SE 10:24-10:51) gives mean ~235 and 5th percentile ~160. Augmented frames: mean
  176-235, 5th percentile 120-185.
* Early-stopped at 86 (best 66), 3.9 h. Val 0.850 / 0.663 (n640 0.846 / 0.660), 24SU test 0.922 / 0.737 (0.924 / 0.742):
  **no cost on the streams**.
* **26SE: 250 passes, recall 0.925 (n640 0.913), precision 0.932 (0.935)**. **All 4 washed-out student-only misses of
  n640 (10:34:41, 10:36:16, 10:36:21, 10:39:52) are now found**; the remaining student-only misses are the two-cars-in-one-pass
  case (10:55:39) and the end-of-race crowd (12:04:49). False passes 17 (n640 16): the same kinds (stationary tracks,
  end-of-race clutter, 0.2-0.4 s blips). CPU 44.5 / 20.7 ms (ONNX 1 / 4 threads), same as n640.
* Recommendation: keep the washout augmentation in every future car student (free on the streams, fixes a real
  fixed-camera failure).

### Tracking at 10 fps instead of 30 (deployment rate; cars_n640_v2, same timeline)

* `--track-fps 10`: **240 passes, recall 0.925 (same), precision 0.971 (30 fps: 0.932), 7 false passes (17)**. The
  passes.py thresholds are in frames (`MIN_FRAMES = 6`, i.e. 0.2 s at 30 fps but 0.6 s at 10 fps), so the 0.2-0.4 s
  blips disappear. Not a threshold change, but note it: a deployment at 10 fps sees this precision; the "right"
  minimum duration should be set in seconds, on non-26SE footage, if pass logic is ever revisited.

### Number-panel detector round 2 (`num_n320_r2`, `datasets/numbers_crop_r2`)

* `students/build_numbers_r2.py`: numbers_crop (hard links, unchanged val / test) + the WP9 proposals of the train events,
  capped at 6 crops per track (evenly over the track): **+20,004 crops / 29,822 boxes** from 4,318 tracks (train 3,095 ->
  23,099 crops). A random sheet of 40 proposal crops: boxes sit on real numbers on nearly all; a few (car 29, car 10,
  car 47) are on a dark panel or a light with no visible digit, and the other numbers on the same car (side plates,
  nose) are still unlabelled, i.e. the same missing-instance noise as round 1.
* Same settings as num_n320 (YOLO11n, 320, batch 128, patience 30): early-stopped at 102 (best 73), ~1 h (35 s/epoch).
  Box mAP: **val 0.762 / 0.365 (r1 0.743 / 0.351), test 0.519 / 0.186 (r1 0.542 / 0.203)**: mAP against the
  grounded-only labels does not show it, but the chain does:
* Chain on val (panel_val.py, recognizer syn_ft, conf 0.1, pad 0.05, 400 tracks x 4 crops): **top-1 89.0% (r1 86.3%)**,
  crops without a box 8% (11%), tau 0.8 **52% / 97.6%** (r1 43% / 95.4%). Test (24SU): top-1 76.5% (72.8%), tau 0.8
  19% / 97.4% (18% / 94.4%). r2 is the panel detector from here on.
* **26SE with the WP10b recognizer `pq_spl_full`, boxed vote (tau 0.9 / k 3 from val), the teacher's passes**
  (`out/eval_recog_pq_spl_full_panels[_all]_boxed[_r2]/`):

| boxes | crops | no abstain: answered / precision / end-to-end | val rule: coverage / precision / end-to-end |
|---|---|---|---|
| num_n320 | 10 VLM crops | 97% / 73.2% / 66.3% | 55.3% / 93.8% / 48.4% |
| num_n320_r2 | 10 VLM crops | 99% / 75.5% / 69.8% | 59.1% / 95.7% / 52.8% |
| num_n320 | all (<= 30) | 100% / 78.2% / 72.6% | 54.5% / 99.2% / 50.4% |
| **num_n320_r2** | **all (<= 30)** | **100% / 82.5% / 76.6%** | **61.7% / 98.6% / 56.7%** |

  The proposal round gives +4 points end-to-end and +7 points of coverage at the val rule. (WP10b found the same
  proposals useless as *recognizer* training data: they help the detector, not the reader.)

### Fully student end-to-end on 26SE (`students/e2e_26se.py`): the headline

Chain: student car detector (5 fps timeline + ByteTrack at 30 fps on events, native resolution) -> passes (passes.py,
unchanged) -> <= 30 crops per pass (every track crop, evenly subsampled) -> num_n320_r2 (conf 0.1, +5%) -> pq_spl_full
-> `vote_boxed` (tau 0.9, k 3, from 25SU val) -> 26SE roster -> id-preferring match to the 252 human laps after
09:18:30. Outputs `out/eval_<run>_e2e/e2e_pq_spl_full/`. The native-resolution re-tracking reproduced the pass metrics
of the 1920-wide runs exactly.

| pipeline | passes | pass recall / precision | identity, no abstain: answered / precision | end-to-end | identity, val rule: answered / precision [95% CI] | end-to-end (val rule) |
|---|---|---|---|---|---|---|
| teacher: YOLO-World v2-x + Qwen3-VL-4B (10 crops) | 241 | 0.933 / 0.975 | 77% / 94.0% (it abstains by itself) | **67.9%** | (share >= 0.5: 65% / 98.7%) | - |
| **students: cars_s640 + num_n320_r2 + pq_spl_full** | 242 | **0.929 / 0.967** | 99% / 82.7% | **75.8%** | **62% / 100% [97.4-100]** (144 / 144) | **57.1%** |
| students: cars_n640_v2 + same | 250 | 0.925 / 0.932 | 99% / 82.7% | 75.8% | 60% / 100% [97.3-100] (139 / 139) | 55.2% |
| students: cars_n640 + same | 246 | 0.913 / 0.935 | 99% / 81.6% | 73.8% | 61% / 99.3% [96.1-99.9] (140 / 141) | 55.6% |

* **The all-student chain finds as many laps as the teacher and names more of them correctly** (75.8% of laps vs 67.9%),
  because it reads ~30 crops per pass instead of 10 and answers on mud cars the VLM declined. At the val-picked rule it is
  an auto-confirm signal: **every one of 144 (s640) / 139 (n640_v2) answers was right**, on 60-62% of the laps.
* Per car (s640, no abstain): 4, 5, 8, 9, 11, 14, 16, 22 never wrong. **Car 1 (mud) 19/32 right, 12 wrong (7 read as
  "11")** and never answered under the val rule; the VLM read car 1 on 4 of 32 passes. Other errors: car 6 (3 -> "5",
  3 -> "1"), car 10 (-> 11 / 12), car 20 (-> 11 / 1), car 3 (-> 1). Most errors involve a "1": the easiest digit to see in
  a vertical bar, as WP10 warned.
* The 10 fps tracking above would raise pass precision further for the n models; not combined with identity here.

### CPU cost of the deployed chain on this 3900X (loaded CPU, measured parts)

| part | unit | 1 thread | 4 threads |
|---|---|---|---|
| car detector YOLO11n (ONNX, 384x640 + letterbox + NMS) | per frame | 45 ms | 21 ms |
| car detector YOLO11s | per frame | 126 ms | 45 ms |
| panel detector num_n320 (ONNX 320) | per car crop | 30 ms | 19 ms |
| recognizer pq_spl_full (WP10b) | per number box | 21 ms | 12 ms (7 ms batched) |

* Continuous detection + tracking at 10 fps: YOLO11n 0.45 s of CPU per second on one thread (45% of a core) or ~0.2 s/s
  on 4 threads; YOLO11s needs 4 threads (0.45 s/s), it is not real-time on one thread (1.26 s/s). At 30 fps only
  YOLO11n on 4 threads fits (0.63 s/s).
* Identity per pass at 30 crops and ~0.8 boxes per crop (24 boxes): panel 0.9 s + recognizer 0.5 s = **~1.4 s on one
  thread, ~0.75 s on 4 threads** (0.47 / 0.25 s at 10 crops). Passes come every ~40 s on average (252 laps in 2.85 h),
  so identity is ~3% of a core; it can run in a worker after the car leaves the frame. In the e2e run (6 threads,
  ultralytics path): 20 ms panel + 11 ms recognizer per crop.
* So a laptop with 4 cores runs YOLO11n at 10-15 fps + the identity chain comfortably; YOLO11s at 10 fps is possible on
  4 threads with little headroom. RAM: ~150 MB (car) + ~110 MB (panel) + ~125 MB (recognizer) peak per process.

### Comparison table (final)

| detector | input | 24SU test mAP50 / 50-95 | 26SE passes | recall | precision | false passes | end-to-end, borrowed VLM ids | end-to-end, all-student chain (no abstain / val rule) | CPU ms ORT 1 / 4 thr (384x640) |
|---|---|---|---|---|---|---|---|---|---|
| YOLO-World v2-x (teacher) | 1280 | reference | 241 | 0.933 | 0.975 | 6 | 67.9% | 76.6% / 56.7% (teacher passes) | GPU only |
| COCO YOLO11x | 1280 | - | 230 | 0.869 | 0.952 | 11 | 62.7% (own VLM) | - | - |
| COCO YOLO11n, untrained (11-09-03 only) | 1280 | - | 15 (24 laps) | 0.58 | 0.93 | 1 | - | - | 50 / 22 |
| student YOLO11n (`cars_n640`) | 640 | 0.924 / 0.742 | 246 | 0.913 | 0.935 | 16 | 65.1% | 73.8% / 55.6% | 45 / 21 |
| student YOLO11n + washout (`cars_n640_v2`) | 640 | 0.922 / 0.737 | 250 | 0.925 | 0.932 | 17 | 65.9% | 75.8% / 55.2% | 45 / 21 |
| ... same, tracking at 10 fps | 640 | | 240 | 0.925 | 0.971 | 7 | | | |
| **student YOLO11s (`cars_s640`)** | 640 | **0.935 / 0.772** | 242 | **0.929** | **0.967** | 8 | 65.9% | **75.8% / 57.1%** | 126 / 45 |

Recommendation: **laptop default = cars_n640_v2 (YOLO11n + washout) at 10 fps**: real-time on one thread, pass recall
0.925; use cars_s640 when 4 free cores are guaranteed (best precision at 30 fps). Number chain: num_n320_r2 +
pq_spl_full + boxed vote; auto-confirm at the val rule, everything else to the human reviewer.

## Session 2: real-time prototype (WP11)

Code: `rt/` (`run.py` main, `capture.py`, `models.py` ONNX Runtime / OpenVINO wrappers + numpy CTC roster scorer,
`tracker.py` numpy ByteTrack, `passlogic.py` online pass logic + crop candidates, `identity.py` async worker, `sinks.py`
JSONL / CSV / LapAppSink stub, `timesync.py`, `stats.py`, `config.toml`; tools `eval_26se.py`, `loadtest.py`,
`bench_laptop.py`; `rt/README.md` = install, config reference, event schema). Runtime deps: onnxruntime, numpy,
opencv-python, ffmpeg (+ psutil for stats / affinity, openvino optional). No torch / ultralytics at run time. All runs
CPU only (`CUDA_VISIBLE_DEVICES=-1`) while another agent trained on the GPU and its data loaders kept ~8 of the 24
logical CPUs busy. Outputs: `out/rt/`.

### Equivalence with the offline pipeline (checked part by part)

* **Car detector**: the ONNX export (384x640) + `rt.models.letterbox` / NMS gives the same boxes as ultralytics' `.pt`
  predict (1,937 detections on 1,800 frames, max coordinate difference 0.001 px).
* **Tracker**: `rt/tracker.py` is a line-by-line port of ultralytics 8.4's BYTETracker (Kalman XYAH, two-stage
  association with fuse_score, unconfirmed tracks reported from their second detection, buffer 30 *frames*,
  end-of-frame duplicate removal, and the quirk that a just-removed lost track stays matchable for one more frame).
  Same detections in -> **0 id mismatches in 1,800 frames, Kalman boxes within 1e-4 px**. lap.lapjv(cost_limit) is
  replaced by an exact Hungarian solve on the augmented matrix.
* **Panel detector**: needs a dynamic-shape export (`students/runs/bench/num_n320_r2_best/320dyn/best.onnx`, made here)
  to use ultralytics' minimum-rectangle letterbox: boxes within 0.5 px of the `.pt`. The fixed 320x320 export
  (`320/best.onnx`, also `deploy/models/panel.onnx`) is not equivalent: same box count on 94% of crops, same argmax
  on 95% of boxed crops (600 crops). Small, but it is model noise; export the panel net with `dynamic=True`.
* **Roster scorer**: numpy CTC forward vs torch `ctc_loss`: max posterior difference 2e-7.
* **Pass logic** (`python rt/eval_26se.py passlogic out/eval_cars_n640_v2_t10`): the offline 10 fps tracks replayed
  frame by frame through the online logic give **257 / 257 passes with identical t0 and 256 with identical t1** vs
  `passes.build`. The streaming rules take the attach decision when a track becomes foreground (t0 and first box are
  final by then). Nested duplicates are tested against coexisting members of open passes. A pass closes when all
  members left the tracker, 10 s passed since its last box, and no younger, not-yet-foreground track started inside
  the window (at most `pending_max_s` = 10 s).

### Whole-race replay (cars_n640_v2, 10 fps, native-resolution crops, 4 files in parallel, `out/rt/replay_n/`)

| | offline e2e (30 fps tracking, all crops <= 30) | offline, 10 fps tracking | **online rt (10 fps)** |
|---|---|---|---|
| passes / matched / laps | 250 / 233 / 252 | 240 / 233 | **240 / 233** |
| pass recall / precision | 0.925 / 0.932 | 0.925 / 0.971 | **0.925 / 0.971** |
| identity no abstain: answered / precision / end-to-end | 99% / 82.7% / 75.8% | - | **98% / 83.4% / 75.8%** |
| val rule (tau 0.9, k 3): answered / precision / end-to-end | 60% / 139/139 / 55.2% | - | **58% / 134/135 = 99.3% [95.9-99.9] / 53.2%** |
| crops read per pass | 29.2 | - | 22.8 |

* **Passes: 238 / 240 identical to the offline 10 fps run** (t0 within 1 s; 13 differ by > 0.05 s). Continuous tracking
  vs offline event windows: online finds car 7 at 09:55:43 and car 10 at 10:24:27; offline finds car 1 at 10:24:25 and
  car 3 at 10:36:23. Same recall. Per file 75/75, 74/90, 21/24, 63/63 laps.
* **Identity**: identical end-to-end without abstention (191 laps). The val rule answers slightly fewer passes. 10 fps
  gives fewer frames per pass, and pick_crops (large + sharp, area > 0.4 x max) leaves 22.8 crops instead of ~29.
  **One wrong auto-confirm**: car 6 at 10:55:28 (a 12 s pass in a pack, between car 15 and car 4) read "5" with 23
  supporting crops, share 0.99. Offline read the same "5" with share 0.84, under tau. This is the known
  stylised-6 error (WP10b: 6 -> "5"). Large, sharp crops agree with each other more than evenly spaced ones, so the
  online crop choice makes a wrong read more confident. Treat auto-confirm as ~99%, not 100%. Never tuned on 26SE.
* Same-number agreement with the offline e2e on the 238 common passes: 208. Auto set: 129 in both, 7 online only,
  12 offline only.
* Provisional reads (PTZ entry signal; same rule as auto-confirm, on <= 16 early crops): 57% of matched passes,
  98.5% precise (wrong: car 1 read "11", and the car-6 pass above), always equal to the final number.
* Timing: `pass_start` 0.5 s after the pass's first frame (6 frames). `pass_end` 10.1 s after the last box (p95 12.9 s;
  this is the re-attach window by design). Identity after pass_end: p50 0.9 s, p95 2.9 s on the saturated replay CPU.
* RAM: peak 285-290 MB per process at 1080p, 377 MB with native 4K frames. Crop buffers (JPEG q92, 48 candidates
  per track, thinned in time) peak at 7-13 MB (34 MB at 4K). The identity queue never exceeded 3 jobs.
* The 4 files in parallel saturated the CPU (system 90-95%): 1080p files ran at 1.0-1.3x real time, the 4K file at
  0.5x (native 4K HEVC decode).

### Real-time load (`rt/loadtest.py`, 11-29-49.mkv, `--realtime`, 10-min segments, high priority, decoder unpinned)

quiet = 11:35-11:45 (14 laps), busy = 11:59-12:09 (19 laps incl. the 12:04 finish, cars stopped among people).
Threads = detector + identity intra-op threads; cap2 / cap3 = process pinned to 2 / 3 physical cores (CPUs 2,4[,6]).
CPU % = % of one logical CPU (100 = one core), mean / p95 over 1 s samples. `out/rt/load/`, `out/rt/load_s/`.

| model | config | segment | dropped | det ms p50 / p90 | lag p95 / max ms | pipeline CPU % | ffmpeg CPU % | RAM peak | id latency p50 / max |
|---|---|---|---|---|---|---|---|---|---|
| n640_v2 | 1+1 | quiet | 0 | 59 / 69 | 74 / 348 | 71 / 88 | 112 | 252 MB | 0.64 / 4.6 s |
| n640_v2 | 4+4 | quiet | 0 | 43 / 52 | 56 / 196 | 126 / 165 | 123 | 257 MB | 0.54 / 4.2 s |
| n640_v2 | cap2 (1+1) | quiet | 0 | 55 / 62 | 66 / 89 | 67 / 81 | 91 | 250 MB | 0.57 / 4.0 s |
| n640_v2 | cap3 (2+1) | quiet | 0 | 43 / 53 | 58 / 79 | 81 / 97 | 100 | 252 MB | 0.58 / 4.1 s |
| n640_v2 | 1+1 | busy | 0 | 60 / 70 | 77 / 350 | 76 / 103 | 115 | 256 MB | 1.01 / 1.9 s |
| n640_v2 | 4+4 | busy | 0 | 43 / 52 | 59 / 225 | 131 / 207 | 124 | 263 MB | 1.10 / 1.6 s |
| n640_v2 | **cap2 (1+1)** | busy | **0** | 54 / 60 | 67 / 108 | **67 / 93** | 93 | 258 MB | 1.06 / 1.7 s |
| n640_v2 | cap3 (2+1) | busy | 0 | 43 / 53 | 60 / 79 | 86 / 119 | 100 | 259 MB | 1.07 / 1.6 s |
| s640 | 1+1 | busy | **2,241 (37%)** | 159 / 177 | 851 / 993 | 112 / 134 | 114 | 356 MB | 0.63 / 2.3 s |
| s640 | 4+4 | busy | 13 | 90 / 106 | 371 / 599 | 286 / 410 | 139 | 344 MB | 1.43 / 3.0 s |
| s640 | cap3 (2+1) | busy | 47 | 93 / 113 | 479 / 629 | 179 / 206 | 104 | 353 MB | 1.09 / 2.1 s |

* **cars_n640_v2 keeps up in every config, including 2 pinned cores in the busy finish stretch**: 0 dropped frames,
  bounded queues (identity queue <= 2), no growth in RAM. On those segments: laps 19/19 (busy) and 14/14 (quiet),
  auto-confirm 14/14 and 12/12 correct (`rt/eval_26se.py metrics --window`). In real time the provisional number
  arrives 0.6-0.75 s (p50) / 1.2-1.5 s (p90) after t_enter, always before the car leaves the frame.
* **cars_s640 is out of budget**: on 1 thread it drops 37% of frames (159 ms per frame). On 4 threads it still drops
  frames, uses ~2.9 cores and lags up to 0.6 s. With drops it produced 1 wrong auto-confirm in the busy segment
  (13/14). Pass recall survived (19/19) even at 37% drops.
* "det ms" in-situ is ~1.3-2x the isolated bench (WP9: 45 / 21 ms), because the 1080p letterbox counts in it and
  ffmpeg + the other agent share the CPU and memory bandwidth. The 26SE files are HEVC 1080p60: their software
  decode alone costs ~1 core (here) and is not representative of a camera (below).

### Component costs on this 3900X and the worst-case laptop (`rt/bench_laptop.py`, `out/rt/bench_laptop_3900x.json`)

Real 1080p frame. Wall ms p50 (CPU ms = all threads of the process, i.e. what the machine pays). Background load ~35%.

| component | 1 thread | 2 threads | 4 threads | laptop worst case (x2): wall at 1 / 2 thr | laptop CPU % of one core at 10 fps |
|---|---|---|---|---|---|
| car detector n640_v2, ORT (letterbox + 384x640 + NMS), per frame | 55 (55) | 40 (68) | 37 (91) | 110 / 80 ms | 110 (1 thr) / 135 (2 thr) |
| same, OpenVINO CPU | 58 (57) | 38 (70) | 28 (97) | 115 / 76 ms | ~ the same |
| car detector s640, ORT | 148 (148) | 97 (176) | 79 (233) | 300 / 195 ms | 300-350: impossible |
| panel detector per car crop | 19 (19) | 15 (24) | 14 (31) | 38 / 30 ms | - |
| recognizer per number box | 24 (24) | 22 (36) | 19 (44) | 48 / 44 ms | - |
| identity per pass (30 crops, 0.8 boxes per crop, 1 thread) | 1.15 s | | | 2.3 s | 38 (busy: 1 pass / 6 s), 6 (race average: 1 pass / 40 s) |
| tracker + pass logic + crop JPEG / sharpness | | | | | ~15 (in-situ total minus det and identity, x2) |
| decode 1080p30 H.264 8 Mb/s, software, 10 fps out | 53-60% of a core | | | | ~100-120 |
| decode 1080p30 MJPEG (USB webcam), software | 61% | | | | ~120 |
| decode 26SE HEVC 1080p60 (the replay files) | 159% | | | | out of budget: use another format |

* Extrapolation: an i5-8250U-class CPU is ~1.3-1.5x slower per core than this 3900X at turbo (single-thread
  benchmarks: approximate, from memory, not measured). Under sustained all-core load a U-series chip falls toward
  its 1.6 GHz base clock, so x2 is the planning number. The only real answer is `rt/bench_laptop.py` on the laptop.
  It has a `--sustain` check for throttling (here: stable, p90 70 -> 61 ms over 60 s).
* **Worst-case laptop budget, cars_n640_v2 at 10 fps (2 detector threads + 1 identity thread)**: detector ~135% +
  identity ~38% (busy) + tracking ~15% ~= 1.9 cores = 47% of 4 cores *before decoding*. Software decode of a 1080p30
  camera adds ~1 core (-> ~72%), which leaves too little for the OS + OBS. On one detector thread the detector alone
  would need ~110 ms per 100 ms frame (drops); hence the default is now 2 + 1 threads.
* Levers, in order: (1) hardware decode: `capture.hwaccel = "d3d11va"` / `"qsv"` (Intel QuickSync on the UHD 620;
  untested here), or set the camera itself to 10-15 fps / 720p (software decode cost scales with the input fps);
  (2) the car detector on the iGPU: `runtime.car_backend = "openvino"`, `car_device = "GPU"` (static 384x640; falls
  back to ORT CPU with a warning). Untested here (no Intel GPU). It would free ~1.3 cores if UHD 620 runs YOLO11n in
  <= 30-40 ms; (3) the identity worker on 1 thread is cheap on average and only bursts at packs (it is async, so it
  delays numbers, never frames).
* RAM is not an issue: 250-260 MB (n) / 345-355 MB (s) per process, steady over 10 min. 8 GB is plenty.

### Gotchas

* `threading.Thread` subclasses must not have attributes named `start` or `_stop` (my first FileCapture had
  `self.start` = the seek offset and crashed `cap.start()`).
* `time.process_time()` on Windows ticks in 15.6 ms steps: measure CPU over a whole loop and divide, never per call.
* ONNX Runtime's idle intra-op threads spin by default. `session.intra_op.allow_spinning = 0` (`ort_spinning = false`)
  keeps them from burning a laptop core between 10 fps frames.
* This PC's clock is **not** synced (`w32tm /query /status`: "Local CMOS Clock", leap indicator 3). `rt/timesync.py`
  warns about exactly this at the start of every live run.
* ffmpeg is on the user PATH via WinGet (`%LOCALAPPDATA%\Microsoft\WinGet\Links`). Git Bash shells may lack it, so
  prepend that path.

### Open questions for the user

1. The laptop: run `python rt/bench_laptop.py --video <a camera recording> --sustain 120` (and `--hwaccel d3d11va`)
   and send back `bench_laptop.json`. Does the iGPU (OpenVINO GPU) work there?
2. Lap app: API or direct `contagem_passagem` insert? Unread passes (`equipe_numero` is NOT NULL)? Should
   `auto_confirm` create a `contagem_decisao` 'confirmar', or only pre-fill the reviewer's screen? A new
   `contagem_fonte` row for the camera? Server-clock anchoring (the `ancora_*` columns).
3. PTZ: which transport (UDP / websocket / HTTP) and which payload? Today: `pass_start` (entry box) + `pass_provisional`
   (number) events.
4. Camera / capture: USB (MJPEG?) or IP camera (RTSP H.264?), resolution and fps. Can it stream at 10-15 fps? It sets
   the decode cost, which is the largest single item on the laptop.

## Session 2: final deployable models on all 5 events + deploy/models (WP9)

`students/build_all_events.py` builds `datasets/cars_all` (45,939 images) and `datasets/numbers_crop_r2_all` (25,068) as
hard links of train + val + test (the sources are untouched; ultralytics' label caches land in the new folders);
`images/val` = 25SU again, for curves only (it is in the training data). `students/queue_final.sh` ran everything.

### Recipe: identical hyperparameters, fixed epochs, no early stopping

* Epochs = the best epoch of the held-out run, i.e. **the same number of passes over the (larger) data**: cars_n640_v2_all
  66 epochs (best of cars_n640_v2), cars_s640_all 55 (cars_s640), num_n320_r2_all 73 (num_n320_r2); `--patience 0`, and
  **`last.pt` is the model** (best.pt is picked on 25SU, which is in training). Unavoidable differences from the held-out
  runs: the linear LR decay now ends at the last epoch (the held-out runs stopped mid-schedule at ~35-45% of lr0), and
  ultralytics' `close_mosaic = 10` now applies to the last 10 epochs (the held-out runs never reached it).
* Recognizer `pq_spl_all`: `train_parseq.py --all-splits` (new option: train + val + test caches, 5,540 'ok' crops vs
  3,502) with the pq_spl_full recipe; PARSeq epochs are fixed sample counts (6,000), so "same passes over the data" =
  5 x 1.58 -> **8 epochs** (51 min on CPU, 4 threads); last.pt. The splice bank stays train-only (its glyphs come from
  the alignment of the train crops). tau / k copied from pq_spl_full (no val left): `pq_spl_all/eval/panel_tau.json`.
* GPU time: num_n320_r2_all 1.0 h, cars_n640_v2_all 4.5 h (~245 s/epoch), cars_s640_all 5.6 h (~360 s/epoch).
  Final curves on 25SU (not held out): n 0.950 / 0.811, s 0.964 / 0.848, panel 0.909 / 0.545.

### 26SE regression check (evaluation only; full eval with the new car models + e2e_26se.py)

Pass metrics come from each car model's own 5 fps timeline + 30 fps tracking. Identity: <= 30 crops per pass, panel boxes
conf 0.1 + 5%, boxed vote tau 0.9 / k 3. "rule" = answered share of matched passes / precision / end-to-end of 252 laps.

| car passes | panel | recognizer | pass recall / precision | no abstain: precision / end-to-end | val rule: answered / precision / end-to-end |
|---|---|---|---|---|---|
| cars_n640_v2 (current) | r2 | pq_spl_full | 0.925 / 0.932 | 82.7% / 75.8% | 60% / 100% / 55.2% |
| cars_s640 (current) | r2 | pq_spl_full | 0.929 / 0.967 | 82.7% / 75.8% | 62% / 100% / 57.1% |
| **cars_n640_v2_all** | r2 | pq_spl_full | **0.925 / 0.967** | 82.3% / 75.8% | 60% / 100% / 55.6% |
| **cars_n640_v2_all** | **r2** | **pq_spl_all** | 0.925 / 0.967 | **84.4% / 77.4%** | **61% / 100% / 56.4%** (142 / 142) |
| cars_n640_v2_all | r2_all | pq_spl_all | 0.925 / 0.967 | 85.0% / 76.6% | 64% / 99.3% / 59.1% (149 / 150) |
| **cars_s640_all** | r2 | pq_spl_full | 0.929 / 0.967 | 85.8% / 79.0% | 60% / 99.3% / 55.6% |
| **cars_s640_all** | **r2** | **pq_spl_all** | 0.929 / 0.967 | **85.3% / 78.6%** | **60% / 99.3% / 55.2%** (139 / 140) |
| cars_s640_all | r2_all | pq_spl_all | 0.929 / 0.967 | 87.5% / 80.6% | 64% / 98.7% / 58.3% (147 / 149) |

On the teacher's passes (harness, all crops): r2 + pq_spl_full top-1 82.1%, rule 62% / 98.6%; r2 + pq_spl_all 83.4%,
64% / 98.7%; r2_all + pq_spl_full 78.3%, 61% / 98.6%; r2_all + pq_spl_all 83.4%, 65% / 98.7%. With 10 crops: 74.9 /
78.7 / 74.9 / 77.0% top-1.

* **cars_n640_v2_all: better.** Same recall, **pass precision 0.932 -> 0.967** (8 false passes instead of 17; the same
  stationary / end-of-race kinds remain). **cars_s640_all: same pass metrics** (identical recall / precision), slightly
  better identity downstream. Both adopted.
* **pq_spl_all: at least as good** (+1.3 points top-1 on the teacher's passes, +1.6 end-to-end on the n passes, -0.4 on
  the s passes = 1 lap, precision unchanged). Adopted.
* **num_n320_r2_all: not adopted.** It is not clearly worse, but it trades precision for coverage at the auto-confirm rule:
  +7 / +8 correct auto-confirms and **+1 / +1 wrong ones** (99.3% / 98.7% instead of 100% / 99.3%), and it finds fewer
  boxes on the 26SE crops (5,803 vs 6,651 on the harness crops). For a human-assist system a wrong auto-confirm costs more
  than a missed one, so the deployed panel stays num_n320_r2. The extra errors (car 13 read "16" at 10:41:21, car 1 read
  "13" at 10:24:26) are in the washed-out stretch where all chains are weakest, and the differences are within the
  Wilson intervals (~±3 points at n ~ 145): this is a judgement call, not a significant result.
* So the deployed set is **car_n = cars_n640_v2_all, car_s = cars_s640_all, panel = num_n320_r2, recog = pq_spl_all**.
  With car_n: pass recall 0.925 / precision 0.967, 77.4% of laps named correctly without abstaining, auto-confirm on 61%
  of passes at 100% (teacher: 0.933 / 0.975 / 67.9%).

### deploy/models (ONNX set for WP11's rt/; `students/make_manifest.py` writes manifest.json)

* `car_n.onnx`, `car_s.onnx`: fixed 1x3x384x640 (`dynamic=False`, simplify, no NMS). `panel.onnx`: **dynamic shapes**
  (`dynamic=True`, imgsz 320). `recog.onnx`: students/recog/export_onnx.py (opset 17, dynamic N and W). The `.onnx`
  files are gitignored; `manifest.json` (committed) has source runs, sha256 prefixes, export flags, pre/post-processing,
  and every pipeline threshold (detector conf 0.15, ByteTrack defaults, 10 fps tracking, passes.py rules incl.
  MIN_FRAMES 6 in frames, crops <= 30 with 5% pad, panel conf 0.1 / pad 0.05, recognizer +10%, vote share tau 0.9 / k 3
  / p 0.5, roster snapping).
* **Fixed-size export bug found by WP11:** a 320x320 panel export padded every crop to a square, while the .pt (ultralytics
  rect mode) pads only to the next multiple of 32, so the box count matched on only 89-94% of crops
  (`students/verify_onnx.py`: 89% same count, max difference up to 72 px). The dynamic export matches on 100% (max
  0.0001 px, 500 26SE crops). The car exports at 384x640 match exactly anyway (99 26SE frames: 100% same count, max 0.0004 px)
  because a 16:9 frame letterboxes to 384x640 in both cases.
* The deploy files themselves were validated end to end: `e2e_26se.py --panel-weights deploy/models/panel.onnx --recog
  pq_spl_all` on the cars_*_all passes reproduces the .pt numbers exactly (n: 77.4% / 61% at 100%; s: 78.6% / 60% at
  99.3%). `e2e_26se.py` now accepts an .onnx panel detector.

## Session 3: field-ready live capture for rt/ (WP-L1, 2026-11-12)

Code: `rt/camsource.py` (camera interface, OpenCV/DirectShow + fake), `rt/livecam.py` (grab/retrieve decimation,
two-clock timestamps, reconnect), `rt/recorder.py`, `rt/status.py`, `rt/supervise.py` + `rt/run_lapline.bat`,
`rt/configs/lapline.example.toml`, tests in `rt/tests/` (pytest; fake camera, fake clock, fake process). Pass logic,
tracker, models and the file capture are unchanged, so the 26SE replay numbers above still hold.

* **Recording disk use, measured on 26SE 1080p frames (grass and trees, 10 frames of 11-29-49.mkv)**: one JPEG is
  611 KB at quality 80 and 822 KB at quality 90. MJPEG copy = 66-89 GB/h at 30 fps, 22-30 GB/h at 10 fps: too much
  for a race day on a laptop disk. H.264 x264 veryfast CRF 23, GOP 2 s, 10 fps: 70 MB per minute = **~4.2 GB/h**
  (1 min of the same file). Hence the default `recording.mode = "h264"` (QuickSync / AMF if they work, else x264),
  only the processed 10 fps; `mjpeg` copy stays an option. A webcam's own MJPEG quality may be lower than q80:
  check the real file size on the laptop.
* **Frame timestamps moved to grab()** for the USB camera (before: `time.time()` after `read()`, i.e. after the
  decode). Every frame also carries `time.monotonic()` of the same grab (`Frame.t_cap`), and a `clock_map` event
  once a minute relates the two clocks (for the beam input, WP-L2).
* **Decimation tolerance must depend on the camera rate**: a fixed quarter of the 100 ms period (25 ms) picked frame
  5 instead of 6 on a 60 fps camera (frame interval 16.7 ms). The tolerance is now 40% of the camera's frame interval,
  capped at 25 ms.
* `queue.Queue.get()` without a timeout blocks Ctrl+C on Windows; `Capture.frames()` now polls with a 0.5 s timeout.
* Not verified on hardware yet (no webcam on this PC): DirectShow raw MJPEG passthrough (`CAP_PROP_CONVERT_RGB = 0`;
  rt checks the first frame and falls back to OpenCV decoding), the DirectShow `auto_exposure` value that means
  "manual" (driver-dependent; the read-back on the status page tells), w32tm "Phase Offset" parsing on a non-English
  Windows (shows "unknown" then; set `clock.ntp_server` for a real SNTP measurement), QSV/AMF on the T14.

### Review fixes on PR #1 (2026-11-12): what OpenCV's DirectShow backend really does

The reviewer read OpenCV 4.x `modules/videoio/src/cap_dshow.cpp`. **Corrections of the bullets above:**

* **Correction: "grab() every frame, retrieve() only the processed ones" saves nothing on DirectShow and busy-loops.**
  DirectShow's `grabFrame()` returns True at once (it only checks that the device is not disconnected); the wait for
  a new frame (up to ~1 s) is inside `retrieve()`, which also decodes, and DirectShow ignores `CONVERT_RGB = 0` for
  MJPG (so `raw_mjpeg` never gave JPEG bytes there). Now `camera.read_mode = auto` picks per backend:
  `retrieve_every_frame` on DirectShow (one grab + one waiting retrieve per camera frame, stamped right after the
  retrieve, 2 of 3 dropped in software: the decode of all 30 fps is paid, ~1.2 cores on a throttled U-series per
  the WP11 table; to be measured on the T14), `grab_then_retrieve` only on V4L2 where grab really blocks.
  `FakeDshowCamera` in rt/camsource.py copies the DirectShow behaviour for the tests. Possible future sources (not
  implemented): MSMF if its grab blocks per frame, or ffmpeg `-f dshow -vcodec mjpeg` copying the camera JPEGs.
* **Correction: timestamps are taken after retrieve() on DirectShow, not at grab().** They include the MJPEG decode
  (~10-20 ms); fold it into `capture.live_latency_s` when the latency is measured (LED / beam, WP-L2).
* **An unplug on DirectShow is edge-triggered**: one grab() returns False, then grab() is True again and retrieve()
  times out every second. The old "30 failed grabs" rule never fired, so the camera was never reopened in-process.
  Loss is now "no good frame for `camera.lost_after_s` = 3 s (monotonic) or 10 failed reads in a row".
* **Correction: `recording.mode = "mjpeg"` (copy) is not possible with this capture** (frames arrive decoded) and is
  refused at start, as is `all_frames`; H.264 only (QSV / AMF probed at start and logged, else x264). The MJPEG
  disk numbers above (66-89 / 22-30 GB/h) stay as measurements but no longer apply to rt. The ~4.2 GB/h figure is
  x264 CRF 23; QSV / AMF sizes are not measured yet.
* Recording files now split at a camera loss or a gap of > 2 frame periods (the file rate is nominal, so a file
  spanning a gap replays shifted), the `.frames.csv` is authoritative and the replay (`FileCapture`) uses it.
  ffmpeg's stderr goes to `<segment>.ffmpeg.log` (an unread pipe can fill and freeze ffmpeg); a watchdog kills an
  ffmpeg that does not accept data for 10 s or exits, retries 10 s later, gives up after 5 failures in a row.
* Python's `HTTPServer` sets SO_REUSEADDR; **on Windows that lets a second program bind a port that is already
  listening**, so a busy status port would go unnoticed. The status server binds with SO_EXCLUSIVEADDRUSE on
  Windows, and with `--require-status` (the launcher passes it) rt exits with code 3 if the port is busy.
* **Correction: the w32tm parsing was English-only**, and `subprocess.run(text=True)` could fail to decode cp850
  bytes of a pt-BR console (reported as "w32tm not available"). It is now read by layout (field order and value
  shapes) from bytes decoded with the console code page. No real pt-BR `w32tm /query /status /verbose` capture was
  available: paste one from the T14 into rt/tests/test_timesync.py to replace the constructed sample.

### Flaky CI on PR #1 (2026-11-12): recorder tests that depended on the runner's disk, and two real recorder races

* **The CI failure was not a race: the tests read the runner's real free disk.** Same commit, two ubuntu-26.04 runs:
  one passed (Azure runner), one failed 4 tests with `rec.segments == []` (an AWS us-east-2 runner, different image
  build). The 4 failing tests were exactly the 4 that build a `Recorder` without `disk_free_gb`, so the disk guard read
  the real disk; under `stop_free_gb` (5 GB) it records nothing, by design and silently in the tests
  (`log=lambda m: None`). Reproduced 100% by giving pytest a 4 GB tmpfs as `--basetemp` (600/600 + 20/20 failed, same
  errors, same ~10 ms run time as in CI). Fix: tests inject `plenty_of_free_disk` (rt/tests/conftest.py).
* **Real race found while stress-testing: a camera loss split the recording at the wrong frame.** `camera_lost()` set
  a flag that the writer acted on at the next frame it wrote, so frames from before the loss still in the queue went
  into the next file. The e2e clip test failed with 2 segments in 2 of ~160 local runs (rare: it needs the writer to be
  behind exactly at the loss). In the field it misplaces the split whenever the writer is behind (CPU load, slow
  encoder). Now each queued frame carries its offer number and the request remembers how many frames were offered
  before the loss; the file closes just before the first later frame (or, when the queue is idle, once the writer has
  taken the last frame from before the loss). Two unit tests replay both orderings by hand and fail every time on the
  old code.
* **Second real race: segment names in the same second.** The "name taken?" check looked only for the `.mkv`, which
  ffmpeg creates when its process gets going, possibly after the next segment is opened; a second segment within the
  same second could get the same name and overwrite the first (ffmpeg `-y`, sidecars). It now also checks the `.json`
  sidecar, which is written before `open_segment()` returns.

## Session 3: beam input for rt/ (WP-L2, AUTOPTZ-23, 2026-11-12)

Code: `rt/beam/` (`sources.py` drivers modem-line / byte-stream / fake, `grouping.py` edges -> breaks + stuck beam,
`associate.py` live pass rule + LED watcher, `service.py` reader + dispatcher threads), `rt/tools/beam_probe.py`,
`rt/tools/fuse_beam.py`, `[beam]` in `rt/config.toml` and `rt/configs/lapline.example.toml` (off by default), a Beam
section on the status page, tests `rt/tests/test_beam_*.py` + `test_fuse_beam.py`. Pass logic, tracker and models are
unchanged (beam results are separate `pass_beam` events), so the 26SE replay numbers above still hold. Written before
the adapter arrived: how the pulse reaches the PC (a modem line or bytes) is still unknown, hence two drivers and the
probe.

* **Clocks**: beam edges are stamped with the same `time.time()` / `time.monotonic()` pair as frames. On DirectShow
  frames are stamped after `retrieve()` (decode included, see the PR #1 review fixes above), so frame times run late
  by the decode time; `assoc_offset_s` and the LED latency absorb it. `t_beam` gets `clock.offset_s` like frames but NOT `capture.live_latency_s` (a camera property), so `t_cross -
  t_beam` = camera latency + where the camera's centre line sits relative to the beam + the pass logic's definition
  of t_cross. fuse_beam.py estimates that offset from clean single matches; `assoc_offset_s` takes it for live use.
* **Isolation**: the reader thread does nothing but read and stamp (it never takes rt's emitter lock or writes a
  file); a dispatcher thread writes the events. rt's `Emitter` holds a non-reentrant lock while calling sinks, so the
  sink that feeds `pass_end` to the beam may only queue; emitting from inside a sink would deadlock.
* **Live association waits for its window**: a pass is decided only after `t_cross - offset + window + group_gap`
  (in practice at once, since `pass_end` comes ~10 s after the car), and at most `window + 5 s` after it arrived so a
  wall-clock jump cannot hold it forever.
* **LED latency is quantised by the 10 fps processing**: a value is grab time of the first processed frame showing the
  LED minus the edge, i.e. latency + 0-100 ms. Use the minimum of many `clean` values. Multi-pulse cars make the
  pairing ambiguous (`clean` = false).
* **Launcher fix (same PR)**: PR #1's launcher killed rt after 30 s of fps = 0 even when rt's own page said the camera
  was lost and rt was reopening it. Now `lost` / `retrying` / `opening` with a live status page is left to rt; an
  unreachable page, or fps = 0 while the camera says `streaming`, is still a stall.
* Checked here without hardware: byte-stream with real pyserial on a Linux pty (3 messages 50 ms apart -> 1 break
  with 3 pulses). Modem-line on a pty fails at the DTR ioctl ("Inappropriate ioctl") and correctly shows `retrying`;
  it needs a real adapter. Not verified: the CP2102's modem-line read rate on Windows (expected ~1 kHz with Python
  3.11+'s high-resolution sleep; the `beam_poll` event and the status page show the real rate).

### Review fixes on PR #5 + on-site beam setup (2026-11-12)

* **Correction: a modem-line unplug was not detected on Windows.** pyserial's `serialwin32` reads CTS/DSR/DCD/RI
  through `GetCommModemStatus` but ignores its return value, so an unplugged adapter just reads "not asserted" and
  nothing raises. The driver now checks `in_waiting` (ClearCommError, which does raise) every 100 ms and before
  reporting any change, plus GetCommModemStatus' own BOOL via ctypes on Windows. The test fake now behaves like
  real pyserial (lines read False, `in_waiting` raises).
* **Correction: break grouping could split one car in two** when the dispatcher advanced the grouper to "now"
  between the reader stamping an edge and queueing it. The grouper now only advances to min(now - 20 ms, the
  reader's published horizon), and a byte-stream message still being assembled holds the horizon back.
* **Correction: byte-stream edges were per read, not per message.** A message split over two USB reads gave two
  pulses (and the probe saw fragments, so its two-message alternation guess failed). Bytes less than 3 character
  times apart (from the baud: 3.1 ms at 9600) are now one message, stamped at its first byte, in the driver and
  the probe.
* **Correction: the probe took the polarity from the majority of time**; a person standing in the beam turned it
  around. It now takes the state at the start of the probe as "clear" and warns when the time share disagrees.
* **Launcher**: "camera lost, rt reopens it" is no longer unlimited: restart rt after 600 s of recovery, or after
  60 s without a new open attempt (`camera.open_attempts` on the status page). rt calls `timeBeginPeriod(1)` on
  Windows and logs `sys.version`.
* **Not done**: marking a pass_beam association as low-confidence when the vision t_cross is the pass logic's
  fallback (mean track time): `pass_end` carries nothing that tells the fallback apart, and the pass logic was not
  to be changed.
* **On-site beam setup**: the status page's `/beam` panel (built-in probe: live lines at ~5 Hz, byte messages,
  Suggest, Apply / Turn off) saves to `<config>.beam.toml` and restarts only the BeamService. Setup mode pauses the
  driver (one COM port owner) and ends by itself after 300 s without a page refresh. Checked here with a headless
  Chrome against fake serial ports; not yet with the real CP2102.
