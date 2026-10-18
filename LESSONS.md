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
