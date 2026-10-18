# QA review tool (WP6)

Estimate teacher label precision per event and per shot type by reviewing ~150 random tracks per event
(target ~15 min per event, i.e. ~5 s per item). Stdlib Python server + one vanilla-JS page, no CDN.

## Run it (from the repo root)

```
.venv\Scripts\python.exe tools\qa\make_batch.py                      # session-1 test batch -> out\qa\26SE_session1.jsonl
.venv\Scripts\python.exe tools\qa\serve.py out\qa\26SE_session1.jsonl  # then open http://127.0.0.1:8765/
.venv\Scripts\python.exe tools\qa\report.py out\qa\26SE_session1.jsonl [--by event,shot_tag,meta.src] [--json r.json]
```

`serve.py <batch> [--port 8765] [--root .] [--reviewer NAME] [--host 0.0.0.0] [--token T]` binds to 127.0.0.1 by
default. To review from another device on the LAN (laptop, tablet), pass `--host 0.0.0.0`. It prints
`http://<lan-ip>:<port>/?t=<token>`, and only that URL works (it sets a cookie; requests without it get 403). On touch
screens, tap the key hints in the footer. Every keypress verdict is
appended (and fsynced) to `<batch>.verdicts.jsonl` immediately. Stop with Ctrl+C at any time; restarting the server and
reloading the page resumes at the first unreviewed item. Several batches can be reviewed in parallel on different ports.

Keys: `1` correct · `2` wrong number, then type the right one + Enter (empty = unknown, Esc = cancel) · `3` not a car /
bad box · `4` can't tell · `B` toggle "number boxes wrong" (sent with the next verdict) · `R` hide/show per-crop reads ·
`Backspace` undo the last verdict and go back to that item (repeatable) · click a crop to zoom.
For an item **without** a proposed number, `1` means "agreed, no readable number" and `2` means "readable: type it".

Reviewing rules (agreed 2026-10-18):
- **Two cars in one box** (the box spans both cars in most crops; check the context frame): `3`, even if one car's
  number is readable and matches. The box is a wrong detector label, and the track's identity is ambiguous. If the box sits on
  one car and a neighbour only peeks in at the edge or overlaps in a crop or two, judge that one car normally. The same goes
  for a readable car in the background: judge only the boxed car (press `B` only if a number box sits on the background car).
- **A track that jumps to another car in a crop or two** (tracker id switch; each box is still a car): judge the majority
  car, `1`/`2` as usual. Not `3`. If it is split so evenly that the track has no clear owner, use `4`.
- **`B` (number boxes wrong)**: only for a drawn box that would teach something wrong: a box on something that is not the
  race number (sponsor, overlay text, shirt, wheel), on a different number than the proposal, or one that cuts off digits.
  Do **not** press it for crops without a box (boxes come from the grounded read, run on only 2–3 crops of some tracks),
  for an unboxed second number on the car (known gap), or for a slightly loose or shifted box that still contains all the
  digits (the recognizer pads and jitters boxes).
- The small text in each crop's corner is the VLM teacher's raw read of that crop ("none" = it saw no legible number).
  Green means it equals the proposal, orange means it differs. The proposal is their majority, snapped to the roster
  (leading zeros ignored: "04" = 4).

The header shows progress, the running precision with its Wilson 95% interval, coverage, not-car / unsure counts and an
ETA from the median time per item.

## Item format (what the teacher, WP5, emits)

One batch file per review (one event, or one event × shot-tag slice), `.jsonl` (one JSON object per line) or `.parquet`
with the same columns. Item order in the file = review order, so **shuffle when writing**.

| field | type | notes |
|---|---|---|
| `item_id` | str | unique in the batch; stable (e.g. `<video_id>:<shot>:<track>`) so verdicts can be joined back |
| `event` | str | event code, e.g. `26BR`; the page loads `rosters/<event>.csv` (`numero,equipe`) for team names |
| `video_id` | str | |
| `shot_tag` | str | track / pit / interview / studio / graphics / crowd / onboard ...; report groups by it |
| `track_key` | str | key into `work/<video_id>/labels.parquet` |
| `proposed_number` | int or null | consensus number after roster snapping and the WP5 thresholds; null = abstained |
| `vote_share` | float | votes for the proposed number / crops read (0 when null) |
| `n_votes` | int | votes for the proposed number |
| `n_crops` | int, optional | crops read for the vote |
| `overlay_number` | int or null | identity from broadcast overlays (WP2b); shown, highlighted when it differs |
| `crops` | [str] | image paths **relative to the repo root** (`--root`), forward slashes, e.g. `work/<id>/crops/t12_f0450.jpg`; up to ~10, time-ordered |
| `crop_reads` | [str] or null, optional | per-crop raw read, drawn in the crop corner (green = equals proposal, orange = other) |
| `number_boxes` | [[[x1,y1,x2,y2], ...] or null per crop] or null | number-panel boxes **in crop pixels** (of the stored crop file), one list per crop, aligned with `crops` |
| `car_box_context` | str or null | path (same rule as crops) to a downscaled full frame with the car box drawn |
| `stratum` | str, optional | sampling stratum (e.g. shot tag) |
| `weight` | float, optional | stratum population / stratum sample size; report.py prints the weighted overall precision |
| `meta` | object | free-form; scalar values are shown small on the page. **`meta.hidden` is never sent to the browser** (put GT / answers there) |

Verdict records (`<batch>.verdicts.jsonl`, last record per `item_id` wins, `verdict: "undo"` clears the item):
`{item_id, verdict: correct|wrong|not_car|unsure|undo, number, boxes_wrong, dt_ms, reviewer, ts}`.

## Report definitions

- precision = correct / (correct + wrong), over items with a proposed number (unsure and not-a-car excluded)
- coverage = items with a proposed number / items judged to be cars (everything except not_car)
- not a car = not_car / reviewed; missed read = no-proposal items where the reviewer typed a number / no-proposal cars
- boxes wrong = items flagged `B` / reviewed items that have number boxes
- All with Wilson 95% intervals, for the whole batch and per `--by` key (item fields or `meta.<key>`).
- If items have `meta.hidden.gt_car` (session-1 batch), the reviewer's verdicts are also compared with the GT
  (confusion table of verdict vs GT, typed numbers vs GT).

`simulate_gt.py <batch> [--limit N] [--error-rate r]` answers a batch from `meta.hidden.gt_car` through the running
server's HTTP API (test harness for resume and for the report's GT comparison).

## Notes for WP5 (teacher)

- Sample per event, stratified by shot tag (`~150` items; a floor of a few items per tag so rare tags still show up),
  and write `stratum` + `weight` so the overall precision can be re-weighted.
- Include abstained tracks (`proposed_number: null`) in the sample, otherwise coverage and missed reads are unmeasured.
- Crops must be the actual files the VLM read (so `number_boxes` coordinates line up); keep them ≤ ~640 px, JPEG.
- Leave GT-like fields (e.g. the overlay identity when you want to test the reviewer blind) in `meta.hidden`.
