# zone/: spotter-driven PTZ tracking (AUTOPTZ-10)

So far this package holds WP-Z1 (AUTOPTZ-15): the PTZ driver, a simulated PTZ, and the bench script that
characterizes a real camera. It uses only the Python standard library; the tests need `pytest`.

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
python -m zone.ptz.fake_camera --port 52399 [--inquiry-mode at_rest] [--refresh-hz 2]
python -m zone.tools.characterize --ip 127.0.0.1 --port 52399 --out reports/fake --quick --yes
```

## How the bench measures

- **Feedback.** It pans at a fixed speed and polls position at 5/10/20/50 Hz. A rate is "ok" only when all of
  these hold: ≥ 90% of the inquiries are answered, the answers arrive at ≥ 80% of the asked rate, the value
  *changes* on ≥ 70% of the answers where a live report would (whole-unit rounding taken into account), the
  motion is smooth (interval-rate cv ≤ 0.35, or within 1 unit of a straight line), and there are no jumps. Per
  rate the summary prints answers/s, updates/s (how often the value really changed), cv, jumps and the reason it
  passed or failed. A camera that answers fast from a slowly refreshed position cache is flagged "STALE CACHE".
  While moving, the inquiry is classified as `yes`, `frozen` (answers, but the value doesn't change until the
  camera stops) or `no`.
- **Speed table.** With live feedback, it moves at each step and fits the slope of the position over the second
  half of the move. Without it (position readable only at rest), it makes two timed moves of T and 2T per direction:
  (travel(2T) − travel(T)) / T is the steady rate with the latency and ramp cancelled out.
- **Latency.** With live feedback at ≥ 10 Hz, it polls around the command: time to the first change, time to 90%
  of the steady rate, then the same for stop. A camera that answers only at rest goes silent when it starts moving
  and answers again once it has stopped; those two edges time the start latency and the stop. The resolution of the
  stop edge is the driver timeout (`--timeout`, 0.2 s by default).
- **Verdict.** Branch A if a poll rate ≥ 10 Hz is "ok" (above) and the position really updates ≥ 10 times a
  second at that rate (5% counting tolerance), otherwise Branch B.

## Tests

```
python -m pytest zone
```

Everything runs on a fake clock (no real sleeps) except a few UDP round trips on 127.0.0.1. Full bench runs execute
against the simulator in well under a second.
