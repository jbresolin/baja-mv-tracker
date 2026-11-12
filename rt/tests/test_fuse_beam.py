"""Tests of the offline fusion script (rt/tools/fuse_beam.py), on synthetic rt JSONL files.

After the event, this script lines up the beam breaks (precise times, no identity) with the vision passes (identity,
less precise times) and, optionally, with the human lap counts. What it must get right:
  * the camera-vs-beam offset: the camera's centre line is not exactly where the beam is, so every vision crossing
    is a roughly constant time after (or before) the beam. It is estimated from "clean" cases (one pass, one break,
    nobody else around);
  * one-to-one matching: each break goes to at most one pass, nearest first;
  * two cars side by side = ONE break for TWO passes ("shared_break");
  * a break with no pass = a marshal or a person crossing, or a car vision did not see ("break_no_pass");
  * a pass with no break = the beam was off / stuck, or a false vision pass ("pass_no_break").
The synthetic run below has all of these, at known times, so every number in the summary can be checked.
"""
import csv
import json

import pytest

from rt.tools import fuse_beam

T0 = 1_800_000_000.0
OFFSET = 0.40               # the synthetic camera sees each car cross 0.40 s after the beam


def write_jsonl(path, events):
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def synthetic_run():
    """Events of one rt run: 5 clean cars, 2 cars side by side, 1 marshal, 1 pass with no beam, 1 stuck episode."""
    ev = []
    pid = bid = 0

    def car(t_beam, number, jitter=0.0, with_pass=True, with_break=True):
        nonlocal pid, bid
        if with_break:
            ev.append(dict(type="beam_break", break_id=bid, t_beam=t_beam, n_edges=6, n_pulses=3, duration_s=0.4,
                           started_blocked=False))
            bid += 1
        if with_pass:
            t = t_beam + OFFSET + jitter
            ev.append(dict(type="pass_end", pass_id=pid, t_enter=t - 1, t_cross=t, t_exit=t + 1))
            ev.append(dict(type="pass_identity", pass_id=pid, number=number, auto_confirm=True, t_cross=t))
            pid += 1

    for k, (num, jit) in enumerate([(11, 0.02), (14, -0.03), (3, 0.01), (22, 0.0), (7, -0.01)]):
        car(T0 + 60 * k, num, jit)                                      # 5 clean cars, a minute apart
    car(T0 + 400, 5)                                                    # car 5 and car 9 side by side:
    car(T0 + 400, 9, jitter=0.15, with_break=False)                     # one break, two passes
    car(T0 + 500, None, with_pass=False)                                # a marshal walks through the beam
    car(T0 + 600, 18, with_break=False)                                 # vision pass, the beam saw nothing
    ev.append(dict(type="beam_stuck", state="stuck", t_since=T0 + 700))
    ev.append(dict(type="beam_stuck", state="cleared", t_since=T0 + 700, duration_s=30))
    ev.append(dict(type="camera_latency", latency_s=0.09, clean=True))
    ev.append(dict(type="camera_latency", latency_s=0.07, clean=True))
    ev.append(dict(type="camera_latency", latency_s=0.01, clean=False))   # not clean: ignored
    ev.append(dict(type="pass_beam", pass_id=0, status="matched"))
    return ev


@pytest.fixture
def run_file(tmp_path):
    p = tmp_path / "lapline_09-00-00.jsonl"
    write_jsonl(p, synthetic_run())
    return p


def test_offset_is_estimated_from_clean_matches(run_file, tmp_path):
    s = fuse_beam.run([run_file], tmp_path / "f.csv")
    o = s["offset_clean_matches"]
    assert o["n"] == 5                                   # the 5 clean cars (not the side-by-side pair)
    assert o["median"] == pytest.approx(OFFSET, abs=1e-6)     # median of 0.42, 0.37, 0.41, 0.40, 0.39
    assert o["p5"] == pytest.approx(0.37, abs=1e-6) and o["p95"] == pytest.approx(0.42, abs=1e-6)
    assert s["offset_used_s"] == pytest.approx(0.40)


def test_every_case_is_classified(run_file, tmp_path):
    s = fuse_beam.run([run_file], tmp_path / "f.csv")
    assert s["counts"] == dict(match=6, shared_break=1, break_no_pass=1, pass_no_break=1)
    assert s["passes"] == 8 and s["breaks"] == 7
    assert s["stuck_episodes"] == 1
    assert s["camera_latency_clean_s"]["n"] == 2 and s["camera_latency_clean_s"]["min"] == 0.07
    assert s["live_association"] == dict(matched=1)


def test_csv_rows_carry_numbers_and_times(run_file, tmp_path):
    out = tmp_path / "f.csv"
    fuse_beam.run([run_file], out)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    shared = by_kind["shared_break"][0]
    assert shared["number"] in ("5", "9")                # one of the pair got the break, the other shares it
    assert shared["break_id"] == "5"                     # the side-by-side break
    marshal = by_kind["break_no_pass"][0]
    assert marshal["pass_id"] == "" and marshal["break_id"] == "6"
    lonely = by_kind["pass_no_break"][0]
    assert lonely["number"] == "18"
    # A pass without a beam time gets the vision time minus the offset as its best crossing time.
    assert float(lonely["t_fused"]) == pytest.approx(T0 + 600, abs=0.01)
    assert all(r["t_fused_iso"] for r in rows)
    assert (tmp_path / "f.csv.summary.json").exists()


def test_given_offset_skips_the_estimate(run_file, tmp_path):
    s = fuse_beam.run([run_file], tmp_path / "f.csv", offset=0.0, window_s=0.3)
    # With offset 0 and a 0.3 s window the 0.4 s camera delay is outside the window: nothing matches.
    assert s["offset_used_s"] == 0.0
    assert s["counts"].get("match", 0) == 0


def test_several_files_of_one_day_are_read_together(tmp_path):
    # rt restarted once: pass_id and break_id start again at 0 in the second file; they must not be mixed up.
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, synthetic_run())
    write_jsonl(b, [dict(e, t_beam=e["t_beam"] + 3600) if "t_beam" in e else
                    dict(e, t_cross=e["t_cross"] + 3600) if "t_cross" in e else e for e in synthetic_run()])
    s = fuse_beam.run([a, b], tmp_path / "f.csv")
    assert s["passes"] == 16 and s["breaks"] == 14
    assert s["counts"]["match"] == 12


def test_a_truncated_last_line_is_skipped(tmp_path):
    # rt killed in the middle of writing a line: the half line must not stop the script.
    p = tmp_path / "cut.jsonl"
    write_jsonl(p, synthetic_run())
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"type": "beam_break", "break_id": 99, "t_be')
    assert fuse_beam.run([p], tmp_path / "f.csv")["breaks"] == 7


def test_comparison_with_human_counts_from_a_dump(run_file, tmp_path):
    # A tiny lap-app dump in the 26SE format (contagem_volta rows: prova, evento, car, lap, ts_ms, dispersion_ms,
    # n_sources, lap_ms, ?, status), on the SERVER clock, which here is 80 s behind this PC (video_minus_server).
    laps = [(11, 1, T0 + 0.8), (14, 1, T0 + 60.5), (3, 1, T0 + 120.6), (22, 1, T0 + 180.9), (7, 1, T0 + 240.7),
            (5, 1, T0 + 400.5), (9, 1, T0 + 401.0), (18, 1, T0 + 600.9),
            (30, 1, T0 + 520.0),            # a lap the humans counted that neither the beam nor vision has
            (40, 1, T0 + 5000.0)]           # far outside this run's time span: ignored
    values = ",".join("('END','26SE',%d,%d,%d,1500,2,NULL,NULL,'contada')" % (car, lap, round((t - 80) * 1000))
                      for car, lap, t in laps)
    dump = tmp_path / "dump.sql"
    dump.write_text("INSERT INTO `contagem_volta` VALUES %s;\n" % values, encoding="utf-8")
    s = fuse_beam.run([run_file], tmp_path / "f.csv", dump=dump, video_minus_server=80.0)
    h = s["human"]
    assert h["laps"] == 9                                # car 40 is outside the run: not counted
    assert h["matched"] == 8                             # every car crossing found its human lap ...
    assert s["counts"]["human_only"] == 1                # ... car 30 has no crossing (the marshal break is 20 s away)
    assert h["with_beam"] == 7                           # 5 clean + the side-by-side pair; car 18 had no beam
    assert h["number_agrees"] == 8 and h["number_differs"] == 0
    # Humans click after the beam: fused - human is negative. Values: -0.8 -0.5 -0.6 -0.9 -0.7 (clean cars),
    # -0.5 / -1.0 (the pair), -0.9 (car 18: vision time minus the 0.40 s offset). Median of the 8 = -0.75.
    assert h["fused_minus_human_s"]["median"] == pytest.approx(-0.75, abs=0.001)


def test_command_line_prints_a_summary(run_file, tmp_path, capsys):
    fuse_beam.main([str(run_file), "--out", str(tmp_path / "x.csv")])
    out = capsys.readouterr().out
    assert "match 6, shared_break (side by side) 1, pass_no_break 1, break_no_pass 1" in out
    assert "5 clean matches" in out


def test_non_numeric_numbers_never_crash_the_comparison():
    # The vision number can be missing or odd ("?", "12a"); the human count could be too. The comparison must then
    # say "unknown" (None), not crash the whole offline report.
    assert fuse_beam.numbers_agree("12", 12) is True
    assert fuse_beam.numbers_agree(12, "13") is False
    assert fuse_beam.numbers_agree("?", 12) is None
    assert fuse_beam.numbers_agree("12a", 12) is None
    assert fuse_beam.numbers_agree(None, 12) is None
    assert fuse_beam.numbers_agree(12, "") is None
