"""Tests of the online pass logic (rt/passlogic.py): when does a car "pass" start and end, and when are two car
tracks the same car?

Background: the tracker gives each car a track id and a box per frame. One car can produce several tracks (the
tracker loses it and finds it again), and two cars can follow each other closely. The pass logic turns tracks into
passes: one car going by = one pass = one count. It emits
  * pass_start: a car became "foreground" (big and low enough in the frame, for 6 frames),
  * pass_provisional: an early number read while the car is still in view,
  * pass_end: the pass is final (the car is gone and 10 s have passed with nothing re-attaching to it),
  * pass_identity: the number read from the pass's best crops.

These scenes are synthetic: a 320x180 frame at 10 fps, with car boxes we place by hand (helpers.run_scene).
"""
import pytest

from .helpers import SyncIdentity, make_identity, moving_box, of_type, run_scene, t_of


def single_car(i0=10, i1=40):
    """One car crossing the frame from left to right in 3 s (frames 10 to 40)."""
    return {1: {i: moving_box(i, i0, i1, 0, 256) for i in range(i0, i1 + 1)}}


def test_one_car_gives_one_pass_with_the_right_times(cfg):
    events = run_scene(cfg, single_car(), n_frames=200)

    # Exactly one start and one end: one car = one count.
    starts, ends = of_type(events, "pass_start"), of_type(events, "pass_end")
    assert len(starts) == 1 and len(ends) == 1

    # pass_start comes when the track has 6 frames (min_frames): the car appeared at frame 10, so at frame 15.
    # t_enter is the first frame of the car (frame 10): that is the time the human counters click.
    assert starts[0]["frame"] == 15
    assert starts[0]["t_enter"] == pytest.approx(t_of(10))

    # The box centre crosses the middle of the frame at frame 25 (x1 = 128, centre = 160 = 320 / 2).
    e = ends[0]
    assert e["t_cross"] == pytest.approx(t_of(25))
    assert e["t_exit"] == pytest.approx(t_of(40))     # the last box
    assert e["dwell_s"] == pytest.approx(3.0)
    assert e["n_frames"] == 31 and e["n_tracks"] == 1
    assert e["exited"] is True                         # the last box touches the right edge of the frame

    # pass_end waits: the track stays "alive" 3 s after its last box (tracker buffer), and then the pass stays open
    # 10 s more (static_gap_s) in case the same car shows up again. Frame 40 + 100 frames (10 s) -> frame 141.
    assert ends[0]["frame"] == 141


def test_two_cars_nose_to_tail_are_two_passes(cfg):
    # Car A crosses during frames 10-40; car B enters 1.5 s later (frames 25-55), while A is still in view.
    # They overlap in time but are far apart in the frame, so they must NOT be merged into one pass.
    tracks = {1: single_car(10, 40)[1], 2: single_car(25, 55)[1]}
    events = run_scene(cfg, tracks, n_frames=250)

    starts, ends = of_type(events, "pass_start"), of_type(events, "pass_end")
    assert len(starts) == 2 and len(ends) == 2
    assert {e["pass_id"] for e in ends} == {0, 1}           # two different passes
    by_id = {e["pass_id"]: e for e in ends}
    assert by_id[0]["t_enter"] == pytest.approx(t_of(10))   # car A
    assert by_id[1]["t_enter"] == pytest.approx(t_of(25))   # car B, 1.5 s later
    assert by_id[0]["tracks"] != by_id[1]["tracks"]


def stopping_car(stop_from=25, stop_to=300, leave_at=330):
    """A car drives in, stops in the middle of the frame for 27.5 s (frames 25-300), then drives out."""
    boxes = {}
    for i in range(10, leave_at + 1):
        if i < stop_from:
            boxes[i] = moving_box(i, 10, stop_from, 0, 128)
        elif i <= stop_to:
            boxes[i] = moving_box(stop_from, 10, stop_from, 0, 128)   # standing still at x1 = 128
        else:
            boxes[i] = moving_box(i, stop_to, leave_at, 128, 256)
    return boxes


def test_car_that_stops_in_frame_is_still_one_pass(cfg):
    events = run_scene(cfg, {1: stopping_car()}, n_frames=500)

    ends = of_type(events, "pass_end")
    assert len(of_type(events, "pass_start")) == 1 and len(ends) == 1
    # The pass lasts from the car's arrival (frame 10) to its last box (frame 330): 32 s.
    assert ends[0]["dwell_s"] == pytest.approx(32.0)
    # And it is not closed while the car stands still: only 10 s after it left (+3 s tracker buffer).
    assert ends[0]["frame"] > 330 + 100


def test_stopped_car_lost_and_found_again_by_the_tracker_is_one_pass(cfg):
    # The tracker loses the stopped car at frame 150 (track 1 ends) and finds it again 6 s later at the same place,
    # as a new track (track 2). Rule (b) of the pass logic: a track that starts where another ended (boxes overlap,
    # IoU > 0.2) within 10 s is the same car. So: still one pass, made of two tracks.
    full = stopping_car()
    tracks = {1: {i: b for i, b in full.items() if i <= 150},
              2: {i: b for i, b in full.items() if i >= 210}}
    events = run_scene(cfg, tracks, n_frames=500)

    ends = of_type(events, "pass_end")
    assert len(of_type(events, "pass_start")) == 1 and len(ends) == 1
    assert ends[0]["n_tracks"] == 2
    assert ends[0]["t_enter"] == pytest.approx(t_of(10))
    assert ends[0]["t_exit"] == pytest.approx(t_of(330))


def test_small_or_far_away_objects_are_not_passes(cfg):
    # A box only 20 px high (0.11 of the frame; the rule wants > 0.2) is something far away, not a car passing the
    # line: no pass at all.
    tracks = {1: {i: moving_box(i, 10, 40, 0, 256, y1=20, bw=30, bh=20) for i in range(10, 41)}}
    events = run_scene(cfg, tracks, n_frames=200)
    assert of_type(events, "pass_start") == []


def test_provisional_and_final_identity_of_a_pass(cfg):
    # Now with the identity worker (fake models that read every number panel as "14"; real vote and roster).
    roster = [4, 14, 41]
    ident = SyncIdentity(make_identity(cfg, roster, emit=None, read="14"))
    events = run_scene(cfg, single_car(), n_frames=200, identity=ident)

    # While the car is still in view, a few crops are read early: the provisional number for the PTZ camera.
    prov = of_type(events, "pass_provisional")
    assert len(prov) == 1 and prov[0]["number"] == 14
    assert prov[0]["frame"] < 40                  # before the car left the frame

    # After pass_end, the pass's best crops are read and voted: number 14, confident enough to auto-confirm.
    ident_ev = of_type(events, "pass_identity")
    assert len(ident_ev) == 1
    e = ident_ev[0]
    assert e["number"] == 14 and e["team"] == "Team 14"
    assert e["auto_confirm"] is True and e["confidence"] == pytest.approx(1.0)
    assert e["provisional_number"] == 14
    # pass_identity repeats the pass times of pass_end, so one line of the JSONL has everything.
    end = of_type(events, "pass_end")[0]
    assert e["t_enter"] == end["t_enter"] and e["t_cross"] == end["t_cross"]
    # The order of the events in the log.
    order = [ev["type"] for ev in events]
    assert order == ["pass_start", "pass_provisional", "pass_end", "pass_identity"]
