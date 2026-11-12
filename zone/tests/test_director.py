"""Tests for zone/director.py: which car is followed and when.

The director only decides; these tests feed it hand-made tracks and a time, and check the decision."""
import pytest

from zone import director as D
from zone.director import Director, DirectorSettings
from zone.progress import TrackView


def tv(tid, s, v=0.25, n=5, age=0.0, off=0):
    """A track as the progress tracker would report it."""
    return TrackView(track_id=tid, s=s, v=v, s_std=0.01, v_std=0.01, lateral_px=0.0, n_updates=n,
                     since_update_s=age, off_corridor_frames=off, predicted=age > 0, box=(), ground=())


def tracks(*views):
    return {v.track_id: v for v in views}


def auto_waiting(settle=1.0, **kw):
    """A director in AUTO, already back at the entry (waiting), at t = 10."""
    d = Director(DirectorSettings(settle_s=settle, **kw))
    d.set_mode(D.AUTO, 0.0)
    assert d.step(0.0, {}).action == D.RECALL_ENTRY      # entering AUTO: go to the entry preset first
    assert d.step(0.5, {}).action == D.NONE              # still settling
    d.step(10.0, {})
    assert d.sub == D.WAITING
    return d


def test_starts_stopped_and_sends_nothing():
    d = Director()
    assert d.mode == D.STOPPED
    assert d.step(0.0, tracks(tv(1, 0.05))).action == D.NONE


def test_passive_sends_nothing():
    d = Director()
    d.set_mode(D.PASSIVE, 0.0)
    assert d.step(0.0, tracks(tv(1, 0.05))).action == D.NONE


def test_locks_the_first_car_crossing_the_entry():
    d = auto_waiting()
    dec = d.step(10.1, tracks(tv(1, -0.10), tv(2, 0.02)))  # car 1 not there yet, car 2 just crossed s = 0
    assert dec.action == D.FOLLOW and dec.track.track_id == 2
    assert d.sub == D.TRACKING and d.locked_id == 2


@pytest.mark.parametrize("bad", [
    tv(1, 0.05, v=0.01),          # too slow (v_min 0.05)
    tv(1, 0.05, v=-0.2),          # going backwards
    tv(1, 0.05, n=2),             # only 2 detections (min 3)
    tv(1, 0.05, age=0.3),         # no fresh detection (predicted)
    tv(1, 0.05, off=1),           # outside the corridor
    tv(1, 0.40),                  # already deep in the zone: past the lock window, too late to catch
])
def test_lock_rule_rejects(bad):
    d = auto_waiting()
    assert d.step(10.1, tracks(bad)).action == D.NONE
    assert d.sub == D.WAITING


def test_other_cars_are_ignored_while_tracking():
    d = auto_waiting()
    d.step(10.1, tracks(tv(1, 0.02)))
    dec = d.step(10.2, tracks(tv(1, 0.05), tv(2, 0.06, v=0.5)))   # a faster car right behind (an overtake)
    assert dec.action == D.FOLLOW and dec.track.track_id == 1


def test_exit_then_return_then_next_car():
    events = []
    d = Director(DirectorSettings(settle_s=1.0), on_event=events.append)
    d.set_mode(D.AUTO, 0.0)
    d.step(0.0, {})
    d.step(5.0, {})
    d.step(5.1, tracks(tv(1, 0.02)))
    dec = d.step(9.0, tracks(tv(1, 1.01)))           # reached the exit
    assert dec.action == D.RECALL_ENTRY and dec.reason == "exit"
    assert d.sub == D.RETURNING and d.passes_followed == 1
    assert d.step(9.5, tracks(tv(2, 0.02))).action == D.NONE   # still settling: car 2 is not taken yet
    d.step(10.1, {})                                  # settle time over
    assert d.sub == D.WAITING
    assert d.step(10.2, tracks(tv(2, 0.03))).track.track_id == 2
    kinds = [e["event"] for e in events]
    assert kinds.count("lock") == 2 and "release" in kinds


def test_drop_when_lost_too_long():
    d = auto_waiting(lost_timeout_s=1.5)
    d.step(10.1, tracks(tv(1, 0.02)))
    assert d.step(10.5, tracks(tv(1, 0.2, age=1.0))).action == D.FOLLOW     # predicted for 1 s: keep following
    dec = d.step(11.0, tracks(tv(1, 0.3, age=1.6)))
    assert dec.action == D.RECALL_ENTRY and dec.reason == "lost"


def test_drop_when_stopped_too_long():
    d = auto_waiting(stopped_timeout_s=3.0)
    d.step(10.1, tracks(tv(1, 0.02)))
    for t in (11.0, 12.0, 13.5):
        assert d.step(t, tracks(tv(1, 0.4, v=0.0))).action == D.FOLLOW
    dec = d.step(14.2, tracks(tv(1, 0.4, v=0.0)))      # slower than stopped_v since t = 11: > 3 s
    assert dec.reason == "stopped"


def test_moving_again_resets_the_stopped_timer():
    d = auto_waiting(stopped_timeout_s=3.0)
    d.step(10.1, tracks(tv(1, 0.02)))
    d.step(11.0, tracks(tv(1, 0.4, v=0.0)))
    d.step(13.0, tracks(tv(1, 0.4, v=0.2)))            # it moved again
    assert d.step(15.0, tracks(tv(1, 0.5, v=0.0))).action == D.FOLLOW


def test_drop_when_off_the_corridor():
    d = auto_waiting(off_corridor_frames=3)
    d.step(10.1, tracks(tv(1, 0.02)))
    assert d.step(10.2, tracks(tv(1, 0.1, off=3))).action == D.FOLLOW
    assert d.step(10.3, tracks(tv(1, 0.1, off=4))).reason == "off_corridor"


def test_drop_when_the_track_disappears():
    d = auto_waiting()
    d.step(10.1, tracks(tv(1, 0.02)))
    assert d.step(10.2, {}).reason == "gone"


def test_stopped_button_sends_a_stop_and_unlocks():
    d = auto_waiting()
    d.step(10.1, tracks(tv(1, 0.02)))
    d.set_mode(D.STOPPED, 10.2)
    assert d.locked_id is None
    assert d.step(10.3, tracks(tv(1, 0.05))).action == D.STOP
    assert d.step(10.4, tracks(tv(1, 0.07))).action == D.NONE   # only once


def test_leaving_auto_for_passive_stops_the_camera():
    d = auto_waiting()
    d.set_mode(D.PASSIVE, 11.0)
    assert d.step(11.1, {}).action == D.STOP


def test_a_new_director_always_starts_stopped():
    # a crash + restart builds a new Director: it must never come back in AUTO
    d = auto_waiting()
    d.step(10.1, tracks(tv(1, 0.02)))
    assert Director().mode == D.STOPPED


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        Director().set_mode("TURBO", 0.0)
