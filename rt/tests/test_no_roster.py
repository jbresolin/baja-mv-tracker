"""Tests of running rt for an event that has no roster yet ("open set" mode).

Background: rt snaps every number it reads to the event's entry list, rosters/<EV>.csv (see test_vote.py). That file
is made by `python teacher/rosters.py <EV>` from the official results site, which may list no teams until the event
starts. Before this fix a missing roster stopped rt at start, and the launcher restarted it every 2 s, forever.

Now a missing, empty or unreadable roster never stops rt. config.load_roster() returns an "open set" instead: every
number from 1 to identity.open_set_max (999), no team names, and a warning. In that mode:
  * numbers are still read and voted on, so each pass still gets a number;
  * auto-confirm is always off (its ~99% precision was measured with a real roster, which also throws away
    misreads that are not entry numbers), so no pass is accepted without a human, and no pass_provisional is sent;
  * the warning is logged and shown in yellow on the status page, and says how to fix it (fetch the roster, then
    restart rt: the roster is only read at start).
"""
import pytest

from rt import config as C
from rt.status import StatusBoard, render_html

from .helpers import SyncIdentity, make_identity, moving_box, of_type, run_scene
from .test_vote import cands


def one_car():
    """One car crossing the frame from left to right in 3 s (frames 10 to 40)."""
    return {1: {i: moving_box(i, 10, 40, 0, 256) for i in range(10, 41)}}


# ------------------------------------------------------------------------------------------------ loading the roster
def test_a_missing_roster_gives_an_open_set_with_a_warning(tmp_path):
    # tmp_path is an empty folder pytest makes for this test: there is no 26SU.csv in it.
    roster = C.load_roster("26SU", rosters_dir=tmp_path)
    assert roster["open_set"] is True
    # Every plausible number can be read: 1, 2, ..., 999.
    assert roster["numbers"][0] == 1 and roster["numbers"][-1] == 999 and len(roster["numbers"]) == 999
    assert roster["teams"] == {}                       # no team names without an entry list
    # The warning names the event, says what it means, how to fix it, and that rt must be restarted.
    w = roster["warning"]
    assert "no roster for 26SU" in w
    assert "not checked against the entry list" in w
    assert "python teacher/rosters.py 26SU" in w
    assert "restart rt" in w


def test_an_empty_roster_file_is_treated_like_a_missing_one(tmp_path):
    # The results page can be empty before the event: the scraper then writes only the header line.
    (tmp_path / "26SU.csv").write_text("numero,equipe,escola,fonte\n", encoding="utf-8")
    roster = C.load_roster("26SU", rosters_dir=tmp_path)
    assert roster["open_set"] is True and "is empty" in roster["warning"]


def test_a_broken_roster_file_does_not_stop_rt_either(tmp_path):
    # A file with a non-number in the "numero" column: still no crash, open set, and the warning says why.
    (tmp_path / "26SU.csv").write_text("numero,equipe\nabc,Team X\n", encoding="utf-8")
    roster = C.load_roster("26SU", rosters_dir=tmp_path)
    assert roster["open_set"] is True and "unreadable" in roster["warning"]


def test_the_open_set_size_can_be_changed(tmp_path):
    # identity.open_set_max = 99 in the config: a smaller set (all numbers seen so far were 1-99), cheaper to snap.
    roster = C.load_roster("26SU", open_set_max=99, rosters_dir=tmp_path)
    assert roster["numbers"] == list(range(1, 100))


def test_an_existing_roster_is_used_exactly_as_before(tmp_path):
    # The normal case must not change: the numbers and team names from the file, no warning.
    (tmp_path / "26SE.csv").write_text("numero,equipe\n14,Equipe A\n4,Equipe B\n", encoding="utf-8")
    roster = C.load_roster("26SE", rosters_dir=tmp_path)
    assert roster["open_set"] is False and roster["warning"] is None
    assert roster["numbers"] == [4, 14]                # sorted
    assert roster["teams"] == {4: "Equipe B", 14: "Equipe A"}


def test_the_real_26se_roster_still_loads():
    # The repo's own rosters/26SE.csv (the benchmark race): read from the default folder, not an open set.
    roster = C.load_roster("26SE")
    numbers, teams = C.read_roster("26SE")
    assert roster["open_set"] is False and roster["numbers"] == numbers and roster["teams"] == teams
    assert len(numbers) > 10


def test_the_default_config_has_the_open_set_size(cfg):
    assert cfg["identity"]["open_set_max"] == 999


# ------------------------------------------------------------------------------------------------ reading numbers
def open_set_worker(cfg, read):
    """An identity worker in open-set mode, with fake models that read every number panel as `read`."""
    roster = C.load_roster("NOSUCHEVENT", rosters_dir="/nonexistent-folder")
    w = make_identity(cfg, roster["numbers"], emit=None, read=read)
    w.teams = roster["teams"]            # make_identity invents team names; an open set has none
    w.open_set = True
    return w


def test_open_set_reads_any_number_but_never_auto_confirms(cfg):
    # 5 crops, all clearly read as 123. With a roster this unanimous vote would be auto-confirmed; without one it
    # must go to a human.
    w = open_set_worker(cfg, "123")
    cs = cands(5)
    for c in cs:
        w.read(c)
    v = w.vote(cs)
    assert v["number"] == 123                          # a number is still produced
    assert v["confidence"] == pytest.approx(1.0)
    assert v["auto_confirm"] is False                  # ... but never accepted automatically
    assert v["team"] == ""                             # no team name known


def test_open_set_worker_flag_comes_from_the_constructor(cfg):
    # run.py passes open_set=True to IdentityWorker; check that the constructor stores it.
    from rt.identity import IdentityWorker
    from .helpers import FakePanel, FakeRecognizer
    w = IdentityWorker(cfg, [1, 2, 3], {}, emit=None, panel=FakePanel(), recognizer=FakeRecognizer("2"),
                       open_set=True)
    assert w.open_set is True


def test_a_pass_still_gets_a_number_without_a_roster(cfg):
    # The whole online path (pass logic + identity) on a synthetic car read as "27": one pass, number 27.
    ident = SyncIdentity(open_set_worker(cfg, "27"))
    events = run_scene(cfg, one_car(), n_frames=200, identity=ident)
    identity_events = of_type(events, "pass_identity")
    assert len(identity_events) == 1
    e = identity_events[0]
    assert e["number"] == 27 and e["auto_confirm"] is False
    assert e["roster_checked"] is False                # the JSONL says the number was not checked against a roster
    # No provisional number for the PTZ entry camera: it is only sent for auto-confirmed reads.
    assert of_type(events, "pass_provisional") == []


def test_with_a_roster_the_pass_says_it_was_checked(cfg):
    # The normal path, for comparison: same scene with a real roster -> auto-confirmed and roster_checked True.
    ident = SyncIdentity(make_identity(cfg, [4, 27, 41], emit=None, read="27"))
    events = run_scene(cfg, one_car(), n_frames=200, identity=ident)
    e = of_type(events, "pass_identity")[0]
    assert e["number"] == 27 and e["auto_confirm"] is True and e["roster_checked"] is True


# ------------------------------------------------------------------------------------------------ status page
def test_the_warning_is_shown_in_yellow_on_the_status_page(cfg):
    board = StatusBoard(cfg)
    warning = C.load_roster("26SU", rosters_dir="/nonexistent-folder")["warning"]
    board.notices.append(warning)                      # what run.py does in open-set mode
    payload = board.payload()
    assert payload["notices"] == [warning]
    page = render_html(payload)
    assert '<div class="notice">no roster for 26SU' in page
    assert ".notice{background:#ffc" in page           # the yellow style
