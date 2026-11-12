"""Tests of number reading after the models: snapping a read to the event roster, and the vote over a pass's crops.

Background: the recognizer reads digits from each crop of a car's number panel, but it can misread. rt never outputs a
number that is not in the event's roster (rosters/<EV>.csv): the read is turned into a probability for each roster
number ("snapping"). Then each pass votes over its ~20-30 crops. The winner is the pass's number; it is
"auto-confirmed" (accepted without a human) only when the vote is clear: share >= tau (0.9) and at least
min_support (3) crops that each read it with probability >= 0.5. These thresholds come from the 25SU validation set.
"""
import numpy as np
import pytest

from rt.config import vote_rule
from rt.models import RosterScorer, vote_boxed
from rt.passlogic import Cand

from .conftest import textured_frame
from .helpers import ctc_logp, make_identity

ROSTER = [4, 14, 41, 104]


def snap(text, roster=ROSTER):
    """Probability of each roster number for a recognizer that is sure it read `text`."""
    P, p_other = RosterScorer(roster)(ctc_logp(text)[:, None, :])
    return P[0], p_other[0]


def test_a_clear_read_snaps_to_that_roster_number():
    P, p_other = snap("14")
    assert ROSTER[int(np.argmax(P))] == 14
    assert P.max() == pytest.approx(1.0, abs=1e-6)
    assert p_other == pytest.approx(0.0, abs=1e-6)     # nothing left for "a number outside the roster"


def test_a_leading_zero_still_matches_a_one_digit_number():
    # Some panels print car 4 as "04". The scorer accepts "n" and "0n" for numbers below 10.
    P, _ = snap("04")
    assert ROSTER[int(np.argmax(P))] == 4


def test_a_number_outside_the_roster_gets_no_probability():
    # "99" is not in the roster: no roster number gets probability, everything goes to "other".
    P, p_other = snap("99")
    assert P.max() < 1e-6
    assert p_other == pytest.approx(1.0, abs=1e-6)


def test_vote_rule_comes_from_the_published_models():
    # The thresholds of the vote are read from the model manifest (deploy/models/manifest.json), not from code.
    from rt import config as C
    rule = vote_rule(C.load())
    assert rule == dict(tau=0.9, min_support=3, score="share")


def test_vote_picks_the_majority_and_reports_how_clear_it_was():
    # 7 crops read 14, 3 crops read 41. Share of the winner = 7 / 10.
    P = [snap("14")[0]] * 7 + [snap("41")[0]] * 3
    j, score, runner_up, n_boxed, support, shares = vote_boxed(P, [True] * 10)
    assert ROSTER[j] == 14
    assert score == pytest.approx(0.7)
    assert runner_up == pytest.approx(0.3)
    assert support == 7 and n_boxed == 10


def test_crops_without_a_number_panel_do_not_vote():
    # has_box = False means the panel detector found no number panel on that crop (mud, wrong angle). Those crops are
    # ignored: here 3 crops read 41 but have no box, so 14 wins with share 1.0.
    P = [snap("14")[0]] * 4 + [snap("41")[0]] * 3
    j, score, _, n_boxed, support, _ = vote_boxed(P, [True] * 4 + [False] * 3)
    assert ROSTER[j] == 14 and score == pytest.approx(1.0) and n_boxed == 4 and support == 4


def cands(n):
    """n crop candidates (the pass logic's crops) with some image in them."""
    return [Cand(t=float(k), area=100.0, sharp=1.0, img=textured_frame(40, 30, k), jpeg_q=92) for k in range(n)]


def test_identity_worker_auto_confirms_a_clear_vote(cfg):
    # 5 crops, all read as 14 by the (fake) recognizer: a unanimous vote, auto-confirmed.
    w = make_identity(cfg, ROSTER, emit=None, read="14")
    cs = cands(5)
    assert w.vote(cs) is None          # nothing read yet: no vote
    for c in cs:
        w.read(c)
    v = w.vote(cs)
    assert v["number"] == 14 and v["auto_confirm"] is True and v["support"] == 5
    assert v["top3"][0] == dict(number=14, share=1.0)


def test_identity_worker_sends_an_unclear_vote_to_review(cfg):
    # 3 of 5 crops read 14, 2 read 41: 14 wins, but a 0.6 share is under tau = 0.9, so a human must confirm it.
    reads = ["14", "41", "14", "41", "14"]
    w = make_identity(cfg, ROSTER, emit=None, read=lambda k: reads[k])
    cs = cands(5)
    for c in cs:
        w.read(c)
    v = w.vote(cs)
    assert v["number"] == 14
    assert v["confidence"] == pytest.approx(0.6)
    assert v["auto_confirm"] is False
    # Suggestions for the reviewer, best first (a 3rd number may follow with a share of ~0).
    assert [x["number"] for x in v["top3"][:2]] == [14, 41]
    assert v["top3"][1]["share"] == pytest.approx(0.4)


def test_identity_worker_with_no_panel_found_gives_no_number(cfg):
    # The panel detector finds nothing on any crop (e.g. car 1, always covered in mud): number None, never a guess.
    w = make_identity(cfg, ROSTER, emit=None, read="14", boxes=False)
    cs = cands(5)
    for c in cs:
        w.read(c)
    v = w.vote(cs)
    assert v["number"] is None and v["auto_confirm"] is False and v["n_boxed"] == 0
