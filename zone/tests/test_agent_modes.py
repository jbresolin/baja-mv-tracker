"""Tests for zone/agent/modes.py: the studio's camera modes (manual / auto / auto_paused / hold) on top of the zone
director's (STOPPED / PASSIVE / AUTO).

How these tests work, for a first-time reader:
* `ModeArbiter` decides the mode. It never touches the real program: it asks a "link" to switch the director and
  to say whether the tracker can run. Here the link is `FakeLink`, a small stand-in that only remembers what it was
  asked, so each rule can be checked on its own.
* Time is a `FakeClock`: it only moves when the test calls `clock.advance(seconds)`.
* "Events" are the messages the studio receives (e.g. mode_changed). `emitted` collects them.

The rules being checked come from the studio's contract (baja-studio docs/agent-api.md, section 7) and the brief:
the studio always wins (its moves pause AUTO), AUTO comes back by itself `resume_after_ms` after the studio's last
move ends, `manual` and `hold` never bring AUTO back by themselves, and the laptop's own buttons still work.
"""
import pytest

from zone.agent.contract import AgentError
from zone.agent.modes import ModeArbiter
from zone.clock import FakeClock


class FakeLink:
    """Pretends to be the zone app. `director` is the director's mode; `ready` says whether AUTO could run;
    `frames` whether the spotter webcam delivers frames. `calls` records every director switch asked for."""

    def __init__(self):
        self.director = "STOPPED"
        self.ready = (True, "")
        self.frames = True
        self.calls = []

    def director_mode(self):
        return self.director

    def tracker_ready(self):
        return self.ready

    def frames_ok(self):
        return self.frames

    def set_director(self, mode, why, stop_camera):
        self.calls.append((mode, stop_camera))
        self.director = mode
        return True, mode


def make(resume_after_ms=1000, mode="manual"):
    """An arbiter on a fake link and a fake clock. -> (arbiter, link, clock, emitted events)."""
    link, clock, emitted = FakeLink(), FakeClock(0.0), []
    arbiter = ModeArbiter(link, clock, lambda event, detail: emitted.append((event, detail)), resume_after_ms,
                          initial_mode=mode)
    return arbiter, link, clock, emitted


# ------------------------------------------------------------------------------------------- studio -> director
def test_studio_auto_is_director_auto_and_manual_is_passive():
    # The studio asks for auto: the zone's director goes to AUTO. Then manual: the director goes to PASSIVE (it keeps
    # watching the cars but sends the camera nothing), and the camera is stopped right away (stop_camera=True),
    # because the zone was moving it.
    arbiter, link, _, emitted = make()
    arbiter.studio_set_mode("auto", None, "studio:s1")
    assert link.director == "AUTO" and arbiter.mode == "auto"
    arbiter.studio_set_mode("manual", None, "studio:s1")
    assert link.director == "PASSIVE" and arbiter.mode == "manual"
    assert link.calls[-1] == ("PASSIVE", True)
    assert [e for e, _ in emitted] == ["mode_changed", "mode_changed"]
    assert emitted[-1][1] == {"from": "auto", "to": "manual", "reason": "requested by the studio"}


def test_manual_without_spotter_frames_is_stopped():
    # No webcam frames: there is nothing to watch, so "manual" leaves the director in STOPPED instead of PASSIVE.
    arbiter, link, _, _ = make()
    link.frames = False
    arbiter.studio_set_mode("manual", None, "studio:s1")
    assert link.director == "STOPPED"


def test_hold_is_like_manual_for_the_director():
    arbiter, link, _, _ = make()
    arbiter.studio_set_mode("auto", None, "studio:s1")
    arbiter.studio_set_mode("hold", None, "studio:s1")
    assert link.director == "PASSIVE" and arbiter.mode == "hold" and arbiter.holder == "studio:s1"


def test_auto_is_refused_when_the_tracker_cannot_run():
    # The link says the tracker can't run (e.g. no zone taught): the studio gets error tracker_not_ready (HTTP 409)
    # with the reason, and nothing changes.
    arbiter, link, _, emitted = make()
    link.ready = (False, "no zone taught")
    with pytest.raises(AgentError) as refused:
        arbiter.studio_set_mode("auto", None, "studio:s1")
    assert refused.value.code == "tracker_not_ready" and refused.value.status == 409
    assert "no zone taught" in refused.value.detail
    assert arbiter.mode == "manual" and link.director == "STOPPED" and emitted == []


def test_a_bad_profile_is_refused_and_changes_nothing():
    arbiter, _, _, _ = make()
    for bad in (-1, 600_001, 1.5, True, "3000"):
        with pytest.raises(AgentError) as refused:
            arbiter.studio_set_mode("auto", {"resume_after_ms": bad}, "studio:s1")
        assert refused.value.code == "bad_request"
    assert arbiter.mode == "manual"


def test_a_profile_replaces_the_previous_one_and_no_profile_keeps_it():
    arbiter, _, _, _ = make(resume_after_ms=3000)
    assert arbiter.reply()["profile"]["resume_after_ms"] == 3000          # the config's default
    arbiter.studio_set_mode("auto", {"resume_after_ms": 500, "zone": "finish"}, "studio:s1")
    arbiter.studio_set_mode("manual", None, "studio:s1")
    assert arbiter.reply()["profile"] == {"resume_after_ms": 500, "zone": "finish"}


# ------------------------------------------------------------------------------------------- pause and resume
def test_studio_motion_in_auto_pauses_and_auto_comes_back_after_resume_after_ms():
    # The studio sends a move that lasts 0.3 s (its ttl) while the zone is in AUTO. The director must go quiet
    # (PASSIVE, and nothing sent to the camera: stop_camera=False, the studio's own command follows) and the mode
    # becomes auto_paused. AUTO comes back 1 s (resume_after_ms) after the move ENDS, so at 0.3 + 1.0 = 1.3 s.
    arbiter, link, clock, emitted = make(resume_after_ms=1000)
    arbiter.studio_set_mode("auto", None, "studio:s1")
    emitted.clear()
    arbiter.studio_motion("studio:s1", motion_ends_at=clock.monotonic() + 0.3)
    assert arbiter.mode == "auto_paused" and link.director == "PASSIVE"
    assert link.calls[-1] == ("PASSIVE", False)
    assert emitted == [("mode_changed", {"from": "auto", "to": "auto_paused", "reason": "studio motion command"}),
                       ("manual_override_started", {"source": "studio:s1"})]
    clock.advance(1.25)
    arbiter.tick()
    assert arbiter.mode == "auto_paused"          # 1.25 s: not yet
    assert arbiter.resume_in_s() == pytest.approx(0.05)
    clock.advance(0.1)
    arbiter.tick()
    assert arbiter.mode == "auto" and link.director == "AUTO"      # 1.35 s: back (the director starts clean)
    assert emitted[-2:] == [("mode_changed", {"from": "auto_paused", "to": "auto", "reason": "resume_after_ms elapsed"}),
                            ("manual_override_ended", {"reason": "timeout"})]


def test_a_held_joystick_keeps_auto_paused_until_it_is_released():
    # A joystick held for 3 s sends a move every 0.1 s (each with a 0.3 s ttl). Each one restarts the countdown,
    # so AUTO only comes back 0.3 + 1.0 s after the last one.
    arbiter, link, clock, _ = make(resume_after_ms=1000)
    arbiter.studio_set_mode("auto", None, "studio:s1")
    for _ in range(30):
        arbiter.studio_motion("studio:s1", clock.monotonic() + 0.3)
        clock.advance(0.1)
        arbiter.tick()
        assert arbiter.mode == "auto_paused"
    clock.advance(1.1)        # 0.1 s after the last move + 1.1 s = 1.2 s < 1.3 s
    arbiter.tick()
    assert arbiter.mode == "auto_paused"
    clock.advance(0.2)
    arbiter.tick()
    assert arbiter.mode == "auto"


def test_a_stop_or_a_recall_starts_the_countdown_at_once():
    arbiter, _, clock, _ = make(resume_after_ms=1000)
    arbiter.studio_set_mode("auto", None, "studio:s1")
    arbiter.studio_motion("studio:s1", clock.monotonic())      # a /stop: the motion ends now
    clock.advance(1.01)
    arbiter.tick()
    assert arbiter.mode == "auto"


def test_put_auto_ends_a_pause_at_once():
    arbiter, link, clock, emitted = make(resume_after_ms=60_000)
    arbiter.studio_set_mode("auto", None, "studio:s1")
    arbiter.studio_motion("studio:s1", clock.monotonic())
    arbiter.studio_set_mode("auto", None, "studio:s1")
    assert arbiter.mode == "auto" and link.director == "AUTO"
    assert emitted[-1] == ("manual_override_ended", {"reason": "auto requested"})


def test_resume_falls_back_to_manual_if_the_tracker_cannot_run_any_more():
    # The webcam died during the pause: when the countdown ends, AUTO can't start, so the camera goes to manual and
    # the studio is told why.
    arbiter, link, clock, emitted = make(resume_after_ms=500)
    arbiter.studio_set_mode("auto", None, "studio:s1")
    arbiter.studio_motion("studio:s1", clock.monotonic())
    link.ready = (False, "no spotter frame for 4 s")
    clock.advance(0.6)
    arbiter.tick()
    assert arbiter.mode == "manual" and link.director == "PASSIVE"
    assert "no spotter frame" in emitted[-2][1]["reason"]
    assert emitted[-1] == ("manual_override_ended", {"reason": "tracker_not_ready"})


def test_studio_motion_in_manual_or_hold_changes_nothing():
    for mode in ("manual", "hold"):
        arbiter, link, clock, emitted = make()
        arbiter.studio_set_mode(mode, None, "studio:s1")
        emitted.clear()
        arbiter.studio_motion("studio:s1", clock.monotonic() + 0.3)
        clock.advance(10)
        arbiter.tick()
        assert arbiter.mode == mode and emitted == []


def test_the_zone_leaving_auto_on_its_own_makes_the_mode_manual():
    # Something on the zone side stopped AUTO (the zone was edited down to one keyframe...): the studio's mode
    # follows at the next tick, with a mode_changed.
    arbiter, link, _, emitted = make()
    arbiter.studio_set_mode("auto", None, "studio:s1")
    link.director = "STOPPED"
    arbiter.tick()
    assert arbiter.mode == "manual"
    assert emitted[-1][1]["reason"] == "the zone tracker left AUTO"


# ------------------------------------------------------------------------------------------- the laptop's page
def test_the_page_buttons_change_the_studio_mode_too():
    # PASSIVO and PARADO are "manual" for the studio, AUTO is "auto". The director follows the page's usual rules
    # (stop_camera=None).
    arbiter, link, _, emitted = make()
    assert arbiter.local_mode("AUTO") == (True, "AUTO")
    assert arbiter.mode == "auto" and link.calls[-1] == ("AUTO", None)
    arbiter.local_mode("PASSIVE")
    assert arbiter.mode == "manual" and link.director == "PASSIVE"
    arbiter.local_mode("STOPPED")
    assert arbiter.mode == "manual" and link.director == "STOPPED"
    assert [d["reason"] for e, d in emitted if e == "mode_changed"] == ["laptop page", "laptop page"]


def test_in_hold_the_page_cannot_bring_auto_back():
    # The contract: in hold, nothing on the agent's side may resume tracking (a button on the laptop neither).
    # PARADO / PASSIVO still work and the camera stays in hold.
    arbiter, link, _, _ = make()
    arbiter.studio_set_mode("hold", None, "studio:s1")
    ok, message = arbiter.local_mode("AUTO")
    assert not ok and "HOLD" in message and "studio:s1" in message
    assert arbiter.mode == "hold" and link.director == "PASSIVE"
    assert arbiter.local_mode("STOPPED")[0]
    assert arbiter.mode == "hold" and link.director == "STOPPED"


def test_the_page_auto_ends_a_studio_pause():
    arbiter, link, clock, emitted = make(resume_after_ms=60_000)
    arbiter.studio_set_mode("auto", None, "studio:s1")
    arbiter.studio_motion("studio:s1", clock.monotonic())
    arbiter.local_mode("AUTO")
    assert arbiter.mode == "auto" and link.director == "AUTO"
    assert emitted[-1] == ("manual_override_ended", {"reason": "laptop page"})


def test_parar_ends_auto_and_a_pause_but_keeps_hold():
    arbiter, _, clock, _ = make()
    arbiter.studio_set_mode("auto", None, "studio:s1")
    arbiter.local_stop()
    assert arbiter.mode == "manual"
    arbiter.studio_set_mode("auto", None, "studio:s1")
    arbiter.studio_motion("studio:s1", clock.monotonic())
    arbiter.local_stop()
    assert arbiter.mode == "manual" and arbiter.resume_at is None
    arbiter.studio_set_mode("hold", None, "studio:s1")
    arbiter.local_stop()
    assert arbiter.mode == "hold"


# ------------------------------------------------------------------------------------------- restart
@pytest.mark.parametrize("last, mode, notice", [
    ("auto", "manual", True),          # the zone restarts in STOPPED: auto can't be restored, and it says so
    ("auto_paused", "manual", True),   # a pause doesn't survive a restart (it counts as auto)
    ("hold", "hold", False),
    ("manual", "manual", False),
    (None, "manual", False),           # never ran before
])
def test_the_mode_after_a_restart(last, mode, notice):
    restored, restore_notice = ModeArbiter.restored(last, "2026-11-21T12:00:00+00:00")
    assert restored == mode
    if notice:
        assert restore_notice == ("2026-11-21T12:00:00+00:00",
                                  {"from": "auto", "to": "manual", "reason": "tracker_not_ready"})
    else:
        assert restore_notice is None


def test_the_restore_notice_is_cleared_by_a_studio_mode_or_a_mode_change():
    for clear in (lambda a: a.studio_set_mode("manual", None, "studio:s1"), lambda a: a.local_mode("AUTO")):
        mode, notice = ModeArbiter.restored("auto", "ts")
        arbiter = ModeArbiter(FakeLink(), FakeClock(), lambda e, d: None, 3000, mode, notice)
        assert arbiter.restore_notice is not None
        clear(arbiter)
        assert arbiter.restore_notice is None


def test_every_mode_change_is_remembered_for_the_next_restart():
    remembered = []
    arbiter = ModeArbiter(FakeLink(), FakeClock(), lambda e, d: None, 3000, on_mode=remembered.append)
    arbiter.studio_set_mode("auto", None, "studio:s1")
    arbiter.studio_motion("studio:s1", 0.0)
    arbiter.studio_set_mode("hold", None, "studio:s1")
    assert remembered == ["auto", "auto_paused", "hold"]
