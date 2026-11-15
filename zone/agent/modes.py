"""Who drives the zone's PTZ: the contract's modes (studio side) on top of the director's (zone side).

Two state machines meet here:

* the **director** (zone/director.py) has STOPPED / PASSIVE / AUTO, as WP-Z2 built it, unchanged;
* the **contract** (baja-studio docs/agent-api.md §7, normative) has manual / auto / auto_paused / hold.

`ModeArbiter` keeps the contract mode and moves the director to match:

    contract      director                                   who drives the PTZ
    --------      --------                                   ------------------
    manual        PASSIVE if spotter frames arrive,          the studio (the zone sends nothing)
                  STOPPED otherwise
    auto          AUTO                                       the zone's tracker
    auto_paused   PASSIVE (keeps estimating, sends nothing)  the studio, until resume_after_ms after its last
                                                             motion ends; then AUTO again, from a clean state
                                                             (back to the entry preset, next car)
    hold          PASSIVE / STOPPED, like manual             the studio; nothing on the zone side may bring
                                                             AUTO back (not even the laptop page's AUTO button)

The laptop's own page stays in charge for the person standing next to it: PARADO / PASSIVO make the contract mode
`manual`, AUTO makes it `auto` (both with a `mode_changed` event), and PARAR is also a stop. In `hold` the page's
AUTO is refused (the contract forbids the agent from resuming on its own triggers) and PARADO / PASSIVO keep `hold`.

The arbiter never talks to the PTZ or the runtime itself: it goes through a `DirectorLink` (zone/agent/link.py), so
these rules are tested with a fake link and a fake clock (zone/tests/test_agent_modes.py).

Thread safety: one lock around every method. Lock order: arbiter lock, then the runtime's lock (taken inside the
link). Nothing calls into the arbiter while holding the runtime's lock.
"""
import threading
from typing import Any, Callable, Optional, Protocol

from zone.agent.contract import (
    AUTO, AUTO_PAUSED, HOLD, MANUAL, MANUAL_OVERRIDE_ENDED, MANUAL_OVERRIDE_STARTED, MODE_CHANGED,
    RESTORE_REASON_TRACKER_NOT_READY, TRACKER_NOT_READY, AgentError, checked_profile,
)

# the director's modes (zone/director.py), repeated here to keep this module free of the runtime's imports
D_STOPPED, D_PASSIVE, D_AUTO = "STOPPED", "PASSIVE", "AUTO"


class DirectorLink(Protocol):
    """What the arbiter needs from the zone app (implemented in zone/agent/link.py)."""

    def director_mode(self) -> str:
        """The director's mode now: "STOPPED", "PASSIVE" or "AUTO"."""

    def tracker_ready(self) -> tuple[bool, str]:
        """Can AUTO run right now? (True, "") or (False, a sentence saying why not)."""

    def frames_ok(self) -> bool:
        """Are spotter frames arriving (so PASSIVE has something to estimate on)?"""

    def set_director(self, mode: str, why: str, stop_camera: Optional[bool]) -> tuple[bool, str]:
        """Switch the director. `stop_camera`: True = stop the PTZ right now, False = send it nothing at all (the
        studio is about to command it), None = the director's usual rules (the laptop page's buttons)."""


class ModeArbiter:
    """The contract mode of the zone's one camera, and the rules of contract §7."""

    def __init__(self, link: DirectorLink, clock, emit: Callable[[str, dict], None],
                 default_resume_after_ms: int, initial_mode: str = MANUAL,
                 restore_notice: Optional[tuple[str, dict]] = None,
                 on_mode: Optional[Callable[[str], None]] = None) -> None:
        """emit(event, detail) sends a contract event; on_mode(mode) is called after every change (to remember the
        mode across restarts); restore_notice is the (ts, detail) of a mode_changed to replay to every studio that
        connects after a restart (contract §9)."""
        self.link, self.clock = link, clock
        self.emit = emit
        self.on_mode = on_mode or (lambda mode: None)
        self.default_resume_after_ms = int(default_resume_after_ms)
        self.mode = initial_mode
        self.profile: dict[str, Any] = {}
        self.resume_at: Optional[float] = None       # monotonic time auto_paused becomes auto again
        self.holder: Optional[str] = None            # who paused or holds the camera ("studio:<X-Studio-Id>")
        self.restore_notice = restore_notice
        self.lock = threading.RLock()

    # ------------------------------------------------------------------------------------------- restart
    @staticmethod
    def restored(last_mode: Optional[str], now_iso: str) -> tuple[str, Optional[tuple[str, dict]]]:
        """The mode to start in after a restart, from the last mode remembered, and the restore notice.

        The zone always restarts with the director in STOPPED (WP-Z2's rule: nothing moves until someone asks), so
        auto is never restored: the contract's answer for "the last mode was auto but the tracker can't run" is
        `manual`, plus a `mode_changed` {from auto, to manual, reason tracker_not_ready} replayed to every studio
        that connects. A pause doesn't survive a restart (auto_paused counts as auto). hold and manual come back."""
        if last_mode in (AUTO, AUTO_PAUSED):
            return MANUAL, (now_iso, {"from": AUTO, "to": MANUAL, "reason": RESTORE_REASON_TRACKER_NOT_READY})
        if last_mode == HOLD:
            return HOLD, None
        return MANUAL, None

    # ------------------------------------------------------------------------------------------- reading
    def resume_after_ms(self) -> int:
        return int(self.profile.get("resume_after_ms", self.default_resume_after_ms))

    def reply(self) -> dict:
        """The answer of GET / PUT /mode: the mode and the profile (with the resume_after_ms in force)."""
        with self.lock:
            return {"mode": self.mode, "profile": {"resume_after_ms": self.resume_after_ms(), **self.profile}}

    def resume_in_s(self) -> Optional[float]:
        """Seconds until auto_paused resumes by itself (None when not paused)."""
        with self.lock:
            if self.mode != AUTO_PAUSED or self.resume_at is None:
                return None
            return max(0.0, self.resume_at - self.clock.monotonic())

    # ------------------------------------------------------------------------------------------- studio commands
    def studio_motion(self, source: str, motion_ends_at: float) -> None:
        """Call BEFORE sending a studio motion command (/move, /stop, preset recall) to the PTZ.

        In auto, the director is silenced first (PASSIVE, nothing sent: the studio's command follows at once) and
        the mode becomes auto_paused. In auto_paused, the countdown restarts from the end of this motion (its
        velocity's ttl, or now for a stop or a recall): contract decision 5. Other modes: nothing changes."""
        with self.lock:
            self.holder = source
            if self.mode == AUTO:
                self.link.set_director(D_PASSIVE, "studio motion (%s)" % source, stop_camera=False)
                self._change(AUTO_PAUSED, "studio motion command")
                self.emit(MANUAL_OVERRIDE_STARTED, {"source": source})
            if self.mode == AUTO_PAUSED:
                self.resume_at = motion_ends_at + self.resume_after_ms() / 1000.0

    def studio_set_mode(self, mode: str, profile: Optional[dict], source: str) -> dict:
        """PUT /mode: manual, auto or hold (auto_paused is refused before it gets here). -> the reply.

        Raises AgentError tracker_not_ready (409) when auto can't run, bad_request for a bad profile; then nothing
        has changed."""
        with self.lock:
            new_profile = checked_profile(profile) if profile is not None else None
            if mode == AUTO:
                ok, why = self.link.tracker_ready()
                if not ok:
                    raise AgentError(TRACKER_NOT_READY, why)
            if new_profile is not None:
                self.profile = new_profile      # a profile replaces the previous one; no profile keeps it
            self.restore_notice = None          # a studio has chosen the mode: no need to tell it about the restart
            old = self.mode
            if mode == AUTO:
                if old != AUTO:
                    ok, why = self.link.set_director(D_AUTO, "studio: auto (%s)" % source, stop_camera=None)
                    if not ok:
                        raise AgentError(TRACKER_NOT_READY, why)
                self.resume_at = None
                self._change(AUTO, "requested by the studio")
                if old == AUTO_PAUSED:
                    self.emit(MANUAL_OVERRIDE_ENDED, {"reason": "auto requested"})
            else:
                self._quiet_director("studio: %s (%s)" % (mode, source))
                self.resume_at = None
                if mode == HOLD:
                    self.holder = source
                self._change(mode, "requested by the studio")
            return self.reply()

    def _quiet_director(self, why: str) -> None:
        """manual / hold: the director estimates (PASSIVE) when there are frames, else STOPPED; never drives. Leaving
        AUTO stops the camera at once (the zone's own motion must end; nothing more is sent after that)."""
        target = D_PASSIVE if self.link.frames_ok() else D_STOPPED
        current = self.link.director_mode()
        if current != target:
            self.link.set_director(target, why, stop_camera=current == D_AUTO)

    # ------------------------------------------------------------------------------------------- time
    def tick(self) -> None:
        """Called several times a second. auto_paused -> auto once the countdown is over (or manual, with the reason,
        if the tracker can't run any more). And if the zone left AUTO on its own (zone edited to fewer than two
        keyframes...), the contract mode follows: manual."""
        with self.lock:
            now = self.clock.monotonic()
            if self.mode == AUTO_PAUSED and self.resume_at is not None and now >= self.resume_at:
                self.resume_at = None
                ok, why = self.link.tracker_ready()
                if ok:
                    ok, why = self.link.set_director(D_AUTO, "resume_after_ms elapsed", stop_camera=None)
                if ok:
                    self._change(AUTO, "resume_after_ms elapsed")
                    self.emit(MANUAL_OVERRIDE_ENDED, {"reason": "timeout"})
                else:
                    self._change(MANUAL, "could not resume auto: %s" % why)
                    self.emit(MANUAL_OVERRIDE_ENDED, {"reason": RESTORE_REASON_TRACKER_NOT_READY})
            elif self.mode == AUTO and self.link.director_mode() != D_AUTO:
                self._change(MANUAL, "the zone tracker left AUTO")

    # ------------------------------------------------------------------------------------------- the laptop page
    def local_mode(self, director_mode: str) -> tuple[bool, str]:
        """The laptop page's PARADO / PASSIVO / AUTO buttons. -> (ok, message for the page)."""
        with self.lock:
            director_mode = str(director_mode).upper()
            if director_mode == D_AUTO and self.mode == HOLD:
                return False, ("o estúdio está segurando a câmera (HOLD, %s): só o estúdio pode voltar ao AUTO"
                               % (self.holder or "estúdio"))
            ok, msg = self.link.set_director(director_mode, "laptop page", stop_camera=None)
            if not ok:
                return ok, msg
            old = self.mode
            self.resume_at = None
            if director_mode == D_AUTO:
                self._change(AUTO, "laptop page")
            elif old != HOLD:
                self._change(MANUAL, "laptop page")
            if old == AUTO_PAUSED and self.mode != AUTO_PAUSED:
                self.emit(MANUAL_OVERRIDE_ENDED, {"reason": "laptop page"})
            return ok, msg

    def local_stop(self) -> None:
        """The laptop page's PARAR (the caller has already stopped the camera and put the director in STOPPED): auto
        and auto_paused end (manual); manual and hold stay."""
        with self.lock:
            self.resume_at = None
            old = self.mode
            if old in (AUTO, AUTO_PAUSED):
                self._change(MANUAL, "PARAR on the laptop page")
            if old == AUTO_PAUSED:
                self.emit(MANUAL_OVERRIDE_ENDED, {"reason": "laptop page"})

    # ------------------------------------------------------------------------------------------- internals
    def _change(self, mode: str, reason: str) -> None:
        if mode == self.mode:
            return
        old, self.mode = self.mode, mode
        self.restore_notice = None
        self.emit(MODE_CHANGED, {"from": old, "to": mode, "reason": reason})
        self.on_mode(mode)
