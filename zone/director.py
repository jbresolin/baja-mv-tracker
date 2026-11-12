"""Director: which car to follow, and when. A small explicit state machine.

Modes (chosen on the page):

* **STOPPED**: nothing is sent to the PTZ. The startup default, always, including after a crash and restart.
* **PASSIVE**: detection, progress and logging run, the speed profile learns, but the PTZ gets no commands. Used to
  learn the profile, and whenever the studio has the PTZ on air.
* **AUTO**, with sub-states:
    - `returning`: the entry preset was recalled; wait `settle_s` for the camera to get there.
    - `waiting`: the PTZ sits at the entry preset and doesn't move.
    - `tracking`: follow the locked car. Lock = the first track that reaches s >= s_lock, moving forward with
      v >= v_min, with at least `min_updates` detections, and not already past s_lock + lock_window (a car that
      is deep in the zone when the camera gets back to the entry can't be caught from there: it's skipped).
      Other cars are ignored (overtakes too, for this test).
      The car is let go when it reaches s >= s_exit (normally 1), or dropped when:
        * lost: no detection for longer than `lost_timeout_s` (meanwhile it's predicted ahead on the speed profile),
        * stopped: |v| below `stopped_v` for longer than `stopped_timeout_s`,
        * off the line: outside the corridor for more than `off_corridor_frames` frames in a row,
        * gone: the track no longer exists at all,
        * diverged: the camera stayed too far from the target for too many cycles (the controller's divergence
          guard; the runtime calls `drop()`).
      Either way: back to `returning`.

The director never talks to the PTZ itself. `step()` returns a `Decision` and the runtime acts on it, which keeps
this module easy to test: feed it fake tracks and a time, check the decision.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from zone.progress import TrackView

STOPPED, PASSIVE, AUTO = "STOPPED", "PASSIVE", "AUTO"
WAITING, TRACKING, RETURNING = "waiting", "tracking", "returning"
MODES = (STOPPED, PASSIVE, AUTO)

# what the runtime should do this cycle
NONE = "none"            # send nothing
FOLLOW = "follow"        # run the controller on `track`
RECALL_ENTRY = "recall"  # recall the entry preset
STOP = "stop"            # stop the camera (pan/tilt/zoom)


@dataclass
class DirectorSettings:
    s_lock: float = 0.0
    lock_window: float = 0.15       # only lock a car between s_lock and s_lock + lock_window (one already deep in the
                                    # zone when the PTZ gets back to the entry is skipped: the camera couldn't catch it)
    v_min: float = 0.05             # s per second; a car crossing the zone in 20 s has 0.05
    min_updates: int = 3
    lost_timeout_s: float = 1.5
    stopped_v: float = 0.02
    stopped_timeout_s: float = 3.0
    off_corridor_frames: int = 3
    s_exit: float = 1.0
    settle_s: float = 1.5           # wait after recalling the entry preset (seeded from the report)


@dataclass
class Decision:
    action: str
    track: Optional[TrackView] = None
    reason: str = ""


class Director:
    def __init__(self, settings: DirectorSettings = None, on_event: Callable[[dict], None] = None):
        self.cfg = settings or DirectorSettings()
        self.on_event = on_event or (lambda ev: None)
        self.mode = STOPPED
        self.sub = WAITING
        self.locked_id: Optional[int] = None
        self.lock_t = None
        self.return_until = 0.0
        self.stopped_since = None
        self.last_drop_reason = ""
        self.passes_followed = 0
        self._pending: Optional[str] = None   # an action requested by set_mode, delivered on the next step

    # ------------------------------------------------------------------------------------------- mode changes
    def set_mode(self, mode, t, why="operator"):
        """Change mode. The resulting PTZ action (stop / recall) is returned by the next `step()`."""
        mode = str(mode).upper()
        if mode not in MODES:
            raise ValueError("unknown mode %r" % mode)
        old = self.mode
        if mode == AUTO and old != AUTO:
            self._go_returning(t, "auto_start")
        elif mode != AUTO:
            self._unlock()
            self.sub = WAITING
            # a STOP is always sent when entering STOPPED (the big red button), and when leaving AUTO for PASSIVE
            if mode == STOPPED or old == AUTO:
                self._pending = STOP
        self.mode = mode
        self._event(t, "mode", old=old, new=mode, why=why)

    def state(self):
        return dict(mode=self.mode, sub=self.sub if self.mode == AUTO else None, locked_id=self.locked_id,
                    last_drop_reason=self.last_drop_reason, passes_followed=self.passes_followed)

    # ------------------------------------------------------------------------------------------- one cycle
    def step(self, t, tracks: Dict[int, TrackView]) -> Decision:
        if self._pending:
            action, self._pending = self._pending, None
            if self.mode != AUTO or action == RECALL_ENTRY:
                return Decision(action, reason="mode change")
        if self.mode != AUTO:
            return Decision(NONE)
        if self.sub == RETURNING:
            if t >= self.return_until:
                self.sub = WAITING
                self._event(t, "sub", new=WAITING)
            else:
                return Decision(NONE)
        if self.sub == WAITING:
            cand = self._candidate(tracks)
            if cand is None:
                return Decision(NONE)
            self.locked_id, self.lock_t, self.stopped_since = cand.track_id, t, None
            self.sub = TRACKING
            self._event(t, "lock", track_id=cand.track_id, s=round(cand.s, 3), v=round(cand.v, 3))
            return Decision(FOLLOW, cand, "lock")
        # TRACKING
        tr = tracks.get(self.locked_id)
        reason = self._drop_reason(t, tr)
        if reason:
            return self._release(t, reason, tr)
        return Decision(FOLLOW, tr)

    def drop(self, t, reason, tr: Optional[TrackView] = None) -> Optional[Decision]:
        """Let the locked car go for a reason found outside the director (the controller's divergence guard).
        -> the RECALL_ENTRY decision, or None when no car is being tracked."""
        if self.mode != AUTO or self.sub != TRACKING:
            return None
        return self._release(t, reason, tr)

    def _candidate(self, tracks):
        c = self.cfg
        ok = [tr for tr in tracks.values()
              if c.s_lock <= tr.s <= c.s_lock + c.lock_window and tr.s < c.s_exit and tr.v >= c.v_min
              and tr.n_updates >= c.min_updates and not tr.predicted and tr.off_corridor_frames == 0]
        if not ok:
            return None
        return max(ok, key=lambda tr: tr.s)   # two at once: the one further along reached s_lock first

    def _drop_reason(self, t, tr: Optional[TrackView]):
        c = self.cfg
        if tr is None:
            return "gone"
        if tr.s >= c.s_exit:
            return "exit"
        if tr.since_update_s > c.lost_timeout_s:
            return "lost"
        if tr.off_corridor_frames > c.off_corridor_frames:
            return "off_corridor"
        if abs(tr.v) < c.stopped_v:
            if self.stopped_since is None:
                self.stopped_since = t
            elif t - self.stopped_since > c.stopped_timeout_s:
                return "stopped"
        else:
            self.stopped_since = None
        return ""

    def _release(self, t, reason, tr):
        self.last_drop_reason = reason
        if reason == "exit":
            self.passes_followed += 1
        self._event(t, "release", track_id=self.locked_id, reason=reason,
                    s=None if tr is None else round(tr.s, 3), held_s=round(t - (self.lock_t or t), 2))
        self._go_returning(t, reason)
        self._pending = None
        return Decision(RECALL_ENTRY, tr, reason)

    def _go_returning(self, t, why):
        self._unlock()
        self.sub = RETURNING
        self.return_until = t + self.cfg.settle_s
        self._pending = RECALL_ENTRY
        self._event(t, "sub", new=RETURNING, why=why)

    def _unlock(self):
        self.locked_id = self.lock_t = self.stopped_since = None

    def _event(self, t, kind, **kw):
        self.on_event(dict(type="director", event=kind, t=round(t, 3), mode=self.mode,
                           sub=self.sub if self.mode == AUTO else None, **kw))
