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
        * lost: no detection for longer than `lost_timeout_s` (meanwhile it's predicted ahead: zone/progress.py,
          frozen speed, typical profile, held at most `progress.max_predict_s_ahead` past the last measured s),
        * stopped: for longer than `stopped_timeout_s`, BOTH slow along the path (|v| below `stopped_ratio` x the
          learned typical speed at its s, or below `stopped_v` before the profile has learned) AND still in the
          image (its ground point moved less than `stopped_px_s` pixels per second over that time). `s` is arc length in the image, so where the
          path runs toward the camera a car at racing speed advances very little s per second: the speed alone
          would call it stopped. The pixel test is what says it really is,
        * off the line: outside the corridor for more than `off_corridor_frames` frames in a row,
        * gone: the track no longer exists at all,
        * diverged: the camera stayed too far from the target for too many cycles (the controller's divergence
          guard; the runtime calls `drop()`).
      Either way: back to `returning`.
      During a dropout (no detection) longer than `loosen_after_s`, the zoom is loosened gradually (up to
      `loosen_max` wider by `lost_timeout_s`: `Decision.zoom_extra`), so a car that reappears off the prediction is
      still in the picture; it comes back over `loosen_restore_s` once the car is seen again. When the car reappears,
      `reacquired()` logs it (with the dropout's length) and, if the tracker gave it a new id that progress
      re-attached to the followed track, follows the new id.

The director never talks to the PTZ itself. `step()` returns a `Decision` and the runtime acts on it, which keeps
this module easy to test: feed it fake tracks and a time, check the decision.
"""
import math
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
    lost_timeout_s: float = 3.0     # race: dust, another car in front, a few bad frames (was 1.5 before 2026-11-15)
    loosen_after_s: float = 0.5     # a dropout longer than this starts loosening the zoom ...
    loosen_max: float = 0.3         # ... up to this much wider (fraction of the zoom-in) at lost_timeout_s ...
    loosen_restore_s: float = 1.0   # ... and back to normal over this long once the car is seen again
    stopped_v: float = 0.02         # "slow" before the speed profile has learned anything (s per second)
    stopped_timeout_s: float = 3.0
    stopped_ratio: float = 0.15     # once learned: "slow" = below this fraction of the typical speed at that s
    stopped_px_s: float = 8.0       # ... and a car whose ground point moves this fast in the image is never stopped
    lock_px_s: float = 15.0         # lock also a car slower than v_min (in s) that goes forward and whose ground
                                    # point moves at least this fast in the image (an entry seen head-on); 0 = off
    off_corridor_frames: int = 3
    s_exit: float = 1.0
    settle_s: float = 1.5           # wait after recalling the entry preset (seeded from the report)


@dataclass
class Decision:
    action: str
    track: Optional[TrackView] = None
    reason: str = ""
    zoom_extra: float = 0.0         # FOLLOW: frame this much wider than usual (0..loosen_max), see Controller.follow


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
        self.stopped_anchor = None            # the ground point (x, y) when the car got slow (the stopped rule)
        self.last_drop_reason = ""
        self.last_drop = None                 # {"reason", "s", "held_s", "t"} of the last release (for the page)
        self.passes_followed = 0
        self.zoom_extra = 0.0                 # current extra zoom-out (dropout loosening), 0 = normal framing
        self._t_zoom = None                   # time of the last zoom_extra update
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

    def discard_pending(self):
        """Forget the action the last `set_mode` queued for the next step (the stop when leaving AUTO, the recall when
        entering it). Used when the studio takes the camera over (zone/agent): its own command follows at once, and a
        stop from the director one cycle later would cut it."""
        self._pending = None

    def state(self):
        return dict(mode=self.mode, sub=self.sub if self.mode == AUTO else None, locked_id=self.locked_id,
                    last_drop_reason=self.last_drop_reason, last_drop=self.last_drop,
                    passes_followed=self.passes_followed, zoom_extra=round(self.zoom_extra, 3))

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
        return Decision(FOLLOW, tr, zoom_extra=self._loosen(t, tr))

    def reacquired(self, t, ev):
        """Progress found the followed car again after a dropout (ev: old_id, new_id, dropout_s, how, s; see
        ProgressTracker._reacquire). With a new id (ByteTrack started a new track for the reappearing car and progress
        re-attached it), the new id is followed from now on. Logged as a `reacquired` event either way. Ignored unless
        it is the car being followed right now."""
        if self.mode != AUTO or self.sub != TRACKING or ev.get("old_id") != self.locked_id:
            return
        self.locked_id = ev.get("new_id", self.locked_id)
        self._event(t, "reacquired", track_id=self.locked_id, old_id=ev.get("old_id"),
                    dropout_s=ev.get("dropout_s"), how=ev.get("how"), s=ev.get("s"))

    def _loosen(self, t, tr: TrackView):
        """Extra zoom-out for this cycle. Grows linearly from 0 at `loosen_after_s` of dropout to `loosen_max` at
        `lost_timeout_s`; once the car is seen again it goes back down at loosen_max per `loosen_restore_s` (never a
        jump: the zoom is an absolute position, a jump would be a visible snap)."""
        c = self.cfg
        span = max(1e-6, c.lost_timeout_s - c.loosen_after_s)
        want = 0.0
        if c.loosen_max > 0 and tr.since_update_s > c.loosen_after_s:
            want = c.loosen_max * min(1.0, (tr.since_update_s - c.loosen_after_s) / span)
        dt = 0.0 if self._t_zoom is None else max(0.0, t - self._t_zoom)
        self._t_zoom = t
        if want >= self.zoom_extra:
            self.zoom_extra = want
        else:
            rate = c.loosen_max / c.loosen_restore_s if c.loosen_restore_s > 0 else math.inf
            self.zoom_extra = max(want, self.zoom_extra - rate * dt)
        return self.zoom_extra

    def drop(self, t, reason, tr: Optional[TrackView] = None) -> Optional[Decision]:
        """Let the locked car go for a reason found outside the director (the controller's divergence guard).
        -> the RECALL_ENTRY decision, or None when no car is being tracked."""
        if self.mode != AUTO or self.sub != TRACKING:
            return None
        return self._release(t, reason, tr)

    def _candidate(self, tracks):
        c = self.cfg
        ok = [tr for tr in tracks.values()
              if c.s_lock <= tr.s <= c.s_lock + c.lock_window and tr.s < c.s_exit and self._moving_forward(tr)
              and tr.n_updates >= c.min_updates and not tr.predicted and tr.off_corridor_frames == 0]
        if not ok:
            return None
        return max(ok, key=lambda tr: tr.s)   # two at once: the one further along reached s_lock first

    def _moving_forward(self, tr):
        """Lock rule's "moving forward": v >= v_min, or (an entry seen head-on, where v in s is small) going forward
        at all while its ground point visibly moves in the image. Prefer an entry that is NOT head-on: the speed
        measured on the approach is what the camera's feed-forward starts from."""
        c = self.cfg
        if tr.v >= c.v_min:
            return True
        px = getattr(tr, "ground_px_s", None)
        return c.lock_px_s > 0 and tr.v > 0 and px is not None and px >= c.lock_px_s

    def is_slow(self, tr):
        """Slow along the path: below stopped_ratio x the learned typical speed at its s, or below stopped_v before the
        speed profile has learned anything."""
        c = self.cfg
        v_typ = getattr(tr, "v_typ", None)
        return abs(tr.v) < (c.stopped_ratio * v_typ if v_typ else c.stopped_v)

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
        if tr.predicted:
            # no detection this cycle: a prediction says nothing about stopping (and one held at the cap has v = 0
            # by design). The stopped timer neither starts nor resets; `lost` covers a car that never comes back.
            return ""
        if self.is_slow(tr):
            if self.stopped_since is None:
                self.stopped_since, self.stopped_anchor = t, tr.ground
            elif t - self.stopped_since > c.stopped_timeout_s:
                # Slow for the whole timeout. Did its ground point stay put in the image over that time? Measured from
                # where it was when it got slow to where it is now (not frame to frame: a detection's few pixels of
                # jitter must not count as motion). A car that crossed pixels on a head-on stretch is moving: the
                # check starts over from here.
                moved = _distance(tr.ground, self.stopped_anchor) / (t - self.stopped_since)
                if moved < c.stopped_px_s:
                    return "stopped"
                self.stopped_since, self.stopped_anchor = t, tr.ground
        else:
            self.stopped_since = None
        return ""

    def _release(self, t, reason, tr):
        self.last_drop_reason = reason
        self.last_drop = dict(reason=reason, s=None if tr is None else round(tr.s, 3),
                              held_s=round(t - (self.lock_t or t), 2), t=round(t, 3))
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
        self.zoom_extra, self._t_zoom = 0.0, None

    def _event(self, t, kind, **kw):
        self.on_event(dict(type="director", event=kind, t=round(t, 3), mode=self.mode,
                           sub=self.sub if self.mode == AUTO else None, **kw))


def _distance(a, b):
    """Pixels between two (x, y) ground points; 0 when either is unknown (no detection yet)."""
    if not a or not b:
        return 0.0
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
