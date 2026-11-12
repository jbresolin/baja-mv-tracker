"""Grouping raw beam edges into one "break" per car (WP-L2). Software only: no hardware hold-off.

Why grouping: a Baja car is a tube frame. As it crosses, the light beam is blocked by the front, passes through a gap
between tubes, is blocked again by the driver, the firewall, a wheel... so ONE car gives SEVERAL pulses. The rule:

  * the first active edge opens a break: its time is THE crossing time (the car's nose cut the beam);
  * every edge that comes less than `group_gap_s` (default 0.25 s) after the previous one joins the same break;
  * the break closes when the beam is clear and has stayed quiet for `group_gap_s`.
    A second car nose-to-tail, just more than `group_gap_s` behind, therefore starts a new break.

Stuck beam: if a break lasts longer than `stuck_s` (someone standing in the beam, a car stopped on the line, a
misaligned sensor, or the wrong polarity in the config) it is not a car. One `beam_stuck` event is produced when that
is detected (state "stuck"), none of its edges become breaks, and a second `beam_stuck` (state "cleared") comes when
the beam has been clear for `group_gap_s` again. So a person standing in the beam for an hour gives two events, not
thousands of breaks.

BreakGrouper is pure logic: it gets edges and the current time, and returns finished records. It never reads a
clock or sleeps, so tests drive it with made-up times.
"""


class BreakGrouper:
    def __init__(self, group_gap_s=0.25, stuck_s=5.0):
        self.gap = float(group_gap_s)
        self.stuck_s = float(stuck_s)
        self.open = None            # the break being built (dict), or None
        self.blocked = False        # is the beam blocked right now (last non-momentary edge was active)?
        self.stuck = None           # the stuck episode (dict) while the beam is stuck, else None
        self.last_edge_mono = None
        self.next_break_id = 0

    # ------------------------------------------------------------------ inputs
    def start_blocked(self, t_mono, t_wall):
        """The beam was already active when the port opened (no edge seen): treat it as a break that started now,
        flagged `started_blocked` (its start time is not a real crossing time). If it stays blocked it becomes a
        stuck episode like any other."""
        self.blocked = True
        self.last_edge_mono = t_mono
        self.open = self._new_break(t_mono, t_wall, started_blocked=True)
        self.open["n_edges"] = 0
        self.open["n_pulses"] = 0

    def feed(self, edge):
        """Add one RawEdge (rt/beam/sources.py). Returns the records finished by it (closed breaks / stuck
        events), each a dict with "kind" = "break" or "stuck"."""
        out = self.advance(edge.t_mono, edge.t_wall)
        self.last_edge_mono = edge.t_mono
        if not edge.momentary:
            self.blocked = edge.level == 1
        if self.stuck is not None:                   # inside a stuck episode: count, never make breaks
            self.stuck["n_edges"] += 1
            return out
        if self.open is None:
            if edge.level == 1:
                self.open = self._new_break(edge.t_mono, edge.t_wall)
            return out                               # an inactive edge with no break open: nothing to group
        b = self.open
        b["n_edges"] += 1
        b["n_pulses"] += int(edge.level == 1)
        b["last_mono"], b["last_wall"] = edge.t_mono, edge.t_wall
        return out

    def advance(self, now_mono, now_wall):
        """Let time pass up to now: close a break that has been quiet long enough, detect / clear a stuck beam.
        Returns the finished records. Called for every edge and regularly (every ~50 ms) without edges."""
        out = []
        b = self.open
        if b is not None and self.stuck is None:
            if now_mono - b["first_mono"] > self.stuck_s and (self.blocked or now_mono - b["last_mono"] < self.gap):
                # Still going after stuck_s: not a car. Drop the break, start a stuck episode.
                self.stuck = dict(kind="stuck", state="stuck", since_mono=b["first_mono"], since_wall=b["first_wall"],
                                  n_edges=b["n_edges"], started_blocked=b["started_blocked"])
                self.open = None
                out.append(dict(self.stuck))
            elif not self.blocked and now_mono - b["last_mono"] >= self.gap:
                self.open = None
                out.append(self._finish(b))
        if self.stuck is not None and not self.blocked and now_mono - (self.last_edge_mono or now_mono) >= self.gap:
            s = self.stuck
            self.stuck = None
            out.append(dict(s, state="cleared", until_mono=self.last_edge_mono,
                            until_wall=now_wall - (now_mono - self.last_edge_mono),
                            duration_s=round(self.last_edge_mono - s["since_mono"], 3)))
        return out

    def disconnect(self, now_mono, now_wall):
        """The port was lost: whatever was open is cut short. An open break is finished with `interrupted` = True;
        a stuck episode is cleared. The beam state is unknown until the port reopens."""
        out = []
        if self.open is not None:
            b = self.open
            self.open = None
            out.append(dict(self._finish(b), interrupted=True))
        if self.stuck is not None:
            s = self.stuck
            self.stuck = None
            out.append(dict(s, state="cleared", reason="port lost", until_mono=now_mono, until_wall=now_wall,
                            duration_s=round(now_mono - s["since_mono"], 3)))
        self.blocked = False
        return out

    # ------------------------------------------------------------------ helpers
    def _new_break(self, t_mono, t_wall, started_blocked=False):
        b = dict(kind="break", break_id=self.next_break_id, first_mono=t_mono, first_wall=t_wall, last_mono=t_mono,
                 last_wall=t_wall, n_edges=1, n_pulses=1, started_blocked=started_blocked, interrupted=False)
        self.next_break_id += 1
        return b

    @staticmethod
    def _finish(b):
        return dict(b, duration_s=round(b["last_mono"] - b["first_mono"], 4))

    def status(self, now_mono):
        """For the status page: is a stuck episode going on, and since how long."""
        if self.stuck is None:
            return dict(stuck=False)
        return dict(stuck=True, stuck_for_s=round(now_mono - self.stuck["since_mono"], 1))
