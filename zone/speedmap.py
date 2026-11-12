"""Speed mapping: from a required rate (raw device units per second) to a PTZ speed step, without chatter.

The camera only takes integer speed steps (VISCA: 1..24 for pan, 1..20 for tilt), and step -> rate is non-linear
and undocumented; the characterization report measured it (the "speed table"). Turning a wanted rate into a step
naively (round to the nearest step every cycle) makes the motor twitch between neighbouring steps, and stop/start
whenever the wanted rate hovers around the slowest step. That looks terrible on air. So, per axis:

1. **Inverse table**: the wanted rate becomes a continuous "ideal step" by interpolating the measured table backwards.
2. **Hysteresis**: keep the current step unless the ideal step is more than (0.5 + hysteresis) steps away from it.
3. **Ramp**: change by at most `max_step_change` steps per control cycle.
4. **Minimum step**: below the slowest step that still moves smoothly (`min_smooth_step`), the choice is "hold the
   minimum step" or "stop and wait", whichever error is smaller: hold it while the wanted rate is above half the
   minimum rate, stop below. A small band (`start_margin`) around that half-way point keeps it from flipping back and
   forth between stop and start.

Steps are signed: + = right / up / tele, - = left / down / wide (the driver's convention).
"""
import math

import numpy as np


class AxisSpeedMap:
    def __init__(self, table, negative_table=None, min_smooth_step=1, hysteresis=0.3, max_step_change=4,
                 start_margin=0.15):
        """table: {step (>0): rate in units/s} for positive moves; negative_table: same for negative moves when the
        camera is asymmetric (rates given as positive numbers). min_smooth_step: slowest step that moves smoothly."""
        self.pos = self._prepare(table)
        self.neg = self._prepare(negative_table) if negative_table else self.pos
        self.max_step = int(max(self.pos[0][-1], self.neg[0][-1]))
        self.min_step = max(1, int(min_smooth_step))
        self.hysteresis = float(hysteresis)
        self.max_step_change = max(1, int(max_step_change))
        self.start_margin = float(start_margin)

    @staticmethod
    def _prepare(table):
        steps = np.array(sorted(int(k) for k in table), dtype=np.float64)
        rates = np.array([abs(float(table[k])) for k in sorted(int(k) for k in table)], dtype=np.float64)
        # a measured table can have a step that measured a hair slower than the one before it; force it monotone so
        # the inverse is well defined
        rates = np.maximum.accumulate(rates)
        return steps, rates

    def _side(self, sign):
        return self.pos if sign >= 0 else self.neg

    # ------------------------------------------------------------------------------------------- table lookups
    def rate_of(self, step):
        """Signed rate (units/s) of a signed step. Steps between table entries are interpolated."""
        if step == 0:
            return 0.0
        steps, rates = self._side(step)
        r = float(np.interp(abs(step), steps, rates))
        return r if step > 0 else -r

    def ideal_step(self, rate):
        """Continuous signed step that would give `rate` (inverse of the table, clamped to the table's range)."""
        if rate == 0:
            return 0.0
        steps, rates = self._side(rate)
        a = abs(rate)
        if a <= rates[0]:
            x = steps[0] * a / rates[0] if rates[0] > 0 else steps[0]
        else:
            # np.interp needs strictly increasing x; equal neighbouring rates are nudged apart
            r = rates + np.arange(len(rates)) * 1e-9
            x = float(np.interp(a, r, steps))
        return math.copysign(x, rate)

    # ------------------------------------------------------------------------------------------- the decision
    def next_step(self, rate, current):
        """The step to send this cycle, given the wanted signed `rate` and the step currently commanded."""
        current = int(current)
        target = self._target(rate, current)
        return self._ramp(current, target)

    def _target(self, rate, current):
        r_min = abs(self.rate_of(self.min_step if rate >= 0 else -self.min_step))
        a = abs(rate)
        moving_same_way = current != 0 and (current > 0) == (rate > 0)
        # 4. minimum-step rule, with a band so we never alternate stop / start
        threshold = r_min * (0.5 - self.start_margin if moving_same_way else 0.5 + self.start_margin)
        if a < threshold or rate == 0:
            return 0
        x = abs(self.ideal_step(rate))
        x = min(max(x, self.min_step), self.max_step)
        sign = 1 if rate > 0 else -1
        # 2. hysteresis: keep the current step if the ideal one is close enough to it
        if moving_same_way and abs(x - abs(current)) <= 0.5 + self.hysteresis:
            return current
        return sign * int(round(x))

    def _ramp(self, current, target):
        # 3. ramp: at most max_step_change per cycle; a reversal first ramps down to a stop
        d = self.max_step_change
        if current != 0 and target != 0 and (current > 0) != (target > 0):
            target = 0
        if target == current:
            return current
        if target == 0:
            nxt = current - d if current > 0 else current + d
            if current > 0 and nxt <= 0 or current < 0 and nxt >= 0 or abs(nxt) < self.min_step:
                return 0
            return nxt
        if current == 0:
            # starting: jump straight to at least the minimum smooth step (anything below it doesn't move smoothly)
            sign = 1 if target > 0 else -1
            return sign * max(self.min_step, min(abs(target), d))
        nxt = current + max(-d, min(d, target - current))
        if abs(nxt) < self.min_step:
            nxt = self.min_step if nxt > 0 else -self.min_step
        return nxt
