"""Which way is "right"? The pan/tilt SIGN: how the numbers a camera reports relate to the way it really turns.

Background for a newcomer
-------------------------
The controller sends speed steps in PHYSICAL directions: a positive pan step turns the camera right (the VISCA
direction byte 0x02), a positive tilt step tilts it up. But the keyframe poses it follows are numbers the camera
*reported* (inquiry) or was *sent* (absolute moves), and cameras disagree on which way those numbers count: many
count pan up when turning right (`pan_sign = +1`), some count it DOWN (`pan_sign = -1`), and a camera mounted upside
down flips both axes. The controller turns "the pose must grow" into "turn right" with this sign. With the wrong sign
it turns AWAY from the car from the very first command (bench, 2026-11-15: "the camera starts moving in a completely
wrong direction"), and in Branch B nothing notices (its dead-reckoned pose uses the same wrong sign, so it believes
it is on target).

Where the sign comes from, most trusted first:
1. `ptz.pan_sign` / `ptz.tilt_sign` in the config (+1 or -1): the operator knows. "auto" (the default) = look below.
2. Measured while teaching (`zone.json` -> `ptz_signs`): with the "inquiry" pose source, every jog the operator holds
   is a little experiment: a right jog that made the reported pan go down means pan_sign = -1 (see
   `teach.InquirySource`). This is the same camera, in the same mounting, as the keyframes: the best evidence.
3. The bench report (`speed_table.pan_sign`), when the bench could measure its speed table.
4. Nothing: +1, flagged "padrao" (default) on the page and in the log, in red when the keyframes depend on it.

Keyframes taught by dead reckoning don't depend on the sign (their numbers were made with the same sign the
controller uses), so an unmeasured sign is only a problem for keyframes taught by inquiry or absolute moves.
"""

# how the sign was decided (shown on the page and written in the control log)
CONFIG, TEACH, REPORT, DEFAULT = "config", "teach", "report", "default"
AXES = ("pan", "tilt")
# keyframe pose sources whose numbers are in the camera's own frame (so they need the right sign)
FRAME_SENSITIVE = ("inquiry", "commanded")


def config_sign(value):
    """The `ptz.pan_sign` config value -> +1 / -1, or None for "auto" (or anything else empty)."""
    if value in (None, "", "auto"):
        return None
    v = int(value)
    if v not in (1, -1):
        raise ValueError("ptz.pan_sign / ptz.tilt_sign must be 1, -1 or \"auto\", not %r" % (value,))
    return v


def report_sign(report, axis):
    """The sign the bench report measured for `axis`, or None if it didn't measure one.

    The report's `model` always has a pan_sign (+1 by default), so +1 alone proves nothing: only the speed table's own
    `pan_sign` (written when the bench measured rates) counts. A model value of -1 can't be a default, so it counts
    too (tests and hand-edited reports)."""
    raw = getattr(report, "raw", None) or {}
    measured = (raw.get("speed_table") or {}).get(axis + "_sign")
    if measured in (1, -1):
        return int(measured)
    model_value = getattr(report.model, axis + "_sign", 1)
    return -1 if model_value == -1 else None


def resolve(cfg_ptz, report):
    """Signs from the config and the report only (the zone's own measurement is applied later, see `apply_zone`).
    -> ({"pan": +1/-1, "tilt": ...}, {"pan": CONFIG / REPORT / DEFAULT, ...})"""
    signs, how = {}, {}
    for axis in AXES:
        forced = config_sign((cfg_ptz or {}).get(axis + "_sign", "auto"))
        measured = report_sign(report, axis)
        if forced is not None:
            signs[axis], how[axis] = forced, CONFIG
        elif measured is not None:
            signs[axis], how[axis] = measured, REPORT
        else:
            signs[axis], how[axis] = 1, DEFAULT
    return signs, how


def apply_zone(settings, zone):
    """Use the signs measured while teaching this zone (zone.ptz_signs) unless the config forces one. Changes
    `settings.controller` in place (the controller and its pose estimator share that object, so a sign measured during
    a teach session takes effect at once). -> list of warnings (pt-BR, for the page and the log)."""
    warnings = []
    measured = dict(getattr(zone, "ptz_signs", None) or {})
    base_how = settings.sign_base_how
    base_sign = settings.sign_base
    for axis in AXES:
        sign, how = base_sign[axis], base_how[axis]
        taught = measured.get(axis)
        if taught in (1, -1) and how == CONFIG and taught != sign:
            warnings.append("sinal de %s: o site.toml força %+d, mas o ensino mediu %+d nesta câmera; se a PTZ girar "
                            "ao contrário, tire ptz.%s_sign do site.toml" % (axis, sign, taught, axis))
        if taught in (1, -1) and how != CONFIG:
            if how == REPORT and taught != sign:
                warnings.append("sinal de %s: o relatório da bancada diz %+d, o ensino mediu %+d; usando %+d (o "
                                "ensino é desta câmera, nesta montagem)" % (axis, sign, taught, taught))
            sign, how = int(taught), TEACH
        setattr(settings.controller, axis + "_sign", sign)
        settings.sign_how[axis] = how
    unmeasured = [a for a in AXES if settings.sign_how[a] == DEFAULT]
    sensitive = any(k.pose_source.split("+")[0] in FRAME_SENSITIVE for k in zone.keyframes())
    if unmeasured and sensitive:
        warnings.append("sinal de %s NÃO MEDIDO (assumido +1): se a PTZ girar para o lado errado, ensine de novo com "
                        "o jog (mede sozinho) ou ponha ptz.%s_sign = -1 no site.toml" % (
                            " e ".join(unmeasured), unmeasured[0]))
    return warnings


def summary(settings):
    """For the page / log: {"pan": {"sign": -1, "how": "teach"}, "tilt": {...}}."""
    return {a: dict(sign=getattr(settings.controller, a + "_sign"), how=settings.sign_how[a]) for a in AXES}


def vote_sign(votes):
    """A list of +1 / -1 observations -> (sign, consistent) or (None, True) without any."""
    if not votes:
        return None, True
    total = sum(votes)
    if total == 0:
        return None, False
    return (1 if total > 0 else -1), all(v == votes[0] for v in votes)
