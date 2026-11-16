"""Sanity check of a taught zone's keyframes: flag poses that look wrong, so the operator re-teaches them.

Why (2026-11-15 bench, real SMTAV PTZ): the operator jogged the camera progressively left to right and taught four
keyframes, but one came out as

    P10 pan 1199 tilt 136 zoom 8724
    P11 pan 1764 tilt  -4 zoom    0    <- wrong: beyond P12 and P13, and fully wide
    P12 pan 1448 tilt 176 zoom 6772
    P13 pan 1636 tilt 100 zoom 9399

and the controller faithfully drove the camera there ("way too far and too fast"). Nothing on the page said that
P11 looked odd. This module looks at the whole set and says which keyframes look odd, and why. It never blocks
anything (a track can really curve back): the page shows a marker on each flagged keyframe with the reason and the
hint "re-ensinar", the log and the health panel get a warning, and the operator can mark a keyframe "conferido"
(`Keyframe.confirmed`) after checking it with "testar", which hides its warnings.

The rules (each one gives one reason, in pt-BR):

1. **Order** (pan, and tilt): along the path the camera normally turns one way only. We find the longest run of
   keyframes (in path order, not necessarily next to each other) whose pan only goes one way, and flag the others.
   In the example: 1199 -> 1448 -> 1636 is the longest run going up, so P11 (1764) is flagged. Same for tilt (a
   track that climbs and then dips can do this for real: then "conferido").
2. **Zoom 0**: a keyframe with zoom exactly 0 (fully wide) while its neighbouring keyframes are not near 0. Zoom 0 is
   also what a stray `90 50 00 00 00 00 FF` from the camera decodes to.
3. **Jump**: the pan changes by more than `jump_fraction` (60 %) of the whole pan range of the keyframes between two
   keyframes next to each other. Only with 3 keyframes or more (with 2, the only step is always 100 %).
"""

ORDER_TOLERANCE = 5            # raw units: a step back smaller than this is not "going back" (reading noise)
JUMP_FRACTION = 0.6            # rule 3: a step bigger than this share of the whole pan range is a "jump"
NEAR_ZERO_ZOOM_SHARE = 0.05    # rule 2: a zoom below this share of the camera's top zoom counts as "near 0"
DEFAULT_ZOOM_MAX = 16384       # the usual top zoom of a VISCA camera, when the report doesn't say
HINT = "confira com 'testar' e, se estiver errado, enquadre de novo e use 're-ensinar'"


def check_keyframes(points, zoom_max=None, jump_fraction=JUMP_FRACTION):
    """Look at the keyframes of a zone (its `points`, in path order).

    -> {point index: [warning, ...]} for the flagged keyframes only, where a warning is
       dict(kind="order_pan" | "order_tilt" | "zoom_zero" | "jump", text=<pt-BR reason>).
    Keyframes the operator marked `confirmed` are never in the result."""
    # (point index, keyframe) of every keyframe, in path order
    keyframes = [(i, p.keyframe) for i, p in enumerate(points) if p.keyframe is not None]
    warnings = {}

    def flag(position, kind, text):
        point_index, keyframe = keyframes[position]
        warnings.setdefault(point_index, []).append(dict(kind=kind, text=text))

    # --- rule 1 (pan): order -------------------------------------------------------------------------------------
    off_order = {"pan": set(), "tilt": set()}

    def check_order(axis, positions):
        """Flag the keyframes at `positions` (indices into `keyframes`) that break the one-way order of `axis`."""
        if len(positions) < 3:
            return
        values = [getattr(keyframes[p][1], axis) for p in positions]
        direction, keep = longest_one_way_run(values, ORDER_TOLERANCE)
        first, last = values[keep[0]], values[keep[-1]]
        for k, position in enumerate(positions):
            if k in keep:
                continue
            off_order[axis].add(position)
            text = "%s fora de ordem: os outros keyframes vão de %d a %d (%s), este tem %s %d" % (
                axis, first, last, "crescendo" if direction > 0 else "diminuindo", axis, values[k])
            if axis == "tilt":
                text += " (pode ser normal se a pista sobe e desce)"
            flag(position, "order_" + axis, text)

    check_order("pan", list(range(len(keyframes))))

    # --- rule 2: zoom exactly 0 while the neighbours aren't near 0 --------------------------------------------------
    near_zero = max(10, NEAR_ZERO_ZOOM_SHARE * (zoom_max or DEFAULT_ZOOM_MAX))
    zoom_zero = set()
    for position, (_, kf) in enumerate(keyframes):
        if kf.zoom != 0:
            continue
        neighbours = [keyframes[j][1].zoom for j in (position - 1, position + 1) if 0 <= j < len(keyframes)]
        if neighbours and all(z > near_zero for z in neighbours):
            zoom_zero.add(position)
            flag(position, "zoom_zero", "zoom 0 (todo aberto) mas os keyframes vizinhos têm zoom %s: suspeito "
                 "(a câmera às vezes manda uma resposta solta que parece 'zoom 0')" % " e ".join(map(str, neighbours)))

    # --- rule 1 (tilt): order, among the keyframes that don't already look wrong ----------------------------------
    # A keyframe already flagged above is left out here, so a wrong pose doesn't become the reference that makes
    # the good ones look out of order (in the bench example P11's tilt -4 would otherwise flag P12 and P13).
    check_order("tilt", [p for p in range(len(keyframes)) if p not in off_order["pan"] and p not in zoom_zero])

    # --- rule 3: a big jump of pan between two keyframes next to each other ----------------------------------------
    if len(keyframes) >= 3:
        pans = [kf.pan for _, kf in keyframes]
        pan_range = max(pans) - min(pans)
        for position in range(1, len(keyframes)):
            step = abs(pans[position] - pans[position - 1])
            if pan_range <= 0 or step <= jump_fraction * pan_range:
                continue
            before, after = keyframes[position - 1][1], keyframes[position][1]
            text = "salto de pan de %d unidades entre P%d e P%d (%d%% de toda a faixa de pan dos keyframes)" % (
                step, before.preset, after.preset, round(100.0 * step / pan_range))
            # Put the marker on the one of the two that is also out of order, if exactly one is (in the bench
            # example: P11). Otherwise we can't tell which one is wrong: mark both.
            pair = [position - 1, position]
            suspects = [p for p in pair if p in off_order["pan"]]
            for p in (suspects if len(suspects) == 1 else pair):
                flag(p, "jump", text)

    # the operator checked these and said they're right
    for point_index in list(warnings):
        if points[point_index].keyframe.confirmed:
            del warnings[point_index]
    return warnings


def longest_one_way_run(values, tolerance=ORDER_TOLERANCE):
    """The longest run of `values` (keeping their order, skipping some) that only goes one way.

    -> (direction, positions): direction +1 (growing) or -1 (shrinking), positions = the indices kept, in order.
    A step back of at most `tolerance` still counts as "the same way". When both directions keep as many, the
    direction from the first value to the last one wins (growing if they are equal).

    Example: [1199, 1764, 1448, 1636] -> (+1, [0, 2, 3]): 1199, 1448, 1636 grow; 1764 is the odd one out."""
    if not values:
        return 1, []
    overall = 1 if values[-1] >= values[0] else -1
    best = None
    for direction in (overall, -overall):      # the overall direction first: it wins a tie
        signed = [direction * v for v in values]
        # length[k] = length of the longest run that ends at position k; previous[k] = the position before k in it
        length, previous = [1] * len(values), [None] * len(values)
        for k in range(len(values)):
            for j in range(k):
                if signed[k] >= signed[j] - tolerance and length[j] + 1 > length[k]:
                    length[k], previous[k] = length[j] + 1, j
        end = max(range(len(values)), key=lambda k: (length[k], -k))   # longest; the earliest end on a tie
        run = []
        while end is not None:
            run.append(end)
            end = previous[end]
        run.reverse()
        if best is None or len(run) > len(best[1]):
            best = (direction, run)
    return best


def summary(points, warnings):
    """One line per flagged keyframe, for the log and the health panel: "P11: pan fora de ordem ...; zoom 0 ..."."""
    lines = []
    for point_index in sorted(warnings):
        kf = points[point_index].keyframe
        lines.append("P%d (ponto %d): %s" % (kf.preset, point_index,
                                             "; ".join(w["text"] for w in warnings[point_index])))
    return lines
