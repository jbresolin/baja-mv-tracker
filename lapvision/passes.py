"""Stage 3: turn tracker output into passes.

Rules (from the race-control requirements):
  * one physical car crossing = one pass, no matter how many number plates are seen
    (all plates/frames only add votes to that pass's identity);
  * a car that stops / is occluded / gets a new tracker id while still in frame is the
    same pass: a new pass only starts after the previous one LEFT the frame.

Implementation:
  * foreground track = big box (height > 25% H) reaching the bottom part of the frame
    (filters cars in the background and parked vehicles behind the fence);
  * fragments are chained when the next fragment starts < MERGE_GAP s after the previous
    ended AND the previous did not end touching a frame edge (i.e. car still in view),
    or when they overlap in time with overlapping boxes (duplicate ids);
  * pass time = when the box centre crosses the frame centre line (falls back to midpoint).
"""
import numpy as np
import pandas as pd

EDGE = 0.02        # fraction of width considered "touching the edge"
MERGE_GAP = 3.0    # s
STATIC_GAP = 10.0  # s, max gap to re-attach a track that restarts where the last one ended
FG_H = 0.20
FG_Y2 = 0.40
MIN_FRAMES = 6


def track_summary(t):
    t = t.sort_values("tod")
    W, H = t.W.iloc[0], t.H.iloc[0]
    cx = (t.x1 + t.x2) / 2 / W
    first, last = t.iloc[0], t.iloc[-1]

    def edge(r):
        return r.x1 < EDGE * W or r.x2 > (1 - EDGE) * W

    # centre-line crossing
    s = np.sign(cx.values - 0.5)
    idx = np.where(np.diff(s) != 0)[0]
    tc = float(t.tod.iloc[idx[0] + 1]) if len(idx) else float(t.tod.median())
    return dict(event=first.event, track=int(first.track), t0=float(first.tod), t1=float(last.tod),
                n=len(t), hmax=float(((t.y2 - t.y1) / H).max()), y2max=float((t.y2 / H).max()),
                cx0=float(cx.iloc[0]), cx1=float(cx.iloc[-1]), start_edge=bool(edge(first)),
                end_edge=bool(edge(last)), tcross=tc, crossed=bool(len(idx)),
                last_box=(last.x1 / W, last.y1 / H, last.x2 / W, last.y2 / H),
                first_box=(first.x1 / W, first.y1 / H, first.x2 / W, first.y2 / H))


def _near(a, b, tol=0.15):
    """boxes (normalised) overlap or are close."""
    ax, bx = (a[0] + a[2]) / 2, (b[0] + b[2]) / 2
    return abs(ax - bx) < max(tol, (a[2] - a[0] + b[2] - b[0]) / 2)


def _dedupe(tracks, fg):
    """Union tracks that coexist in time on the same car (detector put two boxes on one car,
    e.g. a whole-car box and a partial box -> two ByteTrack ids). Criterion: on shared frames the
    smaller box lies mostly inside the larger one (median intersection / smaller area > 0.6)."""
    key = tracks.event + ":" + tracks.track.astype(str)
    rows = {k: g.set_index("tod") for k, g in tracks.assign(key=key).groupby("key")}
    keys = list(fg.key)
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            k = parent[k]
        return k

    for i, a in enumerate(keys):
        A = rows[a]
        for b in keys[i + 1:]:
            B = rows[b]
            common = A.index.intersection(B.index)
            if len(common) < 3:
                continue
            a_, b_ = A.loc[common], B.loc[common]
            ix = (np.minimum(a_.x2, b_.x2) - np.maximum(a_.x1, b_.x1)).clip(lower=0)
            iy = (np.minimum(a_.y2, b_.y2) - np.maximum(a_.y1, b_.y1)).clip(lower=0)
            amin = np.minimum((a_.x2 - a_.x1) * (a_.y2 - a_.y1), (b_.x2 - b_.x1) * (b_.y2 - b_.y1))
            if float(np.median(ix * iy / amin)) > 0.6:
                parent[find(b)] = find(a)
    groups = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    out = []
    for members in groups.values():
        t = pd.concat([rows[k].reset_index() for k in members])
        t = t.assign(area=(t.x2 - t.x1) * (t.y2 - t.y1)).sort_values("area").drop_duplicates("tod", keep="last")
        sm = track_summary(t)
        sm["keys"] = members
        out.append(sm)
    return pd.DataFrame(out)


def build(tracks):
    S = pd.DataFrame([track_summary(g) for _, g in tracks.groupby(["event", "track"])])
    S["key"] = S.event + ":" + S.track.astype(str)
    fg = S[(S.hmax > FG_H) & (S.y2max > FG_Y2) & (S.n >= MIN_FRAMES)]
    fg = _dedupe(tracks, fg).sort_values("t0")
    passes = []  # list of lists of summaries
    for _, r in fg.iterrows():
        attached = False
        for p in reversed(passes[-4:]):
            last = max(p, key=lambda x: x["t1"])
            gap = r.t0 - last["t1"]
            # (a) car still in view (did not end at an edge) and reappears nearby soon
            if gap < MERGE_GAP and not last["end_edge"] and _near(last["last_box"], r.first_box):
                p.append(r.to_dict()); attached = True; break
            # (b) a new track starting where the previous one ended = same car that stopped / was
            #     occluded / sits at the frame edge. A genuinely new car enters from the entry side.
            if gap < STATIC_GAP and _iou(last["last_box"], r.first_box) > 0.2:
                p.append(r.to_dict()); attached = True; break
            # (c) simultaneous duplicate ids on the same car
            if gap < 0 and _near(last["last_box"], r.first_box, 0.05):
                p.append(r.to_dict()); attached = True; break
        if not attached:
            passes.append([r.to_dict()])
    out = []
    for i, p in enumerate(passes):
        P = pd.DataFrame(p)
        crossed = P[P.crossed]
        out.append(dict(pass_id=i, event=P.event.iloc[0],
                        tracks=[k for ks in P["keys"] for k in ks], t0=P.t0.min(),
                        t1=P.t1.max(), tcross=float(crossed.tcross.iloc[0]) if len(crossed) else float(P.tcross.mean()),
                        n=int(P.n.sum()), exited=bool(P.sort_values("t1").end_edge.iloc[-1]),
                        hmax=float(P.hmax.max()), dwell=float(P.t1.max() - P.t0.min())))
    return pd.DataFrame(out)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def pick_crops(tracks, ps, k=8):
    """Best k crops for a pass: large + sharp, spread across time."""
    key = tracks.event + ":" + tracks.track.astype(str)
    t = tracks[key.isin(ps.tracks)].copy()
    t["area"] = (t.x2 - t.x1) * (t.y2 - t.y1)
    t = t[t.area > 0.4 * t.area.max()]
    t["score"] = t.sharp.rank(pct=True) + t.area.rank(pct=True)
    t = t.sort_values("tod")
    chunks = np.array_split(np.arange(len(t)), min(k, len(t)))
    return [t.iloc[c].sort_values("score").iloc[-1] for c in chunks if len(c)]
