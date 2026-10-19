"""Online pass logic: lapvision/passes.py (session 1, as evaluated by WP9) in streaming form.

Offline (`passes.build`) works on finished tracks: foreground filter (max box height > 0.2 H, max bottom > 0.4 H,
>= 6 frames), union of nested duplicate tracks, then in t0 order each track joins one of the last 4 passes if
  (a) gap < 3 s, the previous box did not end at a frame edge and the new one starts near it, or
  (b) gap < 10 s and the new track starts where the previous one ended (IoU > 0.2), or
  (c) it coexists with the previous one (gap < 0) and starts near it,
where `previous` = the pass member with the latest end. Here the same decision is taken when a track *becomes*
foreground (its t0 and first box are final by then; the other side uses the member's state at that moment), nested
duplicates are tested against the coexisting members of open passes, and a pass closes when no rule can attach
anything any more: all members gone from the tracker, > static_gap since the last box, and no younger track that
started inside the window is still waiting to become foreground (bounded by pending_max_s).

Events: pass_start (a new foreground car; PTZ entry signal), pass_end (pass closed, final times), plus the identity
worker's pass_provisional / pass_identity. Crops: each track keeps a bounded buffer of crop candidates (large + sharp,
thinned in time when full, JPEG-encoded); at close `pick_crops` chooses <= max_crops of them as passes.pick_crops.
"""
import time
from collections import deque

import cv2
import numpy as np

from .models import sharpness


# ------------------------------------------------------------------------------------------------ crops
class Cand:
    """One crop candidate. `res` is filled by the identity worker: (P over roster, has_box, n_boxes)."""
    __slots__ = ("t", "area", "sharp", "data", "jpeg", "res")

    def __init__(self, t, area, sharp, img, jpeg_q):
        self.t, self.area, self.sharp, self.res = t, area, sharp, None
        if jpeg_q:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_q)])
            self.data, self.jpeg = buf, True
        else:
            self.data, self.jpeg = img.copy(), False

    def image(self):
        return cv2.imdecode(self.data, cv2.IMREAD_COLOR) if self.jpeg else self.data

    def nbytes(self):
        return int(self.data.nbytes)


def rank_pct(x):
    """pandas Series.rank(pct=True) (average ranks for ties) / n."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if not n:
        return x
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    upper = np.cumsum(cnt)
    avg = upper - (cnt - 1) / 2.0
    return avg[inv] / n


def pick_crops(cands, k):
    """lapvision/passes.pick_crops: area > 0.4 x max, score = rank(sharp) + rank(area), k chunks in time, best per
    chunk."""
    if not cands:
        return []
    amax = max(c.area for c in cands)
    c = sorted([c for c in cands if c.area > 0.4 * amax], key=lambda z: z.t)
    if not c:
        return []
    sc = rank_pct([z.sharp for z in c]) + rank_pct([z.area for z in c])
    out = []
    for ch in np.array_split(np.arange(len(c)), min(k, len(c))):
        if len(ch):
            out.append(c[int(ch[np.argmax(sc[ch])])])  # argmax = first best (pandas' sort().iloc[-1] = last best)
    return out


# ------------------------------------------------------------------------------------------------ tracks
class TrackAcc:
    def __init__(self, tid, W, H, cfg):
        self.tid, self.W, self.H = tid, W, H
        p, i = cfg["passes"], cfg["identity"]
        self.edge = p["edge"]
        self.n = 0
        self.t0 = self.t1 = None
        self.hmax = self.y2max = 0.0
        self.first_box = self.last_box = None
        self.start_edge = self.end_edge = False
        self.prev_sign = None
        self.tcross = None
        self.times = []                     # for the median-time fallback (capped)
        self.boxes = deque(maxlen=600)      # (frame no, abs box): nested-duplicate test
        self.alive = True
        self.qualified = False
        self.pass_ = None
        self.cands = []
        self.max_area = 0.0
        self.cap = int(i["candidates_per_track"])
        self.crop_min_h = float(i["crop_min_h"])
        self.pad = float(i["crop_pad"])
        self.jpeg = int(i["crop_jpeg"])

    def _edge(self, b):
        return b[0] < self.edge or b[2] > 1 - self.edge

    def update(self, t, box, fno, frame):
        x1, y1, x2, y2 = (float(v) for v in box)
        W, H = self.W, self.H
        bn = (x1 / W, y1 / H, x2 / W, y2 / H)
        if self.n == 0:
            self.t0, self.first_box, self.start_edge = t, bn, self._edge(bn)
        self.n += 1
        self.t1, self.last_box, self.end_edge = t, bn, self._edge(bn)
        self.hmax = max(self.hmax, (y2 - y1) / H)
        self.y2max = max(self.y2max, y2 / H)
        s = np.sign((x1 + x2) / 2 / W - 0.5)
        if self.prev_sign is not None and s != self.prev_sign and self.tcross is None:
            self.tcross = t
        self.prev_sign = s
        if len(self.times) < 3000:
            self.times.append(t)
        self.boxes.append((fno, (x1, y1, x2, y2)))
        # crop candidate (as students/eval_26se.py --crops: box + 5% per side, native resolution)
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh
        self.max_area = max(self.max_area, area)
        if bh >= self.crop_min_h * H and area > 0.4 * self.max_area:
            X1, Y1 = int(max(0, x1 - self.pad * bw)), int(max(0, y1 - self.pad * bh))
            X2, Y2 = int(min(W, x2 + self.pad * bw)), int(min(H, y2 + self.pad * bh))
            crop = frame[Y1:Y2, X1:X2]
            if crop.size:
                self.cands.append(Cand(t, area, sharpness(crop), crop, self.jpeg))
                if len(self.cands) > self.cap:
                    self.thin()

    def thin(self):
        """Bounded buffer: drop candidates that can no longer be picked (area <= 0.4 x max), then keep the better of
        each pair of neighbours in time (halves the buffer, keeps the time spread)."""
        c = [z for z in self.cands if z.area > 0.4 * self.max_area]
        if len(c) > self.cap:
            sc = rank_pct([z.sharp for z in c]) + rank_pct([z.area for z in c])
            keep = []
            for i in range(0, len(c), 2):
                j = i if i + 1 >= len(c) or sc[i] >= sc[i + 1] else i + 1
                keep.append(c[j])
            c = keep
        self.cands = c

    def tmed(self):
        return float(np.median(self.times)) if len(self.times) < 3000 else (self.t0 + self.t1) / 2

    def meets_fg(self, p):
        return self.hmax > p["fg_h"] and self.y2max > p["fg_y2"] and self.n >= p["min_frames"]


def _near(a, b, tol=0.15):
    ax, bx = (a[0] + a[2]) / 2, (b[0] + b[2]) / 2
    return abs(ax - bx) < max(tol, (a[2] - a[0] + b[2] - b[0]) / 2)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def nested_ratio(a, b):
    """median over common frames of intersection / smaller area; None if < 3 common frames (passes._dedupe)."""
    da = dict(a.boxes)
    r = []
    for f, bb in b.boxes:
        aa = da.get(f)
        if aa is None:
            continue
        ix = max(0.0, min(aa[2], bb[2]) - max(aa[0], bb[0]))
        iy = max(0.0, min(aa[3], bb[3]) - max(aa[1], bb[1]))
        amin = min((aa[2] - aa[0]) * (aa[3] - aa[1]), (bb[2] - bb[0]) * (bb[3] - bb[1]))
        r.append(ix * iy / amin if amin > 0 else 0.0)
    return float(np.median(r)) if len(r) >= 3 else None


# ------------------------------------------------------------------------------------------------ passes
class Pass:
    def __init__(self, pid, first):
        self.pid = pid
        self.members = [first]
        self.open = True
        self.closed_mono = None
        self.t_closed = None
        self.prov = None            # first confident provisional read (dict)
        self.prov_reads = 0
        self.prov_inflight = False
        self.prov_last = -1e18
        self.done = False           # identity finished
        self.selected = []

    def t0(self):
        return min(m.t0 for m in self.members)

    def t1(self):
        return max(m.t1 for m in self.members)

    def last(self):
        return max(self.members, key=lambda m: m.t1)

    def summary(self):
        M = sorted(self.members, key=lambda m: m.t0)
        crossed = [m for m in M if m.tcross is not None]
        tcross = crossed[0].tcross if crossed else float(np.mean([m.tmed() for m in M]))
        t0, t1 = self.t0(), self.t1()
        return dict(t_enter=t0, t_cross=tcross, t_exit=t1, dwell_s=round(t1 - t0, 3),
                    n_frames=int(sum(m.n for m in M)), n_tracks=len(M), tracks=[m.tid for m in M],
                    exited=bool(self.last().end_edge), hmax=round(max(m.hmax for m in M), 4))

    def cands(self):
        return [c for m in self.members for c in m.cands]


class PassLogic:
    def __init__(self, cfg, emit, identity=None, log=print):
        self.cfg, self.emit, self.identity, self.log = cfg, emit, identity, log
        self.p = cfg["passes"]
        self.idc = cfg["identity"]
        self.accs = {}
        self.passes = []        # recent passes (open + the last few closed)
        self.next_pid = 0
        self.now = None
        self.epoch = 0          # source number (tracker ids restart per source)
        self.n_closed = 0

    def new_source(self):
        """the tracker was reset (next file): every current track is gone."""
        for a in self.accs.values():
            a.alive = False
        self.accs = {}
        self.epoch += 1

    def step(self, fr, outputs, alive_ids):
        self.now = fr.t
        H, W = fr.img.shape[:2]
        for tid, box, score in outputs:
            key = (self.epoch, tid)
            a = self.accs.get(key)
            if a is None:
                a = self.accs[key] = TrackAcc(key, W, H, self.cfg)
            a.update(fr.t, box, fr.idx, fr.img)
        for key in [k for k in self.accs if k[1] not in alive_ids]:
            a = self.accs.pop(key)
            a.alive = False
            if not a.qualified:
                a.cands = []
        new = sorted([a for a in self.accs.values() if not a.qualified and a.meets_fg(self.p)], key=lambda a: a.t0)
        for a in new:
            a.qualified = True
            self._assign(a)
        self._close(fr.t)
        if self.identity is not None and self.idc["provisional"]:
            self._provisional(fr.t)

    def _assign(self, r):
        p = self.p
        recent = [q for q in self.passes[-int(p["recent_passes"]):] if q.open]
        for q in reversed(recent):  # nested duplicate of a coexisting member (passes._dedupe)
            for m in q.members:
                if m.alive or m.t1 >= r.t0:
                    nr = nested_ratio(m, r)
                    if nr is not None and nr > p["nested_ratio"]:
                        return self._attach(q, r, "nested")
        for q in reversed(recent):
            last = q.last()
            gap = r.t0 - last.t1
            if gap < p["merge_gap_s"] and not last.end_edge and _near(last.last_box, r.first_box):
                return self._attach(q, r, "a")
            if gap < p["static_gap_s"] and _iou(last.last_box, r.first_box) > 0.2:
                return self._attach(q, r, "b")
            if gap < 0 and _near(last.last_box, r.first_box, 0.05):
                return self._attach(q, r, "c")
        q = Pass(self.next_pid, r)
        self.next_pid += 1
        r.pass_ = q
        self.passes.append(q)
        self.emit(dict(type="pass_start", pass_id=q.pid, t_enter=r.t0, t_video=self.now,
                       box=[round(v, 4) for v in r.first_box], hmax=round(r.hmax, 4)))

    def _attach(self, q, r, rule):
        q.members.append(r)
        r.pass_ = q

    def _close(self, now, force=False):
        p = self.p
        for q in self.passes:
            if not q.open:
                continue
            if not force:
                if any(m.alive for m in q.members):
                    continue
                t1 = q.t1()
                if now - t1 <= p["static_gap_s"]:
                    continue
                if any(a.alive and not a.qualified and a.t0 <= t1 + p["static_gap_s"]
                       and now - a.t0 <= p["pending_max_s"] for a in self.accs.values()):
                    continue
            q.open = False
            q.closed_mono = time.monotonic()
            q.t_closed = now
            q.selected = pick_crops(q.cands(), int(self.idc["max_crops"]))
            summ = q.summary()
            for m in q.members:     # free everything that was not selected
                m.cands = []
                m.times = []
                m.boxes.clear()
            self.n_closed += 1
            ev = dict(type="pass_end", pass_id=q.pid, t_video=now, **summ, n_crops=len(q.selected))
            self.emit(ev)
            if self.identity is not None:
                self.identity.submit_final(q, ev)
            else:
                q.selected = []
        keep = [q for q in self.passes if q.open]
        closed = [q for q in self.passes if not q.open][-int(p["recent_passes"]):]
        self.passes = sorted(keep + closed, key=lambda q: q.pid)

    def _provisional(self, now):
        i = self.idc
        for q in self.passes:
            if (not q.open or q.prov is not None or q.prov_inflight or q.prov_reads >= i["provisional_max_reads"]
                    or now - q.prov_last < i["provisional_every_s"]):
                continue
            c = q.cands()
            unread = [z for z in c if z.res is None]
            if not unread:
                continue
            amax = max(z.area for z in c)
            pool = [z for z in unread if z.area > 0.4 * amax]
            if not pool:
                continue
            sc = rank_pct([z.sharp for z in pool]) + rank_pct([z.area for z in pool])
            take = [pool[j] for j in np.argsort(-sc, kind="stable")[:int(i["provisional_batch"])]]
            q.prov_last = now
            q.prov_reads += len(take)
            q.prov_inflight = True
            self.identity.submit_provisional(q, take)

    def flush(self):
        """end of input: close every pass now (as if the re-attach windows had expired)."""
        for a in self.accs.values():
            a.alive = False
            if not a.qualified:
                a.cands = []
        self.accs = {}
        if self.now is not None:
            self._close(self.now, force=True)

    def open_count(self):
        return sum(q.open for q in self.passes)

    def crop_bytes(self):
        tot = 0
        for a in self.accs.values():
            tot += sum(c.nbytes() for c in a.cands)
        for q in self.passes:
            if not q.open:
                tot += sum(c.nbytes() for c in q.selected)
            else:
                tot += sum(c.nbytes() for m in q.members if not m.alive for c in m.cands)
        return tot
