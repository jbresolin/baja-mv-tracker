"""ByteTrack in numpy: a faithful port of ultralytics 8.4's BYTETracker (trackers/byte_tracker.py + bytetrack.yaml),
which produced every offline 26SE result (students/eval_26se.py via `model.track(tracker="bytetrack.yaml")`).

Same state machine, Kalman filter (XYAH, constant velocity), two-stage association (high-score dets with IoU fused
with the score, then low-score dets on plain IoU), unconfirmed tracks (a new track is reported from its second
detection on, except on the very first frame), track_buffer in frames (ultralytics does NOT scale it with the frame
rate: 30 frames = 3 s at 10 fps) and the end-of-frame duplicate removal. The reported box is the Kalman state, as
ultralytics reports it. Only the linear assignment differs: lap.lapjv(cost_limit) is replaced by an exact Hungarian
solve on the thresholded matrix (same optimum; ties may break differently).
"""
import numpy as np

NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3


class KalmanXYAH:
    def __init__(self):
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.Hm = np.eye(4, 8)
        self.wp, self.wv = 1.0 / 20, 1.0 / 160

    def initiate(self, m):
        m = np.asarray(m, dtype=np.float64)
        mean = np.r_[m, np.zeros(4)]
        h = m[3]
        std = [2 * self.wp * h, 2 * self.wp * h, 1e-2, 2 * self.wp * h,
               10 * self.wv * h, 10 * self.wv * h, 1e-5, 10 * self.wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self.wp * h, self.wp * h, 1e-2, self.wp * h, self.wv * h, self.wv * h, 1e-5, self.wv * h]
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + np.diag(np.square(std))
        return mean, cov

    def update(self, mean, cov, m):
        h = mean[3]
        R = np.diag(np.square([self.wp * h, self.wp * h, 1e-1, self.wp * h]))
        pm, pc = mean[:4], cov[:4, :4] + R
        K = np.linalg.solve(pc, cov[:, :4].T).T
        mean = mean + (np.asarray(m) - pm) @ K.T
        cov = cov - K @ pc @ K.T
        return mean, cov


def tlwh_to_xyah(t):
    t = np.asarray(t, dtype=np.float64).copy()
    t[:2] += t[2:] / 2
    t[2] /= t[3]
    return t


class STrack:
    _kf = KalmanXYAH()

    def __init__(self, xyxy, score):
        x1, y1, x2, y2 = xyxy
        self._tlwh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
        self.mean = self.cov = None
        self.is_activated = False
        self.score = float(score)
        self.track_id = 0
        self.state = NEW
        self.frame_id = self.start_frame = 0

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.astype(np.float64).copy()
        r = self.mean[:4].copy()
        r[2] *= r[3]
        r[:2] -= r[2:] / 2
        return r

    @property
    def xyxy(self):
        r = self.tlwh
        r[2:] += r[:2]
        return r

    def predict(self):
        m = self.mean.copy()
        if self.state != TRACKED:
            m[7] = 0
        self.mean, self.cov = self._kf.predict(m, self.cov)

    def activate(self, tid, frame_id):
        self.track_id = tid
        self.mean, self.cov = self._kf.initiate(tlwh_to_xyah(self._tlwh))
        self.state = TRACKED
        self.is_activated = frame_id == 1
        self.frame_id = self.start_frame = frame_id

    def re_activate(self, det, frame_id):
        self.mean, self.cov = self._kf.update(self.mean, self.cov, tlwh_to_xyah(det.tlwh))
        self.state = TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = det.score

    def update(self, det, frame_id):
        self.frame_id = frame_id
        self.mean, self.cov = self._kf.update(self.mean, self.cov, tlwh_to_xyah(det.tlwh))
        self.state = TRACKED
        self.is_activated = True
        self.score = det.score


def iou_matrix(a, b):
    """a (N,4), b (M,4) xyxy -> IoU (N,M) (ultralytics bbox_ioa(iou=True), eps 1e-7)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    iw = (np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])).clip(0)
    ih = (np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])).clip(0)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-7)


def hungarian(cost):
    """Minimum-cost assignment for a rectangular matrix (n <= m after transposition). Returns list of (row, col)."""
    cost = np.asarray(cost, dtype=np.float64)
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape
    INF = float("inf")
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = cost[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            cand = np.where(free, minv[1:], INF)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] > 0]
    return [(c, r) for r, c in pairs] if transposed else pairs


def linear_assignment(cost, thresh):
    """lapjv(extend_cost=True, cost_limit=thresh) equivalent: optimal matching where a pair costs <= thresh, every
    element may stay unmatched at cost thresh / 2 each."""
    n, m = cost.shape
    if cost.size == 0:
        return [], list(range(n)), list(range(m))
    # augmented square problem (n+m): real block, row dummies, col dummies, dummy-dummy 0
    big = 1e6
    C = np.full((n + m, n + m), big)
    C[:n, :m] = np.where(cost <= thresh, cost, big)
    C[:n, m:] = np.where(np.eye(n, dtype=bool), thresh / 2, big)
    C[n:, :m] = np.where(np.eye(m, dtype=bool), thresh / 2, big)
    C[n:, m:] = 0.0
    matches = [(r, c) for r, c in hungarian(C) if r < n and c < m and cost[r, c] <= thresh]
    mr = {r for r, _ in matches}
    mc = {c for _, c in matches}
    return matches, [i for i in range(n) if i not in mr], [j for j in range(m) if j not in mc]


class ByteTracker:
    def __init__(self, track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25, track_buffer=30,
                 match_thresh=0.8, fuse_score=True):
        self.high, self.low, self.new = track_high_thresh, track_low_thresh, new_track_thresh
        self.buffer, self.match, self.fuse = int(track_buffer), match_thresh, fuse_score
        self.reset()

    def reset(self):
        self.tracked, self.lost = [], []
        self.frame_id = 0
        self._next = 1
        self._dead = set()        # ids already removed (ultralytics keeps a 1000-track removed buffer)

    def _dists(self, tracks, dets, fuse):
        d = 1 - iou_matrix([t.xyxy for t in tracks], [x.xyxy for x in dets])
        if fuse and d.size:
            d = 1 - (1 - d) * np.array([x.score for x in dets])[None]
        return d

    def update(self, boxes, scores):
        """boxes (N,4) xyxy, scores (N,) -> list of (track_id, xyxy (4,), score) for tracks reported this frame.
        alive_ids() = ids that may still be reported later (tracked or lost)."""
        self.frame_id += 1
        fid = self.frame_id
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        valid = ((boxes[:, 2] - boxes[:, 0]) > 0) & ((boxes[:, 3] - boxes[:, 1]) > 0)
        hi = valid & (scores >= self.high)
        lo = valid & (scores > self.low) & (scores < self.high)
        dets = [STrack(b, s) for b, s in zip(boxes[hi], scores[hi])]
        dets2 = [STrack(b, s) for b, s in zip(boxes[lo], scores[lo])]
        activated, refind, lost, removed = [], [], [], []
        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]
        ids = {t.track_id for t in tracked}
        pool = tracked + [t for t in self.lost if t.track_id not in ids]
        for t in pool:
            t.predict()
        # 1st association: high-score detections
        m, u_track, u_det = linear_assignment(self._dists(pool, dets, self.fuse), self.match)
        for it, idt in m:
            t, d = pool[it], dets[idt]
            if t.state == TRACKED:
                t.update(d, fid)
                activated.append(t)
            else:
                t.re_activate(d, fid)
                refind.append(t)
        # 2nd association: remaining tracked tracks vs low-score detections
        r_tracked = [pool[i] for i in u_track if pool[i].state == TRACKED]
        if r_tracked and dets2:
            m2, u2, _ = linear_assignment(1 - iou_matrix([t.xyxy for t in r_tracked], [d.xyxy for d in dets2]), 0.5)
            for it, idt in m2:
                t, d = r_tracked[it], dets2[idt]
                if t.state == TRACKED:
                    t.update(d, fid)
                    activated.append(t)
                else:
                    t.re_activate(d, fid)
                    refind.append(t)
        else:
            u2 = list(range(len(r_tracked)))
        for it in u2:
            t = r_tracked[it]
            t.state = LOST
            lost.append(t)
        # unconfirmed tracks vs the leftover high-score detections
        dets = [dets[i] for i in u_det]
        if unconfirmed:
            m3, u_unc, u_det = linear_assignment(self._dists(unconfirmed, dets, self.fuse), 0.7)
            for it, idt in m3:
                unconfirmed[it].update(dets[idt], fid)
                activated.append(unconfirmed[it])
            for it in u_unc:
                unconfirmed[it].state = REMOVED
                removed.append(unconfirmed[it])
        else:
            u_det = list(range(len(dets)))
        for i in u_det:
            d = dets[i]
            if d.score < self.new:
                continue
            d.activate(self._next, fid)
            self._next += 1
            activated.append(d)
        for t in self.lost:
            if fid - t.frame_id > self.buffer:
                t.state = REMOVED
                removed.append(t)
        # merge_track_pools
        tr = [t for t in self.tracked if t.state == TRACKED]
        for group in (activated, refind):
            have = {t.track_id for t in tr}
            tr += [t for t in group if t.track_id not in have]
        trids = {t.track_id for t in tr}
        lst = [t for t in self.lost if t.track_id not in trids] + lost
        # ultralytics filters against the removed buffer *before* adding this frame's removals, so a track that just
        # timed out stays in the lost pool for one more frame (and can still be re-found); kept for fidelity
        lst = [t for t in lst if t.track_id not in self._dead]
        tr, lst = _remove_duplicates(tr, lst)
        self.tracked, self.lost = tr, lst
        self._dead |= {t.track_id for t in removed}
        if len(self._dead) > 5000:
            self._dead = set(sorted(self._dead)[-1000:])
        return [(t.track_id, t.xyxy, t.score) for t in self.tracked if t.is_activated]

    def alive_ids(self):
        return {t.track_id for t in self.tracked} | {t.track_id for t in self.lost}


def _remove_duplicates(a, b, thresh=0.15):
    if not a or not b:
        return a, b
    d = 1 - iou_matrix([t.xyxy for t in a], [t.xyxy for t in b])
    dupa, dupb = set(), set()
    for p, q in zip(*np.where(d < thresh)):
        if a[p].frame_id - a[p].start_frame > b[q].frame_id - b[q].start_frame:
            dupb.add(q)
        else:
            dupa.add(p)
    return [t for i, t in enumerate(a) if i not in dupa], [t for i, t in enumerate(b) if i not in dupb]
