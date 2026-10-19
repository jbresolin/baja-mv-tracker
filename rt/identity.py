"""Identity worker thread: car crop -> number-panel detector (conf 0.1, +5% pad) -> recognizer on each panel box
(+10% context) -> P(roster number) per crop (max over its boxes) -> boxed vote (WP10b) -> roster number.

Jobs: provisional (a few crops while the pass is open; first confident read -> pass_provisional, for the PTZ entry
camera) before final (the pass's <= 30 picked crops at close -> pass_identity). Per-crop results are cached on the
candidate, so crops already read provisionally are not read again. Detection never waits for this thread; when the
queue is long, provisional jobs are skipped first.
"""
import itertools
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .config import vote_rule
from .models import Recognizer, RosterScorer, YoloOnnx, enlarge, plain, vote_boxed


class IdentityWorker(threading.Thread):
    def __init__(self, cfg, roster, teams, emit, log=print):
        super().__init__(daemon=True)
        m, r, i = cfg["models"], cfg["runtime"], cfg["identity"]
        th, spin = int(r["threads_identity"]), bool(r["ort_spinning"])
        self.panel = YoloOnnx(m["panel_detector"], int(m["panel_imgsz"]), th, rect=bool(i["panel_rect"]),
                              spinning=spin, backend=r["id_backend"], device=r["id_device"], log=log)
        self.rec = Recognizer(m["recognizer"], th, spin, backend=r["id_backend"], device=r["id_device"], log=log)
        self.roster, self.teams = roster, teams
        self.scorer = RosterScorer(roster)
        self.rule = vote_rule(cfg)
        self.conf, self.pad = float(i["panel_conf"]), float(i["panel_pad"])
        self.qmax = int(i["queue_max"])
        self.emit, self.log = emit, log
        self.q = queue.PriorityQueue()
        self.seq = itertools.count()
        self.stats = dict(crops=0, boxes=0, panel_s=0.0, recog_s=0.0, final_latency=[], prov_skipped=0,
                          provisional=0, finals=0, max_queue=0)
        self.crops_dir = Path(cfg["output"]["crops_dir"]) if cfg["output"].get("crops_dir") else None
        self.now_fn = lambda: None

    # ------------------------------------------------------------------ queue
    def submit_provisional(self, p, cands):
        if self.q.qsize() >= self.qmax:
            self.stats["prov_skipped"] += 1
            p.prov_inflight = False
            return
        self.q.put((0, next(self.seq), ("prov", p, cands, None)))
        self.stats["max_queue"] = max(self.stats["max_queue"], self.q.qsize())

    def submit_final(self, p, end_event):
        self.q.put((1, next(self.seq), ("final", p, p.selected, end_event)))
        self.stats["max_queue"] = max(self.stats["max_queue"], self.q.qsize())

    def finish(self):
        """after the last pass_end: process everything queued, then stop."""
        self.q.put((9, next(self.seq), None))
        self.join()

    def run(self):
        while True:
            _, _, job = self.q.get()
            if job is None:
                return
            kind, p, cands, ev = job
            try:
                if kind == "prov":
                    self._provisional(p, cands)
                else:
                    self._final(p, cands, ev)
            except Exception as e:  # noqa: BLE001 (never kill the worker on one bad crop)
                self.log("identity error on pass %d: %s: %s" % (p.pid, type(e).__name__, e))
                if kind == "prov":
                    p.prov_inflight = False

    # ------------------------------------------------------------------ reading
    def read(self, c):
        if c.res is not None:
            return c.res
        im = c.image()
        H, W = im.shape[:2]
        t = time.perf_counter()
        b, s, _ = self.panel(im, conf=self.conf, iou=0.7)
        t2 = time.perf_counter()
        bx = [enlarge(x, self.pad, W, H) for x in b.tolist()]
        if bx:
            P, _ = self.scorer(self.rec([plain(im, x) for x in bx]))
            P = P.max(0)
        else:
            P = np.zeros(len(self.roster))
        t3 = time.perf_counter()
        st = self.stats
        st["crops"] += 1
        st["boxes"] += len(bx)
        st["panel_s"] += t2 - t
        st["recog_s"] += t3 - t2
        c.res = (P, len(bx) > 0, len(bx))
        return c.res

    def vote(self, cands):
        res = [c.res for c in cands if c.res is not None]
        if not res:
            return None
        j, score, s2, nb, sup, s = vote_boxed([r[0] for r in res], [r[1] for r in res], score=self.rule["score"])
        out = dict(n_read=len(res), n_boxed=nb, n_boxes=int(sum(r[2] for r in res)), support=sup)
        if j < 0:
            return dict(out, number=None, confidence=0.0, auto_confirm=False, top3=[])
        o = np.argsort(-s, kind="stable")[:3]
        top3 = [dict(number=self.roster[k], share=round(float(s[k]), 4)) for k in o if s[k] > 0]
        auto = bool(score >= self.rule["tau"] and sup >= self.rule["min_support"])
        return dict(out, number=self.roster[j], team=self.teams.get(self.roster[j], ""), confidence=round(score, 4),
                    auto_confirm=auto, top3=top3)

    def _provisional(self, p, cands):
        for c in cands:
            self.read(c)
        p.prov_inflight = False
        if p.prov is not None or p.done:
            return
        v = self.vote([c for m in p.members for c in m.cands] + list(p.selected))
        self.stats["provisional"] += 1
        if v and v["auto_confirm"]:
            p.prov = v
            self.emit(dict(type="pass_provisional", pass_id=p.pid, t_video=self.now_fn(),
                           number=v["number"], confidence=v["confidence"], support=v["support"], n_read=v["n_read"],
                           top3=v["top3"]))

    def _final(self, p, cands, end_ev):
        for c in cands:
            self.read(c)
        v = self.vote(cands) or dict(n_read=0, n_boxed=0, n_boxes=0, support=0, number=None, confidence=0.0,
                                     auto_confirm=False, top3=[])
        lat = time.monotonic() - p.closed_mono
        self.stats["final_latency"].append(lat)
        self.stats["finals"] += 1
        if self.crops_dir is not None:
            d = self.crops_dir / ("pass_%05d" % p.pid)
            d.mkdir(parents=True, exist_ok=True)
            for k, c in enumerate(cands):
                cv2.imwrite(str(d / ("%02d_%.2f.jpg" % (k, c.t))), c.image())
        rec = dict(type="pass_identity", pass_id=p.pid, t_video=self.now_fn(),
                   **{k: end_ev[k] for k in ("t_enter", "t_cross", "t_exit", "dwell_s", "n_frames", "n_tracks",
                                             "exited", "hmax")},
                   n_crops=len(cands), **v,
                   provisional_number=p.prov["number"] if p.prov else None,
                   identity_latency_s=round(lat, 3))
        p.done = True
        p.selected = []  # free the crops
        self.emit(rec)
