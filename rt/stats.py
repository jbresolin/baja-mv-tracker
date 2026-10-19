"""Run statistics: CPU / RAM sampling (psutil, optional) and the end-of-run summary written to <out>.stats.json.
CPU % is in "percent of one logical CPU" (100 = one core busy), for this process and for the ffmpeg decoder child."""
import threading
import time

import numpy as np


class Monitor(threading.Thread):
    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        import psutil
        self.psutil = psutil
        self.p = psutil.Process()
        self.period = period
        self.samples = []
        self.child = None
        self._halt = threading.Event()
        self.extra = dict(open_passes=0, crop_mb=0.0, idq=0)
        self.peak = dict(open_passes=0, crop_mb=0.0, idq=0)
        self.cpu0 = self.p.cpu_times()
        self.t0 = time.time()
        self._cp = None

    def note(self, **kw):
        for k, v in kw.items():
            if v is None:
                continue
            self.extra[k] = v
            self.peak[k] = max(self.peak[k], v)

    def run(self):
        self.p.cpu_percent(None)
        self.psutil.cpu_percent(None)
        while not self._halt.wait(self.period):
            cp = None
            try:
                proc = getattr(self.child, "proc", None)
                if proc is not None and proc.poll() is None:
                    if self._cp is None or self._cp.pid != proc.pid:
                        self._cp = self.psutil.Process(proc.pid)
                        self._cp.cpu_percent(None)
                    else:
                        cp = self._cp.cpu_percent(None)
            except Exception:  # noqa: BLE001
                cp = None
            mi = self.p.memory_info()
            self.samples.append(dict(t=time.time(), cpu=self.p.cpu_percent(None), ffmpeg=cp, rss=mi.rss / 2**20,
                                     sys=self.psutil.cpu_percent(None), **self.extra))

    def stop(self):
        self._halt.set()


def _pct(v, q):
    v = [x for x in v if x is not None]
    return round(float(np.percentile(v, q)), 2) if v else None


def _mean(v):
    v = [x for x in v if x is not None]
    return round(float(np.mean(v)), 2) if v else None


def summary(a, cfg, det, idw, mon, n_frames, dropped, tdet, tpost, lags, wall, t_flush, vid_t0, vid_t1, counts):
    vs = (vid_t1 - vid_t0) if vid_t0 is not None else 0.0
    ms = lambda v, q: round(1000 * float(np.percentile(v, q)), 2) if len(v) else None  # noqa: E731
    st = dict(source=a.source, start=a.start, duration=a.duration, realtime=a.realtime,
              car_detector=cfg["models"]["car_detector"], det_desc=det.desc,
              threads_detector=cfg["runtime"]["threads_detector"], threads_identity=cfg["runtime"]["threads_identity"],
              affinity=cfg["runtime"].get("affinity"), frames=n_frames, dropped=dropped,
              drop_rate=round(dropped / max(1, n_frames + dropped), 4), video_s=round(vs, 1), wall_s=round(wall, 1),
              speed_x=round(vs / max(wall, 1e-9), 2),
              det_ms_p50=ms(tdet, 50), det_ms_p90=ms(tdet, 90), det_ms_max=ms(tdet, 100),
              track_pass_ms_p50=ms(tpost, 50), track_pass_ms_p90=ms(tpost, 90), track_pass_ms_max=ms(tpost, 100),
              lag_ms_p50=ms(lags, 50), lag_ms_p95=ms(lags, 95), lag_ms_max=ms(lags, 100),
              passes=counts.get("pass_end", 0), events=counts, flush_after_s=round(t_flush, 1))
    if idw is not None:
        s = idw.stats
        lat = s["final_latency"]
        st["identity"] = dict(crops=s["crops"], boxes=s["boxes"],
                              panel_ms_per_crop=round(1000 * s["panel_s"] / max(1, s["crops"]), 2),
                              recog_ms_per_crop=round(1000 * s["recog_s"] / max(1, s["crops"]), 2),
                              recog_ms_per_box=round(1000 * s["recog_s"] / max(1, s["boxes"]), 2),
                              finals=s["finals"], provisional_jobs=s["provisional"],
                              provisional_skipped=s["prov_skipped"], max_queue=s["max_queue"],
                              latency_s_p50=round(float(np.median(lat)), 3) if lat else None,
                              latency_s_p95=round(float(np.percentile(lat, 95)), 3) if lat else None,
                              latency_s_max=round(float(max(lat)), 3) if lat else None)
    if mon is not None and mon.samples:
        S = mon.samples
        cpu_t = mon.p.cpu_times()
        busy = (cpu_t.user + cpu_t.system) - (mon.cpu0.user + mon.cpu0.system)
        st["cpu"] = dict(proc_mean=_mean([x["cpu"] for x in S]), proc_p95=_pct([x["cpu"] for x in S], 95),
                         proc_cpu_s_per_wall_s=round(100 * busy / max(1e-9, time.time() - mon.t0), 1),
                         ffmpeg_mean=_mean([x["ffmpeg"] for x in S]), ffmpeg_p95=_pct([x["ffmpeg"] for x in S], 95),
                         system_mean=_mean([x["sys"] for x in S]), logical_cpus=mon.psutil.cpu_count())
        st["ram_mb"] = dict(mean=_mean([x["rss"] for x in S]), peak=round(max(x["rss"] for x in S), 1))
        pk = mon.p.memory_info()
        st["ram_mb"]["peak_wset"] = round(getattr(pk, "peak_wset", pk.rss) / 2**20, 1)
        st["peak"] = dict(mon.peak)
    else:
        st["cpu"] = None
    return st
