"""Spotter frames, through rt/'s capture code (imported, not modified: WP-L1 is improving it in parallel; when it
merges, this wrapper picks the new capture up unchanged as long as `rt.capture.open_source` keeps its signature).

Each frame comes out as a `ZoneFrame` with two timestamps:

* `t`: the grab time on the **monotonic** clock (what the filters and the controller use: never jumps),
* `t_wall`: the grab time on the wall clock (epoch seconds, what the logs and recordings are matched on).

For live sources rt stamps frames with time.time() at arrival; `t_cap` is time.monotonic() at the same moment. Arrival
is later than the moment the picture was taken: the webcam exposes, compresses (MJPG), sends over USB and the driver
hands the frame over, typically ~0.1 s in total. That **capture latency** (`rt.capture.live_latency_s`, the same
setting rt uses) is subtracted here from both times of a live frame, so `t` is when the car really was where the
frame shows it. rt already subtracts it from the wall time; we subtract it from the monotonic time too, which is what
progress and the controller's lookahead run on. For a video file played in real time (teaching on a recording),
nothing is subtracted (the file has no capture latency). Files can loop (`loop_file`), so a short
recording can be taught on for as long as needed.
"""
import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class ZoneFrame:
    img: np.ndarray
    t: float        # monotonic seconds at grab
    t_wall: float   # epoch seconds at grab (video PC clock, incl. rt clock.offset_s)
    idx: int


def grab_time(t_arrival_mono, live, capture_latency_s):
    """Monotonic time the picture was taken: the arrival time minus the capture latency, for live sources only."""
    return float(t_arrival_mono) - (float(capture_latency_s) if live else 0.0)


class SpotterSource:
    """Iterate `frames()` until `stop()`. Re-opens a file at its end when loop_file is on."""

    def __init__(self, cfg, rtcfg, log=print):
        self.cfg, self.rtcfg, self.log = cfg, rtcfg, log
        self.source = str(cfg["spotter"]["source"])
        self.loop = bool(cfg["spotter"]["loop_file"])
        self.realtime = bool(cfg["spotter"]["realtime_file"])
        self.stop_ev = threading.Event()
        self.cap = None
        self.k = 0
        self.description = ""
        self.live = None
        self.size = None
        self.capture_latency_s = float(rtcfg["capture"].get("live_latency_s", 0.0))

    def frames(self):
        from rt.capture import open_source
        while not self.stop_ev.is_set():
            self.cap = open_source(self.rtcfg, self.source, realtime=self.realtime, log=self.log)
            self.description = self.cap.describe()
            self.live = self.cap.live
            self.log(self.description)
            self.cap.start()
            for fr in self.cap.frames():
                if self.stop_ev.is_set():
                    break
                self.size = (fr.img.shape[1], fr.img.shape[0])
                # fr.t (wall) already has the capture latency taken off by rt; fr.t_cap (monotonic) doesn't
                yield ZoneFrame(fr.img, grab_time(fr.t_cap, self.cap.live, self.capture_latency_s), fr.t, self.k)
                self.k += 1
            if self.cap.live or not self.loop:
                break
        if self.cap:
            self.cap.stop()

    @property
    def dropped(self):
        return self.cap.dropped if self.cap else 0

    def stop(self):
        self.stop_ev.set()
        if self.cap:
            self.cap.stop()
