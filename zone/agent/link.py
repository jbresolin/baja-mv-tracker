"""`ZoneDirectorLink`: what the contract's mode rules (modes.py) need from the running zone app.

Kept apart from modes.py so the rules can be tested with a fake link, and apart from app.py so the app stays the
zone's (the agent is an add-on to it).
"""
from typing import Callable, Optional

from zone import director as D


class ZoneDirectorLink:
    def __init__(self, zapp, device_offline: Callable[[], bool], frames_max_age_s: float = 3.0) -> None:
        """zapp: the ZoneApp. device_offline(): is the PTZ known not to answer? frames_max_age_s: spotter frames
        older than this mean "no frames" (the webcam is gone)."""
        self.zapp = zapp
        self.device_offline = device_offline
        self.frames_max_age_s = float(frames_max_age_s)

    def director_mode(self) -> str:
        return self.zapp.runtime.director.mode

    def frame_age_s(self) -> Optional[float]:
        """Seconds since the last spotter frame, or None if no frame ever came."""
        if self.zapp.latest is None:
            return None
        return self.zapp.clock.monotonic() - self.zapp.last_frame_mono

    def frames_ok(self) -> bool:
        age = self.frame_age_s()
        return age is not None and age <= self.frames_max_age_s

    def tracker_ready(self) -> tuple[bool, str]:
        """AUTO needs a taught zone (2+ keyframes), spotter frames, and a PTZ that answers. The sentences go to the
        studio's operators (the `detail` of a 409 tracker_not_ready)."""
        rt = self.zapp.runtime
        if rt.path is None or rt.zone is None or rt.zone.entry_preset is None:
            return False, ("no zone taught on the zone laptop: teach at least 2 keyframes on its page "
                           "(http://<laptop>:8090/)")
        age = self.frame_age_s()
        if age is None:
            return False, "no spotter frames yet on the zone laptop (is the spotter webcam connected?)"
        if age > self.frames_max_age_s:
            return False, "no spotter frame for %.0f s on the zone laptop (spotter webcam unplugged?)" % age
        if self.device_offline():
            return False, "the zone's PTZ is not answering (network or power)"
        return True, ""

    def handover_pending(self) -> bool:
        return self.zapp.runtime.handover_requested is not None

    def set_director(self, mode: str, why: str, stop_camera: Optional[bool]) -> tuple[bool, str]:
        """stop_camera None: the director's usual rules (runtime.set_mode, as the page's buttons always did).
        True / False: a hand-over to the studio (runtime.hand_over), stopping the camera now or sending it nothing."""
        mode = str(mode).upper()
        rt = self.zapp.runtime
        if stop_camera is None or mode == D.AUTO:
            ok, msg = rt.set_mode(mode, why=why)
        else:
            ok, msg = rt.hand_over(mode, why, stop_camera)
        if ok:
            self.zapp.teach_after_mode(mode)
        return ok, msg
