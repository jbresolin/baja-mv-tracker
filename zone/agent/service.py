"""`ZoneAgent`: the agent's state that the routes (server.py) serve: who it is, its camera, its command log, the
contract events, and the state pushed to every studio WebSocket.

It is built around a running `ZoneApp` (zone/app.py) and plugs into it with `attach()`:

* the page's PARADO / PASSIVO / AUTO and PARAR go through the agent's mode rules (`local_mode`, `local_stop`);
* the director's lock / release events become tracker_acquired / tracker_lost;
* `ZoneApp.state()` shows the studio's mode and who holds the camera on the laptop page.

Threads: the routes run in uvicorn's threadpool; `tick()` runs in a background thread (`run_ticker`), several times
a second, because it may wait for the PTZ; the WebSocket pushes run on the agent server's event loop and never wait
for the PTZ (they read cached values). Events are collected under a lock and handed to the event loop, which sends
them to every subscriber.
"""
import asyncio
import socket
import threading
from collections import deque
from typing import Optional

from zone.agent.camera import LOCAL_SOURCE, ZoneCamera, iso_now
from zone.agent.contract import CONTRACT_VERSION, DEGRADED, LOG_LENGTH, MODE_CHANGED, OK, UNKNOWN_CAMERA, AgentError
from zone.agent.link import ZoneDirectorLink
from zone.agent.modes import ModeArbiter
from zone.agent.presets import AgentStateFile, PresetBook, state_path_for
from zone import config as C

SOFTWARE_VERSION = "baja-mv-tracker zone 1.0"
TICK_PERIOD_S = 0.1


class ZoneAgent:
    def __init__(self, zapp, agent_cfg: dict, log=print) -> None:
        self.zapp, self.clock = zapp, zapp.clock
        self.cfg = agent_cfg
        self.print = log
        host = socket.gethostname()
        self.token = str(agent_cfg.get("token") or "")
        self.agent_id = str(agent_cfg.get("agent_id") or host)
        self.name = str(agent_cfg.get("name") or "Zone laptop %s" % host)
        self.camera_id = str(agent_cfg.get("camera_id") or "cam1")
        self.started_mono = self.clock.monotonic()
        self._events: list[dict] = []
        self._events_lock = threading.Lock()
        self._log: deque = deque(maxlen=LOG_LENGTH)
        self._log_lock = threading.Lock()
        # the WebSocket side (set by the server's lifespan, on its event loop)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.wake: Optional[asyncio.Event] = None
        self.subscribers: set = set()

        state_path = agent_cfg.get("state_file") or state_path_for(zapp.zone_path)
        self.state_file = AgentStateFile(C.resolve_path(state_path), log=log)
        self.presets = PresetBook(self.state_file, lambda: [k.preset for k in zapp.teach.zone.keyframes()],
                                  int(agent_cfg.get("preset_max", 127)))
        self.camera = ZoneCamera(zapp, self.camera_id, str(agent_cfg.get("camera_label") or "Zone PTZ"),
                                 self.presets, lambda event, detail: self.emit(self.camera_id, event, detail),
                                 float(agent_cfg.get("position_hz", 2.0)))
        self.link = ZoneDirectorLink(zapp, lambda: not self.camera.online(),
                                     float(agent_cfg.get("frames_max_age_s", 3.0)))
        mode, notice = ModeArbiter.restored(self.state_file.last_mode, iso_now(self.clock))
        if notice:
            log("agent: the last mode was %s before this restart; back in %s (the zone always restarts in STOPPED)"
                % (self.state_file.last_mode, mode))
        self.arbiter = ModeArbiter(self.link, self.clock, lambda event, detail: self.emit(self.camera_id, event, detail),
                                   int(agent_cfg.get("resume_after_ms", 3000)), mode, notice,
                                   on_mode=self.state_file.set_last_mode)
        self.state_file.set_last_mode(mode)
        self.camera.arbiter = self.arbiter
        self.camera.studio_connected = lambda: bool(self.subscribers)

    # ------------------------------------------------------------------------------------------- plugging in
    def attach(self) -> "ZoneAgent":
        """Plug into the zone app: its page's mode buttons, PARAR and state, and its director events."""
        self.zapp.agent = self
        self.zapp.event_listeners.append(self.camera.on_zone_event)
        return self

    def run_ticker(self, stop_event: threading.Event) -> None:
        """Background thread: `tick()` until the app stops."""
        while not stop_event.wait(TICK_PERIOD_S):
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 (the agent must never take the zone down)
                self.print("agent tick failed: %s: %s" % (type(e).__name__, e))

    def tick(self) -> None:
        """Device status, position, presets; then the mode rules' clock (auto_paused -> auto)."""
        self.camera.tick()
        self.arbiter.tick()

    # ------------------------------------------------------------------------------------------- the laptop page
    def local_mode(self, director_mode: str) -> tuple[bool, str]:
        ok, msg = self.arbiter.local_mode(director_mode)
        self.record(self.camera_id, LOCAL_SOURCE, "set_mode", {"director": str(director_mode).upper()},
                    "ok" if ok else "error: refused")
        if ok:
            self.camera._commanded_by(LOCAL_SOURCE)
        return ok, msg

    def local_release_hold(self) -> tuple[bool, str]:
        """The page's "liberar HOLD" (confirmed on the page): see ModeArbiter.local_release_hold. Logged."""
        holder = self.arbiter.holder
        ok, msg = self.arbiter.local_release_hold()
        self.record(self.camera_id, LOCAL_SOURCE, "release_hold", {"holder": holder}, "ok" if ok else "error: refused")
        if ok:
            self.camera._commanded_by(LOCAL_SOURCE)
            self.print("agent: HOLD of %s released on the laptop page" % holder)
        return ok, msg

    def local_stop(self) -> None:
        self.camera.local_stop()
        self.arbiter.local_stop()
        self.record(self.camera_id, LOCAL_SOURCE, "stop", {}, "ok")

    def page_state(self) -> dict:
        """What the laptop page shows about the studio."""
        a = self.arbiter
        return {"enabled": True, "error": self.zapp.agent_error, "port": int(self.cfg.get("port", 8890)),
                "mode": a.mode, "holder": a.holder,
                "resume_in_s": None if a.resume_in_s() is None else round(a.resume_in_s(), 1),
                "studio_moving": self.camera.moving() and (self.camera.last_command_source or "").startswith("studio"),
                "last_command_source": self.camera.last_command_source, "studios": len(self.subscribers)}

    # ------------------------------------------------------------------------------------------- agent level
    def camera_for(self, camera_id: str) -> ZoneCamera:
        if camera_id != self.camera_id:
            raise AgentError(UNKNOWN_CAMERA, "no camera %r; this agent has %s" % (camera_id, self.camera_id))
        return self.camera

    def info(self) -> dict:
        return {"agent_id": self.agent_id, "name": self.name, "contract_version": CONTRACT_VERSION,
                "software_version": SOFTWARE_VERSION, "capabilities": [], "cameras": [self.camera.info()]}

    def health(self) -> dict:
        ok = self.camera.online()
        return {"status": OK if ok else DEGRADED, "detail": "" if ok else "offline: %s" % self.camera_id,
                "uptime_s": round(self.clock.monotonic() - self.started_mono, 3),
                "cameras": {self.camera_id: OK if ok else "offline"}}

    # ------------------------------------------------------------------------------------------- command log
    def record(self, module: str, source: str, command: str, args: dict, result: str) -> None:
        with self._log_lock:
            self._log.append({"ts": iso_now(self.clock), "module": module, "source": source, "command": command,
                              "args": args, "result": result})

    def log_tail(self, limit: int) -> list:
        with self._log_lock:
            return list(reversed(self._log))[:limit]

    # ------------------------------------------------------------------------------------------- events and state
    def emit(self, camera_id: Optional[str], event: str, detail: dict) -> None:
        """Queue a contract event for every connected studio (sent by the event loop at once, see `notify`)."""
        message = {"type": "event", "ts": iso_now(self.clock), "camera_id": camera_id, "event": event,
                   "detail": detail}
        with self._events_lock:
            self._events.append(message)
            del self._events[:-200]
        self.zapp._on_event(dict(type="agent", event=event, msg=_describe(event, detail)))
        self.notify()

    def take_events(self) -> list:
        with self._events_lock:
            events, self._events = self._events, []
        return events

    def notify(self) -> None:
        """Wake the push loop (from any thread), so events and a mode change go out now, not at the next push."""
        loop, wake = self.loop, self.wake
        if loop is not None and wake is not None:
            try:
                loop.call_soon_threadsafe(wake.set)
            except RuntimeError:      # the loop is closed (shutting down)
                pass

    def state_message(self) -> dict:
        return {"type": "state", "ts": iso_now(self.clock), "cameras": {self.camera_id: self.camera.state()}}

    def greeting(self) -> list:
        """What a newly accepted WebSocket gets at once: the state, then the restore notice if there is one (contract
        §9: after a restart that could not restore auto, every studio that connects is told, until the mode next
        changes or a studio sets it)."""
        messages = [self.state_message()]
        notice = self.arbiter.restore_notice
        if notice is not None:
            ts, detail = notice
            messages.append({"type": "event", "ts": ts, "camera_id": self.camera_id, "event": MODE_CHANGED,
                             "detail": dict(detail)})
        return messages


def _describe(event: str, detail: dict) -> str:
    """One line for the laptop page's event list."""
    if event == MODE_CHANGED:
        return "estúdio: modo %s → %s (%s)" % (detail.get("from"), detail.get("to"), detail.get("reason"))
    parts = " ".join("%s=%s" % (k, v) for k, v in detail.items())
    return "estúdio: %s %s" % (event, parts)
