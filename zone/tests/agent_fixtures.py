"""Builders shared by the agent API tests (test_agent_*.py): the whole zone app with the studio's agent API on it, a
simulated PTZ, and a fake clock. No camera, no network, no real waiting.

For a first-time reader:
* `ZoneApp` is the zone program (zone/app.py). `ZoneAgent` is the agent API added to it (zone/agent/).
* `studio` is a FastAPI `TestClient`: it sends the requests the studio would send ("POST /cameras/cam1/move"),
  straight into the program, in memory. `page` does the same for the laptop's own page.
* The clock is a `FakeClock`: time stands still until a test calls `rig.advance(seconds)`.
* In the real program a background thread calls `agent.tick()` ten times a second (device status, the "resume after
  a pause" countdown...). Here `rig.advance()` calls it, so a test decides exactly when time passes.
"""
import asyncio
from dataclasses import dataclass

import numpy as np
from fastapi.testclient import TestClient

from zone import config as C
from zone.agent.server import create_agent_app
from zone.agent.service import ZoneAgent
from zone.app import ZoneApp, create_web
from zone.clock import FakeClock
from zone.ptz.interface import ERROR, TIMEOUT, CommandResult
from zone.ptz.sim import PtzModel, SimulatedPtz
from zone.source import ZoneFrame
from zone.tests.z2_fixtures import straight_zone

STUDIO_ID = "test-studio"


@dataclass
class Rig:
    zapp: ZoneApp
    agent: ZoneAgent
    ptz: SimulatedPtz
    clock: FakeClock
    studio: TestClient      # requests as the studio sends them (with X-Studio-Id and the token)
    page: TestClient        # requests as the laptop's own page sends them

    def frame(self):
        """Pretend the spotter webcam just delivered a frame (AUTO needs recent frames)."""
        give_frame(self.zapp)

    def advance(self, seconds, step=0.05, frames=True):
        """Let `seconds` of fake time pass in small steps. At each step: a new spotter frame (unless frames=False),
        the agent's background work (`tick`), and the simulated PTZ moves on."""
        n = max(1, int(round(seconds / step)))
        for _ in range(n):
            self.clock.advance(seconds / n)
            if frames:
                self.frame()
            self.agent.tick()
            self.ptz.tick()

    def connect_studio(self):
        """Pretend a studio has its state WebSocket open (the agent only polls the PTZ's position for a studio)."""
        self.agent.subscribers.add(asyncio.Queue())

    def cycle(self):
        """One control cycle of the zone's frame loop, with no cars in view (what the loop does 10 times a second)."""
        t = self.clock.monotonic()
        return self.zapp.runtime.cycle(ZoneFrame(None, t, self.clock.wall(), 0), tracker_out=[])

    def mode(self):
        """The camera's mode as the studio sees it (manual / auto / auto_paused / hold)."""
        return self.studio.get("/cameras/cam1/mode").json()["mode"]

    def director(self):
        """The zone director's mode (STOPPED / PASSIVE / AUTO)."""
        return self.zapp.runtime.director.mode

    def events(self):
        """The contract events the agent has queued for the studio since the last call: [(event, detail), ...]."""
        return [(e["event"], e["detail"]) for e in self.agent.take_events()]


def give_frame(zapp):
    t = zapp.clock.monotonic()
    zapp.latest = (ZoneFrame(np.zeros((8, 8, 3), np.uint8), t, zapp.clock.wall(), 0), None)
    zapp.last_frame_mono = t


def make_rig(tmp_path, zone=None, branch="A", token="", frames=True, sets=(), state_file=None, ptz=None, clock=None):
    """The zone app around a simulated PTZ, with the agent API attached.

    branch "A": the (pretend) bench report says position inquiries work while moving, so the agent reports positions.
    branch "B": no usable inquiries: `position` is null. token: the agent's X-Agent-Token ("" = none).
    frames: give it one spotter frame now (so AUTO can start)."""
    cfg = C.load(None, ["ptz.driver=sim", "log.cycle_log=false", *sets])
    cfg["agent"]["token"] = token
    if state_file:
        cfg["agent"]["state_file"] = str(state_file)
    model = PtzModel(inquiry_mode="always" if branch == "A" else "never")
    report = C.Report(model, branch, "test report", "test.json", {"model": model.to_dict()})
    settings = C.build(cfg, report)
    clock = clock or FakeClock(0.0)
    ptz = ptz or SimulatedPtz(model, clock)
    zone = zone if zone is not None else straight_zone()
    for k in zone.keyframes():                  # the camera "remembers" the keyframe presets of the teach
        ptz.define_preset(k.preset, k.pan, k.tilt, k.zoom)
    zapp = ZoneApp(cfg, settings, ptz, clock, zone, tmp_path / "zone.json", tmp_path / "out", log=lambda m: None,
                   cycle_log=False)
    agent = ZoneAgent(zapp, cfg["agent"], log=lambda m: None).attach()
    if frames:
        give_frame(zapp)
    headers = {"X-Studio-Id": STUDIO_ID, **({"X-Agent-Token": token} if token else {})}
    studio = TestClient(create_agent_app(agent), headers=headers)
    page = TestClient(create_web(zapp))
    return Rig(zapp, agent, ptz, clock, studio, page)


class DeadPtz:
    """A PTZ that has stopped answering (unplugged, network down): every command times out, every inquiry is None.
    `alive = True` brings it back (then it forwards to the simulator it wraps)."""

    def __init__(self, sim):
        self.sim = sim
        self.alive = False

    def __getattr__(self, name):
        attr = getattr(self.sim, name)
        if not callable(attr) or name in ("tick", "define_preset", "is_moving", "true_pan_tilt"):
            return attr

        def call(*args, **kwargs):
            if self.alive:
                return attr(*args, **kwargs)
            if name.startswith("inquire"):
                return None
            return CommandResult(TIMEOUT)
        return call


class CountingPtz:
    """The simulator, counting the position inquiries it is asked (to check who sends what to the camera)."""

    def __init__(self, sim):
        self.sim = sim
        self.inquiries = 0

    def __getattr__(self, name):
        attr = getattr(self.sim, name)
        if not name.startswith("inquire"):
            return attr

        def call(*args, **kwargs):
            self.inquiries += 1
            return attr(*args, **kwargs)
        return call


class RefusingPtz:
    """The simulator, except that it answers every preset store with a VISCA error (it is there, it just refuses)."""

    def __init__(self, sim):
        self.sim = sim

    def __getattr__(self, name):
        if name == "preset_store":
            return lambda n: CommandResult(ERROR, error_code=0x02)
        return getattr(self.sim, name)
