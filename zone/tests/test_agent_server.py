"""Tests for the agent API's server side (zone/agent/server.py, zone/app.py start_agent): a studio that stops
reading, a studio that goes silent, the stop sent when the zone starts, and an agent API that can't start.

For a first-time reader: a "subscriber queue" holds the messages waiting to go to one studio over its WebSocket. If
the studio stops reading (its PC hung, the cable was pulled without the socket closing), nothing may pile up forever.
"""
import asyncio
import socket
import time

import pytest

pytest.importorskip("fastapi")

from starlette.websockets import WebSocketDisconnect  # noqa: E402

from zone.agent import server  # noqa: E402
from zone.app import agent_server_problem, start_agent, stop_at_startup  # noqa: E402
from zone.clock import FakeClock  # noqa: E402
from zone.ptz.sim import SimulatedPtz  # noqa: E402
from zone.tests.agent_fixtures import make_rig  # noqa: E402


class FakeAgent:
    def __init__(self, queues):
        self.subscribers = set(queues)


def test_a_studio_that_does_not_read_gets_at_most_so_many_state_pushes_but_every_event():
    reading, stuck = asyncio.Queue(), asyncio.Queue()
    agent = FakeAgent([stuck])
    for _ in range(100):                                 # 25 s of state pushes nobody reads
        server._broadcast(agent, {"type": "state"})
    assert stuck.qsize() == server.MAX_QUEUED_STATES
    server._broadcast(agent, {"type": "event", "event": "mode_changed"})
    assert stuck.qsize() == server.MAX_QUEUED_STATES + 1  # an event is never dropped
    agent.subscribers.add(reading)
    server._broadcast(agent, {"type": "state"})
    assert reading.qsize() == 1                          # a studio that reads gets everything


def test_a_silent_studio_is_disconnected(tmp_path, monkeypatch):
    # The studio pings every 2 s. One that says nothing for SILENT_STUDIO_CLOSE_S (10 s; 0.3 s here) is gone:
    # its socket is closed. Real time, but short.
    monkeypatch.setattr(server, "SILENT_STUDIO_CLOSE_S", 0.3)
    rig = make_rig(tmp_path)
    with rig.studio as client, client.websocket_connect("/agent/ws") as ws:
        started = time.monotonic()
        with pytest.raises(WebSocketDisconnect):
            for _ in range(100):
                ws.receive_json()
        assert time.monotonic() - started < 3.0


def test_a_studio_that_pings_stays_connected(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "SILENT_STUDIO_CLOSE_S", 0.6)
    rig = make_rig(tmp_path)
    with rig.studio as client, client.websocket_connect("/agent/ws") as ws:
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:               # 1.5 s, pinging as the studio does (faster here)
            ws.send_json({"type": "ping"})
            message = ws.receive_json()
            while message != {"type": "pong"}:
                message = ws.receive_json()
        ws.send_json({"type": "ping"})                   # still connected: it answers
        while ws.receive_json() != {"type": "pong"}:
            pass


def test_the_zone_stops_the_camera_when_it_starts():
    # A crash kills the dead-man thread with the process, so the camera may still be moving when the zone starts
    # again: the first thing sent is a stop (pan/tilt and zoom).
    clock = FakeClock()
    ptz = SimulatedPtz(clock=clock)
    ptz.pan_tilt_velocity(12, 0)                        # left moving by the process that died
    stop_at_startup(ptz)
    assert [name for _, name, _ in ptz.commands][-2:] == ["stop", "zoom_stop"]
    clock.advance(1.0)
    assert not ptz.is_moving()


def test_a_ptz_that_fails_at_startup_does_not_stop_the_start():
    class Broken:
        def stop(self):
            raise OSError("no route to host")

        def zoom_stop(self):
            raise OSError("no route to host")
    said = []
    stop_at_startup(Broken(), said.append)
    assert len(said) == 2 and "no route to host" in said[0]


def test_missing_server_libraries_are_reported():
    def only(*present):
        return lambda name: object() if name in present else None
    assert agent_server_problem(only("uvicorn", "websockets")) is None
    assert agent_server_problem(only("uvicorn", "wsproto")) is None
    assert "websockets" in agent_server_problem(only("uvicorn"))
    assert "uvicorn" in agent_server_problem(only("websockets"))


def test_a_port_already_taken_shows_on_the_page(tmp_path):
    # Another program holds the agent's port: the zone keeps running, and the page says the studio's API is down.
    pytest.importorskip("uvicorn")
    rig = make_rig(tmp_path)
    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen()
    port = taken.getsockname()[1]
    cfg = dict(rig.zapp.cfg, agent=dict(rig.zapp.cfg["agent"], host="127.0.0.1", port=port))
    try:
        start_agent(rig.zapp, cfg)
        deadline = time.monotonic() + 10
        while rig.zapp.agent_error is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert rig.zapp.agent_error and str(port) in rig.zapp.agent_error
        assert rig.page.get("/api/state").json()["agent"]["error"] == rig.zapp.agent_error
    finally:
        rig.zapp.stop_ev.set()
        taken.close()
