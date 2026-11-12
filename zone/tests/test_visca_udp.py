"""The driver over real UDP sockets, against the fake camera served on 127.0.0.1 in this process.

These prove the socket code (UdpTransport, FakeViscaUdpServer) works end to end. Only exchanges that get an answer
are tested here: a timeout over a real socket would mean really waiting, and timeouts are already covered with
the fake clock in test_visca_driver.py.
"""
import pytest

from zone.clock import FakeClock, SystemClock
from zone.ptz import visca_packets as packets
from zone.ptz.fake_camera import FakeViscaCamera, FakeViscaUdpServer
from zone.ptz.interface import ACKED, COMPLETED, ERROR
from zone.ptz.sim import PtzModel, SimulatedPtz
from zone.ptz.visca import ViscaPtz


@pytest.fixture
def served():
    # The simulator gets a fake clock that nobody advances: the camera stays put, which is all these tests need.
    sim = SimulatedPtz(PtzModel(), FakeClock())
    camera = FakeViscaCamera(sim)
    with FakeViscaUdpServer(camera) as server:
        host, port = server.address
        # A generous timeout: replies on loopback take well under a millisecond, but CI machines can be slow.
        ptz = ViscaPtz(host, port, timeout=2.0, clock=SystemClock(), watchdog=False)
        try:
            yield ptz, camera, sim
        finally:
            ptz.close()


def test_reset_and_commands_over_udp(served):
    ptz, camera, _ = served
    assert ptz.sequence_reset_status == ACKED
    assert ptz.pan_tilt_velocity(4, -2).status == ACKED
    assert ptz.stop().status == ACKED
    assert [name for name, _ in camera.received] == ["sequence_reset", "pan_tilt_drive", "pan_tilt_drive"]


def test_completion_over_udp(served):
    ptz, _, _ = served
    assert ptz.command(packets.home(), wait_for_completion=True).status == COMPLETED


def test_error_over_udp(served):
    ptz, camera, _ = served
    camera.behaviour["zoom_drive"] = "error"
    result = ptz.zoom_velocity(3)
    assert result.status == ERROR and result.error_code == 0x02


def test_inquiries_over_udp(served):
    ptz, _, sim = served
    sim.pan.position = -321.0
    sim.tilt.position = 45.0
    assert ptz.inquire_pan_tilt() == (-321, 45)
    assert ptz.inquire_zoom() == 0
