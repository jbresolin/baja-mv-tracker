"""Tests of the VISCA driver against the fake camera, with a fake clock.

Setup used by most tests (see the `rig` fixture):

    FakeClock ── SimulatedPtz (the camera's motors)
                     │
               FakeViscaCamera (speaks VISCA, can misbehave on demand)
                     │
               FakeTransport (in-memory "network": waiting just moves the fake clock)
                     │
               ViscaPtz (the driver under test)

No real time passes: when the driver waits 200 ms for a reply that never comes, the fake clock jumps 200 ms.
"""
import json
from dataclasses import dataclass

import pytest

from zone.clock import FakeClock
from zone.ptz import visca_packets as packets
from zone.ptz.fake_camera import FakeTransport, FakeViscaCamera
from zone.ptz.interface import ACKED, ANSWERED, COMPLETED, ERROR, TIMEOUT
from zone.ptz.sim import PtzModel, SimulatedPtz
from zone.ptz.visca import ViscaPtz


@dataclass
class Rig:
    clock: FakeClock
    sim: SimulatedPtz
    camera: FakeViscaCamera
    transport: FakeTransport
    ptz: ViscaPtz


def make_rig(model: PtzModel | None = None, log_path: str | None = None, **camera_options) -> Rig:
    clock = FakeClock()
    sim = SimulatedPtz(model or PtzModel(), clock)
    camera = FakeViscaCamera(sim)
    for name, value in camera_options.items():
        setattr(camera, name, value)
    transport = FakeTransport(camera, clock)
    # watchdog=False: the tests call check_deadlines() themselves, at fake times they choose.
    ptz = ViscaPtz("camera", clock=clock, transport=transport, watchdog=False, log_path=log_path)
    return Rig(clock, sim, camera, transport, ptz)


@pytest.fixture
def rig() -> Rig:
    return make_rig()


def sent_payloads(rig: Rig) -> list[bytes]:
    """The VISCA payloads the driver sent (headers stripped), in order."""
    return [d[8:] for d in rig.transport.sent]


# --- connecting and sequence numbers ------------------------------------------------------------------------------


def test_connect_sends_a_sequence_reset_first(rig):
    assert rig.transport.sent[0] == packets.sequence_reset()
    assert rig.ptz.sequence_reset_status == ACKED


def test_sequence_numbers_count_up_from_one_after_the_reset(rig):
    first = rig.ptz.stop()
    second = rig.ptz.stop()
    assert (first.sequence, second.sequence) == (1, 2)
    headers = [int.from_bytes(d[4:8], "big") for d in rig.transport.sent[1:]]
    assert headers == [1, 2]


def test_a_camera_that_ignores_the_reset_is_still_usable():
    rig = make_rig(behaviour={"sequence_reset": "silent"})
    assert rig.ptz.sequence_reset_status == TIMEOUT
    assert rig.ptz.stop().status == ACKED


def test_sequence_error_from_the_camera_triggers_a_new_reset(rig):
    # The camera complains (control reply 0F 01) that our sequence number is wrong.
    rig.transport.camera = _ControlErrorOnce(rig.camera)
    result = rig.ptz.stop()
    assert result.status == ERROR and result.error_code == 0x0F01
    rig.transport.camera = rig.camera
    resets_before = rig.camera.reset_count
    assert rig.ptz.stop().status == ACKED
    assert rig.camera.reset_count == resets_before + 1  # reset sent before the next command


class _ControlErrorOnce:
    """Answers the next request with a control-level 'sequence number' error."""

    def __init__(self, camera: FakeViscaCamera) -> None:
        self.camera = camera

    def handle(self, datagram: bytes) -> list[bytes]:
        return [packets.wrap(b"\x0f\x01", packets.PAYLOAD_CONTROL_REPLY, 0)]


# --- replies ----------------------------------------------------------------------------------------------------


def test_a_command_returns_on_the_ack(rig):
    result = rig.ptz.pan_tilt_velocity(5, 0)
    assert result.status == ACKED and result.ok
    assert result.round_trip == pytest.approx(rig.transport.reply_delay)


def test_wait_for_completion_returns_completed(rig):
    result = rig.ptz.command(packets.home(), wait_for_completion=True)
    assert result.status == COMPLETED


def test_error_reply_is_reported_with_its_code():
    rig = make_rig(behaviour={"pan_tilt_drive": "error"})
    result = rig.ptz.pan_tilt_velocity(3, 0)
    assert result.status == ERROR and not result.ok
    assert result.error_code == 0x02  # syntax error
    assert rig.ptz.stats["errors"] == 1


def test_no_reply_times_out_after_exactly_the_timeout():
    rig = make_rig(behaviour={"pan_tilt_drive": "silent"})
    before = rig.clock.monotonic()
    result = rig.ptz.pan_tilt_velocity(3, 0)
    assert result.status == TIMEOUT
    assert rig.clock.monotonic() - before == pytest.approx(rig.ptz.timeout)
    assert rig.ptz.stats["timeouts"] == 1


# --- S1: one time budget per call ------------------------------------------------------------------------------
#
# The promise: no call blocks the caller longer than `timeout` (200 ms), whatever goes wrong. Before the fix, a call
# could wait for the lock (up to 200 ms), then for a pending sequence reset (200 ms), then for its reply (200 ms).


def test_pending_reset_and_reply_share_one_timeout():
    # A camera that has gone completely silent, and a sequence reset is due before the next command.
    rig = make_rig(behaviour={"sequence_reset": "silent", "pan_tilt_drive": "silent"})
    rig.ptz._needs_reset = True  # as after a "sequence number" error from the camera
    before = rig.clock.monotonic()
    result = rig.ptz.pan_tilt_velocity(3, 0)
    assert result.status == TIMEOUT
    # Both the reset and the command were sent, but the whole call took one timeout, not two.
    assert rig.clock.monotonic() - before == pytest.approx(rig.ptz.timeout)
    assert rig.camera.received[-2:] == [("sequence_reset", b"\x01"), ("pan_tilt_drive", packets.pan_tilt_drive(3, 0))]


class _SlowLock:
    """Stands in for the driver's lock: getting it takes `wait` seconds of fake time, as if another thread held it.

    With `never=True` it is never free: it waits the whole timeout it was given and then reports failure.
    """

    def __init__(self, clock: FakeClock, wait: float, never: bool = False) -> None:
        self.clock, self.wait, self.never = clock, wait, never
        self.timeouts_asked: list[float] = []

    def acquire(self, timeout: float = -1) -> bool:
        self.timeouts_asked.append(timeout)
        if self.never:
            self.clock.advance(timeout)
            return False
        self.clock.advance(min(self.wait, timeout))
        return True

    def release(self) -> None:
        pass


def test_time_spent_waiting_for_the_lock_comes_out_of_the_timeout():
    rig = make_rig(behaviour={"pan_tilt_drive": "silent"})
    rig.ptz._lock = _SlowLock(rig.clock, wait=0.15)  # another thread holds the wire for 150 ms
    before = rig.clock.monotonic()
    assert rig.ptz.pan_tilt_velocity(3, 0).status == TIMEOUT
    # 150 ms for the lock + only the remaining 50 ms for the reply = 200 ms in total (it used to be 350 ms).
    assert rig.clock.monotonic() - before == pytest.approx(rig.ptz.timeout)


def test_a_lock_that_never_frees_times_out_after_one_timeout():
    rig = make_rig()
    rig.ptz._lock = _SlowLock(rig.clock, wait=0.0, never=True)
    before = rig.clock.monotonic()
    assert rig.ptz.stop().status == TIMEOUT
    assert rig.clock.monotonic() - before == pytest.approx(rig.ptz.timeout)
    assert rig.ptz._lock.timeouts_asked == [pytest.approx(rig.ptz.timeout)]


def test_ack_without_completion_gives_acked_when_waiting_for_completion():
    rig = make_rig(behaviour={"home": "ack_only"})
    assert rig.ptz.command(packets.home(), wait_for_completion=True).status == ACKED


def test_a_stale_reply_is_not_mistaken_for_the_current_one(rig):
    old = rig.ptz.absolute_position(100, 0, 5, 5)  # sequence 1
    # A late completion for command 1 arrives while we wait for the answer to the next request.
    rig.transport.inject(packets.wrap(packets.completion(), packets.PAYLOAD_REPLY, old.sequence))
    position = rig.ptz.inquire_pan_tilt()
    assert position is not None  # the inquiry got its own answer, not the stale completion
    assert rig.ptz.completion_time(old.sequence) is not None


def test_completion_time_for_a_late_completion(rig):
    result = rig.ptz.command(packets.home())  # returns on the ACK; the completion stays in the inbox
    assert rig.ptz.completion_time(result.sequence) is not None


def test_camera_that_does_not_echo_the_sequence_number():
    rig = make_rig(echo_sequence=False)
    assert rig.ptz.stop().status == ACKED
    assert rig.ptz.inquire_pan_tilt() == (0, 0)
    assert rig.ptz.stats["unmatched_replies"] >= 2


def test_camera_that_answers_without_the_ip_header():
    rig = make_rig(bare_replies=True)
    assert rig.ptz.stop().status == ACKED
    assert rig.ptz.inquire_zoom() == 0


# --- inquiries --------------------------------------------------------------------------------------------------


def test_inquiry_decodes_signed_pan(rig):
    rig.ptz.pan_tilt_velocity(-10, 4)
    rig.clock.advance(1.0)
    rig.ptz.stop()
    rig.clock.advance(1.0)  # at rest, so the camera's answer and the truth refer to the same position
    pan, tilt = rig.ptz.inquire_pan_tilt()
    true_pan, true_tilt = rig.sim.true_pan_tilt()
    assert pan < 0 < tilt
    assert (pan, tilt) == (round(true_pan), round(true_tilt))
    assert rig.ptz.last_inquiry.status == ANSWERED


def test_unanswered_inquiry_returns_none():
    rig = make_rig(PtzModel(inquiry_mode="at_rest"))
    rig.ptz.pan_tilt_velocity(10, 0)
    rig.clock.advance(1.0)
    assert rig.ptz.inquire_pan_tilt() is None
    assert rig.ptz.last_inquiry.status == TIMEOUT


def test_an_error_answer_to_an_inquiry_returns_none():
    rig = make_rig(behaviour={"inquire_zoom": "error"})
    assert rig.ptz.inquire_zoom() is None
    assert rig.ptz.last_inquiry.status == ERROR


def test_extra_inquiries(rig):
    assert rig.ptz.inquire_max_speed() == (24, 23)
    assert rig.ptz.inquire_version().startswith(bytes([0x90, 0x50]))


# --- the dead-man switch ----------------------------------------------------------------------------------------


def stops_sent(rig: Rig) -> int:
    return sent_payloads(rig).count(packets.pan_tilt_stop())


def test_ttl_sends_stop_when_not_renewed(rig):
    rig.ptz.pan_tilt_velocity(8, 0, ttl=0.3)
    rig.clock.advance(0.2)
    rig.ptz.check_deadlines()
    assert stops_sent(rig) == 0  # not expired yet
    rig.clock.advance(0.2)
    rig.ptz.check_deadlines()
    assert stops_sent(rig) == 1
    assert rig.ptz.stats["ttl_stops"] == 1
    rig.clock.advance(1.0)
    assert not rig.sim.is_moving()  # the camera really stopped


def test_renewing_the_command_keeps_it_moving(rig):
    for _ in range(10):
        rig.ptz.pan_tilt_velocity(8, 0, ttl=0.3)
        rig.clock.advance(0.2)
        rig.ptz.check_deadlines()
    assert stops_sent(rig) == 0
    assert rig.sim.is_moving()


def test_ttl_counts_from_the_moment_of_sending():
    rig = make_rig()
    rig.transport.reply_delay = 0.15  # slow camera: the reply takes 150 ms
    rig.ptz.pan_tilt_velocity(8, 0, ttl=0.3)  # returns 150 ms after sending
    rig.clock.advance(0.16)  # 0.31 s after sending
    rig.ptz.check_deadlines()
    assert stops_sent(rig) == 1


def test_stop_disarms_the_ttl(rig):
    rig.ptz.pan_tilt_velocity(8, 0, ttl=0.3)
    rig.ptz.stop()
    rig.clock.advance(1.0)
    rig.ptz.check_deadlines()
    assert stops_sent(rig) == 1  # only our own stop, no extra dead-man stop
    assert rig.ptz.stats["ttl_stops"] == 0


def test_without_ttl_motion_continues(rig):
    rig.ptz.pan_tilt_velocity(8, 0)
    rig.clock.advance(5.0)
    rig.ptz.check_deadlines()
    assert stops_sent(rig) == 0 and rig.sim.is_moving()


def test_zoom_ttl(rig):
    rig.ptz.zoom_velocity(3, ttl=0.25)
    rig.clock.advance(0.3)
    rig.ptz.check_deadlines()
    assert sent_payloads(rig).count(packets.zoom_stop()) == 1


def test_position_commands_disarm_the_pan_tilt_ttl(rig):
    rig.ptz.pan_tilt_velocity(8, 0, ttl=0.3)
    rig.ptz.preset_recall(2)
    rig.clock.advance(1.0)
    rig.ptz.check_deadlines()
    assert rig.ptz.stats["ttl_stops"] == 0


def test_watchdog_thread_starts_and_stops():
    clock = FakeClock()
    camera = FakeViscaCamera(SimulatedPtz(PtzModel(), clock))
    ptz = ViscaPtz("camera", clock=clock, transport=FakeTransport(camera, clock), watchdog=True)
    ptz.close()
    assert not ptz._watchdog.is_alive()


# --- logging ----------------------------------------------------------------------------------------------------


def test_jsonl_log_has_both_clocks_and_both_directions(tmp_path):
    log = tmp_path / "visca.jsonl"
    rig = make_rig(log_path=str(log))
    rig.ptz.pan_tilt_velocity(5, 0, ttl=0.1)
    rig.clock.advance(0.2)
    rig.ptz.check_deadlines()
    rig.ptz.close()
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert all("mono" in e and "wall" in e for e in entries)
    assert {e.get("dir") for e in entries} >= {"tx", "rx"}
    assert any(e.get("event") == "ttl_stop" for e in entries)
    velocity = next(e for e in entries if e.get("what") == "pan_tilt_velocity")
    assert velocity["hex"].startswith("01 00 00 09")
    assert entries[-1]["wall"] - entries[-1]["mono"] == pytest.approx(1_800_000_000.0)


def test_no_log_file_by_default(tmp_path, rig):
    rig.ptz.stop()
    assert not list(tmp_path.iterdir())
