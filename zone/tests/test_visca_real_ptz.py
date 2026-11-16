"""The VISCA driver against the PTZ the zone drives at the race, as measured by João's probe on 2026-11-15.

What that camera does differently from the textbook (and from our default fake camera):

1. It puts several VISCA/IP messages in ONE datagram, each with its own 8-byte header, e.g.
       01 11 00 03 <seq> 90 48 FF   01 11 00 03 <seq> 90 58 FF   01 11 00 0B <seq> 90 50 <8 nibbles> FF
       (ACK)                        (completion)                 (pan/tilt data)
2. It numbers its replies from ITS OWN counter (~43000, rising), never with the number of our request.
3. It answers an inquiry with ACK + completion, and only sometimes (~17 %) with the data.
4. Its replies come late (0-0.5 s, ~3 s at first) and in batches, so a reply can belong to an earlier request.

Most tests here use `FakeViscaCamera.like_real_ptz()`, the fake camera made to behave like that, on a fake clock
(no real time passes: waiting 200 ms for a reply just moves the fake clock). A few tests hand-craft the exact
bytes from the probe and push them to the driver with `transport.inject(...)`.

Words used below:
- "request": one command or inquiry the driver sends; it carries a sequence number.
- "echo mode": the camera copies our sequence number into its reply (`ptz.sequence_echo is True`).
- "no echo mode": it doesn't (`ptz.sequence_echo is False`); then a reply's number says nothing.
"""
from dataclasses import dataclass

import pytest

from zone.clock import FakeClock
from zone.ptz import visca_packets as packets
from zone.ptz.fake_camera import FakeTransport, FakeViscaCamera, FakeViscaUdpServer
from zone.ptz.interface import ACKED, ANSWERED, COMPLETED, TIMEOUT
from zone.ptz.sim import PtzModel, SimulatedPtz
from zone.ptz.visca import STOP_RETRY_WINDOW_S, ViscaPtz

STOP = packets.pan_tilt_stop()


@dataclass
class Rig:
    clock: FakeClock
    sim: SimulatedPtz
    camera: FakeViscaCamera
    transport: FakeTransport
    ptz: ViscaPtz


def make_rig(like_real: bool = True, **real_options) -> Rig:
    """The driver + the fake camera + the simulated motors, all on one fake clock.

    like_real=True: the fake camera behaves like the probed PTZ. `real_options` go to `like_real_ptz()`, e.g.
    answer_probability=1.0 (always send the data) or first_reply_delay=0.0 (no slow start)."""
    clock = FakeClock()
    sim = SimulatedPtz(PtzModel(), clock)
    camera = FakeViscaCamera(sim)
    if like_real:
        camera.like_real_ptz(**real_options)
    transport = FakeTransport(camera, clock)
    # watchdog=False: no background thread; the tests call check_deadlines() themselves at fake times they choose.
    ptz = ViscaPtz("camera", clock=clock, transport=transport, watchdog=False)
    return Rig(clock, sim, camera, transport, ptz)


def reply(payload_hex: str, sequence: int) -> bytes:
    """One reply message (8-byte header + payload), numbered `sequence`, as the camera sends it."""
    return packets.wrap(bytes.fromhex(payload_hex), packets.PAYLOAD_REPLY, sequence)


ACK_HEX, COMPLETION_HEX = "90 48 FF", "90 58 FF"  # the real camera answers on "socket" 8
PAN_TILT_DATA_HEX = "90 50 00 01 0A 05 0F 0F 0F 0F FF"  # pan 0x01A5 = 421, tilt 0xFFFF = -1 (from the probe)
ZOOM_DATA_HEX = "90 50 00 00 00 00 FF"  # zoom 0 (from the probe): 7 bytes, a pan/tilt answer is 11


class SilentCamera:
    """Stands in for the fake camera: receives everything, answers nothing. The test injects the replies."""

    def handle(self, datagram: bytes) -> list[bytes]:
        return []


def make_scripted_rig(sequence_echo: bool | None = False) -> Rig:
    """A driver whose camera never answers by itself: each test injects the exact reply datagrams it wants.

    sequence_echo: what the driver already believes about the camera (False = "no echo" mode, as the real PTZ;
    None = it doesn't know yet, as right after connecting)."""
    rig = make_rig(like_real=False)
    rig.transport.camera = SilentCamera()
    rig.ptz.sequence_echo = sequence_echo
    return rig


# --- rule 1: every message of a datagram is read ---------------------------------------------------------------


def test_an_answer_behind_an_ack_and_a_completion_in_the_same_datagram_is_found():
    # The probe's exact datagram: ACK, completion, then the pan/tilt data, all in one. The old driver read only
    # the first message (or nothing at all), so the data behind the ACK was lost.
    rig = make_scripted_rig(sequence_echo=None)
    datagram = reply(ACK_HEX, 43234) + reply(COMPLETION_HEX, 43235) + reply(PAN_TILT_DATA_HEX, 43236)
    rig.transport.inject(datagram, delay=0.05)  # arrives 50 ms after the inquiry goes out
    assert rig.ptz.inquire_pan_tilt() == (421, -1)


def test_in_echo_mode_the_completion_after_the_ack_is_still_remembered():
    # An echoing camera that also batches: [ACK(n), completion(n)] in one datagram. The command returns on the ACK;
    # the completion behind it must not be thrown away: completion_time(n) needs it (the bench times moves by it).
    rig = make_scripted_rig(sequence_echo=True)
    rig.transport.inject(reply("90 41 FF", 1) + reply("90 51 FF", 1), delay=0.01)
    result = rig.ptz.command(packets.home())  # this is request number 1
    assert result.status == ACKED and result.sequence == 1
    assert rig.ptz.completion_time(1) is not None


# --- rule 2: detecting a camera that numbers replies itself -----------------------------------------------------


def test_replies_numbered_from_the_cameras_own_counter_switch_to_no_echo_mode():
    rig = make_rig()
    assert rig.ptz.sequence_echo is None  # nothing has been answered yet (the reset's answer goes elsewhere)
    rig.ptz.stop()
    rig.clock.advance(4.0)  # the real camera's first replies take ~3 s
    rig.ptz.stop()
    # Two replies in a row carrying numbers we never sent (43000-something): this camera doesn't echo.
    assert rig.ptz.sequence_echo is False


def test_a_number_we_never_sent_once_is_not_enough_to_decide():
    # One stray number could be a very late reply from before a reset; it takes two in a row.
    rig = make_scripted_rig(sequence_echo=None)
    rig.transport.inject(reply(ACK_HEX, 43191), delay=0.01)
    rig.ptz.stop()
    assert rig.ptz.sequence_echo is None


def test_an_echoing_camera_is_not_switched_by_one_stray_number_but_is_by_a_run_of_them():
    rig = make_scripted_rig(sequence_echo=True)
    rig.transport.inject(reply(ACK_HEX, 43191), delay=0.01)
    rig.ptz.stop()
    assert rig.ptz.sequence_echo is True  # one stray: still echo mode (it was dropped)
    # A run of 8 never-sent numbers in a row (with no echo in between) means it doesn't echo after all. This
    # protects against the camera's own counter happening to hit one of our numbers right at the start.
    for k in range(8):
        rig.transport.inject(reply(ACK_HEX, 50000 + k), delay=0.01)
        rig.ptz.stop()
    assert rig.ptz.sequence_echo is False


def test_in_no_echo_mode_a_reply_that_happens_to_carry_our_number_does_not_switch_back():
    # The camera's counter and ours both count up: sooner or later they meet. That's a coincidence, not an echo.
    rig = make_scripted_rig(sequence_echo=False)
    next_number = rig.ptz._sequence
    rig.transport.inject(reply(ACK_HEX, next_number), delay=0.01)
    assert rig.ptz.stop().status == ACKED
    assert rig.ptz.sequence_echo is False


# --- rule 2: in "no echo" mode, what answers what ----------------------------------------------------------------


def test_an_inquiry_ignores_ack_and_completion_and_waits_for_the_data():
    rig = make_scripted_rig()
    rig.transport.inject(reply(ACK_HEX, 43000) + reply(COMPLETION_HEX, 43001), delay=0.02)
    rig.transport.inject(reply(PAN_TILT_DATA_HEX, 43002), delay=0.08)  # the data comes later, in its own datagram
    result = rig.ptz.inquire_pan_tilt()
    assert result == (421, -1)
    assert rig.ptz.last_inquiry.status == ANSWERED
    assert rig.ptz.stats["ignored_replies"] == 2  # the ACK and the completion


def test_an_inquiry_without_data_is_a_timeout_not_an_error():
    # The real camera skips the data ~83 % of the time. That's "no answer this time", never a camera fault.
    rig = make_scripted_rig()
    rig.transport.inject(reply(ACK_HEX, 43000) + reply(COMPLETION_HEX, 43001), delay=0.02)
    assert rig.ptz.inquire_pan_tilt() is None
    assert rig.ptz.last_inquiry.status == TIMEOUT
    assert rig.ptz.stats["errors"] == 0


def test_an_error_while_waiting_for_inquiry_data_is_ignored():
    # Without sequence numbers an error can't be told apart from an earlier command's: it doesn't answer the inquiry.
    rig = make_scripted_rig()
    rig.transport.inject(reply("90 68 41 FF", 43000), delay=0.02)
    assert rig.ptz.inquire_zoom() is None
    assert rig.ptz.last_inquiry.status == TIMEOUT


def test_a_zoom_shaped_answer_is_not_taken_for_a_pan_tilt_answer():
    # A late zoom answer (7 bytes) arrives while we wait for pan/tilt (11 bytes): it belongs to an earlier inquiry.
    # The old driver called that an ERROR (and gave up); now it is skipped and the real answer is still found.
    rig = make_scripted_rig()
    rig.transport.inject(reply(ZOOM_DATA_HEX, 43000), delay=0.02)
    rig.transport.inject(reply(PAN_TILT_DATA_HEX, 43001), delay=0.05)
    assert rig.ptz.inquire_pan_tilt() == (421, -1)


def test_a_pan_tilt_shaped_answer_is_not_taken_for_a_zoom_answer():
    rig = make_scripted_rig()
    rig.transport.inject(reply(PAN_TILT_DATA_HEX, 43000), delay=0.02)
    assert rig.ptz.inquire_zoom() is None
    assert rig.ptz.last_inquiry.status == TIMEOUT


def test_a_position_is_not_taken_for_the_version():
    # The version inquiry has no fixed answer shape, but a position-sized answer (11 or 7 bytes) can't be it: the
    # real PTZ sends a pan/tilt + 7-byte pair whatever was asked, so that is an earlier inquiry's data.
    rig = make_scripted_rig()
    rig.transport.inject(reply(PAN_TILT_DATA_HEX, 43000) + reply(ZOOM_DATA_HEX, 43001), delay=0.02)
    assert rig.ptz.inquire_version() is None


def test_a_command_is_answered_by_any_ack_whatever_its_number():
    rig = make_scripted_rig()
    rig.transport.inject(reply(ACK_HEX, 43000), delay=0.03)
    result = rig.ptz.pan_tilt_velocity(3, 0)
    assert result.status == ACKED
    assert result.round_trip == pytest.approx(0.03)


def test_a_command_skips_inquiry_data_and_takes_the_completion():
    rig = make_scripted_rig()
    rig.transport.inject(reply(PAN_TILT_DATA_HEX, 43000) + reply(COMPLETION_HEX, 43001), delay=0.03)
    assert rig.ptz.pan_tilt_velocity(3, 0).status == COMPLETED


def test_replies_already_waiting_when_a_request_goes_out_are_dropped():
    # A late batch from an earlier inquiry is sitting in the socket: a pan/tilt answer claiming (999, 999). It
    # arrived BEFORE our new inquiry was sent, so it can't be its answer; the driver drops it before sending.
    rig = make_scripted_rig()
    stale = packets.encode_pan_tilt_answer(999, 999)
    rig.transport.inject(packets.wrap(stale, packets.PAYLOAD_REPLY, 43000), delay=0.0)
    assert rig.ptz.inquire_pan_tilt() is None  # the camera sends nothing new, so: no answer
    assert rig.ptz.stats["dropped_waiting"] == 1


def test_a_stray_zoom_zero_numbered_0_waiting_before_the_first_inquiry_is_not_taken():
    # Right after connecting, the driver doesn't know yet whether the camera copies our sequence numbers. A stray
    # `90 50 00 00 00 00 FF` numbered 0 is already waiting in the socket (the camera sent it on its own). Before the
    # fix it both switched the driver to "no echo" mode (a 0 is never one of our numbers) AND was taken as the
    # answer "zoom 0" (2026-11-15 bench: a keyframe stored with zoom 0). It arrived BEFORE we asked: dropped.
    rig = make_scripted_rig(sequence_echo=None)
    rig.transport.inject(reply(ZOOM_DATA_HEX, 0), delay=0.0)
    assert rig.ptz.inquire_zoom() is None  # the camera says nothing new, so: no answer (and certainly not 0)
    assert rig.ptz.stats["dropped_waiting"] == 1


def test_a_7_byte_message_glued_behind_a_pan_tilt_answer_is_not_the_zoom():
    # While our zoom question waits, a datagram arrives with a pan/tilt answer and right behind it a 7-byte
    # `90 50 00 00 00 00 FF`: that is a data PAIR answering an earlier pan/tilt question (we ask one thing at a
    # time). Its second half looks like "zoom 0" but is not our zoom: ignored.
    rig = make_scripted_rig()
    rig.transport.inject(reply(PAN_TILT_DATA_HEX, 43000) + reply(ZOOM_DATA_HEX, 43001), delay=0.02)
    assert rig.ptz.inquire_zoom() is None
    # ...while a 7-byte answer on its own is still a zoom answer
    rig.transport.inject(reply("90 50 00 02 00 00 FF", 43002), delay=0.02)
    assert rig.ptz.inquire_zoom() == 0x200


def test_last_reply_at_notes_a_reply_without_data():
    # The zone's health checks use it: an inquiry that got only ACK + completion still shows the camera is there.
    rig = make_scripted_rig()
    rig.transport.inject(reply(ACK_HEX, 43000), delay=0.02)
    before = rig.clock.monotonic()
    assert rig.ptz.inquire_pan_tilt() is None
    assert rig.ptz.last_reply_at == pytest.approx(before + 0.02)


# --- the whole thing, against the fake camera that behaves like the real one ------------------------------------


def test_against_the_real_like_camera_every_answer_is_the_right_one():
    # The camera sits still at pan -321, tilt 45, zoom 9999 (all different, so a mix-up would show). We ask pan/tilt
    # and zoom alternately, 200 times each. Many get no answer (that's this camera), but every answer we DO get must
    # be the right value of the right kind, and nothing is reported as an error.
    # Zoom: this camera's only zoom-sized message is the second half of a data pair, `90 50 00 00 00 00 FF` whatever
    # the zoom (the probe). The driver must never take it as "zoom 0" (2026-11-15 bench: a keyframe got zoom 0), so
    # here the zoom inquiry is simply never answered.
    rig = make_rig()
    rig.sim.pan.position, rig.sim.tilt.position, rig.sim.zoom.position = -321.0, 45.0, 9999.0
    pan_tilt_answers, zoom_answers, statuses = [], [], set()
    for _ in range(200):
        pan_tilt = rig.ptz.inquire_pan_tilt()
        statuses.add(rig.ptz.last_inquiry.status)
        zoom = rig.ptz.inquire_zoom()
        statuses.add(rig.ptz.last_inquiry.status)
        pan_tilt_answers += [pan_tilt] if pan_tilt is not None else []
        zoom_answers += [zoom] if zoom is not None else []
        rig.clock.advance(0.1)
    assert pan_tilt_answers  # some came through...
    assert set(pan_tilt_answers) == {(-321, 45)}  # ...and every one is right
    assert zoom_answers == []  # never the pair's fake "zoom 0"
    assert statuses <= {ANSWERED, TIMEOUT}  # never ERROR
    assert rig.ptz.sequence_echo is False


def test_commands_still_move_the_camera():
    # The probe: a speed-1 pan for 2 s moved the camera. Commands work even if their replies are late.
    rig = make_rig()
    for _ in range(20):
        rig.ptz.pan_tilt_velocity(4, 0, ttl=0.5)
        rig.clock.advance(0.1)
    rig.ptz.stop()
    rig.clock.advance(1.0)
    assert rig.sim.true_pan_tilt()[0] > 10


# --- rule 3: no endless stop resends ------------------------------------------------------------------------------


def test_lost_stops_are_retried_for_two_seconds_at_most_and_at_a_modest_rate():
    # The studio froze this camera's video with an endless stop resend. Our driver re-sends an unacknowledged stop
    # from its dead-man, but only for STOP_RETRY_WINDOW_S (2 s), each try waiting its 200 ms timeout.
    # Here every stop is lost on the way, and no other reply is pending (no slow start, no delay).
    rig = make_rig(first_reply_delay=0.0, reply_delay_range=(0.0, 0.0))
    rig.ptz.pan_tilt_velocity(8, 0, ttl=0.1)
    rig.clock.advance(1.0)  # the velocity command's replies have long arrived
    rig.ptz.drain_replies()
    rig.camera.lose_next[STOP] = 1000
    started = rig.clock.monotonic()
    for _ in range(60):  # 60 dead-man ticks, 6+ s of fake time
        rig.ptz.check_deadlines()
        rig.clock.advance(0.01)
    stops = rig.camera.lost.count(STOP)
    assert 8 <= stops <= 12  # ~5 a second for 2 s, then it gives up
    assert rig.ptz.stats["stop_failures"] == 1
    # And none after the give-up: nothing more is sent however long we wait.
    rig.clock.advance(5.0)
    rig.ptz.check_deadlines()
    assert rig.camera.lost.count(STOP) == stops
    assert rig.clock.monotonic() - started > STOP_RETRY_WINDOW_S


def test_a_lost_stop_is_retried_and_the_next_ack_ends_the_retries():
    rig = make_rig(first_reply_delay=0.0, reply_delay_range=(0.0, 0.0))
    rig.ptz.pan_tilt_velocity(8, 0)
    rig.clock.advance(1.0)
    rig.ptz.drain_replies()
    rig.camera.lose_next[STOP] = 1
    assert rig.ptz.stop().status == TIMEOUT  # lost
    rig.ptz.check_deadlines()  # retried: arrives, ACKed (whatever its number)
    rig.clock.advance(1.0)
    assert not rig.sim.is_moving()
    for _ in range(10):
        rig.ptz.check_deadlines()
    assert rig.ptz.stats["stop_retries"] == 1  # one retry, then quiet


# --- the fake camera's real-PTZ mode itself ------------------------------------------------------------------------


def test_the_fake_numbers_replies_from_its_own_counter_and_batches_them():
    clock = FakeClock()
    camera = FakeViscaCamera(SimulatedPtz(PtzModel(), clock)).like_real_ptz(answer_probability=1.0)
    request = packets.wrap(packets.pan_tilt_position_inquiry(), packets.PAYLOAD_INQUIRY, 7)
    assert camera.handle(request) == []  # nothing right away: the first replies wait ~3 s
    camera.handle(packets.wrap(packets.zoom_position_inquiry(), packets.PAYLOAD_INQUIRY, 8))
    clock.advance(3.1)
    datagrams = camera.take_due(clock.monotonic())
    messages = [m for d in datagrams for m in packets.parse_replies(d)]
    # Both inquiries' replies came out together, at most 4 messages per datagram:
    assert len(datagrams) == 2 and all(len(packets.parse_replies(d)) <= 4 for d in datagrams)
    # The first inquiry: ACK + completion only. Its data comes as a PAIR (pan/tilt 11 bytes, then a 7-byte one)
    # in front of the SECOND request's ACK + completion, as the second probe saw it.
    assert [m.kind for m in messages] == ["ack", "completion", "answer", "answer", "ack", "completion"]
    assert [len(m.payload) for m in messages[2:4]] == [11, 7]
    # ...all numbered from the camera's own counter, never 7 or 8.
    numbers = [m.sequence for m in messages]
    assert len(set(numbers)) == len(numbers) and min(numbers) > 43000


def test_the_fake_sends_no_data_while_moving():
    # Second probe: during a 3 s pan at medium speed, zero data answers, only ACK + completions.
    clock = FakeClock()
    sim = SimulatedPtz(PtzModel(), clock)
    camera = FakeViscaCamera(sim).like_real_ptz(answer_probability=1.0, first_reply_delay=0.0)
    sim.pan_tilt_velocity(8, 0)
    clock.advance(0.5)
    for k in range(20):
        camera.handle(packets.wrap(packets.pan_tilt_position_inquiry(), packets.PAYLOAD_INQUIRY, k + 1))
        clock.advance(0.1)
    assert sim.is_moving() and camera.answers_sent == 0


def test_the_fake_drops_the_data_with_the_given_probability():
    clock = FakeClock()
    camera = FakeViscaCamera(SimulatedPtz(PtzModel(), clock)).like_real_ptz(first_reply_delay=0.0)
    for k in range(1000):
        camera.handle(packets.wrap(packets.pan_tilt_position_inquiry(), packets.PAYLOAD_INQUIRY, k + 1))
    assert camera.inquiries == 1000
    assert 120 <= camera.answers_sent <= 220  # ~17 %


def test_the_fake_does_not_answer_the_sequence_reset_to_the_sender():
    # The real camera answered the reset on port 52381 only, so the driver never sees the answer.
    rig = make_rig()
    assert rig.ptz.sequence_reset_status == TIMEOUT
    assert rig.camera.reset_count == 1


def test_the_default_fake_is_unchanged():
    # Other tests rely on the default fake: one reply per datagram, right away, our own number echoed.
    clock = FakeClock()
    camera = FakeViscaCamera(SimulatedPtz(PtzModel(), clock))
    replies = camera.handle(packets.wrap(packets.pan_tilt_position_inquiry(), packets.PAYLOAD_INQUIRY, 5))
    assert len(replies) == 1 and packets.parse_reply(replies[0]).sequence == 5
    assert packets.parse_reply(replies[0]).kind == "answer"


def test_the_real_like_fake_over_real_udp():
    # The same over real sockets (the standalone fake and the agent rig run like this): the server sends the
    # delayed batches by itself. Short delays here so the test takes well under a second of real time.
    from zone.clock import SystemClock
    sim = SimulatedPtz(PtzModel(), SystemClock())
    camera = FakeViscaCamera(sim).like_real_ptz(answer_probability=1.0, first_reply_delay=0.05,
                                                reply_delay_range=(0.0, 0.05))
    with FakeViscaUdpServer(camera) as server:
        host, port = server.address
        ptz = ViscaPtz(host, port, timeout=1.0, clock=SystemClock(), watchdog=False)
        try:
            assert ptz.stop().ok
            # An inquiry's data rides with the NEXT request's replies: the 2nd or 3rd inquiry gets it.
            assert (0, 0) in [ptz.inquire_pan_tilt() for _ in range(3)]
            assert ptz.sequence_echo is False
            # The pair's 7-byte half (`90 50 00 00 00 00 FF`) looks like "zoom 0" but is not the zoom: never taken.
            assert [ptz.inquire_zoom() for _ in range(3)] == [None, None, None]
        finally:
            ptz.close()


def test_the_command_line_flag():
    import argparse

    from zone.ptz.fake_camera import add_real_ptz_arguments, camera_from_arguments
    ap = argparse.ArgumentParser()
    add_real_ptz_arguments(ap)
    sim = SimulatedPtz(PtzModel(), FakeClock())
    assert camera_from_arguments(sim, ap.parse_args([])).delayed is False  # default: the plain fake
    camera = camera_from_arguments(sim, ap.parse_args(["--like-real-ptz", "--answer-probability", "0.5"]))
    assert camera.delayed and camera.own_counter and camera.answer_probability == 0.5
