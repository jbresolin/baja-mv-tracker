"""Byte-level tests of the VISCA packet builders and reply parsers.

How to read these: each test builds a packet and compares it with the bytes written out by hand, in hex.
`bytes.fromhex("81 01 06 01")` is just a readable way to write b"\\x81\\x01\\x06\\x01". The expected bytes come
from ptz_test.py (which drove the real SMTAV) or from the Sony VISCA command list.
"""
from zone.ptz import visca_packets as packets


def hx(text: str) -> bytes:
    return bytes.fromhex(text)


# --- the 8-byte header ------------------------------------------------------------------------------------------


def test_wrap_puts_type_length_and_sequence_in_front():
    # payload type 0x0100 (command), length 9, sequence 0x01020304, then the payload itself
    datagram = packets.wrap(packets.pan_tilt_stop(), packets.PAYLOAD_COMMAND, 0x01020304)
    assert datagram[:8] == hx("01 00 00 09 01 02 03 04")
    assert datagram[8:] == packets.pan_tilt_stop()


def test_inquiry_header_uses_payload_type_0110():
    assert packets.wrap(packets.zoom_position_inquiry(), packets.PAYLOAD_INQUIRY, 7)[:8] == hx("01 10 00 05 00 00 00 07")


def test_sequence_reset_is_the_control_message():
    assert packets.sequence_reset() == hx("02 00 00 01 00 00 00 00 01")


# --- commands ---------------------------------------------------------------------------------------------------


def test_stop_matches_ptz_test_py():
    assert packets.pan_tilt_stop() == hx("81 01 06 01 00 00 03 03 FF")


def test_pan_right_tilt_down():
    # pan speed 5 to the right (02), tilt speed 3 down (02)
    assert packets.pan_tilt_drive(5, -3) == hx("81 01 06 01 05 03 02 02 FF")


def test_pan_left_tilt_up():
    assert packets.pan_tilt_drive(-24, 23) == hx("81 01 06 01 18 17 01 01 FF")


def test_a_zero_axis_holds_with_minimum_speed_byte():
    # tilt only: pan direction 03 (hold), speed byte 1 as ptz_test.py sent it
    assert packets.pan_tilt_drive(0, 2) == hx("81 01 06 01 01 02 03 01 FF")


def test_both_zero_is_the_stop_packet():
    assert packets.pan_tilt_drive(0, 0) == packets.pan_tilt_stop()


def test_speeds_are_clamped_to_the_spec():
    assert packets.pan_tilt_drive(99, -99) == hx("81 01 06 01 18 17 02 02 FF")


def test_zoom_tele_wide_stop():
    assert packets.zoom_drive(5) == hx("81 01 04 07 25 FF")
    assert packets.zoom_drive(-2) == hx("81 01 04 07 32 FF")
    assert packets.zoom_drive(0) == hx("81 01 04 07 00 FF")
    assert packets.zoom_drive(12) == hx("81 01 04 07 27 FF")  # clamped to 7


def test_zoom_direct_spreads_the_position_over_nibbles():
    assert packets.zoom_direct(0x1234) == hx("81 01 04 47 01 02 03 04 FF")


def test_absolute_position_encodes_negative_pan_as_twos_complement():
    # pan -1 = 0xFFFF, tilt 0x0100; speeds 0x0C / 0x0A
    assert packets.absolute_position(-1, 0x100, 12, 10) == hx("81 01 06 02 0C 0A 0F 0F 0F 0F 00 01 00 00 FF")


def test_relative_position_uses_command_03():
    assert packets.relative_position(16, -16, 1, 1)[:4] == hx("81 01 06 03")


def test_presets_and_home():
    assert packets.preset_recall(3) == hx("81 01 04 3F 02 03 FF")
    assert packets.preset_store(3) == hx("81 01 04 3F 01 03 FF")
    assert packets.home() == hx("81 01 06 04 FF")


def test_inquiries():
    assert packets.pan_tilt_position_inquiry() == hx("81 09 06 12 FF")
    assert packets.zoom_position_inquiry() == hx("81 09 04 47 FF")


# --- replies ----------------------------------------------------------------------------------------------------


def test_parse_reply_reads_the_header():
    reply = packets.parse_reply(hx("01 11 00 03 00 00 00 2A 90 41 FF"))
    assert reply is not None
    assert reply.payload_type == packets.PAYLOAD_REPLY
    assert reply.sequence == 42
    assert reply.kind == packets.ACK


def test_parse_reply_kinds():
    def kind(payload: str) -> str:
        return packets.parse_reply(packets.wrap(hx(payload), packets.PAYLOAD_REPLY, 1)).kind

    assert kind("90 41 FF") == packets.ACK
    assert kind("90 51 FF") == packets.COMPLETION
    assert kind("90 50 00 00 01 00 FF") == packets.ANSWER
    assert kind("90 61 02 FF") == packets.ERROR
    assert kind("80 41 FF") == packets.UNKNOWN  # not from a camera


def test_error_code_is_the_third_byte():
    reply = packets.parse_reply(packets.wrap(hx("90 61 41 FF"), packets.PAYLOAD_REPLY, 1))
    assert reply.error_code == 0x41  # "command not executable"


def test_bare_visca_reply_without_header_is_accepted():
    reply = packets.parse_reply(hx("90 41 FF"))
    assert reply is not None and reply.sequence is None and reply.kind == packets.ACK


def test_garbage_is_rejected():
    assert packets.parse_reply(b"hello") is None
    assert packets.parse_reply(b"") is None


def test_several_messages_in_one_datagram_are_all_read():
    # A datagram from the real PTZ (probe of 2026-11-15): ACK, completion and the pan/tilt data, each message with
    # its own 8-byte header (type 01 11, its length, the camera's own sequence number), back to back.
    datagram = hx("01 11 00 03 00 00 a8 e2 90 48 ff"
                  "01 11 00 03 00 00 a8 e3 90 58 ff"
                  "01 11 00 0b 00 00 a8 e4 90 50 00 01 0a 05 0f 0f 0f 0f ff")
    replies = packets.parse_replies(datagram)
    assert [r.kind for r in replies] == [packets.ACK, packets.COMPLETION, packets.ANSWER]
    assert [r.sequence for r in replies] == [0xA8E2, 0xA8E3, 0xA8E4]
    assert packets.parse_pan_tilt_position(replies[2].payload) == (0x01A5, -1)
    # parse_reply (one message) gives the first of them; it used to give None for such a datagram.
    assert packets.parse_reply(datagram).kind == packets.ACK


def test_a_single_message_datagram_is_one_reply():
    assert len(packets.parse_replies(hx("01 11 00 03 00 00 00 2A 90 41 FF"))) == 1


def test_a_cut_off_last_message_is_ignored():
    # The second message says it has 11 payload bytes, but only 3 arrived: keep the first, drop the broken one.
    datagram = hx("01 11 00 03 00 00 00 01 90 48 ff 01 11 00 0b 00 00 00 02 90 50 00")
    assert [r.kind for r in packets.parse_replies(datagram)] == [packets.ACK]


def test_several_bare_payloads_are_split_at_their_terminators():
    replies = packets.parse_replies(hx("90 41 FF 90 51 FF"))
    assert [r.kind for r in replies] == [packets.ACK, packets.COMPLETION]
    assert all(r.sequence is None for r in replies)
    assert packets.parse_replies(hx("90 41 FF 12 34")) == []  # a non-VISCA tail: none of it is trusted


def test_control_reply():
    reply = packets.parse_reply(hx("02 01 00 01 00 00 00 00 01"))
    assert reply.kind == packets.CONTROL


def test_pan_tilt_position_positive_and_negative():
    # pan 0x0123, tilt 0xFFFE (= -2)
    payload = hx("90 50 00 01 02 03 0F 0F 0F 0E FF")
    assert packets.parse_pan_tilt_position(payload) == (0x123, -2)


def test_0x8000_is_negative():
    # ptz_test.py used "> 0x8000" and read this as +32768; it is the most negative 16-bit value
    payload = hx("90 50 08 00 00 00 00 00 00 00 FF")
    assert packets.parse_pan_tilt_position(payload) == (-32768, 0)


def test_five_nibble_pan_is_understood():
    # some Sony models report pan with 5 nibbles: pan 0xFFFFF (= -1), tilt 0x0010
    payload = hx("90 50 0F 0F 0F 0F 0F 00 00 01 00 FF")
    assert packets.parse_pan_tilt_position(payload) == (-1, 16)


def test_wrong_length_position_is_rejected():
    assert packets.parse_pan_tilt_position(hx("90 50 00 01 FF")) is None
    assert packets.parse_pan_tilt_position(hx("90 41 FF")) is None


def test_zoom_position():
    assert packets.parse_zoom_position(hx("90 50 04 00 00 00 FF")) == 0x4000
    assert packets.parse_zoom_position(hx("90 50 04 00 FF")) is None


def test_encode_decode_round_trip():
    payload = packets.encode_pan_tilt_answer(-1234, 567)
    assert packets.parse_pan_tilt_position(payload) == (-1234, 567)
    assert packets.parse_zoom_position(packets.encode_zoom_answer(9999)) == 9999


def test_drive_speeds_are_clamped_to_the_cameras_top_steps():
    # A camera whose top tilt step is 0x14 (20) is never sent more, even if asked for 23.
    assert packets.pan_tilt_drive(30, 23, pan_max=0x18, tilt_max=0x14) == hx("81 01 06 01 18 14 02 01 FF")
    # A top step above the protocol's own ceiling is ignored: tilt never goes above 0x17.
    assert packets.pan_tilt_drive(1, 30, tilt_max=0x30)[5] == 0x17
    assert packets.absolute_position(0, 0, 24, 23, tilt_max=0x14)[5] == 0x14
    assert packets.relative_position(0, 0, 24, 23, pan_max=0x10)[4] == 0x10
