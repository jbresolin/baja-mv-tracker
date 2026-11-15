"""VISCA over IP, byte by byte: packet builders, the 8-byte header, and reply parsers.

Pure functions and data, no sockets or threads, so every byte can be tested directly.

Background for a newcomer
-------------------------
VISCA is Sony's camera-control protocol. A command is a short list of bytes:

    81 01 06 01 VV WW XX YY FF
    ^^ camera address 1 (0x80 + 1)       ^^ terminator, always 0xFF

"VISCA over IP" sends that same payload in a UDP datagram to port 52381, after an 8-byte header:

    bytes 0-1  payload type: 0x0100 command, 0x0110 inquiry, 0x0111 reply, 0x0200 control, 0x0201 control reply
    bytes 2-3  payload length
    bytes 4-7  sequence number (the camera copies it into its reply, so we know which request it answers)

The camera answers a command twice: ACK (`90 4y FF`, "accepted") and then completion (`90 5y FF`, "done"). An
inquiry gets one answer carrying data (`90 50 ... FF`). A refusal is an error (`90 6y ee FF`). For a velocity
command, completion comes right after the ACK. For a move to an absolute position, it comes when the move ends.

Sources: the command bytes come from `ptz_test.py`, the Pelco-D→VISCA bridge that drove the SMTAV at the March 2026
event (LIVESTACK reference code). Absolute/relative position, zoom direct, home and the extra inquiries come from
the Sony VISCA command list, which the SMTAV follows as far as we know. The bench script checks each of them.
"""
from __future__ import annotations

from dataclasses import dataclass

CAMERA_ADDRESS = 1  # VISCA over IP has one camera per IP address; it answers on address 1
COMMAND_HEADER = 0x80 | CAMERA_ADDRESS  # 0x81
REPLY_HEADER = (CAMERA_ADDRESS + 8) << 4  # replies come "from" address 8+1, written as 0x90
TERMINATOR = 0xFF

PAYLOAD_COMMAND = 0x0100
PAYLOAD_INQUIRY = 0x0110
PAYLOAD_REPLY = 0x0111
PAYLOAD_CONTROL = 0x0200
PAYLOAD_CONTROL_REPLY = 0x0201
HEADER_LENGTH = 8
VISCA_PORT = 52381

# Speed steps from the VISCA spec: pan 0x01-0x18 (1-24), tilt 0x01-0x17 (1-23), zoom 0-7. These are the protocol's
# ceilings; no packet ever carries more.
# ptz_test.py clamped pan/tilt to 18/17 *decimal*, which looks like hex misread as decimal.
PAN_STEP_MAX = 0x18
TILT_STEP_MAX = 0x17
ZOOM_STEP_MAX = 7
# A given camera's top steps can be lower; many Sony-style cameras top out at pan 0x18, tilt 0x14. The driver uses
# these until it learns better (the max-speed inquiry, or the numbers in a bench report), so a camera that has
# never been asked is not sent a tilt speed it may refuse.
DEFAULT_PAN_STEP_MAX = 0x18
DEFAULT_TILT_STEP_MAX = 0x14

PAN_LEFT, PAN_RIGHT, PAN_HOLD = 0x01, 0x02, 0x03
TILT_UP, TILT_DOWN, TILT_HOLD = 0x01, 0x02, 0x03
ZOOM_TELE, ZOOM_WIDE = 0x20, 0x30  # OR-ed with the zoom speed 0-7

PRESET_MAX = 0x7F


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _nibbles(value: int, count: int) -> bytes:
    """`value` as `count` bytes that each carry 4 bits (0x0n), most significant first.

    VISCA spreads numbers over "nibble bytes": 0x1234 becomes 01 02 03 04. Negative numbers are sent as
    two's complement of the field width, so -1 in 4 nibbles is 0F 0F 0F 0F.
    """
    value &= (1 << (4 * count)) - 1
    return bytes((value >> (4 * shift)) & 0x0F for shift in reversed(range(count)))


def _from_nibbles(data: bytes) -> int:
    value = 0
    for byte in data:
        value = (value << 4) | (byte & 0x0F)
    return value


def _signed(value: int, nibble_count: int) -> int:
    """Read an unsigned field of `nibble_count` nibbles as two's complement."""
    bits = 4 * nibble_count
    return value - (1 << bits) if value >= 1 << (bits - 1) else value


# --- commands ------------------------------------------------------------------------------------------------


def pan_tilt_drive(pan_step: int, tilt_step: int, pan_max: int = PAN_STEP_MAX, tilt_max: int = TILT_STEP_MAX) -> bytes:
    """Drive pan and tilt at signed speed steps: + is right / up, - is left / down, 0 holds that axis.

    Speeds are clamped to `pan_max` / `tilt_max` (the camera's top steps; never above the protocol's).
    Both zero gives the stop packet, byte for byte the same as ptz_test.py's `cmd_stop`.
    """
    if pan_step == 0 and tilt_step == 0:
        return pan_tilt_stop()
    pan_direction = PAN_HOLD if pan_step == 0 else (PAN_RIGHT if pan_step > 0 else PAN_LEFT)
    tilt_direction = TILT_HOLD if tilt_step == 0 else (TILT_UP if tilt_step > 0 else TILT_DOWN)
    # A held axis still needs a valid speed byte; ptz_test.py sent the minimum (1) there.
    pan_speed = _clamp(abs(pan_step), 1, min(pan_max, PAN_STEP_MAX))
    tilt_speed = _clamp(abs(tilt_step), 1, min(tilt_max, TILT_STEP_MAX))
    return bytes([COMMAND_HEADER, 0x01, 0x06, 0x01, pan_speed, tilt_speed, pan_direction, tilt_direction, TERMINATOR])


def pan_tilt_stop() -> bytes:
    """Stop pan and tilt (speeds 0, both directions "hold"), as ptz_test.py sent it."""
    return bytes([COMMAND_HEADER, 0x01, 0x06, 0x01, 0x00, 0x00, PAN_HOLD, TILT_HOLD, TERMINATOR])


def zoom_drive(step: int) -> bytes:
    """Zoom at a signed speed: + tele (in), - wide (out), 0 stop. |step| is the VISCA zoom speed, capped at 7."""
    if step == 0:
        return zoom_stop()
    speed = _clamp(abs(step), 0, ZOOM_STEP_MAX)
    return bytes([COMMAND_HEADER, 0x01, 0x04, 0x07, (ZOOM_TELE if step > 0 else ZOOM_WIDE) | speed, TERMINATOR])


def zoom_stop() -> bytes:
    return bytes([COMMAND_HEADER, 0x01, 0x04, 0x07, 0x00, TERMINATOR])


def zoom_direct(position: int) -> bytes:
    """Zoom to an absolute position (4 nibbles, 0..0xFFFF; cameras use 0 = wide up to their tele maximum)."""
    return bytes([COMMAND_HEADER, 0x01, 0x04, 0x47]) + _nibbles(_clamp(position, 0, 0xFFFF), 4) + bytes([TERMINATOR])


def absolute_position(pan: int, tilt: int, pan_speed: int, tilt_speed: int,
                      pan_max: int = PAN_STEP_MAX, tilt_max: int = TILT_STEP_MAX) -> bytes:
    """Move to signed raw (pan, tilt) at the given speed steps (clamped to `pan_max` / `tilt_max`)."""
    return (
        bytes([COMMAND_HEADER, 0x01, 0x06, 0x02, _clamp(pan_speed, 1, min(pan_max, PAN_STEP_MAX)),
               _clamp(tilt_speed, 1, min(tilt_max, TILT_STEP_MAX))])
        + _nibbles(pan, 4)
        + _nibbles(tilt, 4)
        + bytes([TERMINATOR])
    )


def relative_position(pan: int, tilt: int, pan_speed: int, tilt_speed: int,
                      pan_max: int = PAN_STEP_MAX, tilt_max: int = TILT_STEP_MAX) -> bytes:
    """Move by signed raw (pan, tilt) from where the camera is (speeds clamped to `pan_max` / `tilt_max`)."""
    return (
        bytes([COMMAND_HEADER, 0x01, 0x06, 0x03, _clamp(pan_speed, 1, min(pan_max, PAN_STEP_MAX)),
               _clamp(tilt_speed, 1, min(tilt_max, TILT_STEP_MAX))])
        + _nibbles(pan, 4)
        + _nibbles(tilt, 4)
        + bytes([TERMINATOR])
    )


def home() -> bytes:
    """Go to the camera's home position (pan 0, tilt 0)."""
    return bytes([COMMAND_HEADER, 0x01, 0x06, 0x04, TERMINATOR])


def preset_recall(n: int) -> bytes:
    """Recall preset `n` (0-based, 0-127). Note the Pelco-D controller counts from 1."""
    return bytes([COMMAND_HEADER, 0x01, 0x04, 0x3F, 0x02, n & PRESET_MAX, TERMINATOR])


def preset_store(n: int) -> bytes:
    """Store the current position as preset `n` (0-based, 0-127). Overwrites what was there."""
    return bytes([COMMAND_HEADER, 0x01, 0x04, 0x3F, 0x01, n & PRESET_MAX, TERMINATOR])


# --- inquiries -------------------------------------------------------------------------------------------------


def pan_tilt_position_inquiry() -> bytes:
    return bytes([COMMAND_HEADER, 0x09, 0x06, 0x12, TERMINATOR])


def zoom_position_inquiry() -> bytes:
    return bytes([COMMAND_HEADER, 0x09, 0x04, 0x47, TERMINATOR])


def pan_tilt_max_speed_inquiry() -> bytes:
    """Answer `90 50 ww zz FF`: the camera's top pan (ww) and tilt (zz) speed steps."""
    return bytes([COMMAND_HEADER, 0x09, 0x06, 0x11, TERMINATOR])


def version_inquiry() -> bytes:
    """Answer `90 50 ...`: vendor, model and ROM version (raw bytes go into the bench report)."""
    return bytes([COMMAND_HEADER, 0x09, 0x00, 0x02, TERMINATOR])


# --- the VISCA/IP header ---------------------------------------------------------------------------------------


def wrap(payload: bytes, payload_type: int, sequence: int) -> bytes:
    """Put the 8-byte VISCA/IP header in front of `payload`."""
    return (
        payload_type.to_bytes(2, "big")
        + len(payload).to_bytes(2, "big")
        + (sequence & 0xFFFFFFFF).to_bytes(4, "big")
        + payload
    )


def sequence_reset() -> bytes:
    """The control message that resets the camera's sequence counter (payload type 0x0200, payload 0x01).

    The sequence number in its header is ignored by the camera; after it, numbering starts again at 1.
    """
    return wrap(b"\x01", PAYLOAD_CONTROL, 0)


# --- replies ---------------------------------------------------------------------------------------------------

ACK = "ack"
COMPLETION = "completion"
ANSWER = "answer"  # an inquiry answer: 90 50 <data> FF
ERROR = "error"
CONTROL = "control"  # a reply to the sequence reset (or a control-level error)
UNKNOWN = "unknown"

# VISCA error codes (third byte of `90 6y ee FF`).
ERROR_NAMES = {
    0x01: "message length error",
    0x02: "syntax error",
    0x03: "command buffer full",
    0x04: "command cancelled",
    0x05: "no socket",
    0x41: "command not executable",
}


@dataclass(frozen=True)
class Reply:
    """One datagram from the camera."""

    payload_type: int | None  # None when the camera sent a bare VISCA payload without the IP header
    sequence: int | None
    payload: bytes

    @property
    def kind(self) -> str:
        if self.payload_type in (PAYLOAD_CONTROL, PAYLOAD_CONTROL_REPLY):
            return CONTROL
        p = self.payload
        if len(p) < 3 or p[0] & 0xF0 != REPLY_HEADER or p[-1] != TERMINATOR:
            return UNKNOWN
        high = p[1] & 0xF0
        if high == 0x40:
            return ACK
        if high == 0x50:
            return COMPLETION if len(p) == 3 else ANSWER
        if high == 0x60:
            return ERROR
        return UNKNOWN

    @property
    def error_code(self) -> int | None:
        if self.kind == ERROR and len(self.payload) >= 4:
            return self.payload[2]
        return None


_KNOWN_PAYLOAD_TYPES = (PAYLOAD_REPLY, PAYLOAD_CONTROL_REPLY, PAYLOAD_COMMAND, PAYLOAD_INQUIRY, PAYLOAD_CONTROL)


def parse_replies(datagram: bytes) -> list[Reply]:
    """Every VISCA/IP message in one datagram, in order (an empty list if it is not VISCA at all).

    A datagram usually holds one message: the 8-byte header, then `length` payload bytes. But some cameras (the
    PTZ probed on 2026-11-15) put several messages in one datagram, each with its own header and length, e.g.
    ACK + completion + inquiry answer:

        01 11 00 03 <seq> 90 48 FF | 01 11 00 03 <seq> 90 58 FF | 01 11 00 0B <seq> 90 50 <8 nibbles> FF

    Reading only the first message would lose the answer behind the ACK. Bytes that don't form a whole message
    at the end (a truncated datagram) are ignored.

    Some cheap cameras answer with the bare VISCA payload (no 8-byte header). That is accepted too, with
    `payload_type` and `sequence` set to None; several bare payloads in a row are split at their 0xFF terminators.
    """
    replies: list[Reply] = []
    offset = 0
    while len(datagram) - offset >= HEADER_LENGTH + 1:
        payload_type = int.from_bytes(datagram[offset:offset + 2], "big")
        length = int.from_bytes(datagram[offset + 2:offset + 4], "big")
        end = offset + HEADER_LENGTH + length
        if payload_type not in _KNOWN_PAYLOAD_TYPES or length == 0 or end > len(datagram):
            break
        sequence = int.from_bytes(datagram[offset + 4:offset + 8], "big")
        replies.append(Reply(payload_type, sequence, bytes(datagram[offset + HEADER_LENGTH:end])))
        offset = end
    if replies:
        return replies
    # No header: bare VISCA payloads, each one 9x ... FF.
    start = 0
    for index, byte in enumerate(datagram):
        if byte == TERMINATOR:
            chunk = bytes(datagram[start:index + 1])
            if len(chunk) >= 3 and chunk[0] & 0xF0 == REPLY_HEADER:
                replies.append(Reply(None, None, chunk))
            else:
                return []  # not VISCA: don't guess at the rest
            start = index + 1
    return replies if start == len(datagram) else []


def parse_reply(datagram: bytes) -> Reply | None:
    """The first VISCA/IP message of a datagram (see `parse_replies`), or None if it is not VISCA at all.

    Fine for requests (one message each). For what a camera sends back, use `parse_replies`: a reply datagram
    can hold several messages.
    """
    replies = parse_replies(datagram)
    return replies[0] if replies else None


def parse_pan_tilt_position(payload: bytes) -> tuple[int, int] | None:
    """Signed raw (pan, tilt) from `90 50 <pan nibbles> <tilt nibbles> FF`, or None.

    Most cameras use 4 nibbles per axis. Newer Sony models use 5 for pan and 4 for tilt, which is detected from
    the length. ptz_test.py read the sign with `> 0x8000`, which reads 0x8000 itself as positive. Here it is
    proper two's complement.
    """
    if len(payload) < 11 or payload[0] & 0xF0 != REPLY_HEADER or payload[1] != 0x50 or payload[-1] != TERMINATOR:
        return None
    data = payload[2:-1]
    if len(data) == 8:
        pan_nibbles, tilt_nibbles = 4, 4
    elif len(data) == 9:
        pan_nibbles, tilt_nibbles = 5, 4
    else:
        return None
    pan = _signed(_from_nibbles(data[:pan_nibbles]), pan_nibbles)
    tilt = _signed(_from_nibbles(data[pan_nibbles:]), tilt_nibbles)
    return pan, tilt


def parse_zoom_position(payload: bytes) -> int | None:
    """Raw zoom position from `90 50 0p 0q 0r 0s FF`, or None."""
    if len(payload) != 7 or payload[0] & 0xF0 != REPLY_HEADER or payload[1] != 0x50 or payload[-1] != TERMINATOR:
        return None
    return _from_nibbles(payload[2:6])


def parse_max_speed(payload: bytes) -> tuple[int, int] | None:
    """(top pan step, top tilt step) from `90 50 ww zz FF`, or None."""
    if len(payload) != 5 or payload[0] & 0xF0 != REPLY_HEADER or payload[1] != 0x50:
        return None
    return payload[2], payload[3]


# --- helpers the fake camera uses to answer ---------------------------------------------------------------------


def encode_pan_tilt_answer(pan: int, tilt: int) -> bytes:
    return bytes([REPLY_HEADER, 0x50]) + _nibbles(pan, 4) + _nibbles(tilt, 4) + bytes([TERMINATOR])


def encode_zoom_answer(zoom: int) -> bytes:
    return bytes([REPLY_HEADER, 0x50]) + _nibbles(zoom, 4) + bytes([TERMINATOR])


def decode_signed_nibbles(data: bytes) -> int:
    """Signed value of a nibble field (the fake camera reads absolute-position targets with it)."""
    return _signed(_from_nibbles(data), len(data))


def decode_nibbles(data: bytes) -> int:
    return _from_nibbles(data)


def ack(socket_number: int = 1) -> bytes:
    return bytes([REPLY_HEADER, 0x40 | socket_number, TERMINATOR])


def completion(socket_number: int = 1) -> bytes:
    return bytes([REPLY_HEADER, 0x50 | socket_number, TERMINATOR])


def error(code: int, socket_number: int = 1) -> bytes:
    return bytes([REPLY_HEADER, 0x60 | socket_number, code, TERMINATOR])
