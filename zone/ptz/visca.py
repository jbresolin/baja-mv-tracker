"""VISCA-over-IP driver for the SMTAV PTZs (and any camera that speaks Sony's VISCA over UDP port 52381).

How it works, for a newcomer
----------------------------
Each method builds the VISCA payload (see `visca_packets`), puts the 8-byte header with the next sequence number
in front of it, sends it as one UDP datagram, and waits for the camera's reply, but never longer than `timeout`
(200 ms by default). The reply that decides the result is:

- for a command: the first ACK ("accepted"), completion ("done") or error. A completion that arrives later, such
  as the end of an absolute move, is still noted; see `completion_time()`.
- for an inquiry: the answer carrying data.

Only one request is on the wire at a time: a lock makes the driver thread-safe, so the controller and the dead-man
thread can both call it. The timeout is one budget for the whole call, counted from when it starts: waiting for the
lock, a sequence reset that may be due, and waiting for the reply all come out of the same `timeout`.

The dead-man switch (TTL)
-------------------------
The camera itself has no dead-man: if the controller crashes after "pan right", the camera keeps turning. So
`pan_tilt_velocity(..., ttl=0.3)` sets a deadline: if no newer pan/tilt command arrives within 0.3 s, a background
thread sends `stop`. The controller keeps the motion alive by repeating its command, which it does every cycle anyway.

Logging
-------
With `log_path`, every datagram sent and received is appended to a JSONL file with both clocks: `mono` (monotonic,
for durations) and `wall` (Unix time, to line up with video recordings). The bench script turns this on.
"""
from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from collections import deque
from typing import Callable, Protocol

from zone.clock import Clock, SystemClock
from zone.ptz import visca_packets as packets
from zone.ptz.interface import ACKED, ANSWERED, COMPLETED, ERROR, TIMEOUT, CommandResult

DEFAULT_TIMEOUT = 0.2
WATCHDOG_PERIOD = 0.01  # how often the dead-man thread checks deadlines (seconds, real time)
RECENT_SEQUENCES = 64  # how many sent sequence numbers we remember, to recognise late replies


class Transport(Protocol):
    """A datagram link to one camera. The real one is UDP; tests plug in an in-memory fake."""

    def send(self, datagram: bytes) -> None:
        """Send one datagram. Network errors count as a lost packet: must not raise."""

    def receive(self, timeout: float) -> bytes | None:
        """Wait up to `timeout` seconds for one datagram; None if nothing came."""

    def close(self) -> None:
        """Release the socket."""


class UdpTransport:
    """The real network: a UDP socket that sends to the camera and accepts replies from the camera's IP.

    Deliberately not a "connected" UDP socket: a connected socket drops replies that come back from a different
    source port, and some cameras reply from one. ptz_test.py used sendto/recvfrom with the SMTAV the same way.
    """

    def __init__(self, host: str, port: int = packets.VISCA_PORT) -> None:
        self._address = (socket.gethostbyname(host), port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind(("", 0))

    def send(self, datagram: bytes) -> None:
        # An unreachable camera shows up as missing replies, not as an exception here.
        with contextlib.suppress(OSError):
            self._socket.sendto(datagram, self._address)

    def receive(self, timeout: float) -> bytes | None:
        deadline = time.monotonic() + max(timeout, 0.0)
        while True:
            remaining = deadline - time.monotonic()
            self._socket.settimeout(max(remaining, 1e-4))
            try:
                datagram, sender = self._socket.recvfrom(2048)
            except (socket.timeout, BlockingIOError, InterruptedError):
                return None
            except OSError:
                # Windows reports an earlier ICMP "port unreachable" on the next recv (WSAECONNRESET):
                # not a reply, keep waiting for the rest of the timeout.
                datagram, sender = None, None
            if datagram is not None and sender[0] == self._address[0]:
                return datagram
            if time.monotonic() >= deadline:
                return None

    def close(self) -> None:
        self._socket.close()


class ViscaPtz:
    """One VISCA-over-IP camera. Implements `zone.ptz.interface.Ptz`.

    Arguments:
        host, port: the camera's address.
        timeout: the longest any call waits for the camera (seconds).
        clock: where time comes from (tests pass a `FakeClock`).
        log_path: append every datagram to this JSONL file (None = no log).
        transport: the datagram link; default a real UDP socket to host:port.
        watchdog: run the dead-man thread. Tests pass False and call `check_deadlines()` themselves.
        reset_sequence: send the sequence-reset control message on connect.
    """

    def __init__(
        self,
        host: str,
        port: int = packets.VISCA_PORT,
        timeout: float = DEFAULT_TIMEOUT,
        clock: Clock | None = None,
        log_path: str | None = None,
        transport: Transport | None = None,
        watchdog: bool = True,
        reset_sequence: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.clock: Clock = clock or SystemClock()
        self._transport: Transport = transport or UdpTransport(host, port)
        self._lock = threading.Lock()  # one request on the wire at a time
        self._log_lock = threading.Lock()
        self._log_file = open(log_path, "a", encoding="utf-8") if log_path else None  # noqa: SIM115
        self._sequence = 1
        self._recent: deque[int] = deque(maxlen=RECENT_SEQUENCES)
        self._late_completions: dict[int, tuple[float, str]] = {}
        self._needs_reset = False
        # Dead-man deadlines (clock.monotonic() values), None when nothing is armed.
        self._pan_tilt_deadline: float | None = None
        self._zoom_deadline: float | None = None
        self._closed = threading.Event()
        # What the driver noticed about the camera. The bench report copies these.
        self.stats = {"sent": 0, "timeouts": 0, "errors": 0, "unmatched_replies": 0, "ttl_stops": 0}
        self.last_inquiry: CommandResult | None = None
        self.sequence_reset_status: str | None = None
        if reset_sequence:
            self.reset_sequence()
        self._watchdog: threading.Thread | None = None
        if watchdog:
            self._watchdog = threading.Thread(target=self._watchdog_loop, name="visca-deadman", daemon=True)
            self._watchdog.start()

    # --- the Ptz interface ---------------------------------------------------------------------------------------

    def pan_tilt_velocity(self, pan_step: int, tilt_step: int, ttl: float | None = None) -> CommandResult:
        moving = pan_step != 0 or tilt_step != 0
        return self.command(packets.pan_tilt_drive(pan_step, tilt_step), what="pan_tilt_velocity",
                            pan_tilt_ttl=ttl if moving else None)

    def stop(self) -> CommandResult:
        return self.command(packets.pan_tilt_stop(), what="stop", pan_tilt_ttl=None)

    def zoom_velocity(self, step: int, ttl: float | None = None) -> CommandResult:
        return self.command(packets.zoom_drive(step), what="zoom_velocity", zoom_ttl=ttl if step != 0 else None)

    def zoom_stop(self) -> CommandResult:
        return self.command(packets.zoom_stop(), what="zoom_stop", zoom_ttl=None)

    def zoom_direct(self, position: int) -> CommandResult:
        return self.command(packets.zoom_direct(position), what="zoom_direct", zoom_ttl=None)

    def absolute_position(self, pan: int, tilt: int, pan_speed: int, tilt_speed: int) -> CommandResult:
        return self.command(packets.absolute_position(pan, tilt, pan_speed, tilt_speed), what="absolute_position",
                            pan_tilt_ttl=None)

    def relative_position(self, pan: int, tilt: int, pan_speed: int, tilt_speed: int) -> CommandResult:
        return self.command(packets.relative_position(pan, tilt, pan_speed, tilt_speed), what="relative_position",
                            pan_tilt_ttl=None)

    def inquire_pan_tilt(self) -> tuple[int, int] | None:
        result = self.inquire(packets.pan_tilt_position_inquiry(), packets.parse_pan_tilt_position, "inquire_pan_tilt")
        return result.value if result.ok else None  # type: ignore[return-value]

    def inquire_zoom(self) -> int | None:
        result = self.inquire(packets.zoom_position_inquiry(), packets.parse_zoom_position, "inquire_zoom")
        return result.value if result.ok else None  # type: ignore[return-value]

    def preset_recall(self, n: int) -> CommandResult:
        return self.command(packets.preset_recall(n), what="preset_recall", pan_tilt_ttl=None, zoom_ttl=None)

    def preset_store(self, n: int) -> CommandResult:
        return self.command(packets.preset_store(n), what="preset_store")

    def home(self) -> CommandResult:
        return self.command(packets.home(), what="home", pan_tilt_ttl=None)

    def close(self) -> None:
        """Stop the dead-man thread and close the socket and the log. Does not move the camera."""
        self._closed.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=1.0)
        self._transport.close()
        with self._log_lock:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None

    # --- extras the bench script uses ----------------------------------------------------------------------------

    def inquire_max_speed(self) -> tuple[int, int] | None:
        result = self.inquire(packets.pan_tilt_max_speed_inquiry(), packets.parse_max_speed, "inquire_max_speed")
        return result.value if result.ok else None  # type: ignore[return-value]

    def inquire_version(self) -> bytes | None:
        result = self.inquire(packets.version_inquiry(), None, "inquire_version")
        return result.payload if result.ok else None

    def completion_time(self, sequence: int) -> float | None:
        """When the late completion for command `sequence` arrived (clock.monotonic()), if it has.

        Reads any datagrams already waiting first. Used to time absolute moves: the camera sends "done" when the
        move ends.
        """
        self.drain_replies()
        entry = self._late_completions.get(sequence)
        return entry[0] if entry and entry[1] == packets.COMPLETION else None

    def drain_replies(self) -> None:
        """Read (without waiting) any replies that arrived after their command stopped waiting."""
        if not self._lock.acquire(timeout=self.timeout):
            return
        try:
            while (datagram := self._transport.receive(0.0)) is not None:
                self._note_stray(datagram, self.clock.monotonic())
        finally:
            self._lock.release()

    def reset_sequence(self) -> str:
        """Send the sequence-reset control message; numbering restarts at 1. Returns "acked" or "timeout"."""
        with self._lock:
            return self._reset_sequence_locked()

    def _reset_sequence_locked(self, deadline: float | None = None) -> str:
        """Send the reset and wait for its ACK until `deadline` (clock.monotonic(); default: one timeout from now)."""
        sent_at = self.clock.monotonic()
        datagram = packets.sequence_reset()
        self._transport.send(datagram)
        self._log("tx", datagram, sent_at, what="sequence_reset")
        self._sequence = 1
        self._recent.clear()
        status = TIMEOUT
        if deadline is None:
            deadline = sent_at + self.timeout
        while (remaining := deadline - self.clock.monotonic()) > 0:
            reply_datagram = self._transport.receive(remaining)
            if reply_datagram is None:
                break
            self._log("rx", reply_datagram, self.clock.monotonic())
            reply = packets.parse_reply(reply_datagram)
            if reply is not None and reply.kind == packets.CONTROL:
                status = ACKED
                break
        self._needs_reset = False
        self.sequence_reset_status = status
        return status

    # --- the request/response core -------------------------------------------------------------------------------

    _KEEP = object()  # "leave this dead-man deadline as it is"

    def command(
        self,
        payload: bytes,
        what: str = "command",
        wait_for_completion: bool = False,
        pan_tilt_ttl: object = _KEEP,
        zoom_ttl: object = _KEEP,
    ) -> CommandResult:
        """Send a VISCA command and wait for its ACK (or its completion, with `wait_for_completion`).

        `pan_tilt_ttl` / `zoom_ttl`: None disarms that dead-man; a number arms it for that many seconds from the
        moment of sending. Leaving them out keeps the current deadline. They are set under the same lock as the
        send, so the dead-man thread can never slip an old stop in after a newer velocity command.
        """
        def arm(now: float) -> None:
            if pan_tilt_ttl is not ViscaPtz._KEEP:
                self._pan_tilt_deadline = None if pan_tilt_ttl is None else now + float(pan_tilt_ttl)  # type: ignore[arg-type]
            if zoom_ttl is not ViscaPtz._KEEP:
                self._zoom_deadline = None if zoom_ttl is None else now + float(zoom_ttl)  # type: ignore[arg-type]

        return self._request(payload, packets.PAYLOAD_COMMAND, None, what, wait_for_completion, arm)

    def inquire(
        self, payload: bytes, parser: Callable[[bytes], object] | None, what: str = "inquiry"
    ) -> CommandResult:
        """Send a VISCA inquiry; the result's `value` is `parser(answer payload)`, or the raw payload without one."""
        result = self._request(payload, packets.PAYLOAD_INQUIRY, parser, what, False, None)
        self.last_inquiry = result
        return result

    def _request(
        self,
        payload: bytes,
        payload_type: int,
        parser: Callable[[bytes], object] | None,
        what: str,
        wait_for_completion: bool,
        arm: Callable[[float], None] | None,
    ) -> CommandResult:
        started = self.clock.monotonic()
        # One deadline for the whole call. Everything below waits only for what is left of it, so the caller is
        # never blocked longer than `timeout` (before, lock wait + sequence reset + reply wait could add up to 3x).
        deadline = started + self.timeout
        if not self._lock.acquire(timeout=max(0.0, deadline - self.clock.monotonic())):
            # Someone else holds the wire for longer than our whole budget: give up rather than block.
            self.stats["timeouts"] += 1
            return CommandResult(TIMEOUT, sent_at=started)
        try:
            return self._exchange_locked(payload, payload_type, parser, what, wait_for_completion, arm, deadline)
        finally:
            self._lock.release()

    def _exchange_locked(
        self,
        payload: bytes,
        payload_type: int,
        parser: Callable[[bytes], object] | None,
        what: str,
        wait_for_completion: bool,
        arm: Callable[[float], None] | None,
        deadline: float | None = None,
    ) -> CommandResult:
        """Send one request and wait for its reply until `deadline` (default: one timeout from now).
        The caller holds `self._lock`.

        If a sequence reset is due it goes first, out of the same budget. If the reset used it all up, the request
        is still sent (a stop should go out even then) but not waited for.
        """
        if deadline is None:
            deadline = self.clock.monotonic() + self.timeout
        if self._needs_reset:
            self._reset_sequence_locked(deadline)
        sequence = self._sequence
        self._sequence = ((self._sequence + 1) & 0xFFFFFFFF) or 1
        self._recent.append(sequence)
        datagram = packets.wrap(payload, payload_type, sequence)
        sent_at = self.clock.monotonic()
        if arm is not None:
            arm(sent_at)
        self._transport.send(datagram)
        self.stats["sent"] += 1
        self._log("tx", datagram, sent_at, what=what, seq=sequence)
        return self._await_reply(sequence, sent_at, payload_type == packets.PAYLOAD_INQUIRY, parser,
                                 wait_for_completion, deadline)

    def _await_reply(
        self,
        sequence: int,
        sent_at: float,
        is_inquiry: bool,
        parser: Callable[[bytes], object] | None,
        wait_for_completion: bool,
        deadline: float,
    ) -> CommandResult:
        acked: CommandResult | None = None
        while (remaining := deadline - self.clock.monotonic()) > 0:
            datagram = self._transport.receive(remaining)
            if datagram is None:
                break
            now = self.clock.monotonic()
            self._log("rx", datagram, now)
            reply = packets.parse_reply(datagram)
            if reply is None:
                continue
            if reply.kind == packets.CONTROL:
                # 0F 01 = the camera's sequence counter disagrees with ours: reset before the next request.
                if reply.payload[:2] == b"\x0f\x01":
                    self._needs_reset = True
                    self.stats["errors"] += 1
                    return CommandResult(ERROR, sequence, sent_at, now, error_code=0x0F01, payload=reply.payload)
                continue
            if reply.sequence is not None and reply.sequence != sequence:
                if reply.sequence in self._recent:
                    self._remember_late(reply, now)  # an answer to an earlier request
                    continue
                # A sequence number we never sent: some cameras don't echo it. Take it as ours, but count it.
                self.stats["unmatched_replies"] += 1
            kind = reply.kind
            if kind == packets.ERROR:
                self.stats["errors"] += 1
                return CommandResult(ERROR, sequence, sent_at, now, reply.error_code, reply.payload)
            if is_inquiry:
                if kind == packets.ANSWER:
                    value = parser(reply.payload) if parser else reply.payload
                    if parser is not None and value is None:
                        # It answered, but not in a format we understand: report it as an error.
                        return CommandResult(ERROR, sequence, sent_at, now, None, reply.payload)
                    return CommandResult(ANSWERED, sequence, sent_at, now, payload=reply.payload, value=value)
                continue
            if kind == packets.COMPLETION:
                return CommandResult(COMPLETED, sequence, sent_at, now, payload=reply.payload)
            if kind == packets.ACK:
                acked = CommandResult(ACKED, sequence, sent_at, now, payload=reply.payload)
                if not wait_for_completion:
                    return acked
        if acked is not None:
            return acked
        self.stats["timeouts"] += 1
        self._log_event("timeout", sequence=sequence)
        return CommandResult(TIMEOUT, sequence, sent_at)

    def _note_stray(self, datagram: bytes, now: float) -> None:
        self._log("rx", datagram, now)
        reply = packets.parse_reply(datagram)
        if reply is not None and reply.sequence is not None and reply.sequence in self._recent:
            self._remember_late(reply, now)

    def _remember_late(self, reply: packets.Reply, now: float) -> None:
        if reply.sequence is not None and reply.kind in (packets.COMPLETION, packets.ERROR):
            self._late_completions[reply.sequence] = (now, reply.kind)
            if len(self._late_completions) > RECENT_SEQUENCES:
                self._late_completions.pop(next(iter(self._late_completions)))

    # --- the dead-man switch -------------------------------------------------------------------------------------

    def check_deadlines(self) -> None:
        """Send stop / zoom stop for every TTL that has run out. The watchdog thread calls this ~100x a second.

        The check and the stop happen under the request lock, so a velocity command that renews the deadline is
        either entirely before (and the deadline is no longer expired) or entirely after (and moves the camera
        again).
        """
        if not self._lock.acquire(timeout=self.timeout):
            return
        try:
            now = self.clock.monotonic()
            if self._pan_tilt_deadline is not None and now >= self._pan_tilt_deadline:
                self._pan_tilt_deadline = None
                self.stats["ttl_stops"] += 1
                self._log_event("ttl_stop", axis="pan_tilt")
                self._exchange_locked(packets.pan_tilt_stop(), packets.PAYLOAD_COMMAND, None, "ttl_stop", False, None)
            if self._zoom_deadline is not None and now >= self._zoom_deadline:
                self._zoom_deadline = None
                self.stats["ttl_stops"] += 1
                self._log_event("ttl_stop", axis="zoom")
                self._exchange_locked(packets.zoom_stop(), packets.PAYLOAD_COMMAND, None, "ttl_zoom_stop", False, None)
        finally:
            self._lock.release()

    def _watchdog_loop(self) -> None:
        while not self._closed.wait(WATCHDOG_PERIOD):
            try:
                self.check_deadlines()
            except Exception as exc:  # noqa: BLE001 - the dead-man must outlive any single failure
                self._log_event("watchdog_error", error=repr(exc))

    # --- logging -------------------------------------------------------------------------------------------------

    def _log(self, direction: str, datagram: bytes, mono: float, **extra: object) -> None:
        if self._log_file is None:
            return
        entry: dict[str, object] = {"mono": round(mono, 6), "wall": round(self.clock.wall(), 6), "dir": direction,
                                    "hex": datagram.hex(" ")}
        if direction == "rx":
            reply = packets.parse_reply(datagram)
            entry["kind"] = reply.kind if reply else "garbage"
            if reply is not None and reply.sequence is not None:
                entry["seq"] = reply.sequence
        entry.update(extra)
        self._write(entry)

    def _log_event(self, event: str, **extra: object) -> None:
        if self._log_file is None:
            return
        entry: dict[str, object] = {"mono": round(self.clock.monotonic(), 6), "wall": round(self.clock.wall(), 6),
                                    "event": event}
        entry.update(extra)
        self._write(entry)

    def _write(self, entry: dict[str, object]) -> None:
        with self._log_lock:
            if self._log_file is not None:
                self._log_file.write(json.dumps(entry) + "\n")
                self._log_file.flush()
