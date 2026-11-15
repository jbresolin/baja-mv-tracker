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

A stop can get lost (it is one UDP datagram). So a stop that the camera doesn't acknowledge, whether the dead-man's
or one the caller sent with `stop()` / `zoom_stop()`, stays "armed": the dead-man thread sends it again on every
tick until the camera ACKs it, for up to `STOP_RETRY_WINDOW_S` (2 s), then gives up and logs it. A newer command for
that axis cancels the retries. When the Python process exits normally (including Ctrl+C) an `atexit` handler sends
stop and zoom stop as a last resort. A process that is killed (Task Manager, closing the console window, a power
cut) runs no Python code at all, so nothing can stop the camera then.

Cameras that don't copy our sequence number
-------------------------------------------
Most cameras copy the request's sequence number into their reply. Some don't, and then a reply's number can't
say which request it answers: cheap ones answer with 0, and the PTZ probed on 2026-11-15 numbers every reply from
**its own counter** (about 43000 and rising, never a number we sent). Two replies in a row with numbers we never
sent (or one numbered 0) put the driver in "no echo" mode (`sequence_echo = False`). That camera also:

- puts several messages in one datagram (`visca_packets.parse_replies` reads all of them, never only the first);
- answers an inquiry with ACK + completion (`90 4y FF`, `90 5y FF`) and only sometimes (~17 %) with the data
  (`90 50 ... FF`);
- answers late (0-0.5 s, ~3 s at first) and in batches, so a reply may belong to an earlier request;
- sends position data only at rest, as a pair (11-byte pan/tilt + 7-byte `90 50`, the latter also when no zoom was
  asked), usually in the batch of a *later* request (second probe, 2026-11-15). It gives no usable position
  feedback: the bench concludes Branch B, and in Branch B nothing in the zone reads positions.

So in "no echo" mode:

- an inquiry is answered only by a `90 50` answer its parser accepts (the right length: 11 bytes for pan/tilt, 7
  for zoom). ACKs, completions, errors and answers of the wrong length are ignored (`stats["ignored_replies"]`).
  No data within the timeout is a TIMEOUT of that inquiry, as for any other camera: not an error;
- a command is answered by the first ACK, completion or error, whatever its number (nothing better is possible);
- whatever is already waiting when a request is sent is read and dropped first: it can't be that request's reply.

`last_reply_at` is when anything last came from the camera: a camera that ACKs an inquiry but sends no data is
still there (the zone's health probe uses it).

Logging
-------
With `log_path`, every datagram sent and received is appended to a JSONL file with both clocks: `mono` (monotonic,
for durations) and `wall` (Unix time, to line up with video recordings). The bench script turns this on.
"""
from __future__ import annotations

import atexit
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
RECEIVE_BUFFER = 65535  # bytes per datagram: a batch of several replies must never be cut short
# Replies numbered with numbers we never sent, in a row, that make the driver stop trusting sequence numbers:
# while it doesn't know yet (a camera with its own counter does this from its first reply), and after the camera
# had echoed one (an own counter can hit one of our numbers by chance, once; an echoing camera never sends a run).
UNKNOWN_IN_A_ROW_UNDECIDED = 2
UNKNOWN_IN_A_ROW_AFTER_ECHO = 8
# Lengths of position answers (zoom 7; pan/tilt 11, or 12 with a 5-nibble pan): in "no echo" mode these are never
# taken as the answer to a raw inquiry such as the version.
POSITION_ANSWER_LENGTHS = (7, 11, 12)
DRAIN_LIMIT = 64  # at most this many waiting datagrams are dropped before a request (no endless loop on a flood)
STOP_RETRY_WINDOW_S = 2.0  # keep re-sending an unacknowledged stop for this long, then give up (and log it)


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
                datagram, sender = self._socket.recvfrom(RECEIVE_BUFFER)
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
        pan_step_max, tilt_step_max: the camera's top speed steps, e.g. from a bench report. Speeds above them are
            clamped. Left out: the max-speed inquiry's answer once `inquire_max_speed()` has been called, until
            then pan 0x18 / tilt 0x14.
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
        pan_step_max: int | None = None,
        tilt_step_max: int | None = None,
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
        # Does the camera copy our sequence number into its replies? None until we know. Most do (True). Some
        # cheap ones always answer with sequence 0 (False): for those, a reply's number can't tell which request
        # it answers, so we take whatever comes as the answer to the current one.
        self.sequence_echo: bool | None = None
        self._unknown_in_a_row = 0  # replies in a row carrying a number we never sent (see _camera_ignores_sequence)
        self.last_reply_at: float | None = None  # clock.monotonic() when anything last came from the camera
        # Dead-man deadlines (clock.monotonic() values), None when nothing is armed.
        self._pan_tilt_deadline: float | None = None
        self._zoom_deadline: float | None = None
        # While a stop is being retried (it wasn't ACKed), the time we give up retrying it. None: not retrying.
        self._pan_tilt_stop_retry_until: float | None = None
        self._zoom_stop_retry_until: float | None = None
        self._closed = threading.Event()
        # What the driver noticed about the camera. The bench report copies these.
        # stop_retries: stops sent again because the previous one wasn't ACKed; stop_failures: gave up after 2 s.
        # ignored_replies: "no echo" mode only, replies that can't answer the request waited for (an inquiry's ACK,
        # an answer of the wrong length...); dropped_waiting: replies already waiting when a request went out.
        self.stats = {"sent": 0, "timeouts": 0, "errors": 0, "unmatched_replies": 0, "ttl_stops": 0,
                      "stop_retries": 0, "stop_failures": 0, "ignored_replies": 0, "dropped_waiting": 0}
        self.last_inquiry: CommandResult | None = None
        self.sequence_reset_status: str | None = None
        # Top speed steps: speeds sent are clamped to these. `step_max_source` says where they came from.
        self.pan_step_max = pan_step_max or packets.DEFAULT_PAN_STEP_MAX
        self.tilt_step_max = tilt_step_max or packets.DEFAULT_TILT_STEP_MAX
        self.step_max_source = "given" if (pan_step_max or tilt_step_max) else "default"
        if reset_sequence:
            self.reset_sequence()
        # Last resort on a normal exit (end of script, an exception, Ctrl+C): stop the camera. close() removes it.
        atexit.register(self._stop_at_exit)
        self._watchdog: threading.Thread | None = None
        if watchdog:
            self._watchdog = threading.Thread(target=self._watchdog_loop, name="visca-deadman", daemon=True)
            self._watchdog.start()

    # --- the Ptz interface ---------------------------------------------------------------------------------------

    def pan_tilt_velocity(self, pan_step: int, tilt_step: int, ttl: float | None = None) -> CommandResult:
        if pan_step == 0 and tilt_step == 0:
            return self.stop()  # the same packet; going through stop() gives it the retries
        return self.command(packets.pan_tilt_drive(pan_step, tilt_step, self.pan_step_max, self.tilt_step_max),
                            what="pan_tilt_velocity", pan_tilt_ttl=ttl)

    def stop(self) -> CommandResult:
        """Stop pan and tilt. If the camera doesn't ACK it, the dead-man thread keeps re-sending it (up to 2 s)."""
        return self._stop_command(packets.pan_tilt_stop(), "stop", "pan_tilt")

    def zoom_velocity(self, step: int, ttl: float | None = None) -> CommandResult:
        if step == 0:
            return self.zoom_stop()
        return self.command(packets.zoom_drive(step), what="zoom_velocity", zoom_ttl=ttl)

    def zoom_stop(self) -> CommandResult:
        """Stop zooming, with the same retries as `stop()`."""
        return self._stop_command(packets.zoom_stop(), "zoom_stop", "zoom")

    def zoom_direct(self, position: int) -> CommandResult:
        return self.command(packets.zoom_direct(position), what="zoom_direct", zoom_ttl=None)

    def absolute_position(self, pan: int, tilt: int, pan_speed: int, tilt_speed: int) -> CommandResult:
        return self.command(packets.absolute_position(pan, tilt, pan_speed, tilt_speed, self.pan_step_max,
                                                      self.tilt_step_max), what="absolute_position", pan_tilt_ttl=None)

    def relative_position(self, pan: int, tilt: int, pan_speed: int, tilt_speed: int) -> CommandResult:
        return self.command(packets.relative_position(pan, tilt, pan_speed, tilt_speed, self.pan_step_max,
                                                      self.tilt_step_max), what="relative_position", pan_tilt_ttl=None)

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
        atexit.unregister(self._stop_at_exit)
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
        """Ask the camera for its top (pan, tilt) speed steps. A plausible answer becomes the clamp for later
        commands, unless top steps were given explicitly (constructor or `set_step_max`)."""
        result = self.inquire(packets.pan_tilt_max_speed_inquiry(), packets.parse_max_speed, "inquire_max_speed")
        if not result.ok:
            return None
        pan_max, tilt_max = result.value  # type: ignore[misc]
        if self.step_max_source != "given" and 1 <= pan_max <= packets.PAN_STEP_MAX and 1 <= tilt_max <= 0x18:
            self.pan_step_max, self.tilt_step_max = pan_max, min(tilt_max, packets.TILT_STEP_MAX)
            self.step_max_source = "camera"
        return result.value  # type: ignore[return-value]

    def set_step_max(self, pan_step_max: int, tilt_step_max: int) -> None:
        """Set the top speed steps explicitly (e.g. from a bench report or the command line); speeds above them
        are clamped from now on, and the max-speed inquiry no longer changes them."""
        self.pan_step_max = max(1, min(pan_step_max, packets.PAN_STEP_MAX))
        self.tilt_step_max = max(1, min(tilt_step_max, packets.TILT_STEP_MAX))
        self.step_max_source = "given"

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
            for _ in range(DRAIN_LIMIT):
                datagram = self._transport.receive(0.0)
                if datagram is None:
                    break
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
        # `_recent` is deliberately kept: a reply to a request sent before the reset can still arrive late, and it
        # must be recognised as late rather than taken for the answer to a new request. Numbers drop out of it as
        # they are reused (see _exchange_locked).
        status = TIMEOUT
        if deadline is None:
            deadline = sent_at + self.timeout
        while (remaining := deadline - self.clock.monotonic()) > 0:
            reply_datagram = self._transport.receive(remaining)
            if reply_datagram is None:
                break
            self.last_reply_at = self.clock.monotonic()
            self._log("rx", reply_datagram, self.last_reply_at)
            if any(reply.kind == packets.CONTROL for reply in packets.parse_replies(reply_datagram)):
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
            # A newer command for an axis replaces whatever was pending for it, including a stop being retried.
            if pan_tilt_ttl is not ViscaPtz._KEEP:
                self._pan_tilt_deadline = None if pan_tilt_ttl is None else now + float(pan_tilt_ttl)  # type: ignore[arg-type]
                self._pan_tilt_stop_retry_until = None
            if zoom_ttl is not ViscaPtz._KEEP:
                self._zoom_deadline = None if zoom_ttl is None else now + float(zoom_ttl)  # type: ignore[arg-type]
                self._zoom_stop_retry_until = None

        return self._request(payload, packets.PAYLOAD_COMMAND, None, what, wait_for_completion, arm)

    def _stop_command(self, payload: bytes, what: str, axis: str) -> CommandResult:
        """Send a stop for `axis` ("pan_tilt" or "zoom"). If it isn't ACKed, leave that axis's dead-man deadline
        expired so the dead-man thread re-sends the stop on its next tick (see `check_deadlines`).

        Both steps happen under the request lock, so a velocity command can't slip in between and be stopped."""
        def arm(now: float) -> None:
            setattr(self, f"_{axis}_deadline", None)
            setattr(self, f"_{axis}_stop_retry_until", None)

        def after(result: CommandResult) -> None:
            if not result.ok and not self._closed.is_set():
                sent_at = result.sent_at if result.sent_at is not None else self.clock.monotonic()
                setattr(self, f"_{axis}_deadline", sent_at)  # already expired: retry on the next tick
                setattr(self, f"_{axis}_stop_retry_until", sent_at + STOP_RETRY_WINDOW_S)
                self._log_event("stop_not_acked", axis=axis, status=result.status)

        return self._request(payload, packets.PAYLOAD_COMMAND, None, what, False, arm, after)

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
        after: Callable[[CommandResult], None] | None = None,
    ) -> CommandResult:
        """Send one request and wait for its reply, holding the lock. `after(result)` runs while still holding it."""
        started = self.clock.monotonic()
        # One deadline for the whole call. Everything below waits only for what is left of it, so the caller is
        # never blocked longer than `timeout` (before, lock wait + sequence reset + reply wait could add up to 3x).
        deadline = started + self.timeout
        if not self._lock.acquire(timeout=max(0.0, deadline - self.clock.monotonic())):
            # Someone else holds the wire for longer than our whole budget: give up rather than block.
            self.stats["timeouts"] += 1
            return CommandResult(TIMEOUT, sent_at=started)
        try:
            result = self._exchange_locked(payload, payload_type, parser, what, wait_for_completion, arm, deadline)
            if after is not None:
                after(result)
            return result
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
        if self.sequence_echo is False:
            # Without sequence numbers to tell replies apart, whatever already waits is an earlier request's reply
            # (a late batch): read it now so it can't be taken for this request's.
            self._drop_waiting_locked()
        sequence = self._sequence
        self._sequence = ((self._sequence + 1) & 0xFFFFFFFF) or 1
        if sequence in self._recent:
            self._recent.remove(sequence)  # reused after a reset: from now on it means this request
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
        """Wait until `deadline` for the reply that decides request `sequence`. Every message of every datagram is
        looked at (a datagram can hold several); the messages after the deciding one are noted as strays."""
        acked: CommandResult | None = None
        while (remaining := deadline - self.clock.monotonic()) > 0:
            datagram = self._transport.receive(remaining)
            if datagram is None:
                break
            now = self.clock.monotonic()
            self.last_reply_at = now
            self._log("rx", datagram, now)
            replies = packets.parse_replies(datagram)
            for index, reply in enumerate(replies):
                result = self._match_reply(reply, sequence, sent_at, now, is_inquiry, parser)
                if result is None:
                    continue
                if result.status == ACKED and wait_for_completion:
                    acked = acked or result  # keep waiting for "done"
                    continue
                for rest in replies[index + 1:]:
                    self._note_stray_reply(rest, now)
                return result
        if acked is not None:
            return acked
        self.stats["timeouts"] += 1
        self._log_event("timeout", sequence=sequence)
        return CommandResult(TIMEOUT, sequence, sent_at)

    def _match_reply(
        self,
        reply: packets.Reply,
        sequence: int,
        sent_at: float,
        now: float,
        is_inquiry: bool,
        parser: Callable[[bytes], object] | None,
    ) -> CommandResult | None:
        """What one reply message means for request `sequence`: the result it decides, or None (not ours, or not
        an answer to this kind of request: keep waiting)."""
        if reply.kind == packets.CONTROL:
            # 0F 01 = the camera's sequence counter disagrees with ours: reset before the next request.
            if reply.payload[:2] == b"\x0f\x01":
                self._needs_reset = True
                self.stats["errors"] += 1
                return CommandResult(ERROR, sequence, sent_at, now, error_code=0x0F01, payload=reply.payload)
            return None
        if reply.sequence is not None and reply.sequence != sequence:
            if self.sequence_echo is not False and reply.sequence in self._recent:
                self._remember_late(reply, now)  # an answer to an earlier request
                return None
            # A sequence number we never sent. Counted either way; taken as the answer only from a camera
            # known not to echo sequence numbers, otherwise dropped (it may be a late reply from before a
            # reset, or from another program talking to the camera).
            self.stats["unmatched_replies"] += 1
            if not self._camera_ignores_sequence(reply.sequence):
                self._log_event("unmatched_reply_dropped", seq=reply.sequence, waiting_for=sequence)
                return None
        elif reply.sequence == sequence and self.sequence_echo is not False:
            # Our own number came back. (In "no echo" mode a match is a coincidence of the camera's counter.)
            self.sequence_echo = True
            self._unknown_in_a_row = 0
        no_echo = self.sequence_echo is False
        kind = reply.kind
        if is_inquiry:
            if kind == packets.ANSWER:
                value = parser(reply.payload) if parser else reply.payload
                if parser is None and no_echo and len(reply.payload) in POSITION_ANSWER_LENGTHS:
                    # A raw inquiry (the version) can't be answered by a position: that's an earlier answer.
                    return self._ignore(reply, sequence, "position-shaped answer to a raw inquiry")
                if parser is not None and value is None:
                    if no_echo:
                        # Not the shape this inquiry's answer has (e.g. a zoom answer while asking pan/tilt): the
                        # late answer to an earlier inquiry. Keep waiting for ours.
                        return self._ignore(reply, sequence, "answer of another shape")
                    # It answered, but not in a format we understand: report it as an error.
                    return CommandResult(ERROR, sequence, sent_at, now, None, reply.payload)
                return CommandResult(ANSWERED, sequence, sent_at, now, payload=reply.payload, value=value)
            if no_echo:
                # This camera ACKs and completes inquiries too, and late replies arrive in batches: only data
                # answers an inquiry. An error can't be told apart from an earlier command's.
                return self._ignore(reply, sequence, kind)
            if kind == packets.ERROR:
                self.stats["errors"] += 1
                return CommandResult(ERROR, sequence, sent_at, now, reply.error_code, reply.payload)
            return None
        if kind == packets.ERROR:
            self.stats["errors"] += 1
            return CommandResult(ERROR, sequence, sent_at, now, reply.error_code, reply.payload)
        if kind == packets.COMPLETION:
            return CommandResult(COMPLETED, sequence, sent_at, now, payload=reply.payload)
        if kind == packets.ACK:
            return CommandResult(ACKED, sequence, sent_at, now, payload=reply.payload)
        if no_echo and kind == packets.ANSWER:
            return self._ignore(reply, sequence, "inquiry data while waiting for a command")
        return None

    def _ignore(self, reply: packets.Reply, sequence: int, why: str) -> None:
        """"No echo" mode: a reply that can't be the answer to request `sequence`. Counted, logged, skipped."""
        self.stats["ignored_replies"] += 1
        self._log_event("reply_ignored", why=why, payload=reply.payload.hex(" "), waiting_for=sequence)
        return None

    def _drop_waiting_locked(self) -> None:
        """Read and drop the datagrams already waiting (at most DRAIN_LIMIT). The caller holds the lock."""
        for _ in range(DRAIN_LIMIT):
            datagram = self._transport.receive(0.0)
            if datagram is None:
                return
            now = self.clock.monotonic()
            self.last_reply_at = now
            self._log("rx", datagram, now, dropped="arrived before the request was sent")
            self.stats["dropped_waiting"] += 1

    def _camera_ignores_sequence(self, unknown: int) -> bool:
        """Decide, from a reply carrying a number we never sent, whether this camera just doesn't echo sequence
        numbers ("no echo" mode, `sequence_echo = False`, for good). True: take the reply anyway.

        - Already in "no echo" mode: True.
        - Not decided yet: a reply numbered 0 (we never send 0), or UNKNOWN_IN_A_ROW_UNDECIDED (2) never-sent
          numbers in a row: a camera answering with 0, or numbering replies from its own counter (both seen).
        - The camera had echoed our number: one stray number (0 included) changes nothing, it is dropped. Only a
          run of UNKNOWN_IN_A_ROW_AFTER_ECHO (8) in a row means it doesn't echo after all: a camera with its own
          counter can hit one of our numbers by chance, an echoing camera never sends a run like that.
        """
        if self.sequence_echo is False:
            return True
        self._unknown_in_a_row += 1
        if self.sequence_echo is None:
            decided = unknown == 0 or self._unknown_in_a_row >= UNKNOWN_IN_A_ROW_UNDECIDED
        else:
            decided = self._unknown_in_a_row >= UNKNOWN_IN_A_ROW_AFTER_ECHO
        if decided:
            self.sequence_echo = False
            self._log_event("camera_does_not_echo_sequence", seq=unknown, in_a_row=self._unknown_in_a_row)
        return decided

    def _note_stray(self, datagram: bytes, now: float) -> None:
        self.last_reply_at = now
        self._log("rx", datagram, now)
        for reply in packets.parse_replies(datagram):
            self._note_stray_reply(reply, now)

    def _note_stray_reply(self, reply: packets.Reply, now: float) -> None:
        """A reply nobody is waiting for: remember a late completion for `completion_time()`. Only when the camera
        echoes sequence numbers: otherwise a number says nothing about which request it belongs to."""
        if (self.sequence_echo is not False and reply.sequence is not None
                and reply.sequence in self._recent):
            self._remember_late(reply, now)

    def _remember_late(self, reply: packets.Reply, now: float) -> None:
        if reply.sequence is not None and reply.kind in (packets.COMPLETION, packets.ERROR):
            self._late_completions[reply.sequence] = (now, reply.kind)
            if len(self._late_completions) > RECENT_SEQUENCES:
                self._late_completions.pop(next(iter(self._late_completions)))

    # --- the dead-man switch -------------------------------------------------------------------------------------

    def check_deadlines(self) -> None:
        """Send stop / zoom stop for every TTL that has run out, and re-send stops that weren't ACKed. The watchdog
        thread calls this ~100x a second.

        The check and the stop happen under the request lock, so a velocity command that renews the deadline is
        either entirely before (and the deadline is no longer expired) or entirely after (and moves the camera
        again).
        """
        if not self._lock.acquire(timeout=self.timeout):
            return
        try:
            self._expire_locked("pan_tilt", packets.pan_tilt_stop(), "ttl_stop")
            self._expire_locked("zoom", packets.zoom_stop(), "ttl_zoom_stop")
        finally:
            self._lock.release()

    def _expire_locked(self, axis: str, stop_payload: bytes, what: str) -> None:
        """If `axis`'s deadline has passed, send its stop. The deadline stays armed (so the next tick sends the stop
        again) until the camera ACKs it, or until STOP_RETRY_WINDOW_S after the first attempt. The caller holds the
        lock."""
        deadline = getattr(self, f"_{axis}_deadline")
        now = self.clock.monotonic()
        if deadline is None or now < deadline:
            return
        retry_until = getattr(self, f"_{axis}_stop_retry_until")
        if retry_until is None:
            # The TTL ran out just now: the first dead-man stop.
            self.stats["ttl_stops"] += 1
            self._log_event("ttl_stop", axis=axis)
            retry_until = now + STOP_RETRY_WINDOW_S
            setattr(self, f"_{axis}_stop_retry_until", retry_until)
        else:
            self.stats["stop_retries"] += 1
            self._log_event("stop_retry", axis=axis)
        result = self._exchange_locked(stop_payload, packets.PAYLOAD_COMMAND, None, what, False, None)
        if result.ok:
            setattr(self, f"_{axis}_deadline", None)
            setattr(self, f"_{axis}_stop_retry_until", None)
        elif self.clock.monotonic() >= retry_until:
            setattr(self, f"_{axis}_deadline", None)
            setattr(self, f"_{axis}_stop_retry_until", None)
            self.stats["stop_failures"] += 1
            self._log_event("stop_gave_up", axis=axis, seconds=STOP_RETRY_WINDOW_S)

    def _stop_at_exit(self) -> None:
        """Registered with `atexit`: stop pan/tilt and zoom when Python exits without calling close()."""
        if self._closed.is_set():
            return
        for stop in (self.stop, self.zoom_stop):
            with contextlib.suppress(Exception):
                stop()

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
            replies = packets.parse_replies(datagram)
            entry["kind"] = replies[0].kind if replies else "garbage"
            if replies and replies[0].sequence is not None:
                entry["seq"] = replies[0].sequence
            if len(replies) > 1:  # several messages in one datagram: all of them, in order
                entry["kinds"] = [reply.kind for reply in replies]
                entry["seqs"] = [reply.sequence for reply in replies]
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
