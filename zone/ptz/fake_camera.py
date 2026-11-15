"""A fake VISCA-over-IP camera: it speaks the protocol, and its "motors" are a `SimulatedPtz`.

Three ways to use it:

1. **In memory, with a fake clock** (most tests): `FakeTransport(camera, clock)` plugs straight into `ViscaPtz` in
   place of the UDP socket. Waiting for a reply that never comes just moves the fake clock forward, so timeout tests
   take no real time.
2. **As a UDP server in the test process**: `FakeViscaUdpServer(camera)` listens on 127.0.0.1 and a free port; a
   real `ViscaPtz` talks to it over real sockets.
3. **Standalone**, to dry-run the bench script without a camera:

       python -m zone.ptz.fake_camera --port 52381
       python -m zone.tools.characterize --ip 127.0.0.1 --out reports/fake --yes

Misbehaviour on demand, per command name (see `COMMAND_NAMES`), through `camera.behaviour[name]`:
"normal" (ACK + completion), "error" (syntax error), "silent" (no reply at all), "ack_only" (ACK, never completion).
`camera.echo_sequence = False` makes it answer with sequence 0, like some cheap cameras. `camera.bare_replies = True`
makes it drop the 8-byte header. `camera.lose_next[payload] = N` loses the next N requests with exactly that VISCA
payload on the way, as a flaky network would: the camera never sees them, so it neither acts nor replies.

**Like the real PTZ** (`camera.like_real_ptz()`, or `--like-real-ptz` on the command line): what the probe of
2026-11-15 measured on the PTZ the zone drives at the race **while its own web page was open in a browser** (see
LESSONS.md and its same-day correction: with the page closed the camera answers every inquiry, also while moving).
Keep it as a stress test of a busy camera, not as a model of the race PTZ:

- replies are numbered from the camera's own counter (~43000, rising ~40/s), never the request's number;
- the sequence reset is answered to port 52381 only, so the sender never sees the answer;
- an inquiry is answered with ACK + completion (`90 48 FF`, `90 58 FF`), and the data (`90 50 ...`) only with
  probability `answer_probability` (measured: 43 of 259, ~17 %), only at rest (none during a 3 s pan, second
  probe), and then as a **pair**, whatever was asked: the 11-byte pan/tilt answer and a 7-byte `90 50` (zoom-sized),
  sent in the batch of the *next* request (`data_pairs`);
- replies wait: the first ones `first_reply_delay` (~3 s), then 0-0.5 s, they go out on a 50 ms tick, in order,
  and all the messages ready at a tick share datagrams (up to `max_batch` messages each): late and batched.

Without it (the default) every test camera behaves as before: one reply per datagram, right away.
"""
from __future__ import annotations

import argparse
import random
import socket
import threading
from collections import deque

from zone.clock import Clock, SystemClock
from zone.ptz import visca_packets as packets
from zone.ptz.sim import PtzModel, SimulatedPtz

COMMAND_NAMES = (
    "pan_tilt_drive", "zoom_drive", "zoom_direct", "absolute_position", "relative_position", "home",
    "preset_recall", "preset_store", "inquire_pan_tilt", "inquire_zoom", "inquire_max_speed", "inquire_version",
    "sequence_reset",
)
SYNTAX_ERROR = 0x02
VERSION_ANSWER = bytes([0x90, 0x50, 0x00, 0x01, 0x05, 0x11, 0x01, 0x00, 0x02, 0xFF])  # made-up vendor/model/ROM


class FakeViscaCamera:
    """Turns VISCA datagrams into `SimulatedPtz` calls and builds the replies."""

    def __init__(self, sim: SimulatedPtz) -> None:
        self.sim = sim
        self.behaviour: dict[str, str] = {}
        self.echo_sequence = True
        self.bare_replies = False
        self.received: list[tuple[str, bytes]] = []  # (command name, payload) in arrival order
        self.lose_next: dict[bytes, int] = {}  # payload -> how many more of these get lost before arriving
        self.lost: list[bytes] = []  # payloads that were lost, in order
        self.reset_count = 0
        # The real PTZ's quirks (see like_real_ptz); all off by default.
        self.own_counter = False  # number replies from our own counter instead of echoing the request's
        self.inquiry_ack = False  # ACK + completion for inquiries too; the data only with answer_probability
        self.answer_probability = 1.0
        self.reset_answered = True  # False: the reset's answer goes elsewhere (port 52381), the sender sees none
        self.delayed = False  # replies wait in an outbox (first_reply_delay, then reply_delay_range), see below
        self.first_reply_delay = 0.0
        self.reply_delay_range = (0.0, 0.0)
        self.tick_s = 0.05  # delayed replies leave on this tick
        self.max_batch = 4  # messages per datagram at most (the real camera sent up to 4)
        self.random = random.Random(0)  # seeded: the same run gives the same answers
        self.counter = 43000
        self.data_pairs = False  # data as a (pan/tilt, zoom) pair with the next request's replies, never while moving
        self._pending_pair: list[bytes] = []
        self.inquiries = 0
        self.answers_sent = 0
        self._outbox: deque[tuple[float, bytes]] = deque()  # (time it may leave, one wrapped message), in order
        self._last_ready = float("-inf")
        self._replied_once = False

    def like_real_ptz(self, answer_probability: float = 0.17, first_reply_delay: float = 3.0,
                      reply_delay_range: tuple[float, float] = (0.0, 0.5), seed: int = 0) -> FakeViscaCamera:
        """Behave like the PTZ probed on 2026-11-15 (see the module docstring). Returns the camera."""
        self.own_counter = True
        self.inquiry_ack = True
        self.data_pairs = True
        self.answer_probability = answer_probability
        self.reset_answered = False
        self.delayed = True
        self.first_reply_delay = first_reply_delay
        self.reply_delay_range = reply_delay_range
        self.random = random.Random(seed)
        return self

    # --- the outbox (delayed, batched replies) ------------------------------------------------------------------

    def now(self) -> float:
        return self.sim.clock.monotonic()

    def next_due(self) -> float | None:
        """When the next delayed reply may leave (None: nothing waiting)."""
        return self._outbox[0][0] if self._outbox else None

    def take_due(self, now: float) -> list[bytes]:
        """The delayed replies whose time has come, as datagrams of up to `max_batch` messages each."""
        messages = []
        while self._outbox and self._outbox[0][0] <= now:
            messages.append(self._outbox.popleft()[1])
        return [b"".join(messages[i:i + self.max_batch]) for i in range(0, len(messages), self.max_batch)]

    def _post(self, messages: list[bytes]) -> list[bytes]:
        """Send `messages` (wrapped replies to one request): now, one datagram each, or later through the outbox.
        A data pair left by an earlier inquiry (`data_pairs`) goes first, in this request's batch."""
        if self._pending_pair:
            messages = [self._wrap(payload, 0) for payload in self._pending_pair] + messages
            self._pending_pair = []
        if not self.delayed or not messages:
            return messages
        now = self.now()
        delay = self.first_reply_delay if not self._replied_once else self.random.uniform(*self.reply_delay_range)
        self._replied_once = True
        # Leave on the next tick, never before an earlier request's replies (the camera answers in order).
        ready = max(self._last_ready, -(-(now + delay) // self.tick_s) * self.tick_s)
        self._last_ready = ready
        for message in messages:
            self._outbox.append((ready, message))
        return self.take_due(now)

    # --- requests ---------------------------------------------------------------------------------------------

    def handle(self, datagram: bytes) -> list[bytes]:
        """Process one request datagram; return the reply datagrams to send now (possibly none). With `delayed`,
        the replies go to the outbox instead: `take_due()` hands them out when their time comes."""
        request = packets.parse_reply(datagram)  # the same header layout, so the same splitter works
        if request is None or request.payload_type is None:
            return []
        if request.payload_type == packets.PAYLOAD_CONTROL:
            self.received.append(("sequence_reset", request.payload))
            self.reset_count += 1
            if self.behaviour.get("sequence_reset", "normal") == "silent" or not self.reset_answered:
                return []
            return [packets.wrap(b"\x01", packets.PAYLOAD_CONTROL_REPLY, request.sequence or 0)]
        payload = request.payload
        if self.lose_next.get(payload, 0) > 0:
            self.lose_next[payload] -= 1
            self.lost.append(payload)
            return []
        name, answer = self._execute(payload)
        refused = name.endswith(":refused")
        name = name.removesuffix(":refused")
        self.received.append((name, payload))
        mode = "error" if refused else self.behaviour.get(name, "normal")
        sequence = request.sequence if self.echo_sequence else 0
        if mode == "silent":
            return []
        if mode == "error" or name == "unknown":
            return self._post([self._wrap(packets.error(SYNTAX_ERROR), sequence)])
        if request.payload_type == packets.PAYLOAD_INQUIRY:
            self.inquiries += 1
            if answer is not None and self.random.random() >= self.answer_probability:
                answer = None  # the real PTZ sends the data for only ~17 % of its inquiries
            pair: list[bytes] = []
            if self.data_pairs:
                # The real PTZ: data only at rest, as a pan/tilt + zoom-sized pair whatever was asked, and late:
                # it rides with the replies to the next request.
                if answer is not None and not self.sim.is_moving():
                    pan_tilt, zoom = self.sim.inquire_pan_tilt(), self.sim.inquire_zoom()
                    if pan_tilt is not None and zoom is not None:
                        pair = [packets.encode_pan_tilt_answer(*pan_tilt), packets.encode_zoom_answer(zoom)]
                        self.answers_sent += 1
                answer = None
            if self.inquiry_ack:
                replies = [self._wrap(packets.ack(8), sequence), self._wrap(packets.completion(8), sequence)]
            else:
                replies = []
            if answer is not None:
                self.answers_sent += 1
                replies.append(self._wrap(answer, sequence))
            # A camera that can't answer right now (e.g. moving, "at_rest" mode) simply stays silent.
            sent_now = self._post(replies)
            if pair:
                self._pending_pair = pair  # with the next request's replies
            return sent_now
        socket_number = 8 if self.inquiry_ack else 1  # the real PTZ answers on socket 8 (sometimes 7 or 9)
        replies = [self._wrap(packets.ack(socket_number), sequence)]
        if mode != "ack_only":
            replies.append(self._wrap(packets.completion(socket_number), sequence))
        return self._post(replies)

    def _wrap(self, payload: bytes, sequence: int) -> bytes:
        if self.bare_replies:
            return payload
        if self.own_counter:
            # The real PTZ's counter: about 43000, rising ~40 a second on its own, +1 per message.
            self.counter = max(self.counter + 1, 43000 + int(40 * self.now()))
            sequence = self.counter
        return packets.wrap(payload, packets.PAYLOAD_REPLY, sequence)

    def _execute(self, p: bytes) -> tuple[str, bytes | None]:
        """Run the command on the simulator. Returns (command name, inquiry answer or None)."""
        sim = self.sim
        if p[:4] == b"\x81\x01\x06\x01" and len(p) == 9:
            pan_step = 0 if p[6] == packets.PAN_HOLD else (p[4] if p[6] == packets.PAN_RIGHT else -p[4])
            tilt_step = 0 if p[7] == packets.TILT_HOLD else (p[5] if p[7] == packets.TILT_UP else -p[5])
            if pan_step == 0 and tilt_step == 0:
                sim.stop()
            else:
                sim.pan_tilt_velocity(pan_step, tilt_step)
            return "pan_tilt_drive", None
        if p[:4] == b"\x81\x01\x04\x07" and len(p) == 6:
            code = p[4]
            if code == 0x00:
                sim.zoom_stop()
            else:
                speed = max(1, code & 0x0F)
                sim.zoom_velocity(speed if code & 0xF0 == packets.ZOOM_TELE else -speed)
            return "zoom_drive", None
        if p[:4] == b"\x81\x01\x04\x47" and len(p) == 9:
            sim.zoom_direct(packets.decode_nibbles(p[4:8]))
            return "zoom_direct", None
        if p[:4] in (b"\x81\x01\x06\x02", b"\x81\x01\x06\x03") and len(p) == 15:
            pan = packets.decode_signed_nibbles(p[6:10])
            tilt = packets.decode_signed_nibbles(p[10:14])
            if p[3] == 0x02:
                result = sim.absolute_position(pan, tilt, p[4], p[5])
                name = "absolute_position"
            else:
                result = sim.relative_position(pan, tilt, p[4], p[5])
                name = "relative_position"
            # The simulator refuses when its model says the camera lacks the command: answer with an error.
            return (name if result.ok else name + ":refused"), None
        if p == b"\x81\x01\x06\x04\xff":
            sim.home()
            return "home", None
        if p[:5] == b"\x81\x01\x04\x3f\x02" and len(p) == 7:
            sim.preset_recall(p[5])
            return "preset_recall", None
        if p[:5] == b"\x81\x01\x04\x3f\x01" and len(p) == 7:
            sim.preset_store(p[5])
            return "preset_store", None
        if p == b"\x81\x09\x06\x12\xff":
            value = sim.inquire_pan_tilt()
            return "inquire_pan_tilt", None if value is None else packets.encode_pan_tilt_answer(*value)
        if p == b"\x81\x09\x04\x47\xff":
            zoom = sim.inquire_zoom()
            return "inquire_zoom", None if zoom is None else packets.encode_zoom_answer(zoom)
        if p == b"\x81\x09\x06\x11\xff":
            pan_max, tilt_max = sim.model.pan_step_max, sim.model.tilt_step_max
            return "inquire_max_speed", bytes([0x90, 0x50, pan_max, tilt_max, 0xFF])
        if p == b"\x81\x09\x00\x02\xff":
            return "inquire_version", VERSION_ANSWER
        return "unknown", None


class FakeTransport:
    """An in-memory `Transport` for `ViscaPtz`: requests go straight to a `FakeViscaCamera`.

    Replies arrive `reply_delay` seconds after the request. Waiting for one moves the clock forward by sleeping on
    it, so with a `FakeClock` no real time passes.
    """

    def __init__(self, camera: FakeViscaCamera, clock: Clock, reply_delay: float = 0.002) -> None:
        self.camera = camera
        self.clock = clock
        self.reply_delay = reply_delay
        self.sent: list[bytes] = []
        self._inbox: deque[tuple[float, bytes]] = deque()
        self.closed = False

    def send(self, datagram: bytes) -> None:
        self.sent.append(datagram)
        when = self.clock.monotonic() + self.reply_delay
        for reply in self.camera.handle(datagram):
            self._inbox.append((when, reply))

    def inject(self, datagram: bytes, delay: float = 0.0) -> None:
        """Queue an unsolicited datagram (e.g. a stale reply) for the driver to receive."""
        self._inbox.append((self.clock.monotonic() + delay, datagram))

    def _collect(self, until: float) -> None:
        """Move the camera's delayed replies due by `until` into the inbox, at the time they leave the camera."""
        next_due = getattr(self.camera, "next_due", None)
        while next_due is not None and (due := next_due()) is not None and due <= until:
            for datagram in self.camera.take_due(due):
                self._inbox.append((due, datagram))

    def receive(self, timeout: float) -> bytes | None:
        now = self.clock.monotonic()
        self._collect(now + timeout)
        if self._inbox and self._inbox[0][0] <= now + timeout:
            when, datagram = self._inbox.popleft()
            self.clock.sleep(max(0.0, when - now))
            return datagram
        self.clock.sleep(timeout)
        return None

    def close(self) -> None:
        self.closed = True


class FakeViscaUdpServer:
    """Serve a `FakeViscaCamera` over real UDP (default 127.0.0.1 on a free port). Use as a context manager."""

    def __init__(self, camera: FakeViscaCamera, host: str = "127.0.0.1", port: int = 0) -> None:
        self.camera = camera
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((host, port))
        self._socket.settimeout(0.05)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="fake-visca", daemon=True)
        self._lock = threading.Lock()

    @property
    def address(self) -> tuple[str, int]:
        return self._socket.getsockname()

    def start(self) -> FakeViscaUdpServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._socket.close()

    def __enter__(self) -> FakeViscaUdpServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _serve(self) -> None:
        sender = None  # the last client: delayed replies go to it
        if self.camera.delayed:
            self._socket.settimeout(0.01)  # wake often enough to send delayed replies on their 50 ms tick
        while not self._stop.is_set():
            replies: list[bytes] = []
            try:
                datagram, sender = self._socket.recvfrom(2048)
                with self._lock:
                    replies = self.camera.handle(datagram)
            except (socket.timeout, OSError):
                pass
            if sender is not None and self.camera.delayed:
                with self._lock:
                    replies += self.camera.take_due(self.camera.now())
            for reply in replies:
                try:
                    self._socket.sendto(reply, sender)
                except OSError:
                    pass


def add_real_ptz_arguments(ap: argparse.ArgumentParser) -> None:
    """The command-line flags for `like_real_ptz` (also used by zone.tools.agent_rig)."""
    ap.add_argument("--like-real-ptz", action="store_true",
                    help="behave like the PTZ probed on 2026-11-15: own sequence counter, late batched replies, "
                         "ACK + completion for inquiries and the data only sometimes")
    ap.add_argument("--answer-probability", type=float, default=0.17,
                    help="with --like-real-ptz: chance that an inquiry gets its data (default 0.17, as measured)")
    ap.add_argument("--first-reply-delay", type=float, default=3.0,
                    help="with --like-real-ptz: delay of the very first reply, s (default 3, as measured)")


def camera_from_arguments(sim: SimulatedPtz, args: argparse.Namespace) -> FakeViscaCamera:
    camera = FakeViscaCamera(sim)
    if args.like_real_ptz:
        camera.like_real_ptz(answer_probability=args.answer_probability, first_reply_delay=args.first_reply_delay)
    return camera


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="A fake VISCA-over-IP PTZ for dry runs of the bench script.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=packets.VISCA_PORT)
    ap.add_argument("--inquiry-mode", default="always", choices=["always", "at_rest", "never", "frozen_while_moving"])
    ap.add_argument("--refresh-hz", type=float, default=None,
                    help="stale position cache: refresh the reported position only this often (default: always live)")
    add_real_ptz_arguments(ap)
    args = ap.parse_args(argv)
    sim = SimulatedPtz(PtzModel(inquiry_mode=args.inquiry_mode, inquiry_refresh_hz=args.refresh_hz), SystemClock())
    camera = camera_from_arguments(sim, args)
    server = FakeViscaUdpServer(camera, args.host, args.port).start()
    print(f"fake VISCA camera on {server.address[0]}:{server.address[1]} (inquiry {args.inquiry_mode}"
          f"{', like the real PTZ' if camera.delayed else ''}); Ctrl+C to stop")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
