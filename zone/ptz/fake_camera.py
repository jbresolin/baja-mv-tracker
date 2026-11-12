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
makes it drop the 8-byte header.
"""
from __future__ import annotations

import argparse
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
        self.reset_count = 0

    def handle(self, datagram: bytes) -> list[bytes]:
        """Process one request datagram; return the reply datagrams (possibly none)."""
        request = packets.parse_reply(datagram)  # the same header layout, so the same splitter works
        if request is None or request.payload_type is None:
            return []
        if request.payload_type == packets.PAYLOAD_CONTROL:
            self.received.append(("sequence_reset", request.payload))
            self.reset_count += 1
            if self.behaviour.get("sequence_reset", "normal") == "silent":
                return []
            return [packets.wrap(b"\x01", packets.PAYLOAD_CONTROL_REPLY, request.sequence or 0)]
        payload = request.payload
        name, answer = self._execute(payload)
        refused = name.endswith(":refused")
        name = name.removesuffix(":refused")
        self.received.append((name, payload))
        mode = "error" if refused else self.behaviour.get(name, "normal")
        sequence = request.sequence if self.echo_sequence else 0
        if mode == "silent":
            return []
        if mode == "error" or name == "unknown":
            return [self._wrap(packets.error(SYNTAX_ERROR), sequence)]
        if request.payload_type == packets.PAYLOAD_INQUIRY:
            # A camera that can't answer right now (e.g. moving, "at_rest" mode) simply stays silent.
            return [] if answer is None else [self._wrap(answer, sequence)]
        replies = [self._wrap(packets.ack(), sequence)]
        if mode != "ack_only":
            replies.append(self._wrap(packets.completion(), sequence))
        return replies

    def _wrap(self, payload: bytes, sequence: int) -> bytes:
        return payload if self.bare_replies else packets.wrap(payload, packets.PAYLOAD_REPLY, sequence)

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

    def receive(self, timeout: float) -> bytes | None:
        now = self.clock.monotonic()
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
        while not self._stop.is_set():
            try:
                datagram, sender = self._socket.recvfrom(2048)
            except (socket.timeout, OSError):
                continue
            with self._lock:
                replies = self.camera.handle(datagram)
            for reply in replies:
                try:
                    self._socket.sendto(reply, sender)
                except OSError:
                    pass


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="A fake VISCA-over-IP PTZ for dry runs of the bench script.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=packets.VISCA_PORT)
    ap.add_argument("--inquiry-mode", default="always", choices=["always", "at_rest", "never", "frozen_while_moving"])
    ap.add_argument("--refresh-hz", type=float, default=None,
                    help="stale position cache: refresh the reported position only this often (default: always live)")
    args = ap.parse_args(argv)
    sim = SimulatedPtz(PtzModel(inquiry_mode=args.inquiry_mode, inquiry_refresh_hz=args.refresh_hz), SystemClock())
    server = FakeViscaUdpServer(FakeViscaCamera(sim), args.host, args.port).start()
    print(f"fake VISCA camera on {server.address[0]}:{server.address[1]} (inquiry {args.inquiry_mode}); Ctrl+C to stop")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
