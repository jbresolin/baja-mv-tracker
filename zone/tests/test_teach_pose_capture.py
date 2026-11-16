"""Teaching a keyframe must store the pose the camera is REALLY at, never a doubtful one.

The bug these tests reproduce (2026-11-15 bench, real SMTAV PTZ over VISCA, pose source "inquiry"): the operator
jogged the camera from left to right and taught four keyframes. One of them was stored as pan 1764 / zoom 0 while
its neighbours were around pan 1200..1640 / zoom 6800..9400. The controller then drove the camera to that bad pose,
"way too far and too fast". The teach used to read the pose ONCE, and one reading can be wrong:

* the camera may still be moving (coasting) when it is read;
* a late answer to an EARLIER question (the pose before the last jog) can arrive first;
* that camera sometimes sends, on its own, a 7-byte message `90 50 00 00 00 00 FF`, which looks exactly like the
  answer "zoom 0".

What the teach does now (zone/teach.py, `InquirySource.capture`): stop the camera, wait a moment, read the pose up
to 6 times and only accept it when 3 readings in a row agree. Otherwise it refuses the keyframe with "a câmera não
respondeu uma posição estável; tente de novo".

How these tests work, for a first-time reader:
* `FakeClock`: time stands still unless the code (or the test) moves it. Waiting is instant and repeatable.
* `SimulatedPtz`: the camera's motors. It turns only when the fake clock moves.
* `FakeViscaCamera`: speaks the camera's network language (VISCA) on top of the motors. With `own_counter = True`
  it numbers its replies from its own counter, like the real PTZ, so the driver can't use the numbers to tell
  which reply answers which question ("no echo" mode).
* `ViscaPtz`: the REAL driver the zone uses at the race, plugged into the fake camera through an in-memory "network"
  (`FakeTransport`) instead of UDP.
* `ScriptedTransport` (below) lets a test slip an extra, bad message in just before the camera's real answer to a
  given question: that is how we reproduce the stray `90 50 00 00 00 00 FF` and the late stale answer.
"""
import pytest

from zone.clock import FakeClock
from zone.ptz import visca_packets as packets
from zone.ptz.fake_camera import FakeTransport, FakeViscaCamera
from zone.ptz.sim import PtzModel, SimulatedPtz
from zone.ptz.visca import ViscaPtz
from zone.teach import InquirySource, TeachError, TeachSession
from zone.zonefile import Zone

ZOOM_ZERO_HEX = "90 50 00 00 00 00 FF"     # the camera's stray message: looks like "zoom is 0"
UNSTABLE = "a câmera não respondeu uma posição estável; tente de novo"


def message(payload, sequence=0):
    """One reply message as the camera sends it: an 8-byte header (with a sequence number) + the VISCA payload."""
    return packets.wrap(payload, packets.PAYLOAD_REPLY, sequence)


class ScriptedTransport(FakeTransport):
    """The in-memory network, plus: `slip_in(question, datagram)` makes `datagram` arrive just BEFORE the camera's
    real answer to the next `question` the driver sends (a VISCA payload such as the zoom inquiry). The driver then
    sees the bad message first, exactly as on the bench."""

    def __init__(self, camera, clock):
        super().__init__(camera, clock, reply_delay=0.002)   # the camera answers 2 ms after a question
        self.slips = {}                                       # question payload -> datagrams to slip in, in order

    def slip_in(self, question, datagram, times=1):
        self.slips.setdefault(question, []).extend([datagram] * times)

    def send(self, datagram):
        request = packets.parse_reply(datagram)
        waiting = self.slips.get(request.payload) if request is not None else None
        if waiting:
            extra = waiting.pop(0)
            if extra:   # b"" = "nothing this time"
                # 1 ms after the question: before the real answer (2 ms), after the driver dropped what was waiting
                self.inject(extra, delay=0.001)
        super().send(datagram)


class Bench:
    """The teach setup of the 2026-11-15 bench, in memory: the real driver, a camera that doesn't echo sequence
    numbers, the "inquiry" pose source and a teach session."""

    def __init__(self, model=None):
        self.clock = FakeClock(0.0)
        self.sim = SimulatedPtz(model or PtzModel(), self.clock)
        self.camera = FakeViscaCamera(self.sim)
        self.camera.own_counter = True           # replies numbered from the camera's own counter, like the SMTAV
        self.transport = ScriptedTransport(self.camera, self.clock)
        # watchdog=False: no background thread; the test decides when time passes
        self.ptz = ViscaPtz("camera", clock=self.clock, transport=self.transport, watchdog=False)
        self.source = InquirySource(self.ptz, self.clock)
        self.session = TeachSession(Zone(), self.ptz, self.source, preset_base=10)

    def jog(self, pan=6, tilt=0, zoom=0, seconds=1.0, rest=True):
        """Hold a jog button for `seconds` (the page repeats the jog every 150 ms), then let go.
        rest=False: don't wait after letting go, so the camera is still coasting."""
        held = 0.0
        while held < seconds:
            self.source.jog(pan, tilt, zoom)
            self.clock.advance(0.15)
            held += 0.15
        self.source.jog_stop()
        if rest:
            self.clock.advance(1.0)

    def true_pose(self):
        """Where the simulated camera really is (rounded to whole raw units, as the camera reports it)."""
        pan, tilt = self.sim.true_pan_tilt()
        return round(pan), round(tilt), round(self.sim.true_zoom())


def close_to(stored_keyframe, pose, pan_tilt=2, zoom=10):
    return (abs(stored_keyframe.pan - pose[0]) <= pan_tilt and abs(stored_keyframe.tilt - pose[1]) <= pan_tilt
            and abs(stored_keyframe.zoom - pose[2]) <= zoom)


# --------------------------------------------------------------------------------------------- the normal case
def test_a_keyframe_stores_where_the_camera_really_is():
    bench = Bench()
    bench.jog(pan=6, zoom=3)                         # turn right and zoom in a bit
    point = bench.session.add_point(100, 200, keyframe=True)
    assert bench.ptz.sequence_echo is False          # the driver knows the camera doesn't echo our numbers
    assert close_to(point.keyframe, bench.true_pose())
    assert point.keyframe.zoom > 0                   # we did zoom in: a real zoom, not 0


# --------------------------------------------------------------------------------------- the stray "zoom 0"
def test_a_stray_zoom_zero_waiting_before_the_capture_is_never_stored():
    # The camera sent `90 50 00 00 00 00 FF` on its own a moment before the operator clicked "Adicionar keyframe".
    # It is sitting in the network buffer: it arrived before any zoom question was asked, so it can't be an answer.
    bench = Bench()
    bench.jog(pan=6, zoom=3)
    bench.transport.inject(message(bytes.fromhex(ZOOM_ZERO_HEX), 0))
    point = bench.session.add_point(100, 200, keyframe=True)
    assert point.keyframe.zoom != 0
    assert close_to(point.keyframe, bench.true_pose())


def test_a_stray_zoom_zero_arriving_while_capturing_is_outvoted():
    # Worse: the stray message arrives right after we asked the zoom, just before the real answer. The driver can't
    # tell them apart (no sequence numbers), so ONE reading says "zoom 0". The other readings don't agree with it,
    # so it is never stored: we keep only a pose read the same way 3 times in a row.
    bench = Bench()
    bench.jog(pan=6, zoom=3)
    bench.transport.slip_in(packets.zoom_position_inquiry(), message(bytes.fromhex(ZOOM_ZERO_HEX), 43999))
    point = bench.session.add_point(100, 200, keyframe=True)
    assert point.keyframe.zoom != 0
    assert close_to(point.keyframe, bench.true_pose())


def test_a_camera_that_keeps_sending_stray_zoom_zeros_gets_the_keyframe_refused():
    # If the stray message comes before every other zoom answer, the readings never agree 3 times in a row:
    # refuse, with a clear message, rather than store a doubtful pose. Nothing is added to the zone.
    bench = Bench()
    bench.jog(pan=6, zoom=3)
    for _ in range(3):   # readings 1, 3 and 5 get the stray "zoom 0"; 2, 4 and 6 the real zoom
        bench.transport.slip_in(packets.zoom_position_inquiry(), message(bytes.fromhex(ZOOM_ZERO_HEX), 43999))
        bench.transport.slip_in(packets.zoom_position_inquiry(), b"")   # b"": nothing slipped in that time
    with pytest.raises(TeachError, match=UNSTABLE):
        bench.session.add_point(100, 200, keyframe=True)
    assert bench.session.zone.points == []


# ------------------------------------------------------------------------------------ a late, stale answer
def test_a_late_answer_with_the_previous_position_is_outvoted():
    # The operator jogged from position A to position B. An answer about A (to a question asked before the jog)
    # arrives late, right when we ask where the camera is now. That first reading says "A"; the next ones say "B".
    bench = Bench()
    bench.jog(pan=-6)
    before = bench.true_pose()                       # position A
    bench.jog(pan=8, seconds=1.5)                    # now at position B, far to the right
    stale = packets.encode_pan_tilt_answer(before[0], before[1])
    bench.transport.slip_in(packets.pan_tilt_position_inquiry(), message(stale, 43999))
    point = bench.session.add_point(100, 200, keyframe=True)
    assert close_to(point.keyframe, bench.true_pose())
    assert abs(point.keyframe.pan - before[0]) > 50  # really B, not the stale A


# ------------------------------------------------------------------------------------------ still moving
def test_a_capture_right_after_the_jog_waits_for_the_camera_to_stop():
    # The operator lets go of the jog button and clicks "Adicionar keyframe" at once: the camera is still
    # coasting. The teach stops it, waits, and stores where it came to rest (not a pose from the way there).
    bench = Bench()
    bench.jog(pan=12, seconds=1.0, rest=False)       # fast, and no waiting after letting go
    assert bench.sim.is_moving()                     # yes, still moving when the click comes
    point = bench.session.add_point(100, 200, keyframe=True)
    final = bench.true_pose()
    bench.clock.advance(2.0)
    assert bench.true_pose()[:2] == final[:2]        # it really was at rest when the pose was taken...
    assert close_to(point.keyframe, final)           # ...and that rest pose is what was stored


def test_a_camera_that_does_not_stop_gets_the_keyframe_refused():
    # The stop commands get lost on the network (3 tries) and the camera keeps turning: every reading is
    # different. The keyframe is refused instead of storing a pose taken on the move.
    bench = Bench()
    bench.ptz.pan_tilt_velocity(6, 0)                # turning right, with no dead-man (no TTL)
    bench.clock.advance(0.5)
    bench.camera.lose_next[packets.pan_tilt_stop()] = 3
    with pytest.raises(TeachError, match=UNSTABLE):
        bench.session.add_point(100, 200, keyframe=True)
    assert bench.session.zone.points == []
    bench.ptz.stop()                                 # (tidy up: really stop the simulated camera)


def test_a_camera_that_answers_nothing_says_so():
    # No answer at all is a different problem (network, wrong address, web page open...): the old message stays.
    bench = Bench(PtzModel(inquiry_mode="never"))
    with pytest.raises(TeachError, match="não consegui ler a posição"):
        bench.session.add_point(100, 200, keyframe=True)


def test_the_capture_takes_at_most_a_few_seconds():
    # The operator waits for the click to finish: even when it fails, it must not hang.
    bench = Bench()
    bench.ptz.pan_tilt_velocity(6, 0)
    bench.camera.lose_next[packets.pan_tilt_stop()] = 3
    start = bench.clock.monotonic()
    with pytest.raises(TeachError):
        bench.session.add_point(100, 200, keyframe=True)
    assert bench.clock.monotonic() - start < 4.0
    bench.ptz.stop()


# ------------------------------------------------------------------------------------------------ "testar"
def test_testar_confirms_a_good_keyframe():
    bench = Bench()
    bench.jog(pan=6, zoom=3)
    bench.session.add_point(100, 200, keyframe=True)
    bench.jog(pan=-8, seconds=2)                     # move away
    check = bench.session.goto(0)                    # recall the preset, wait, read back
    assert check["checked"] and check["ok"]
    assert bench.session.tests[10] is check          # the page shows the last result next to the keyframe


def test_testar_shows_a_keyframe_stored_with_a_bad_pose():
    # Simulate the bench's bad keyframe: the preset in the camera is fine, but the pose stored in the zone (the one
    # the controller follows) is far off. "testar" reads where the camera actually went and warns.
    bench = Bench()
    bench.jog(pan=6, zoom=3)
    point = bench.session.add_point(100, 200, keyframe=True)
    point.keyframe.pan += 565                        # the stored pose is wrong (like P11: pan 1764, zoom 0)
    point.keyframe.zoom = 0
    check = bench.session.goto(0)
    assert check["checked"] and not check["ok"]
    assert "re-ensinar" in check["message"] and "pan lido" in check["message"] and "zoom lido" in check["message"]


def test_reteaching_clears_the_old_testar_result():
    bench = Bench()
    bench.jog(pan=6)
    point = bench.session.add_point(100, 200, keyframe=True)
    point.keyframe.pan += 500
    assert not bench.session.goto(0)["ok"]
    bench.session.reteach(0)                         # the operator re-teaches it: the old warning goes away
    assert 10 not in bench.session.tests
