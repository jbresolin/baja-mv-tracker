"""Click-teach: build the zone by jogging the PTZ to a framing and clicking the matching point in the spotter image.

Workflow on the page (zone/static/index.html):
  1. "Adicionar ponto" then click the image: a path-only point (to follow curves).
  2. Jog the PTZ to frame a car at some spot, "Adicionar keyframe", click that spot in the spotter image: the current
     PTZ pose is read, a preset is stored (numbered from teach.preset_base), and the point becomes a keyframe.
  3. Fix things: move / delete / reorder points, re-teach one keyframe (go to it, adjust, save), test a keyframe
     (recall its preset), change the corridor width. "Salvar" writes zone.json (the previous one is backed up).

Where a keyframe's pose comes from (`PoseSource`, chosen from the characterization report, in this order):
  * `InquirySource`: ask the camera (pan/tilt/zoom inquiry). Jogging uses velocity commands.
  * `CommandedSource`: absolute positioning works but inquiry doesn't: the camera is only moved with absolute
    "nudges", so its pose is known by construction. It needs a known start: the page's "Ir para home" button.
  * `DeadReckoningSource`: neither works: integrate the velocity commands through the speed table, starting from the
    last recalled keyframe (or "Zerar aqui"). Poses taught this way are marked approximate.
"""
import math
import threading

from zone.speedmap import AxisSpeedMap
from zone.zonefile import Keyframe, PathPoint, Zone


class TeachError(Exception):
    """A teach action that can't be done right now (shown to the operator as is, in pt-BR)."""


# ======================================================================================================== pose sources
class PoseSource:
    name = ""
    approximate = False
    velocity_jog = True        # False: the page sends nudges instead of press-and-hold jogs

    def __init__(self, ptz, clock, ttl_s=0.3):
        self.ptz, self.clock, self.ttl_s = ptz, clock, float(ttl_s)
        self.lock = threading.Lock()

    def read(self):
        """-> (pan, tilt, zoom) or None when the pose can't be known right now."""
        raise NotImplementedError

    def jog(self, pan_step, tilt_step, zoom_step):
        """Press-and-hold jog (the page repeats it every ~150 ms while a button is held; the TTL stops the camera
        if the page goes away)."""
        self.ptz.pan_tilt_velocity(int(pan_step), int(tilt_step), ttl=self.ttl_s)
        if zoom_step:
            self.ptz.zoom_velocity(int(zoom_step), ttl=self.ttl_s)

    def jog_stop(self):
        self.ptz.stop()
        self.ptz.zoom_stop()

    def nudge(self, dpan, dtilt, dzoom, speed):
        raise TeachError("este modo usa os botões de movimento contínuo")

    def recalled(self, kf: Keyframe):
        """A keyframe's preset was recalled: the camera is (about to be) at its pose."""

    def home(self):
        raise TeachError("este modo não precisa de home")

    def zero_here(self):
        raise TeachError("este modo não precisa de zerar")

    def status(self):
        return dict(name=self.name, approximate=self.approximate, velocity_jog=self.velocity_jog, pose=self.read())


class InquirySource(PoseSource):
    """Ask the camera. Also measures the pan/tilt SIGN as a side effect of the operator's jogs (zone/direction.py):
    the pose is read when a held jog starts (or changes direction) and when it ends. A right jog after which the
    reported pan went DOWN is one vote for pan_sign = -1. The operator jogs a lot while teaching, so a zone taught
    this way always knows its signs, even without a bench report."""
    name = "inquiry"
    MIN_DELTA = 4                        # raw units: a jog that moved the reading less than this is not a vote

    def __init__(self, ptz, clock, ttl_s=0.3):
        super().__init__(ptz, clock, ttl_s)
        self.approximate = False
        self.zoom_readable = None        # unknown until the first read; False -> the page asks for the zoom
        self.sign_votes = {"pan": [], "tilt": []}   # +1 / -1 per finished jog that moved that axis
        self._segment = None             # (pan direction, tilt direction, pose at the start) of the jog being held

    def jog(self, pan_step, tilt_step, zoom_step):
        direction = (_sgn(pan_step), _sgn(tilt_step))
        with self.lock:
            if self._segment is None or self._segment[:2] != direction:
                self._close_segment()
                if any(direction):
                    self._segment = direction + (self._read_pan_tilt(),)
        super().jog(pan_step, tilt_step, zoom_step)

    def jog_stop(self):
        super().jog_stop()
        with self.lock:
            self._close_segment()

    def _read_pan_tilt(self):
        try:
            return self.ptz.inquire_pan_tilt()
        except Exception:  # noqa: BLE001 (no reading = no vote, never an error for the operator)
            return None

    def _close_segment(self):
        """The held jog ended (or changed direction): compare the reading now with the one at its start."""
        seg, self._segment = self._segment, None
        if seg is None or seg[2] is None:
            return
        now = self._read_pan_tilt()
        if now is None:
            return
        for i, axis in enumerate(("pan", "tilt")):
            step_direction = seg[i]
            delta = now[i] - seg[2][i]
            if step_direction and abs(delta) >= self.MIN_DELTA:
                # + step moved the reading up -> +1; down -> -1
                self.sign_votes[axis].append(step_direction * (1 if delta > 0 else -1))

    def measured_signs(self):
        """{"pan": -1, "pan_votes": 3, "pan_consistent": True, ...} for the axes with at least one vote."""
        from zone.direction import vote_sign
        out = {}
        for axis, votes in self.sign_votes.items():
            sign, consistent = vote_sign(votes)
            if sign is not None:
                out.update({axis: sign, axis + "_votes": len(votes), axis + "_consistent": consistent})
        return out

    def read(self):
        """-> (pan, tilt, zoom), with zoom None when the camera doesn't answer the zoom inquiry (some don't). The
        teach session then asks the operator for the zoom instead of silently storing a made-up value."""
        pt = self.ptz.inquire_pan_tilt()
        if pt is None:
            return None
        z = self.ptz.inquire_zoom()
        self.zoom_readable = z is not None
        return int(pt[0]), int(pt[1]), int(z) if z is not None else None


class CommandedSource(PoseSource):
    """Only absolute moves: pose known by construction."""
    name = "commanded"
    velocity_jog = False

    def __init__(self, ptz, clock, ttl_s=0.3, limits=None, home_pose=(0, 0, 0)):
        super().__init__(ptz, clock, ttl_s)
        self.pose = None
        self.limits = limits or {}
        self.home_pose = tuple(int(v) for v in home_pose)

    def read(self):
        return None if self.pose is None else tuple(int(v) for v in self.pose)

    def _clamp(self, axis, v):
        lo, hi = self.limits.get(axis, (-10 ** 9, 10 ** 9))
        return int(min(max(v, lo), hi))

    def home(self):
        p, t, z = self.home_pose
        self.ptz.absolute_position(p, t, 12, 10)
        self.ptz.zoom_direct(z)
        self.pose = (p, t, z)

    def nudge(self, dpan, dtilt, dzoom, speed):
        if self.pose is None:
            raise TeachError("posição desconhecida: clique em 'Ir para home' primeiro")
        p, t, z = self.pose
        p, t = self._clamp("pan", p + dpan), self._clamp("tilt", t + dtilt)
        if dpan or dtilt:
            self.ptz.absolute_position(p, t, int(speed), int(speed))
        if dzoom:
            z = self._clamp("zoom", z + dzoom)
            self.ptz.zoom_direct(z)
        self.pose = (p, t, z)

    def jog(self, pan_step, tilt_step, zoom_step):
        raise TeachError("neste modo use os botões de ajuste fino (movimento absoluto)")

    def recalled(self, kf: Keyframe):
        self.pose = (kf.pan, kf.tilt, kf.zoom)


class DeadReckoningSource(PoseSource):
    """Integrates the commanded velocities. A velocity command lasts until the next command or its TTL."""
    name = "dead_reckoning"
    approximate = True

    def __init__(self, ptz, clock, pan_map: AxisSpeedMap, tilt_map: AxisSpeedMap, zoom_table=None, ttl_s=0.3,
                 limits=None, pan_sign=1, tilt_sign=1):
        super().__init__(ptz, clock, ttl_s)
        self.pan_sign, self.tilt_sign = int(pan_sign), int(tilt_sign)   # reported frame vs physical step direction
        self.pan_map, self.tilt_map = pan_map, tilt_map
        self.zoom_map = AxisSpeedMap(zoom_table) if zoom_table else None
        self.limits = limits or {}
        self.pose = None                     # [pan, tilt, zoom] floats
        self.cmd = (0, 0, 0)
        self.t_cmd = clock.monotonic()

    def _advance(self):
        now = self.clock.monotonic()
        if self.pose is not None and any(self.cmd):
            end = min(now, self.t_cmd + self.ttl_s)
            dt = max(0.0, end - self.t_cmd)
            ps, ts, zs = self.cmd
            self.pose[0] += self.pan_sign * self.pan_map.rate_of(ps) * dt
            self.pose[1] += self.tilt_sign * self.tilt_map.rate_of(ts) * dt
            if self.zoom_map:
                self.pose[2] += self.zoom_map.rate_of(zs) * dt
            for i, ax in enumerate(("pan", "tilt", "zoom")):
                lo, hi = self.limits.get(ax, (-math.inf, math.inf))
                self.pose[i] = min(max(self.pose[i], lo), hi)
        self.t_cmd = now

    def jog(self, pan_step, tilt_step, zoom_step):
        with self.lock:
            self._advance()
            super().jog(pan_step, tilt_step, zoom_step)
            self.cmd = (int(pan_step), int(tilt_step), int(zoom_step))

    def jog_stop(self):
        with self.lock:
            self._advance()
            super().jog_stop()
            self.cmd = (0, 0, 0)

    def read(self):
        with self.lock:
            self._advance()
            return None if self.pose is None else tuple(int(round(v)) for v in self.pose)

    def recalled(self, kf: Keyframe):
        with self.lock:
            self.pose = [float(kf.pan), float(kf.tilt), float(kf.zoom)]
            self.cmd = (0, 0, 0)
            self.t_cmd = self.clock.monotonic()

    def zero_here(self):
        with self.lock:
            self.pose = [0.0, 0.0, 0.0]
            self.cmd = (0, 0, 0)
            self.t_cmd = self.clock.monotonic()


def make_pose_source(kind, ptz, clock, settings):
    m = settings.report.model
    ttl = settings.controller.ttl_s
    limits = dict(pan=tuple(m.pan_limits), tilt=tuple(m.tilt_limits), zoom=tuple(m.zoom_limits))
    if kind == "inquiry":
        return InquirySource(ptz, clock, ttl)
    if kind == "commanded":
        return CommandedSource(ptz, clock, ttl, limits)
    if kind == "dead_reckoning":
        return DeadReckoningSource(ptz, clock, settings.pan_map, settings.tilt_map, m.zoom_speed_table, ttl, limits,
                                   settings.controller.pan_sign, settings.controller.tilt_sign)
    raise ValueError("unknown pose source %r" % kind)


# ======================================================================================================== teach edits
class TeachSession:
    """Edits a Zone. Every change calls `on_change(zone_copy)` (the runtime rebuilds its path from it)."""

    def __init__(self, zone: Zone, ptz, source: PoseSource, preset_base=10, on_change=None):
        self.zone = zone
        self.ptz = ptz
        self.source = source
        self.preset_base = int(preset_base)
        self.on_change = on_change or (lambda z: None)
        self.dirty = False
        self.lock = threading.RLock()

    def _changed(self):
        self.dirty = True
        # the pan/tilt sign the pose source measured from the jogs (InquirySource): saved with the zone, so the
        # controller turns the right way (zone/direction.py). An axis not measured this session keeps its old value.
        measured = getattr(self.source, "measured_signs", lambda: {})()
        if measured:
            self.zone.ptz_signs = dict(self.zone.ptz_signs, **measured, how="teach_jog")
        self.on_change(self.zone.copy())

    def _check_index(self, i):
        if not 0 <= int(i) < len(self.zone.points):
            raise TeachError("ponto %s não existe" % i)
        return int(i)

    def _capture_keyframe(self, preset, zoom=None):
        """Read the current pose and store it in `preset`. `zoom`: the operator's zoom value, used only when the
        camera can't tell its zoom (no zoom inquiry); then it is required, never guessed."""
        pose = self.source.read()
        if pose is None:
            raise TeachError("não consegui ler a posição da PTZ (modo %s)" % self.source.name)
        pan, tilt, z = pose
        how = self.source.name
        if z is None:
            if zoom is None or str(zoom).strip() == "":
                raise TeachError("esta PTZ não informa o zoom: digite o zoom deste keyframe no campo 'zoom' e tente "
                                 "de novo")
            z = _zoom_value(zoom)
            how += "+zoom_manual"
        r = self.ptz.preset_store(int(preset))
        if r is not None and not getattr(r, "ok", True):
            raise TeachError("a PTZ não confirmou a gravação do preset %d" % preset)
        return Keyframe(pan, tilt, z, int(preset), self.source.approximate, how)

    def add_point(self, x, y, keyframe=False, insert_at=None, zoom=None):
        """Add a path point (or a keyframe at the current PTZ framing). Appended at the end unless insert_at."""
        with self.lock:
            kf = self._capture_keyframe(self.zone.next_free_preset(self.preset_base), zoom) if keyframe else None
            pt = PathPoint(float(x), float(y), kf)
            if insert_at is None:
                self.zone.points.append(pt)
            else:
                self.zone.points.insert(max(0, min(int(insert_at), len(self.zone.points))), pt)
            self._changed()
            return pt

    def move_point(self, i, x, y):
        with self.lock:
            i = self._check_index(i)
            self.zone.points[i].x, self.zone.points[i].y = float(x), float(y)
            self._changed()

    def delete_point(self, i):
        with self.lock:
            i = self._check_index(i)
            del self.zone.points[i]
            self._changed()

    def reorder(self, i, delta):
        """Move point i up (delta -1) or down (+1) the list."""
        with self.lock:
            i = self._check_index(i)
            j = i + int(delta)
            if not 0 <= j < len(self.zone.points):
                raise TeachError("não dá para mover além da ponta")
            p = self.zone.points
            p[i], p[j] = p[j], p[i]
            self._changed()

    def toggle_keyframe(self, i):
        """Turn a keyframe back into a path-only point (its preset stays in the camera, unused)."""
        with self.lock:
            i = self._check_index(i)
            self.zone.points[i].keyframe = None
            self._changed()

    def make_keyframe(self, i, zoom=None):
        """Attach the current PTZ framing to an existing path point."""
        with self.lock:
            i = self._check_index(i)
            p = self.zone.points[i]
            preset = p.keyframe.preset if p.keyframe else self.zone.next_free_preset(self.preset_base)
            p.keyframe = self._capture_keyframe(preset, zoom)
            self._changed()

    def reteach(self, i, zoom=None):
        """Store the current framing in keyframe i (same preset number)."""
        with self.lock:
            i = self._check_index(i)
            p = self.zone.points[i]
            if p.keyframe is None:
                raise TeachError("o ponto %d não é keyframe" % i)
            p.keyframe = self._capture_keyframe(p.keyframe.preset, zoom)
            self._changed()

    def set_keyframe_zoom(self, i, zoom):
        """Type in the zoom of keyframe i (for cameras that can't tell their zoom). The preset is not touched."""
        with self.lock:
            i = self._check_index(i)
            kf = self.zone.points[i].keyframe
            if kf is None:
                raise TeachError("o ponto %d não é keyframe" % i)
            kf.zoom = _zoom_value(zoom)
            if "zoom_manual" not in kf.pose_source:
                kf.pose_source += "+zoom_manual"
            self._changed()

    def goto(self, i):
        """'Testar keyframe': recall its preset so the operator can check (or re-teach) the framing."""
        with self.lock:
            i = self._check_index(i)
            kf = self.zone.points[i].keyframe
            if kf is None:
                raise TeachError("o ponto %d não é keyframe" % i)
            self.ptz.preset_recall(kf.preset)
            self.source.recalled(kf)

    def set_corridor(self, half_width_px):
        with self.lock:
            v = float(half_width_px)
            if not 5 <= v <= 1000:
                raise TeachError("largura do corredor fora de 5..1000 px")
            self.zone.corridor_half_width_px = v
            self._changed()

    def save(self, path, image_size=None, report=None, cameras=None):
        from zone import zonefile
        with self.lock:
            if image_size:
                self.zone.image_size = list(image_size)
            if report is not None:
                self.zone.report = dict(report)
            if cameras is not None:
                self.zone.cameras = dict(cameras)
            backup = zonefile.save(self.zone, path)
            self.dirty = False
            return backup


def _sgn(x):
    return (x > 0) - (x < 0)


def _zoom_value(zoom):
    try:
        z = int(round(float(zoom)))
    except (TypeError, ValueError):
        raise TeachError("zoom inválido: %r" % (zoom,)) from None
    if not 0 <= z <= 65535:
        raise TeachError("zoom fora de 0..65535: %d" % z)
    return z
