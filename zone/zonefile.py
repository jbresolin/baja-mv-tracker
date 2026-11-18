"""The taught zone: what `zone.json` holds, and how it is saved and loaded.

A zone is an ordered list of points clicked on the spotter image, along the racing line, from upstream of the zone to
its exit. Some points are *keyframes*: they also carry the PTZ pose (pan, tilt, zoom in raw device units) and the
preset number the PTZ stored that framing in. The others are *path-only* points that just follow curves.

The geometry built from these points (progress `s`, projection, pose interpolation) lives in `zone/path.py`. This
module only holds the data and the file format, so it can be tested on its own.

Saving keeps a backup of the previous file on every save, in a `zone_backups/` folder next to it, so a bad teach can
always be undone by copying a backup back.
"""
import copy
import datetime as dt
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

FORMAT_VERSION = 1


@dataclass
class Keyframe:
    """A PTZ framing attached to one path point."""
    pan: int                    # raw device units
    tilt: int
    zoom: int
    preset: int                 # PTZ preset number this framing is stored in
    approximate: bool = False   # True when the pose was dead-reckoned (no inquiry, no absolute moves)
    pose_source: str = ""       # "inquiry" | "commanded" | "dead_reckoning"
    # The operator looked at this keyframe ('testar') and said its pose is right although the sanity check flags it
    # (zone/keyframe_check.py: e.g. the track really curves back). Hides its warnings. Re-teaching clears it.
    confirmed: bool = False

    def to_dict(self):
        d = dict(pan=int(self.pan), tilt=int(self.tilt), zoom=int(self.zoom), preset=int(self.preset),
                 approximate=bool(self.approximate), pose_source=self.pose_source)
        if self.confirmed:
            d["confirmed"] = True
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(pan=int(d["pan"]), tilt=int(d["tilt"]), zoom=int(d["zoom"]), preset=int(d["preset"]),
                   approximate=bool(d.get("approximate", False)), pose_source=str(d.get("pose_source", "")),
                   confirmed=bool(d.get("confirmed", False)))


@dataclass
class PathPoint:
    """One clicked point on the spotter image (pixels). `keyframe` is None for a path-only point.

    `half_width_px`: this point's own corridor half-width (pixels), or None to use the zone's global
    `corridor_half_width_px`. Between points that have one, the width is interpolated along the path (see
    `Zone.point_half_widths`). Lets the corridor be narrow where the path passes near another stretch of track and
    wide where the cars spread out."""
    x: float
    y: float
    keyframe: Optional[Keyframe] = None
    half_width_px: Optional[float] = None

    @property
    def is_keyframe(self):
        return self.keyframe is not None

    def to_dict(self):
        d = dict(x=round(float(self.x), 2), y=round(float(self.y), 2))
        if self.half_width_px is not None:
            d["half_width_px"] = round(float(self.half_width_px), 1)   # omitted when unset: old files stay as they were
        if self.keyframe is not None:
            d["keyframe"] = self.keyframe.to_dict()
        return d

    @classmethod
    def from_dict(cls, d):
        kf = d.get("keyframe")
        hw = d.get("half_width_px")
        return cls(float(d["x"]), float(d["y"]), Keyframe.from_dict(kf) if kf else None,
                   None if hw is None else float(hw))


@dataclass
class Zone:
    points: List[PathPoint] = field(default_factory=list)
    corridor_half_width_px: float = 60.0
    image_size: Optional[List[int]] = None        # [width, height] of the spotter image the points were clicked on
    cameras: dict = field(default_factory=dict)   # {"spotter": "...", "ptz": "..."}: ids, for the record
    report: dict = field(default_factory=dict)    # the characterization report the poses were taken with
    notes: str = ""
    # pan/tilt sign measured while teaching (zone/direction.py): {"pan": -1, "tilt": 1, "pan_votes": 4, ...}
    # +1 = the reported value grows when turning right / up. Empty = not measured (e.g. no position inquiry)
    ptz_signs: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ derived
    def keyframe_indices(self):
        return [i for i, p in enumerate(self.points) if p.is_keyframe]

    def keyframes(self):
        return [p.keyframe for p in self.points if p.is_keyframe]

    @property
    def entry_preset(self):
        """The first keyframe's preset: where the PTZ waits for the next car. None until a keyframe exists."""
        kfs = self.keyframes()
        return kfs[0].preset if kfs else None

    def point_half_widths(self):
        """The corridor half-width (pixels) at every point, in point order.

        * Points with their own `half_width_px` keep it.
        * Points between two such points get a value interpolated linearly by arc length (distance along the path)
          between them.
        * Points before the first / after the last explicit value hold that nearest explicit value.
        * No point has one: every point gets the global `corridor_half_width_px` (the behaviour of old zone files).
        Between two consecutive points the corridor then varies linearly too (PathModel.half_width_at_arc)."""
        n = len(self.points)
        if n == 0:
            return []
        arcs = [0.0]
        for a, b in zip(self.points, self.points[1:]):
            arcs.append(arcs[-1] + ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5)
        known = [(arcs[i], float(p.half_width_px)) for i, p in enumerate(self.points) if p.half_width_px is not None]
        if not known:
            return [float(self.corridor_half_width_px)] * n
        out = []
        for a in arcs:
            if a <= known[0][0]:
                out.append(known[0][1])
            elif a >= known[-1][0]:
                out.append(known[-1][1])
            else:
                for (a0, w0), (a1, w1) in zip(known, known[1:]):
                    if a0 <= a <= a1:
                        out.append(w0 if a1 - a0 <= 1e-9 else w0 + (w1 - w0) * (a - a0) / (a1 - a0))
                        break
        return out

    def used_presets(self):
        return {k.preset for k in self.keyframes()}

    def next_free_preset(self, base=10):
        """Lowest preset number >= base that no keyframe uses yet."""
        used = self.used_presets()
        n = int(base)
        while n in used:
            n += 1
        return n

    def copy(self):
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ file format
    def to_dict(self):
        return dict(format_version=FORMAT_VERSION,
                    image_size=list(self.image_size) if self.image_size else None,
                    corridor_half_width_px=float(self.corridor_half_width_px),
                    entry_preset=self.entry_preset,
                    cameras=dict(self.cameras), report=dict(self.report), notes=self.notes,
                    ptz_signs=dict(self.ptz_signs),
                    points=[p.to_dict() for p in self.points])

    @classmethod
    def from_dict(cls, d):
        v = int(d.get("format_version", 1))
        if v > FORMAT_VERSION:
            raise ValueError("zone file format %d is newer than this program (%d)" % (v, FORMAT_VERSION))
        return cls(points=[PathPoint.from_dict(p) for p in d.get("points", [])],
                   corridor_half_width_px=float(d.get("corridor_half_width_px", 60.0)),
                   image_size=d.get("image_size"), cameras=dict(d.get("cameras") or {}),
                   report=dict(d.get("report") or {}), notes=str(d.get("notes", "")),
                   ptz_signs=dict(d.get("ptz_signs") or {}))


def backup_dir(path):
    return Path(path).parent / "zone_backups"


def save(zone: Zone, path, now=None):
    """Write `zone` to `path`. If the file already exists, it is first copied to zone_backups/<name>.<time>.json.
    The write goes to a temporary file that then replaces the old one, so a crash never leaves half a file.
    Returns the backup path (or None when there was nothing to back up)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        stamp = (now or dt.datetime.now()).strftime("%Y%m%d-%H%M%S-%f")[:-3]
        bdir = backup_dir(path)
        bdir.mkdir(exist_ok=True)
        backup = bdir / ("%s.%s%s" % (path.stem, stamp, path.suffix))
        k = 1
        while backup.exists():  # two saves in the same millisecond (tests) must not overwrite each other
            backup = bdir / ("%s.%s-%d%s" % (path.stem, stamp, k, path.suffix))
            k += 1
        shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(zone.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return backup


def load(path) -> Zone:
    return Zone.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
