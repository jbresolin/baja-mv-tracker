"""The presets the studio sees, and the small file that remembers them (and the last mode) across restarts.

A VISCA camera can't be asked which presets it has stored, so the agent keeps the list itself:

* **keyframe presets** come from the taught zone (zone.json): presets 10, 11, ... (teach.preset_base), named after
  their keyframe ("Zone entry (keyframe 1)"). The studio may recall them, but never store over them: that would move
  a keyframe's framing without updating its pose, and break the teach. Renaming them is refused too (their name
  says what they are).
* **studio presets** are the ones the studio stored through the API, with the labels it gave them. They live in
  `zone.agent.json`, next to zone.json (like zone.speed.json).

The same file remembers the last contract mode, so that after a restart the agent can say what it restored (contract
§9; see modes.ModeArbiter.restored).
"""
import json
import os
import threading
from pathlib import Path
from typing import Callable, Iterable, Optional

from zone.agent.contract import BAD_REQUEST, UNKNOWN_PRESET, AgentError

PRESET_MAX = 127     # VISCA preset numbers are 7 bits in the recall / store packets


def state_path_for(zone_path) -> Path:
    """zone.json -> zone.agent.json (next to it)."""
    p = Path(zone_path)
    return p.with_name(p.stem + ".agent.json")


class AgentStateFile:
    """`zone.agent.json`: {"presets": {"<n>": "<label>"}, "last_mode": "manual"}. Written atomically (a temp file,
    then a rename), so a crash mid-write never leaves half a file. A missing or unreadable file is an empty state:
    the race must not stop because of it (the problem is logged)."""

    def __init__(self, path, log=print) -> None:
        self.path = Path(path)
        self.log = log
        self.lock = threading.Lock()
        self.presets: dict[int, str] = {}
        self.last_mode: Optional[str] = None
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.presets = {int(n): str(label) for n, label in (raw.get("presets") or {}).items()}
                self.last_mode = raw.get("last_mode")
            except (OSError, ValueError, AttributeError) as e:
                self.log("agent state file %s not loaded (%s): starting empty" % (self.path, e))

    def set_preset(self, n: int, label: str) -> None:
        with self.lock:
            self.presets[int(n)] = label
            self._save()

    def set_last_mode(self, mode: str) -> None:
        with self.lock:
            if mode != self.last_mode:
                self.last_mode = mode
                self._save()

    def _save(self) -> None:
        data = {"presets": {str(n): label for n, label in sorted(self.presets.items())}, "last_mode": self.last_mode}
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            self.log("agent state file %s not saved: %s" % (self.path, e))


def keyframe_label(index: int, count: int) -> str:
    """The name the studio sees for keyframe `index` (0-based) of `count`."""
    if index == 0:
        return "Zone entry (keyframe 1)"
    if index == count - 1:
        return "Zone exit (keyframe %d)" % count
    return "Zone keyframe %d" % (index + 1)


def number_ranges(numbers: Iterable[int]) -> str:
    """[0, 1, 2, 3, 7, 8] -> "0-3, 7-8"."""
    out, run = [], []
    for n in sorted(numbers):
        if run and n == run[-1] + 1:
            run.append(n)
            continue
        if run:
            out.append(str(run[0]) if len(run) == 1 else "%d-%d" % (run[0], run[-1]))
        run = [n]
    if run:
        out.append(str(run[0]) if len(run) == 1 else "%d-%d" % (run[0], run[-1]))
    return ", ".join(out)


class PresetBook:
    """The camera's preset list as the studio sees it: keyframe presets (read live from the zone being taught) plus
    the studio's own (from the state file). A number that is both is a keyframe: the keyframe wins."""

    def __init__(self, state: AgentStateFile, keyframe_presets: Callable[[], list[int]],
                 preset_max: int = PRESET_MAX) -> None:
        """keyframe_presets() -> the preset numbers of the zone's keyframes, in path order (entry first)."""
        self.state = state
        self.keyframe_presets = keyframe_presets
        self.preset_max = preset_max

    def keyframes(self) -> dict[int, str]:
        numbers = [int(n) for n in self.keyframe_presets() if n is not None]
        return {n: keyframe_label(i, len(numbers)) for i, n in enumerate(numbers)}

    def listing(self) -> list[dict]:
        """[{"n": 10, "label": "Zone entry (keyframe 1)"}, ...], sorted by number."""
        merged = dict(self.state.presets)
        merged.update(self.keyframes())
        return [{"n": n, "label": label} for n, label in sorted(merged.items())]

    def is_stored(self, n: int) -> bool:
        return n in self.keyframes() or n in self.state.presets

    def check_number(self, n: int) -> None:
        if n > self.preset_max:
            raise AgentError(BAD_REQUEST, "this PTZ stores presets 0 to %d, not %d" % (self.preset_max, n))

    def check_recall(self, n: int) -> None:
        self.check_number(n)
        if not self.is_stored(n):
            raise AgentError(UNKNOWN_PRESET, "preset %d is not stored (stored: %s)"
                             % (n, number_ranges(p["n"] for p in self.listing()) or "none"))

    def check_store(self, n: int) -> None:
        """Refuse to store over a keyframe preset: that would break the zone's teach."""
        self.check_number(n)
        keyframes = self.keyframes()
        if n in keyframes:
            free = number_ranges(set(range(self.preset_max + 1)) - set(keyframes))
            raise AgentError(BAD_REQUEST, "preset %d is the zone tracker's \"%s\": storing over it would break the "
                             "taught zone. Use another number (free for the studio: %s), or re-teach the keyframe on "
                             "the zone laptop's page" % (n, keyframes[n], free))

    def stored(self, n: int) -> dict:
        """Record a successful store. An existing label is kept (contract §6), a new preset is "Preset <n>"."""
        label = self.state.presets.get(n) or "Preset %d" % n
        self.state.set_preset(n, label)
        return {"n": n, "label": label}

    def rename(self, n: int, label: str) -> dict:
        self.check_number(n)
        keyframes = self.keyframes()
        if n in keyframes:
            raise AgentError(BAD_REQUEST, "preset %d is the zone tracker's \"%s\"; its name comes from the taught "
                             "zone and can't be changed here" % (n, keyframes[n]))
        if n not in self.state.presets:
            raise AgentError(UNKNOWN_PRESET, "preset %d is not stored" % n)
        self.state.set_preset(n, label)
        return {"n": n, "label": label}

    def signature(self) -> tuple:
        """Changes whenever the listing changes (to send presets_changed when the zone is re-taught)."""
        return tuple((p["n"], p["label"]) for p in self.listing())
