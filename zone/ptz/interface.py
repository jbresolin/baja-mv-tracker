"""The one PTZ interface: what the controller (WP-Z2) and the bench script may ask of a camera.

`visca.ViscaPtz` (a real camera) and `sim.SimulatedPtz` (a model of one) both implement it, so the same
controller code can run against either.

Units are the camera's raw device units throughout. "Normalized" units (degrees, 0..1 zoom) are a separate
conversion done with the bench report's numbers (`PtzModel.pan_units_per_degree` etc.), never inside the driver.

Sign conventions (the same in every implementation):
- pan step > 0 turns right, < 0 turns left
- tilt step > 0 tilts up, < 0 tilts down
- zoom step > 0 zooms in (tele), < 0 zooms out (wide)
- a step of 0 stops (or holds) that axis

Whether the camera's *reported* raw pan grows to the right or to the left is a fact about the camera.
The bench report records it (`pan_sign` / `tilt_sign` in the model).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# CommandResult.status values.
COMPLETED = "completed"  # the camera said "done"
ACKED = "acked"  # the camera accepted it; "done" had not come yet when we stopped waiting
ANSWERED = "answered"  # an inquiry got its data
ERROR = "error"  # the camera refused it (error_code says why)
TIMEOUT = "timeout"  # no answer within the driver's timeout


@dataclass(frozen=True)
class CommandResult:
    """What happened to one command or inquiry."""

    status: str
    sequence: int | None = None  # the VISCA/IP sequence number it was sent with (None for the simulator)
    sent_at: float | None = None  # clock.monotonic() when sent
    replied_at: float | None = None  # clock.monotonic() of the reply that decided `status`
    error_code: int | None = None
    payload: bytes | None = None  # the deciding reply's VISCA payload (for inquiries, the data)
    value: object = None  # a decoded inquiry answer, e.g. (pan, tilt) or zoom

    @property
    def ok(self) -> bool:
        return self.status in (COMPLETED, ACKED, ANSWERED)

    @property
    def round_trip(self) -> float | None:
        if self.sent_at is None or self.replied_at is None:
            return None
        return self.replied_at - self.sent_at


class Ptz(Protocol):
    """A pan-tilt-zoom camera, in raw device units. Every method returns within the driver's timeout."""

    def pan_tilt_velocity(self, pan_step: int, tilt_step: int, ttl: float | None = None) -> CommandResult:
        """Move at signed speed steps until told otherwise.

        With `ttl` (seconds), the camera is stopped if no newer pan/tilt command arrives within `ttl`: the dead-man
        switch. Without it, the motion continues until `stop()`.
        """

    def stop(self) -> CommandResult:
        """Stop pan and tilt."""

    def zoom_velocity(self, step: int, ttl: float | None = None) -> CommandResult:
        """Zoom at a signed speed step (|step| 1..7); 0 stops. `ttl` as for pan/tilt."""

    def zoom_stop(self) -> CommandResult:
        """Stop zooming."""

    def zoom_direct(self, position: int) -> CommandResult:
        """Zoom to a raw position."""

    def absolute_position(self, pan: int, tilt: int, pan_speed: int, tilt_speed: int) -> CommandResult:
        """Move to raw (pan, tilt) at the given speed steps."""

    def relative_position(self, pan: int, tilt: int, pan_speed: int, tilt_speed: int) -> CommandResult:
        """Move by raw (pan, tilt) from the current position."""

    def inquire_pan_tilt(self) -> tuple[int, int] | None:
        """Raw (pan, tilt), or None if the camera did not answer (or does not answer now)."""

    def inquire_zoom(self) -> int | None:
        """Raw zoom, or None."""

    def preset_recall(self, n: int) -> CommandResult:
        """Go to stored preset `n` (0-based)."""

    def preset_store(self, n: int) -> CommandResult:
        """Store the current pose as preset `n` (0-based). Overwrites it."""

    def home(self) -> CommandResult:
        """Go to the camera's home pose."""

    def close(self) -> None:
        """Stop background work and release the connection (does not move the camera)."""
