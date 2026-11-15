"""The parts of the studio's agent API contract 1.0 (baja-studio `docs/agent-api.md`) the zone node needs.

The studio's own machine-readable twin is `server/baja_studio/agents/contract.py` in baja-studio. This file copies only
what the zone uses, with the same names, so the two are easy to compare. When the contract changes (a new MINOR
version only adds things), compare them again.

Request bodies are Pydantic models (Pydantic comes with FastAPI). Like the studio's, they ignore fields they don't
know (contract §1: both sides ignore unknown JSON fields).
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "1.0"
AGENT_PORT = 8890
TOKEN_HEADER = "X-Agent-Token"
STUDIO_ID_HEADER = "X-Studio-Id"

# Timings (contract §4)
PUSH_INTERVAL_S = 0.25            # we push state 4x a second: twice the contract's minimum (2 Hz), like the fake agent
DEFAULT_TTL_MS = 300
MAX_TTL_MS = 2000
DEFAULT_RESUME_AFTER_MS = 3000
MAX_RESUME_AFTER_MS = 600_000
MAX_PRESET_NUMBER = 255           # preset numbers on the wire; each camera's limits.preset_max may be lower
LOG_LENGTH = 500                  # how many command log entries GET /agent/log can return

# The `reason` of the mode_changed an agent sends when it could not restore auto after a restart (contract §9)
RESTORE_REASON_TRACKER_NOT_READY = "tracker_not_ready"

# Modes (contract §7)
MANUAL, AUTO, AUTO_PAUSED, HOLD = "manual", "auto", "auto_paused", "hold"
MODES = (MANUAL, AUTO, AUTO_PAUSED, HOLD)

# Tracker states (contract §5)
IDLE, SEARCHING, TRACKING, LOST = "idle", "searching", "tracking", "lost"

# Health (contract §3)
OK, DEGRADED, OFFLINE = "ok", "degraded", "offline"

# Events (contract §5)
MODE_CHANGED = "mode_changed"
MANUAL_OVERRIDE_STARTED = "manual_override_started"
MANUAL_OVERRIDE_ENDED = "manual_override_ended"
TRACKER_ACQUIRED = "tracker_acquired"
TRACKER_LOST = "tracker_lost"
DEVICE_OFFLINE = "device_offline"
DEVICE_ONLINE = "device_online"
PRESETS_CHANGED = "presets_changed"

# Error codes and their HTTP status (contract §9)
BAD_REQUEST = "bad_request"
NOT_SUPPORTED = "not_supported"
UNKNOWN_CONTROL = "unknown_control"
UNAUTHORIZED = "unauthorized"
UNKNOWN_CAMERA = "unknown_camera"
UNKNOWN_PRESET = "unknown_preset"
TRACKER_NOT_READY = "tracker_not_ready"
DEVICE_OFFLINE_ERROR = "device_offline"
ERROR_HTTP_STATUS = {
    BAD_REQUEST: 400, NOT_SUPPORTED: 400, UNKNOWN_CONTROL: 400, UNAUTHORIZED: 401, UNKNOWN_CAMERA: 404,
    UNKNOWN_PRESET: 404, TRACKER_NOT_READY: 409, DEVICE_OFFLINE_ERROR: 503,
}


class AgentError(Exception):
    """A command the agent refuses. The server turns it into `{"error": code, "detail": detail}` with the contract's
    HTTP status. `detail` is a sentence an operator can act on."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail

    @property
    def status(self) -> int:
        return ERROR_HTTP_STATUS[self.code]


class ContractModel(BaseModel):
    """Base of the request bodies: unknown fields are ignored, not errors."""

    model_config = ConfigDict(extra="ignore")


class MoveRequest(ContractModel):
    """`POST /cameras/{id}/move`: normalised velocities -1..1 and a time-to-live (the dead-man)."""

    pan: float = Field(default=0.0, ge=-1.0, le=1.0)
    tilt: float = Field(default=0.0, ge=-1.0, le=1.0)
    zoom: float = Field(default=0.0, ge=-1.0, le=1.0)
    ttl_ms: int = Field(default=DEFAULT_TTL_MS, ge=1, le=MAX_TTL_MS)


class ModeRequest(ContractModel):
    """`PUT /cameras/{id}/mode`. `auto_paused` can't be asked for (it isn't in the list, so it's a 400)."""

    mode: Literal["manual", "auto", "hold"]
    profile: dict[str, Any] | None = None


class PresetLabelRequest(ContractModel):
    """`PUT /cameras/{id}/presets/{n}`: a new name, 1-60 characters."""

    label: str = Field(min_length=1, max_length=60)


def checked_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """A copy of a mode profile, refused (400) unless `resume_after_ms`, when given, is a whole number of ms from 0 to
    600000. The other keys (`zone`, `priority_targets`, anything else) are kept as they are and not used here."""
    if "resume_after_ms" in profile:
        value = profile["resume_after_ms"]
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= MAX_RESUME_AFTER_MS:
            raise AgentError(BAD_REQUEST, "profile.resume_after_ms must be a whole number of ms from 0 to %d"
                             % MAX_RESUME_AFTER_MS)
    return dict(profile)
