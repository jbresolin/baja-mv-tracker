"""The zone node's agent API (AUTOPTZ-24): the studio (baja-studio) drives the zone's PTZ through it.

The studio's contract is `docs/agent-api.md` (version 1.0) in jbresolin/baja-studio. This package serves it from the
zone process itself (`python -m zone.app`), on its own port (default 8890), next to the zone's own page (8090). It
shares the zone's one PTZ driver and director: there is no second VISCA client on the laptop.

Modules, one concern each:

* `contract.py`: the contract's constants, error codes and request bodies.
* `modes.py`: the contract's modes (manual / auto / auto_paused / hold) and how they map onto the director's
  (STOPPED / PASSIVE / AUTO). Pure logic with an injected clock, so it is tested without waiting.
* `link.py`: what `modes.py` needs from the zone app (switch the director, "can the tracker run?").
* `presets.py`: the presets the studio sees (the zone's keyframe presets + the ones the studio stored) and the
  small state file that remembers them and the last mode across restarts.
* `camera.py`: the one camera this agent owns: velocities to speed steps, presets, position, state.
* `service.py`: the command log, the events, the state pushes to every studio WebSocket.
* `server.py`: the HTTP and WebSocket routes (FastAPI).
"""
