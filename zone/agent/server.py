"""The agent API's HTTP and WebSocket routes (baja-studio docs/agent-api.md, contract 1.0), served by FastAPI.

    python -m zone.app ...      starts this on agent.port (default 8890) next to the zone's page (8090)

Rules from the contract that live here:

* With a token configured (`agent.token`), every request needs `X-Agent-Token`, except `GET /agent/info` (a subnet
  scan knows no token). A WebSocket without it is refused at the handshake (HTTP 401).
* Every error is `{"error": "<code>", "detail": "<sentence>"}` with the contract's status; a malformed body or an
  out-of-range value is `400 bad_request` (not FastAPI's 422).
* Every command (§6) is logged in `GET /agent/log`, refused ones too, with `X-Studio-Id` as its source.
* A studio's socket is closed after 10 s without a word from it (it pings every 2 s), and a studio that doesn't
  read gets no more than MAX_QUEUED_STATES state pushes queued (events are always kept).
* State is pushed to every WebSocket at once on accept, then every 0.25 s (the contract asks for >= 2 Hz); a push
  due right after a mode change waits 0.1 s for the dust to settle. Events go out as they happen. `{"type": "ping"}` is answered `{"type": "pong"}`.
"""
import asyncio
import contextlib
import json
from contextlib import asynccontextmanager
from typing import Annotated, Any, Callable

from fastapi import Body, FastAPI, Path, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from zone.agent.contract import (
    MAX_PRESET_NUMBER, MODE_CHANGED, PUSH_INTERVAL_S, STUDIO_ID_HEADER, TOKEN_HEADER, UNAUTHORIZED, UNKNOWN_CONTROL, AgentError,
    ModeRequest, MoveRequest, PresetLabelRequest,
)

PresetNumber = Annotated[int, Path(ge=0, le=MAX_PRESET_NUMBER)]
STATE_SETTLE_S = 0.1          # a state push due within this long after a mode change waits (see _push_forever)
MAX_QUEUED_STATES = 40        # a studio that doesn't read (10 s of pushes behind) gets no more state pushes...
SILENT_STUDIO_CLOSE_S = 10.0  # ... and its socket is closed after this long without a word (the studio pings every 2 s)


def error_response(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse({"error": code, "detail": detail}, status_code=status)


def create_agent_app(agent) -> FastAPI:
    """The agent API around a `ZoneAgent` (zone/agent/service.py). State pushes start when the app starts serving."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        agent.loop = asyncio.get_running_loop()
        agent.wake = asyncio.Event()
        pusher = asyncio.create_task(_push_forever(agent))
        try:
            yield
        finally:
            pusher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pusher
            for queue in list(agent.subscribers):
                queue.put_nowait(None)          # close every WebSocket
            agent.loop = agent.wake = None

    app = FastAPI(title="zone agent API (contract 1.0)", lifespan=lifespan)
    _add_error_handlers(app)

    @app.middleware("http")
    async def check_token(request: Request, call_next):
        if agent.token and request.url.path != "/agent/info" and request.headers.get(TOKEN_HEADER) != agent.token:
            return error_response(401, UNAUTHORIZED, "missing or wrong %s" % TOKEN_HEADER)
        return await call_next(request)

    def run(request: Request, camera_id: str, command: str, args: dict, action: Callable[[Any, str], Any]):
        """Run a studio command on the camera and log it, refused or not."""
        source = "studio:%s" % request.headers.get(STUDIO_ID_HEADER, "unknown")
        try:
            result = action(agent.camera_for(camera_id), source)
        except AgentError as refused:
            agent.record(camera_id, source, command, args, "error: %s" % refused.code)
            raise
        agent.record(camera_id, source, command, args, "ok")
        return result

    # ------------------------------------------------------------------------------------------- agent level
    @app.get("/agent/info")
    def get_info():
        return agent.info()

    @app.get("/agent/health")
    def get_health():
        return agent.health()

    @app.get("/agent/log")
    def get_log(limit: Annotated[int, Query(ge=1, le=1000)] = 100):
        return agent.log_tail(limit)

    @app.websocket("/agent/ws")
    async def state_socket(websocket: WebSocket):
        await _serve_state_socket(agent, websocket)

    # ------------------------------------------------------------------------------------------- motion
    # Plain `def` routes: FastAPI runs them in its thread pool, so a PTZ call that waits for the camera never blocks
    # the event loop that pushes the state.
    @app.post("/cameras/{camera_id}/move")
    def move(camera_id: str, body: MoveRequest, request: Request):
        return run(request, camera_id, "move", body.model_dump(),
                   lambda cam, src: cam.move(body.pan, body.tilt, body.zoom, body.ttl_ms, src))

    @app.post("/cameras/{camera_id}/stop")
    def stop(camera_id: str, request: Request):
        return run(request, camera_id, "stop", {}, lambda cam, src: cam.stop(src))

    # ------------------------------------------------------------------------------------------- presets
    @app.post("/cameras/{camera_id}/presets/{n}/recall")
    def recall_preset(camera_id: str, n: PresetNumber, request: Request):
        return run(request, camera_id, "recall_preset", {"n": n}, lambda cam, src: cam.recall_preset(n, src))

    @app.post("/cameras/{camera_id}/presets/{n}/store")
    def store_preset(camera_id: str, n: PresetNumber, request: Request):
        return run(request, camera_id, "store_preset", {"n": n}, lambda cam, src: cam.store_preset(n, src))

    @app.put("/cameras/{camera_id}/presets/{n}")
    def label_preset(camera_id: str, n: PresetNumber, body: PresetLabelRequest, request: Request):
        return run(request, camera_id, "label_preset", {"n": n, "label": body.label},
                   lambda cam, src: cam.label_preset(n, body.label))

    # ------------------------------------------------------------------------------------------- position, mode
    @app.get("/cameras/{camera_id}/position")
    def get_position(camera_id: str):
        return agent.camera_for(camera_id).position()

    @app.get("/cameras/{camera_id}/mode")
    def get_mode(camera_id: str):
        agent.camera_for(camera_id)
        return agent.arbiter.reply()

    @app.put("/cameras/{camera_id}/mode")
    def set_mode(camera_id: str, body: ModeRequest, request: Request):
        args = body.model_dump(exclude_none=True)

        def action(cam, src):
            reply = agent.arbiter.studio_set_mode(body.mode, body.profile, src)
            cam._commanded_by(src)
            return reply
        return run(request, camera_id, "set_mode", args, action)

    # ------------------------------------------------------------------------------------------- resync
    @app.post("/cameras/{camera_id}/resync")
    def resync(camera_id: str, request: Request):
        return run(request, camera_id, "resync", {}, lambda cam, src: cam.resync(src))

    # ------------------------------------------------------------------------------------------- not in this package
    @app.get("/cameras/{camera_id}/image")
    def get_image(camera_id: str):
        agent.camera_for(camera_id)
        return {}                         # image_controls is [] : no controls, so an empty object

    @app.put("/cameras/{camera_id}/image")
    def set_image(camera_id: str, request: Request, body: dict = Body(default_factory=dict)):
        def action(cam, src):
            if body:
                raise AgentError(UNKNOWN_CONTROL, "%s has no image control %s; it has none (image controls are not "
                                 "served by the zone agent)" % (cam.id, ", ".join(sorted(body))))
            return {}
        return run(request, camera_id, "set_image", {name: value for name, value in body.items()}, action)

    @app.get("/cameras/{camera_id}/crop")
    def get_crop(camera_id: str):
        agent.camera_for(camera_id).not_supported("crop")

    @app.put("/cameras/{camera_id}/crop")
    def set_crop(camera_id: str, request: Request, body: Any = Body(default=None)):
        return run(request, camera_id, "set_crop", body if isinstance(body, dict) else {},
                   lambda cam, src: cam.not_supported("crop"))

    return app


def _add_error_handlers(app: FastAPI) -> None:
    """Every error leaves as {error, detail} with the contract's status code."""

    @app.exception_handler(AgentError)
    async def refused(request: Request, error: AgentError):
        return error_response(error.status, error.code, error.detail)

    @app.exception_handler(RequestValidationError)
    async def bad_body(request: Request, invalid: RequestValidationError):
        problems = "; ".join("%s: %s" % (".".join(str(part) for part in problem["loc"]), problem["msg"])
                             for problem in invalid.errors())
        return error_response(400, "bad_request", problems)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException):
        code = "not_found" if error.status_code == 404 else "http_error"
        return error_response(error.status_code, code, str(error.detail))


async def _push_forever(agent) -> None:
    """Events as soon as they happen; the state of the camera every PUSH_INTERVAL_S, on a fixed grid.

    A state push that falls due within STATE_SETTLE_S (0.1 s) of a mode change waits until then: two commands a
    studio sends back to back (auto, then manual a few milliseconds later) show up as their result, never as a state
    that lasted 5 ms. The grid itself never moves (pushes stay at the same phase), and a push is never later than
    0.1 s, so the contract's 2 Hz minimum holds."""
    loop = asyncio.get_running_loop()
    next_slot = loop.time()          # the grid: next_slot, next_slot + 0.25, ...
    deferred = False                 # a push fell due during a settle: send it when the settle is over
    hold_until = 0.0
    while True:
        try:
            for event in agent.take_events():
                _broadcast(agent, event)
                if event.get("event") == MODE_CHANGED:
                    hold_until = loop.time() + STATE_SETTLE_S
            now = loop.time()
            if now >= next_slot:
                deferred = True
                while next_slot <= now:
                    next_slot += PUSH_INTERVAL_S
            if deferred and now >= hold_until:
                _broadcast(agent, agent.state_message())
                deferred = False
        except Exception as e:  # noqa: BLE001 (a bad push must not end the pushes for good)
            agent.print("agent push failed: %s: %s" % (type(e).__name__, e))
            deferred = False
        wake_at = hold_until if deferred else next_slot
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(agent.wake.wait(), max(0.0, wake_at - loop.time()))
        agent.wake.clear()


def _broadcast(agent, message: dict) -> None:
    """Queue a message for every studio. A studio that has stopped reading (a dead link, a hung studio) must not make
    the queue grow forever: once MAX_QUEUED_STATES messages wait, state pushes are dropped for it (the next one
    says everything anyway); events are always kept."""
    for queue in list(agent.subscribers):
        if message.get("type") == "state" and queue.qsize() >= MAX_QUEUED_STATES:
            continue
        queue.put_nowait(message)


async def _serve_state_socket(agent, websocket: WebSocket) -> None:
    """One studio's state WebSocket: pushes out, pings in, until either side closes."""
    if agent.token and websocket.headers.get(TOKEN_HEADER) != agent.token:
        refusal = error_response(401, UNAUTHORIZED, "missing or wrong %s" % TOKEN_HEADER)
        try:
            await websocket.send_denial_response(refusal)
        except RuntimeError:          # the server can't answer the handshake with HTTP: refuse plainly
            await websocket.close(code=4401)
        return
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    agent.subscribers.add(queue)
    for message in agent.greeting():
        queue.put_nowait(message)

    async def send_messages():
        while (message := await queue.get()) is not None:
            await websocket.send_json(message)
        await websocket.close()

    loop = asyncio.get_running_loop()
    last_heard = loop.time()

    async def answer_pings():
        nonlocal last_heard
        while True:
            text = await websocket.receive_text()
            last_heard = loop.time()
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and message.get("type") == "ping":
                queue.put_nowait({"type": "pong"})

    async def close_if_silent():
        """The studio pings every 2 s. Nothing for SILENT_STUDIO_CLOSE_S: it's gone (a link that died without
        closing): close the socket, which frees its queue."""
        while loop.time() - last_heard < SILENT_STUDIO_CLOSE_S:
            await asyncio.sleep(min(1.0, SILENT_STUDIO_CLOSE_S / 4))

    tasks = [asyncio.create_task(send_messages()), asyncio.create_task(answer_pings()),
             asyncio.create_task(close_if_silent())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        agent.subscribers.discard(queue)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                await task
        with contextlib.suppress(Exception):
            await websocket.close()       # already closed in most cases; this ends the silent studio's socket
