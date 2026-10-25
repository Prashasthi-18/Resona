"""FastAPI server: serves the dashboard, streams the engine over a WebSocket,
and exposes the phone-alert setup API.

WebSocket protocol (JSON)
  server -> client: state | frame | whistle_start | whistle_end | target_reached
                    | reset | source_ended | error | notify_result
  client -> server: {"cmd": "start" | "stop" | "reset"}
                    {"cmd": "set_target", "value": 3}
                    {"cmd": "set_sensitivity", "value": 0.5}
                    {"cmd": "set_detector", "value": "dsp" | "cnn"}

Alert setup (REST)
  GET    /api/notify                    current setup
  POST   /api/notify/telegram           {"token": "..."}  check and save a bot token
  POST   /api/notify/telegram/link      look for the user's Start message, save the chat
  DELETE /api/notify/telegram
  POST   /api/notify/ntfy               {"topic": optional}  create or set a topic
  DELETE /api/notify/ntfy
  POST   /api/notify/prefs              {"each_whistle": true}
  POST   /api/notify/test               send a test alert to every connected channel
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .config import Settings
from .engine import Engine
from .notify import Notifier, NotifyError

log = logging.getLogger("resona.server")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class Hub:
    """Fan-out of engine messages (from worker threads) to every connected browser."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.queue: asyncio.Queue | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.listeners = []

    def emit(self, msg: dict) -> None:
        for fn in self.listeners:
            try:
                fn(msg)
            except Exception:
                log.exception("Listener failed")
        if self.loop and self.queue:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)

    async def pump(self) -> None:
        while True:
            msg = await self.queue.get()
            text = json.dumps(msg)
            dead = []
            for ws in self.clients:
                try:
                    await ws.send_text(text)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)


def create_app(settings: Settings, source_factory, detector: str = "dsp",
               autostart: bool = False, notifier: Notifier | None = None) -> FastAPI:
    hub = Hub()
    engine = Engine(settings, source_factory, hub.emit, detector=detector)
    notifier = notifier or Notifier()
    notifier.on_result = lambda r: hub.emit({"type": "notify_result", **r})
    hub.listeners.append(lambda m: notifier.handle_event(m, settings.target_count))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        hub.queue = asyncio.Queue(maxsize=2000)
        task = asyncio.create_task(hub.pump())
        if autostart:
            engine.start()
        yield
        engine.shutdown()
        task.cancel()

    app = FastAPI(title="Resona", lifespan=lifespan)
    app.state.engine, app.state.notifier = engine, notifier

    def fail(exc: Exception, code: int = 400):
        return JSONResponse({"error": str(exc)}, status_code=code)

    async def body(request: Request) -> dict:
        try:
            data = await request.json()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/state")
    async def state():
        return engine.state()

    # ------------------------------------------------------------ phone alerts
    @app.get("/api/notify")
    async def notify_status():
        return notifier.status()

    @app.post("/api/notify/telegram")
    async def telegram_connect(request: Request):
        token = (await body(request)).get("token", "")
        try:
            return await asyncio.to_thread(notifier.connect_telegram, token)
        except NotifyError as e:
            return fail(e)

    @app.post("/api/notify/telegram/link")
    async def telegram_link():
        try:
            return await asyncio.to_thread(notifier.link_telegram)
        except NotifyError as e:
            return fail(e)

    @app.delete("/api/notify/telegram")
    async def telegram_disconnect():
        return notifier.disconnect_telegram()

    @app.post("/api/notify/ntfy")
    async def ntfy_set(request: Request):
        data = await body(request)
        try:
            return notifier.set_ntfy(data.get("topic"), data.get("server"))
        except NotifyError as e:
            return fail(e)

    @app.delete("/api/notify/ntfy")
    async def ntfy_disconnect():
        return notifier.disconnect_ntfy()

    @app.post("/api/notify/prefs")
    async def notify_prefs(request: Request):
        return notifier.set_prefs(bool((await body(request)).get("each_whistle", True)))

    @app.post("/api/notify/test")
    async def notify_test():
        try:
            return await asyncio.to_thread(
                notifier.send_now, "Test alert from Resona",
                "If you can read this, whistle alerts will reach this phone.", False, ["white_check_mark"])
        except NotifyError as e:
            return fail(e)

    # --------------------------------------------------------------- live feed
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        hub.clients.add(ws)
        await ws.send_text(json.dumps(engine.state()))
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                cmd, value = msg.get("cmd"), msg.get("value")
                try:
                    if cmd == "start":
                        await asyncio.to_thread(engine.start)
                    elif cmd == "stop":
                        await asyncio.to_thread(engine.stop)
                    elif cmd == "reset":
                        engine.reset()
                    elif cmd == "set_target":
                        engine.set_target(int(value))
                    elif cmd == "set_sensitivity":
                        engine.set_sensitivity(float(value))
                    elif cmd == "set_detector":
                        engine.set_detector(str(value))
                except Exception as exc:
                    log.exception("Command failed")
                    hub.emit({"type": "error", "message": f"Couldn’t run '{cmd}': {exc}"})
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(ws)

    return app
