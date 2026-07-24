from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from db.models import Database
from engine.events import get_event_bus
from utils.config import settings
from utils.logging import setup_logging
from utils.tts_runtime import check_gradio_tts_on_startup

from .api.audio import router as audio_router
from .api.live import router as live_router
from .api.results import router as results_router

setup_logging()

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Probe Gradio clone TTS once at boot; fall back to Groq when unhealthy
    app.state.tts_status = await check_gradio_tts_on_startup()
    yield


app = FastAPI(
    title="Realtime Agents Eval Dashboard",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.db = Database(settings.db_path)
app.state.tts_status = {"ok": False, "checked": False, "provider": "groq", "message": ""}
# Dashboard owns the bus; do not HTTP-forward events back to ourselves
_bus = get_event_bus()
_bus.forward_enabled = False
app.state.bus = _bus

# Share DB with live checkpoint persistence
from engine.checkpoint_session import get_checkpoint_manager

get_checkpoint_manager().db = app.state.db

app.include_router(results_router)
app.include_router(audio_router)
app.include_router(live_router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/results")
async def results_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "results.html")


@app.post("/api/events")
async def ingest_event(request: Request) -> dict[str, Any]:
    """CLI / out-of-process runners POST live events here."""
    payload = await request.json()
    event_type = payload.pop("type", "message")
    event = await app.state.bus.emit(event_type, forward=False, **payload)
    return {"ok": True, "event": event}


@app.get("/api/events/history")
async def event_history(limit: int = 100) -> list[dict[str, Any]]:
    hist = app.state.bus.history
    return hist[-limit:]


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    await websocket.accept()
    bus = app.state.bus
    queue = await bus.subscribe()
    try:
        # Send recent history first
        for event in bus.history[-50:]:
            await websocket.send_json(event)
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        raise
    finally:
        await bus.unsubscribe(queue)


def create_app() -> FastAPI:
    return app
