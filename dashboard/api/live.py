from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from engine.checkpoint_session import get_checkpoint_manager
from utils.config import settings
from utils.tts_runtime import get_tts_status, preferred_live_tts

router = APIRouter(prefix="/api/live", tags=["live-checkpoint"])


class StartBody(BaseModel):
    scenario: str = Field(
        default="scenarios/myntra_support_rohit.yaml",
        description="Path to realtime scenario YAML",
    )
    variation: Optional[str] = "delayed_kurta"
    agent: Optional[str] = "gpt_realtime"
    stt: Optional[str] = "groq"
    tts: Optional[str] = "auto"


class ApproveTranscriptBody(BaseModel):
    transcript: Optional[str] = None


class ApproveLlmBody(BaseModel):
    next_user_utterance: Optional[str] = None


class RegenerateTtsBody(BaseModel):
    text: Optional[str] = None


@router.get("/tts-status")
async def tts_status() -> dict[str, Any]:
    """Startup Gradio probe result + resolved live TTS provider."""
    status = get_tts_status()
    status["resolved_provider"] = preferred_live_tts("auto")
    return status


@router.get("/state")
async def live_state() -> dict[str, Any]:
    mgr = get_checkpoint_manager()
    if not mgr.session:
        return {"state": "idle", "awaiting": None}
    return mgr.session.public_state()


@router.post("/start")
async def live_start(body: StartBody) -> dict[str, Any]:
    mgr = get_checkpoint_manager()
    try:
        return await mgr.start(
            body.scenario,
            variation=body.variation,
            agent_name=body.agent,
            stt_provider=body.stt,
            tts_provider=body.tts,
        )
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/approve-transcript")
async def approve_transcript(body: ApproveTranscriptBody) -> dict[str, Any]:
    mgr = get_checkpoint_manager()
    try:
        return await mgr.approve_transcript(body.transcript)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/approve-llm")
async def approve_llm(body: ApproveLlmBody) -> dict[str, Any]:
    mgr = get_checkpoint_manager()
    try:
        return await mgr.approve_llm(body.next_user_utterance)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/regenerate-tts")
async def regenerate_tts(body: RegenerateTtsBody) -> dict[str, Any]:
    mgr = get_checkpoint_manager()
    try:
        return await mgr.regenerate_tts(body.text)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/send-tts")
async def send_tts() -> dict[str, Any]:
    mgr = get_checkpoint_manager()
    try:
        return await mgr.send_tts()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/stop")
async def live_stop() -> dict[str, Any]:
    mgr = get_checkpoint_manager()
    return await mgr.stop()


@router.get("/audio/{session_id}/{kind}")
async def live_audio(session_id: str, kind: str) -> FileResponse:
    mgr = get_checkpoint_manager()
    sess = mgr.session
    if not sess or sess.session_id != session_id:
        raise HTTPException(404, "Session not found")
    if kind == "agent":
        rel = sess.agent_audio_path
    elif kind == "tts":
        rel = sess.tts_audio_path
    else:
        raise HTTPException(400, "kind must be agent|tts")
    if not rel:
        raise HTTPException(404, "Audio not ready")
    path = Path(rel)
    if not path.is_absolute():
        path = settings.project_root / path
    if not path.exists():
        raise HTTPException(404, f"Missing file {path}")
    return FileResponse(path, media_type="audio/wav", filename=path.name)
