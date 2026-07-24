from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from utils.config import settings

router = APIRouter(prefix="/api", tags=["audio"])


@router.get("/audio/{turn_id}")
def get_audio(turn_id: int, request: Request, type: str = "agent") -> FileResponse:
    """Serve stored WAV. type=agent|test"""
    db = request.app.state.db
    turn = db.get_turn(turn_id)
    if not turn:
        raise HTTPException(404, "Turn not found")

    audio_type = "agent_output" if type in ("agent", "agent_output") else "test_input"
    rec = db.get_audio_for_turn(turn_id, audio_type)
    path_str = None
    if rec and rec.get("wav_path"):
        path_str = rec["wav_path"]
    elif type in ("agent", "agent_output"):
        path_str = turn.get("agent_audio_path")
    else:
        path_str = turn.get("test_audio_path")

    if not path_str:
        raise HTTPException(404, "Audio path not found for this turn")

    path = Path(path_str)
    if not path.is_absolute():
        path = settings.project_root / path
    if not path.exists():
        # Also try recordings/ relative to CWD
        alt = Path(path_str)
        if alt.exists():
            path = alt
        else:
            raise HTTPException(404, f"Audio file missing: {path}")

    return FileResponse(path, media_type="audio/wav", filename=path.name)
