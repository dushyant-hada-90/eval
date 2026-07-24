from __future__ import annotations

import csv
import io
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

router = APIRouter(prefix="/api", tags=["results"])


def _db(request: Request):
    return request.app.state.db


@router.get("/evaluations")
def list_evaluations(request: Request, limit: int = 100) -> list[dict[str, Any]]:
    return _db(request).list_evaluations(limit=limit)


@router.get("/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: int, request: Request) -> dict[str, Any]:
    row = _db(request).get_evaluation(evaluation_id)
    if not row:
        raise HTTPException(404, "Evaluation not found")
    return row


@router.get("/evaluations/{evaluation_id}/turns")
def list_turns(evaluation_id: int, request: Request) -> list[dict[str, Any]]:
    if not _db(request).get_evaluation(evaluation_id):
        raise HTTPException(404, "Evaluation not found")
    return _db(request).list_turns(evaluation_id)


@router.delete("/evaluations/{evaluation_id}")
def delete_evaluation(evaluation_id: int, request: Request) -> dict[str, Any]:
    if not _db(request).delete_evaluation(evaluation_id):
        raise HTTPException(404, "Evaluation not found")
    return {"ok": True, "deleted_id": evaluation_id}


@router.get("/evaluations/{evaluation_id}/timeline")
def get_timeline(evaluation_id: int, request: Request) -> dict[str, Any]:
    from engine.timeline import build_timeline

    try:
        return build_timeline(_db(request), evaluation_id)
    except KeyError as exc:
        raise HTTPException(404, "Evaluation not found") from exc


@router.get("/evaluations/{evaluation_id}/session-audio")
def get_session_audio(
    evaluation_id: int, request: Request, refresh: bool = False
) -> FileResponse:
    """Merged session WAV (agent + our audio). FileResponse enables HTTP range seeking."""
    from audio.session_mix import ensure_session_mix_file

    if not _db(request).get_evaluation(evaluation_id):
        raise HTTPException(404, "Evaluation not found")
    try:
        path, meta = ensure_session_mix_file(
            _db(request), evaluation_id, force=refresh
        )
    except KeyError as exc:
        raise HTTPException(404, "Evaluation not found") from exc
    except Exception as exc:
        raise HTTPException(400, f"Could not mix session audio: {exc}") from exc
    if meta.get("clips_placed", 0) <= 0:
        raise HTTPException(404, "No audio clips found for this evaluation")
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=f"eval_{evaluation_id}_session.wav",
        headers={
            "Accept-Ranges": "bytes",
            "X-Clips-Placed": str(meta.get("clips_placed", 0)),
            "X-Clips-User": str(meta.get("clips_user", 0)),
            "X-Clips-Agent": str(meta.get("clips_agent", 0)),
            "Cache-Control": "no-cache" if refresh else "private, max-age=60",
        },
    )


@router.get("/turns/{turn_id}")
def get_turn(turn_id: int, request: Request) -> dict[str, Any]:
    row = _db(request).get_turn(turn_id)
    if not row:
        raise HTTPException(404, "Turn not found")
    return row


@router.get("/export.csv")
def export_csv(request: Request) -> StreamingResponse:
    rows = _db(request).all_turns_flat()
    buf = io.StringIO()
    fields = [
        "evaluation_id",
        "scenario_id",
        "eval_type",
        "agent_name",
        "provider_name",
        "model_name",
        "variation_name",
        "run_timestamp",
        "turn_id",
        "turn_number",
        "user_intent",
        "ttf_ms",
        "ftl_ms",
        "latency_ms",
        "wer",
        "cer",
        "reference_text",
        "test_prompt",
        "agent_transcript",
        "intent_alignment_score",
        "question_asking_score",
        "tone_score",
        "context_retention_score",
        "avg_score",
        "scoring_reasoning",
    ]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=evaluations.csv"},
    )
