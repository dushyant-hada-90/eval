from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from db.models import Database
from utils.config import settings

# UI color groups — results page only shows agent + caller (judge) audio
GROUP_META = {
    "agent": {"label": "Agent audio", "color": "#3ecf8e"},
    "user": {"label": "Caller audio", "color": "#f0b429"},
    "latency": {"label": "Model wait", "color": "#c084fc"},
    "connection": {"label": "Connection", "color": "#5b8def"},
    "human": {"label": "Human checkpoint", "color": "#8b9bb4"},
    "system": {"label": "Pipeline", "color": "#38bdf8"},
    "error": {"label": "Error", "color": "#f07178"},
}


def build_timeline(db: Database, evaluation_id: int) -> dict[str, Any]:
    """
    Conversation audio timeline only: agent ↔ caller clips in speak order.

    Ignores wall-clock session_events (those include LLM / STT / TTS gen /
    human-approval gaps which scramble ordering vs the actual call audio).
    """
    ev = db.get_evaluation(evaluation_id)
    if not ev:
        raise KeyError("evaluation not found")

    events = build_audio_conversation_timeline(db, evaluation_id)
    segments = [e for e in events if (e.get("duration_ms") or 0) > 0]
    points = [e for e in events if not (e.get("duration_ms") or 0) > 0]

    end_candidates = [0.0]
    for e in events:
        t0 = float(e.get("t_ms") or 0)
        dur = float(e.get("duration_ms") or 0)
        end_candidates.append(t0 + max(dur, 0))
    total_ms = max(end_candidates)

    return {
        "evaluation_id": evaluation_id,
        "source": "audio_conversation",
        "total_ms": total_ms,
        "groups": {
            "latency": GROUP_META["latency"],
            "agent": GROUP_META["agent"],
            "user": GROUP_META["user"],
        },
        "events": events,
        "segments": segments,
        "points": points,
        "evaluation": {
            "id": ev.get("id"),
            "scenario_id": ev.get("scenario_id"),
            "eval_type": ev.get("eval_type"),
            "agent_name": ev.get("agent_name"),
            "variation_name": ev.get("variation_name"),
            "run_timestamp": ev.get("run_timestamp"),
            "avg_ttf_ms": ev.get("avg_ttf_ms"),
            "avg_ftl_ms": ev.get("avg_ftl_ms"),
            "total_turns": ev.get("total_turns"),
        },
    }


def build_audio_conversation_timeline(
    db: Database, evaluation_id: int
) -> list[dict[str, Any]]:
    """
    Sequential speak order:

      agent(turn1) → caller(turn2's test audio) → agent(turn2) → caller(turn3) → …

    Caller audio for the reply that *follows* agent turn N is stored on turn N+1
    as test_audio_path (live checkpoint). Durations come from audio_recordings
    when available, else WAV length / fallback.
    """
    turns = sorted(db.list_turns(evaluation_id), key=lambda r: int(r["turn_number"]))
    audio_rows = db.list_audio_for_evaluation(evaluation_id)
    by_turn: dict[int, dict[str, Any]] = {}
    for a in audio_rows:
        tn = int(a.get("turn_number") or 0)
        by_turn.setdefault(tn, {})
        if a.get("audio_type") == "agent_output":
            by_turn[tn]["agent_dur"] = a.get("duration_ms")
            by_turn[tn]["agent_path"] = a.get("wav_path")
        elif a.get("audio_type") == "test_input":
            by_turn[tn]["user_dur"] = a.get("duration_ms")
            by_turn[tn]["user_path"] = a.get("wav_path")

    events: list[dict[str, Any]] = []
    t = 0.0

    def add(
        *,
        event_type: str,
        label: str,
        group: str,
        duration_ms: float,
        turn_number: int,
        detail: str = "",
        wav_path: Optional[str] = None,
    ) -> None:
        nonlocal t
        dur = max(0.0, float(duration_ms))
        events.append(
            {
                "id": None,
                "turn_number": turn_number,
                "event_type": event_type,
                "label": label,
                "group": group,
                "color": GROUP_META.get(group, {}).get("color", "#8b9bb4"),
                "t_ms": t,
                "duration_ms": dur,
                "detail": detail,
                "wav_path": wav_path,
            }
        )
        t += dur

    for turn in turns:
        tn = int(turn["turn_number"])
        meta = by_turn.get(tn, {})
        agent_path = meta.get("agent_path") or turn.get("agent_audio_path")
        agent_dur = _duration_ms(
            meta.get("agent_dur"), agent_path, fallback=1500.0
        )
        transcript = (turn.get("agent_transcript") or "")[:140]
        ttf = turn.get("ttf_ms")
        ftl = turn.get("ftl_ms")

        # Caller audio that *preceded* this agent reply (not on greeting turn)
        user_path = meta.get("user_path") or turn.get("test_audio_path")
        user_text = turn.get("test_prompt") or ""
        has_user = (
            tn > 1
            and user_path
            and user_text != "(agent greeting)"
        )
        if has_user:
            user_dur = _duration_ms(
                meta.get("user_dur"), user_path, fallback=1200.0
            )
            add(
                event_type="user_audio",
                label=f"Caller (before agent turn {tn})",
                group="user",
                duration_ms=user_dur,
                turn_number=tn,
                detail=user_text[:140],
                wav_path=user_path,
            )

        # Agent-only wait before first audio token (excludes judge/STT/TTS/approval)
        if tn == 1 and ttf is not None and float(ttf) > 0:
            add(
                event_type="ttf_wait",
                label=f"TTF — wait to first token ({float(ttf):.0f} ms)",
                group="latency",
                duration_ms=float(ttf),
                turn_number=tn,
                detail=f"Session ready → first agent audio · TTF={float(ttf):.1f} ms",
            )
        elif has_user and ftl is not None and float(ftl) > 0:
            add(
                event_type="ftl_wait",
                label=f"FTL — wait to first token ({float(ftl):.0f} ms)",
                group="latency",
                duration_ms=float(ftl),
                turn_number=tn,
                detail=(
                    f"Caller audio fully sent → first agent audio · "
                    f"FTL={float(ftl):.1f} ms"
                ),
            )

        if agent_path or agent_dur:
            add(
                event_type="agent_audio",
                label=f"Agent turn {tn}",
                group="agent",
                duration_ms=agent_dur,
                turn_number=tn,
                detail=transcript,
                wav_path=agent_path,
            )

    return events


def _duration_ms(
    recorded: Optional[float], path_str: Optional[str], *, fallback: float
) -> float:
    if recorded is not None and float(recorded) > 0:
        return float(recorded)
    path = _resolve(path_str)
    if path is None:
        return fallback
    try:
        import wave

        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            return (frames / float(rate)) * 1000.0
    except Exception:
        return fallback


def _resolve(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = settings.project_root / path
    if path.exists():
        return path
    alt = Path(path_str)
    return alt if alt.exists() else None
