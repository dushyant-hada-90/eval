from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from audio.pcm_converter import PCMConverter
from db.models import Database
from engine.timeline import build_timeline
from utils.config import settings


def _resolve_wav(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = settings.project_root / path
    if path.exists():
        return path
    alt = Path(path_str)
    return alt if alt.exists() else None


def _load_pcm(path: Path, sample_rate: int) -> bytes:
    pcm, _ = PCMConverter.wav_to_pcm(path.read_bytes(), target_rate=sample_rate)
    return pcm


def build_session_mix(
    db: Database,
    evaluation_id: int,
    *,
    sample_rate: int = 24000,
) -> tuple[bytes, dict[str, Any]]:
    """
    Mix agent + caller WAVs in conversation order (same positions as the
    audio-only timeline). No approval / LLM / TTS-gen silence.
    """
    tl = build_timeline(db, evaluation_id)
    speech = [
        e
        for e in tl.get("events") or []
        if e.get("group") in {"agent", "user"}
        and float(e.get("duration_ms") or 0) > 0
        and e.get("wav_path")
    ]

    total_ms = max(float(tl.get("total_ms") or 0), 500.0)
    n_samples = int(round((total_ms + 250.0) * sample_rate / 1000.0))
    mix = np.zeros(max(n_samples, sample_rate // 4), dtype=np.float32)
    placed = 0
    placed_user = 0
    placed_agent = 0

    for e in speech:
        path = _resolve_wav(e.get("wav_path"))
        if not path:
            continue
        try:
            pcm = _load_pcm(path, sample_rate)
        except Exception:
            continue
        start = int(round(float(e["t_ms"]) * sample_rate / 1000.0))
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        end = start + len(samples)
        if start < 0:
            samples = samples[-start:]
            start = 0
            end = start + len(samples)
        if end > len(mix):
            mix = np.concatenate(
                [mix, np.zeros(end - len(mix) + sample_rate // 10, dtype=np.float32)]
            )
        mix[start:end] += samples
        placed += 1
        if e.get("group") == "user":
            placed_user += 1
        else:
            placed_agent += 1

    # Fallback: concat in turn order if timeline had no paths
    if placed == 0:
        cursor = 0
        gap = int(0.05 * sample_rate)
        for turn in sorted(db.list_turns(evaluation_id), key=lambda r: r["turn_number"]):
            for path_key, kind in (
                ("test_audio_path", "user"),
                ("agent_audio_path", "agent"),
            ):
                if kind == "user" and turn.get("test_prompt") == "(agent greeting)":
                    continue
                if kind == "user" and int(turn["turn_number"]) <= 1:
                    continue
                path = _resolve_wav(turn.get(path_key))
                if not path:
                    continue
                try:
                    pcm = _load_pcm(path, sample_rate)
                except Exception:
                    continue
                samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
                end = cursor + len(samples)
                if end > len(mix):
                    mix = np.concatenate(
                        [mix, np.zeros(end - len(mix) + gap, dtype=np.float32)]
                    )
                mix[cursor:end] += samples
                cursor = end + gap
                placed += 1
                if kind == "user":
                    placed_user += 1
                else:
                    placed_agent += 1

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 32767:
        mix *= 32767.0 / peak
    pcm_arr = np.clip(mix, -32768, 32767).astype(np.int16)
    if placed:
        nz = np.nonzero(np.abs(pcm_arr) > 50)[0]
        if len(nz):
            cut = min(len(pcm_arr), int(nz[-1] + 0.15 * sample_rate))
            pcm_arr = pcm_arr[:cut]
    pcm_out = pcm_arr.tobytes()
    wav = PCMConverter.pcm_to_wav(pcm_out, sample_rate=sample_rate)
    meta = {
        "evaluation_id": evaluation_id,
        "clips_placed": placed,
        "clips_user": placed_user,
        "clips_agent": placed_agent,
        "duration_ms": PCMConverter.duration_ms(pcm_out, sample_rate),
        "timeline_total_ms": float(tl.get("total_ms") or 0),
        "sample_rate": sample_rate,
        "aligned": True,
    }
    return wav, meta


def cached_session_mix_path(evaluation_id: int) -> Path:
    out_dir = settings.recordings_dir / "_session_mix"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"eval_{evaluation_id}_session.wav"


def ensure_session_mix_file(
    db: Database, evaluation_id: int, *, force: bool = False
) -> tuple[Path, dict[str, Any]]:
    path = cached_session_mix_path(evaluation_id)
    meta_path = path.with_suffix(".meta.json")
    if path.exists() and meta_path.exists() and not force:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("clips_placed", 0) > 0 and meta.get("aligned"):
                return path, meta
        except Exception:
            pass
    wav, meta = build_session_mix(db, evaluation_id)
    path.write_bytes(wav)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return path, meta
