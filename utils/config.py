from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=False)


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _str(name: str, default: str = "") -> str:
    raw = os.getenv(name, default)
    if raw is None:
        return default
    return raw.strip().strip('"').strip("'")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_realtime_model: str
    openai_realtime_voice: str
    openai_stt_model: str
    openai_tts_model: str
    openai_tts_voice: str
    gemini_api_key: str
    gemini_realtime_model: str
    gemini_realtime_voice: str
    google_stt_model: str
    google_tts_model: str
    google_tts_voice: str
    groq_api_key: str
    groq_stt_model: str
    groq_llm_model: str
    groq_tts_model: str
    groq_tts_voice: str
    sarvam_api_key: str
    sarvam_stt_model: str
    sarvam_tts_model: str
    sarvam_tts_voice: str
    sarvam_language: str
    default_stt_provider: str
    default_tts_provider: str
    gradio_tts_url: str
    gradio_tts_path: str
    gradio_tts_fn_index: int
    audio_sample_rate: int
    audio_channels: int
    audio_codec: str
    db_path: Path
    recordings_dir: Path
    dashboard_host: str
    dashboard_port: int
    log_level: str
    project_root: Path


def load_settings() -> Settings:
    db = os.getenv("DB_PATH", "./eval_results.db")
    recordings = os.getenv("RECORDINGS_DIR", "./recordings")
    db_path = Path(db)
    recordings_dir = Path(recordings)
    if not db_path.is_absolute():
        db_path = _ROOT / db_path
    if not recordings_dir.is_absolute():
        recordings_dir = _ROOT / recordings_dir
    return Settings(
        openai_api_key=_str("OPENAI_API_KEY"),
        openai_realtime_model=_str("OPENAI_REALTIME_MODEL", "gpt-realtime"),
        openai_realtime_voice=_str("OPENAI_REALTIME_VOICE", "alloy"),
        openai_stt_model=_str("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
        openai_tts_model=_str("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        openai_tts_voice=_str("OPENAI_TTS_VOICE", "alloy"),
        gemini_api_key=_str("GEMINI_API_KEY") or _str("GOOGLE_API_KEY"),
        gemini_realtime_model=_str(
            "GEMINI_REALTIME_MODEL", "gemini-3.1-flash-live-preview"
        ),
        gemini_realtime_voice=_str("GEMINI_REALTIME_VOICE", "Puck"),
        google_stt_model=_str("GOOGLE_STT_MODEL", "gemini-2.5-flash"),
        google_tts_model=_str("GOOGLE_TTS_MODEL", "gemini-2.5-flash-preview-tts"),
        google_tts_voice=_str("GOOGLE_TTS_VOICE", "Kore"),
        groq_api_key=_str("GROQ_API_KEY"),
        groq_stt_model=_str("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
        groq_llm_model=_str("GROQ_LLM_MODEL", "llama-3.3-70b-versatile"),
        groq_tts_model=_str("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english"),
        groq_tts_voice=_str("GROQ_TTS_VOICE", "austin"),
        sarvam_api_key=_str("SARVAM_API_KEY"),
        sarvam_stt_model=_str("SARVAM_STT_MODEL", "saaras:v3"),
        sarvam_tts_model=_str("SARVAM_TTS_MODEL", "bulbul:v3"),
        sarvam_tts_voice=_str("SARVAM_TTS_VOICE", "shubh"),
        sarvam_language=_str("SARVAM_LANGUAGE", "en-IN"),
        default_stt_provider=_str("DEFAULT_STT_PROVIDER", "groq"),
        default_tts_provider=_str(
            "DEFAULT_TTS_PROVIDER",
            "gradio" if _str("GRADIO_TTS_URL") else "groq",
        ),
        gradio_tts_url=_str("GRADIO_TTS_URL"),
        gradio_tts_path=_str("GRADIO_TTS_PATH", "auto"),
        gradio_tts_fn_index=_int("GRADIO_TTS_FN_INDEX", 0),
        audio_sample_rate=_int("AUDIO_SAMPLE_RATE", 24000),
        audio_channels=_int("AUDIO_CHANNELS", 1),
        audio_codec=_str("AUDIO_CODEC", "pcm16"),
        db_path=db_path,
        recordings_dir=recordings_dir,
        dashboard_host=_str("DASHBOARD_HOST", "0.0.0.0"),
        dashboard_port=_int("DASHBOARD_PORT", 8000),
        log_level=_str("LOG_LEVEL", "INFO"),
        project_root=_ROOT,
    )


settings = load_settings()
