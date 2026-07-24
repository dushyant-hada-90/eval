from __future__ import annotations

import time

from google import genai
from google.genai import types

from providers.registry import stt_registry
from utils.config import settings

from .base import AbstractSTTAdapter, STTResult


@stt_registry.register("google", "gemini")
class GoogleSTTAdapter(AbstractSTTAdapter):
    """Transcribe via Gemini (API key) — no GCP Speech project required."""

    name = "google"
    default_model = "gemini-2.5-flash"

    def __init__(self, model: str | None = None, language: str | None = None) -> None:
        super().__init__(
            model=model or settings.google_stt_model or self.default_model,
            language=language,
        )
        key = settings.gemini_api_key
        if not key:
            raise RuntimeError("GEMINI_API_KEY is required for google STT")
        self.client = genai.Client(api_key=key)

    async def transcribe_wav(self, wav_bytes: bytes) -> STTResult:
        lang_hint = f" Language hint: {self.language}." if self.language else ""
        prompt = (
            "Transcribe the audio exactly. Return only the transcript text, "
            "no quotes or commentary." + lang_hint
        )
        t0 = time.perf_counter()
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = (getattr(response, "text", None) or "").strip()
        return STTResult(
            text=text,
            latency_ms=latency_ms,
            provider=self.name,
            model=self.model,
            language=self.language,
        )
