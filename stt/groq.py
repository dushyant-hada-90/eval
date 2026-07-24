from __future__ import annotations

import io
import time

from groq import AsyncGroq

from providers.registry import stt_registry
from utils.config import settings

from .base import AbstractSTTAdapter, STTResult


@stt_registry.register("groq")
class GroqSTTAdapter(AbstractSTTAdapter):
    name = "groq"
    default_model = "whisper-large-v3-turbo"

    def __init__(self, model: str | None = None, language: str | None = None) -> None:
        super().__init__(
            model=model or settings.groq_stt_model or self.default_model,
            language=language,
        )
        self.client = AsyncGroq(api_key=settings.groq_api_key)

    async def transcribe_wav(self, wav_bytes: bytes) -> STTResult:
        bio = io.BytesIO(wav_bytes)
        bio.name = "audio.wav"
        kwargs = {
            "file": bio,
            "model": self.model,
            "response_format": "text",
        }
        if self.language:
            kwargs["language"] = self.language.split("-")[0]
        t0 = time.perf_counter()
        result = await self.client.audio.transcriptions.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        text = result.strip() if isinstance(result, str) else str(
            getattr(result, "text", result)
        ).strip()
        return STTResult(
            text=text,
            latency_ms=latency_ms,
            provider=self.name,
            model=self.model,
            language=self.language,
        )
