from __future__ import annotations

import asyncio
import io
import time

from providers.registry import stt_registry
from utils.config import settings

from .base import AbstractSTTAdapter, STTResult


@stt_registry.register("sarvam")
class SarvamSTTAdapter(AbstractSTTAdapter):
    name = "sarvam"
    default_model = "saaras:v3"

    def __init__(self, model: str | None = None, language: str | None = None) -> None:
        super().__init__(
            model=model or settings.sarvam_stt_model or self.default_model,
            language=language or settings.sarvam_language or "en-IN",
        )
        if not settings.sarvam_api_key:
            raise RuntimeError("SARVAM_API_KEY is required for sarvam STT")

    def _client(self):
        from sarvamai import SarvamAI

        return SarvamAI(api_subscription_key=settings.sarvam_api_key)

    async def transcribe_wav(self, wav_bytes: bytes) -> STTResult:
        def _sync() -> str:
            client = self._client()
            bio = io.BytesIO(wav_bytes)
            bio.name = "audio.wav"
            resp = client.speech_to_text.transcribe(
                file=bio,
                model=self.model,
                mode="transcribe",
                language_code=self.language,
            )
            return getattr(resp, "transcript", None) or str(resp)

        t0 = time.perf_counter()
        text = await asyncio.to_thread(_sync)
        latency_ms = (time.perf_counter() - t0) * 1000
        return STTResult(
            text=text.strip(),
            latency_ms=latency_ms,
            provider=self.name,
            model=self.model,
            language=self.language,
        )
