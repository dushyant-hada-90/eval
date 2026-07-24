from __future__ import annotations

import io
import time

from openai import AsyncOpenAI

from providers.registry import stt_registry
from utils.config import settings

from .base import AbstractSTTAdapter, STTResult


@stt_registry.register("openai", "gpt", "whisper")
class OpenAISTTAdapter(AbstractSTTAdapter):
    name = "openai"
    default_model = "gpt-4o-mini-transcribe"

    def __init__(self, model: str | None = None, language: str | None = None) -> None:
        super().__init__(
            model=model or settings.openai_stt_model or self.default_model,
            language=language,
        )
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for openai STT")
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def transcribe_wav(self, wav_bytes: bytes) -> STTResult:
        bio = io.BytesIO(wav_bytes)
        bio.name = "audio.wav"
        kwargs = {"file": bio, "model": self.model}
        if self.language:
            kwargs["language"] = self.language.split("-")[0]
        t0 = time.perf_counter()
        result = await self.client.audio.transcriptions.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        text = getattr(result, "text", None) or str(result)
        return STTResult(
            text=text.strip(),
            latency_ms=latency_ms,
            provider=self.name,
            model=self.model,
            language=self.language,
        )
