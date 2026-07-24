from __future__ import annotations

import time

from openai import AsyncOpenAI

from providers.registry import tts_registry
from utils.config import settings

from .base import AbstractTTSAdapter


@tts_registry.register("openai", "gpt")
class OpenAITTSAdapter(AbstractTTSAdapter):
    name = "openai"
    default_model = "gpt-4o-mini-tts"
    default_voice = "alloy"
    native_sample_rate = 24000

    def __init__(
        self,
        model: str | None = None,
        voice: str | None = None,
        language: str | None = None,
    ) -> None:
        super().__init__(
            model=model or settings.openai_tts_model or self.default_model,
            voice=voice or settings.openai_tts_voice or self.default_voice,
            language=language,
        )
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for openai TTS")
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def synthesize_wav(self, text: str) -> tuple[bytes, float]:
        t0 = time.perf_counter()
        response = await self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="wav",
        )
        data = await response.aread()
        latency_ms = (time.perf_counter() - t0) * 1000
        return data, latency_ms
