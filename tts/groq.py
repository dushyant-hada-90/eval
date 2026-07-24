from __future__ import annotations

import time

from groq import AsyncGroq

from providers.registry import tts_registry
from utils.config import settings

from .base import AbstractTTSAdapter


@tts_registry.register("groq")
class GroqTTSAdapter(AbstractTTSAdapter):
    name = "groq"
    default_model = "canopylabs/orpheus-v1-english"
    default_voice = "austin"
    native_sample_rate = 24000

    def __init__(
        self,
        model: str | None = None,
        voice: str | None = None,
        language: str | None = None,
    ) -> None:
        super().__init__(
            model=model or settings.groq_tts_model or self.default_model,
            voice=voice or settings.groq_tts_voice or self.default_voice,
            language=language,
        )
        self.client = AsyncGroq(api_key=settings.groq_api_key)

    async def synthesize_wav(self, text: str) -> tuple[bytes, float]:
        t0 = time.perf_counter()
        response = await self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="wav",
        )
        data = await _read_binary(response)
        latency_ms = (time.perf_counter() - t0) * 1000
        if not data:
            raise RuntimeError("Empty Groq TTS response")
        return data, latency_ms


async def _read_binary(response: object) -> bytes:
    if isinstance(response, (bytes, bytearray)):
        return bytes(response)
    for attr in ("aread", "read"):
        method = getattr(response, attr, None)
        if callable(method):
            data = method()
            if hasattr(data, "__await__"):
                data = await data
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
    content = getattr(response, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    if hasattr(response, "write_to_file"):
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            maybe = response.write_to_file(str(path))
            if hasattr(maybe, "__await__"):
                await maybe
            return path.read_bytes()
        finally:
            path.unlink(missing_ok=True)
    return b""
