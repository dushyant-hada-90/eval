from __future__ import annotations

import asyncio
import base64
import time

from audio.pcm_converter import PCMConverter
from providers.registry import tts_registry
from utils.config import settings

from .base import AbstractTTSAdapter


@tts_registry.register("sarvam")
class SarvamTTSAdapter(AbstractTTSAdapter):
    name = "sarvam"
    default_model = "bulbul:v3"
    default_voice = "shubh"
    native_sample_rate = 22050

    def __init__(
        self,
        model: str | None = None,
        voice: str | None = None,
        language: str | None = None,
    ) -> None:
        super().__init__(
            model=model or settings.sarvam_tts_model or self.default_model,
            voice=voice or settings.sarvam_tts_voice or self.default_voice,
            language=language or settings.sarvam_language or "en-IN",
        )
        if not settings.sarvam_api_key:
            raise RuntimeError("SARVAM_API_KEY is required for sarvam TTS")

    def _client(self):
        from sarvamai import SarvamAI

        return SarvamAI(api_subscription_key=settings.sarvam_api_key)

    async def synthesize_wav(self, text: str) -> tuple[bytes, float]:
        def _sync() -> bytes:
            client = self._client()
            resp = client.text_to_speech.convert(
                text=text,
                target_language_code=self.language,
                model=self.model,
                speaker=self.voice,
                speech_sample_rate=24000,
            )
            audios = getattr(resp, "audios", None) or []
            if not audios:
                raise RuntimeError(f"Empty Sarvam TTS response: {resp}")
            raw = audios[0]
            if isinstance(raw, (bytes, bytearray)):
                data = bytes(raw)
            else:
                data = base64.b64decode(raw)
            # Sarvam may return wav or raw pcm; normalize to wav
            if data[:4] == b"RIFF":
                return data
            return PCMConverter.pcm_to_wav(data, sample_rate=24000)

        t0 = time.perf_counter()
        wav = await asyncio.to_thread(_sync)
        latency_ms = (time.perf_counter() - t0) * 1000
        return wav, latency_ms
