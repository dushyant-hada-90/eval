from __future__ import annotations

import base64
import time

from google import genai
from google.genai import types

from audio.pcm_converter import PCMConverter
from providers.registry import tts_registry
from utils.config import settings

from .base import AbstractTTSAdapter


@tts_registry.register("google", "gemini")
class GoogleTTSAdapter(AbstractTTSAdapter):
    """Gemini native audio TTS via API key."""

    name = "google"
    default_model = "gemini-2.5-flash-preview-tts"
    default_voice = "Kore"
    native_sample_rate = 24000

    def __init__(
        self,
        model: str | None = None,
        voice: str | None = None,
        language: str | None = None,
    ) -> None:
        super().__init__(
            model=model or settings.google_tts_model or self.default_model,
            voice=voice or settings.google_tts_voice or self.default_voice,
            language=language,
        )
        key = settings.gemini_api_key
        if not key:
            raise RuntimeError("GEMINI_API_KEY is required for google TTS")
        self.client = genai.Client(api_key=key)

    async def synthesize_wav(self, text: str) -> tuple[bytes, float]:
        t0 = time.perf_counter()
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice
                        )
                    )
                ),
            ),
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        pcm = _extract_audio(response)
        if not pcm:
            raise RuntimeError("Google TTS returned no audio")
        wav = PCMConverter.pcm_to_wav(pcm, sample_rate=self.native_sample_rate)
        return wav, latency_ms


def _extract_audio(response: object) -> bytes:
    # google-genai: response.candidates[0].content.parts[].inline_data.data
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline else None
            if not data:
                continue
            if isinstance(data, str):
                return base64.b64decode(data)
            return bytes(data)
    return b""
