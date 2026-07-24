from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from audio.pcm_converter import PCMConverter


@dataclass
class TTSResult:
    pcm_bytes: bytes
    wav_bytes: bytes
    sample_rate: int
    latency_ms: float
    provider: str
    model: str
    voice: Optional[str] = None


class AbstractTTSAdapter(ABC):
    """Text-to-speech provider contract."""

    name: str = "base"
    default_model: str = ""
    default_voice: str = ""
    # Native synthesis rate before optional resample
    native_sample_rate: int = 24000

    def __init__(
        self,
        model: str | None = None,
        voice: str | None = None,
        language: str | None = None,
    ) -> None:
        self.model = model or self.default_model
        self.voice = voice or self.default_voice
        self.language = language

    @abstractmethod
    async def synthesize_wav(self, text: str) -> tuple[bytes, float]:
        """
        Returns (wav_bytes, latency_ms).
        latency_ms = wall time of the provider call only.
        """

    async def synthesize_pcm(
        self, text: str, target_sample_rate: int | None = None
    ) -> TTSResult:
        wav_raw, latency_ms = await self.synthesize_wav(text)
        rate = target_sample_rate or self.native_sample_rate
        pcm, out_rate = PCMConverter.wav_to_pcm(wav_raw, target_rate=rate)
        wav_out = PCMConverter.pcm_to_wav(pcm, sample_rate=out_rate)
        return TTSResult(
            pcm_bytes=pcm,
            wav_bytes=wav_out,
            sample_rate=out_rate,
            latency_ms=latency_ms,
            provider=self.name,
            model=self.model,
            voice=self.voice,
        )
