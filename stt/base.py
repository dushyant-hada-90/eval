from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class STTResult:
    text: str
    latency_ms: float
    provider: str
    model: str
    language: Optional[str] = None
    raw: Optional[object] = None


class AbstractSTTAdapter(ABC):
    """Speech-to-text provider contract."""

    name: str = "base"
    default_model: str = ""

    def __init__(self, model: str | None = None, language: str | None = None) -> None:
        self.model = model or self.default_model
        self.language = language

    @abstractmethod
    async def transcribe_wav(self, wav_bytes: bytes) -> STTResult:
        """Transcribe a WAV blob. latency_ms must be provider call time only."""

    async def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int) -> STTResult:
        from audio.pcm_converter import PCMConverter

        wav = PCMConverter.pcm_to_wav(pcm_bytes, sample_rate=sample_rate)
        return await self.transcribe_wav(wav)
