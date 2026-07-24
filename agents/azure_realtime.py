from __future__ import annotations

from typing import AsyncIterator, Optional, Tuple

from .base import AbstractAgentAdapter


class AzureRealtimeAdapter(AbstractAgentAdapter):
    """Stub for future Azure Realtime support."""

    name = "azure_realtime"
    input_sample_rate = 24000
    output_sample_rate = 24000

    async def start(self, realtime_prompt: str) -> float:
        raise NotImplementedError("AzureRealtimeAdapter is not implemented yet")

    async def send_audio(self, pcm_bytes: bytes) -> float:
        raise NotImplementedError("AzureRealtimeAdapter is not implemented yet")

    async def receive_audio_stream(
        self,
    ) -> AsyncIterator[Tuple[bytes, Optional[float]]]:
        raise NotImplementedError("AzureRealtimeAdapter is not implemented yet")
        yield b"", None  # pragma: no cover

    async def close(self) -> None:
        return None
