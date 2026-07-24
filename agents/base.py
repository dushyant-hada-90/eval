from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Tuple


class AbstractAgentAdapter(ABC):
    """Shared interface for realtime voice agent providers."""

    name: str = "base"
    # Provider-native PCM rates; engine resamples Groq TTS to input_sample_rate
    # and stores/transcribes agent audio at output_sample_rate.
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000

    @abstractmethod
    async def start(self, realtime_prompt: str) -> float:
        """
        Open connection, configure session with realtime_prompt.
        Returns perf_counter timestamp when agent is ready (startup mark).
        """

    @abstractmethod
    async def send_audio(self, pcm_bytes: bytes) -> float:
        """
        Send PCM16 audio to the agent and commit the turn.
        Returns perf_counter timestamp when audio was fully sent (FTL start).
        """

    async def trigger_response(
        self, hint: str = "Greet the user briefly."
    ) -> float:
        """
        Ask the agent to speak without user audio (e.g. opening greeting).
        Returns perf_counter timestamp when the trigger was sent.
        """
        raise NotImplementedError(
            f"{self.name} does not support trigger_response()"
        )

    @abstractmethod
    async def receive_audio_stream(
        self,
    ) -> AsyncIterator[Tuple[bytes, Optional[float]]]:
        """
        Yield (audio_chunk, first_token_timestamp_or_none).
        first_token_timestamp is set only on the first non-empty chunk.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close WebSocket / cleanup."""
