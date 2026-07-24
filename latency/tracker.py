from __future__ import annotations

import time
from typing import Optional


class LatencyTracker:
    """High-resolution agent-only latency tracker (excludes Groq pipeline)."""

    def __init__(self) -> None:
        self.agent_start_time: Optional[float] = None
        self.tts_sent_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self._first_token_recorded = False

    def reset_turn(self) -> None:
        """Clear per-turn timestamps; keep agent_start_time for TTF."""
        self.tts_sent_time = None
        self.first_token_time = None
        self._first_token_recorded = False

    def record_agent_startup(self) -> float:
        self.agent_start_time = time.perf_counter()
        return self.agent_start_time

    def record_tts_sent(self) -> float:
        self.tts_sent_time = time.perf_counter()
        return self.tts_sent_time

    def record_first_token(self) -> Optional[float]:
        """Record first audio token once per turn. Returns timestamp or None if already set."""
        if self._first_token_recorded:
            return None
        self.first_token_time = time.perf_counter()
        self._first_token_recorded = True
        return self.first_token_time

    @property
    def ttf_ms(self) -> Optional[float]:
        """Time to First Token from agent startup (ms)."""
        if self.first_token_time is not None and self.agent_start_time is not None:
            return (self.first_token_time - self.agent_start_time) * 1000
        return None

    @property
    def ftl_ms(self) -> Optional[float]:
        """First Token Latency after TTS fully sent (ms)."""
        if self.first_token_time is not None and self.tts_sent_time is not None:
            return (self.first_token_time - self.tts_sent_time) * 1000
        return None
