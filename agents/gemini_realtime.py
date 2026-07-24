from __future__ import annotations

import time
from typing import Any, AsyncIterator, Optional, Tuple

from google import genai
from google.genai import types

from utils.config import settings
from utils.logging import get_logger

from .base import AbstractAgentAdapter

logger = get_logger(__name__)


class GeminiRealtimeAdapter(AbstractAgentAdapter):
    """
    Gemini Live API adapter (google-genai).

    Audio contract:
      - input:  PCM16 mono @ 16 kHz
      - output: PCM16 mono @ 24 kHz
    """

    name = "gemini_realtime"
    input_sample_rate = 16000
    output_sample_rate = 24000

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        chunk_ms: int = 40,
    ) -> None:
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_realtime_model
        self.voice = voice or settings.gemini_realtime_voice
        self.chunk_ms = chunk_ms
        self._client: Optional[genai.Client] = None
        self._session: Any = None
        self._session_cm: Any = None
        self._closed = False

    async def start(self, realtime_prompt: str) -> float:
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env "
                "(Google AI Studio / Gemini API key)."
            )

        self._closed = False
        self._client = genai.Client(api_key=self.api_key)

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=realtime_prompt,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice
                    )
                )
            ),
            # Manual turn boundaries so FTL starts only after we finish sending TTS
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True
                )
            ),
        )

        self._session_cm = self._client.aio.live.connect(
            model=self.model, config=config
        )
        self._session = await self._session_cm.__aenter__()
        startup_ts = time.perf_counter()
        logger.info("Gemini Live session ready (model=%s)", self.model)
        return startup_ts

    async def trigger_response(self, hint: str = "Greet the user briefly.") -> float:
        if not self._session:
            raise RuntimeError("Adapter not started")
        await self._session.send_realtime_input(text=hint)
        return time.perf_counter()

    async def send_audio(self, pcm_bytes: bytes) -> float:
        if not self._session:
            raise RuntimeError("Adapter not started")

        # Resample if caller sent non-16kHz PCM (engine should match input_sample_rate)
        chunk_bytes = max(int(self.input_sample_rate * 2 * self.chunk_ms / 1000), 2)
        if chunk_bytes % 2:
            chunk_bytes += 1

        await self._session.send_realtime_input(activity_start=types.ActivityStart())

        for i in range(0, len(pcm_bytes), chunk_bytes):
            chunk = pcm_bytes[i : i + chunk_bytes]
            if not chunk:
                continue
            await self._session.send_realtime_input(
                audio=types.Blob(
                    data=chunk,
                    mime_type=f"audio/pcm;rate={self.input_sample_rate}",
                )
            )

        await self._session.send_realtime_input(activity_end=types.ActivityEnd())
        sent_ts = time.perf_counter()
        logger.debug("Sent %d PCM bytes to Gemini Live @ %d Hz", len(pcm_bytes), self.input_sample_rate)
        return sent_ts

    async def receive_audio_stream(
        self,
    ) -> AsyncIterator[Tuple[bytes, Optional[float]]]:
        if not self._session:
            raise RuntimeError("Adapter not started")

        first_ts: Optional[float] = None
        deadline = time.perf_counter() + 90.0

        async for msg in self._session.receive():
            if time.perf_counter() > deadline:
                raise TimeoutError("Timed out waiting for Gemini audio")

            sc = getattr(msg, "server_content", None)
            if sc is None:
                if getattr(msg, "error", None):
                    raise RuntimeError(f"Gemini Live error: {msg.error}")
                continue

            model_turn = getattr(sc, "model_turn", None)
            if model_turn and getattr(model_turn, "parts", None):
                for part in model_turn.parts:
                    inline = getattr(part, "inline_data", None)
                    data = getattr(inline, "data", None) if inline else None
                    if not data:
                        continue
                    if isinstance(data, str):
                        import base64

                        data = base64.b64decode(data)
                    ts: Optional[float] = None
                    if first_ts is None:
                        first_ts = time.perf_counter()
                        ts = first_ts
                    yield bytes(data), ts

            if getattr(sc, "turn_complete", False):
                break

    async def close(self) -> None:
        self._closed = True
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception as exc:
                logger.debug("Gemini session close: %s", exc)
            self._session_cm = None
            self._session = None
        self._client = None
