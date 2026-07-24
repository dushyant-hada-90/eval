from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, AsyncIterator, Optional, Tuple

import websockets
from websockets.asyncio.client import ClientConnection

from utils.config import settings
from utils.logging import get_logger

from .base import AbstractAgentAdapter

logger = get_logger(__name__)

AUDIO_DELTA_TYPES = {
    "response.output_audio.delta",
    "response.audio.delta",  # legacy alias still emitted by some models
}


class GPTRealtimeAdapter(AbstractAgentAdapter):
    """OpenAI Realtime GA WebSocket adapter (no OpenAI-Beta header)."""

    name = "gpt_realtime"
    input_sample_rate = 24000
    output_sample_rate = 24000

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        sample_rate: int | None = None,
        chunk_ms: int = 40,
    ) -> None:
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_realtime_model
        self.voice = voice or settings.openai_realtime_voice
        self.sample_rate = sample_rate or self.input_sample_rate
        self.input_sample_rate = self.sample_rate
        self.output_sample_rate = self.sample_rate
        self.chunk_ms = chunk_ms
        self._ws: Optional[ClientConnection] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._closed = False

    async def start(self, realtime_prompt: str) -> float:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        # GA: do NOT send OpenAI-Beta: realtime=v1 (causes beta_api_shape_disabled)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self._closed = False
        self._ws = await websockets.connect(
            url,
            additional_headers=headers,
            max_size=16 * 1024 * 1024,
        )
        self._recv_task = asyncio.create_task(self._reader_loop())

        await self._wait_for_event_types({"session.created"}, timeout=30.0)
        await self._send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": realtime_prompt,
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.sample_rate,
                            },
                            "turn_detection": None,
                        },
                        "output": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.sample_rate,
                            },
                            "voice": self.voice,
                        },
                    },
                },
            }
        )
        await self._wait_for_event_types({"session.updated"}, timeout=30.0)
        startup_ts = time.perf_counter()
        logger.info("GPT Realtime GA session ready (model=%s)", self.model)
        return startup_ts

    async def trigger_response(self, hint: str = "Greet the user briefly.") -> float:
        if not self._ws:
            raise RuntimeError("Adapter not started")
        self._drain_queue()
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": hint}],
                },
            }
        )
        await self._send({"type": "response.create"})
        return time.perf_counter()

    async def send_audio(self, pcm_bytes: bytes) -> float:
        if not self._ws:
            raise RuntimeError("Adapter not started")

        self._drain_queue()
        chunk_bytes = max(int(self.sample_rate * 2 * self.chunk_ms / 1000), 2)
        if chunk_bytes % 2:
            chunk_bytes += 1

        for i in range(0, len(pcm_bytes), chunk_bytes):
            chunk = pcm_bytes[i : i + chunk_bytes]
            if not chunk:
                continue
            await self._send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )

        await self._send({"type": "input_audio_buffer.commit"})
        await self._send({"type": "response.create"})
        sent_ts = time.perf_counter()
        logger.debug("Sent %d PCM bytes to GPT Realtime", len(pcm_bytes))
        return sent_ts

    async def receive_audio_stream(
        self,
    ) -> AsyncIterator[Tuple[bytes, Optional[float]]]:
        first_ts: Optional[float] = None
        while True:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=60.0)
            except asyncio.TimeoutError as exc:
                raise TimeoutError("Timed out waiting for agent audio") from exc

            etype = event.get("type", "")
            if etype in AUDIO_DELTA_TYPES:
                delta = event.get("delta") or ""
                if not delta:
                    continue
                chunk = base64.b64decode(delta)
                if not chunk:
                    continue
                ts: Optional[float] = None
                if first_ts is None:
                    first_ts = time.perf_counter()
                    ts = first_ts
                yield chunk, ts
            elif etype == "response.done":
                break
            elif etype in {"response.audio.done", "response.output_audio.done"}:
                continue
            elif etype == "error":
                raise RuntimeError(f"Realtime API error: {event.get('error') or event}")
            elif etype == "response.cancelled":
                break

    async def close(self) -> None:
        self._closed = True
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._ws:
            await self._ws.close()
            self._ws = None

    def _drain_queue(self) -> None:
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _send(self, payload: dict[str, Any]) -> None:
        assert self._ws is not None
        await self._ws.send(json.dumps(payload))

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if self._closed:
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON realtime frame")
                    continue
                if event.get("type") == "error":
                    logger.error("Realtime error event: %s", event)
                await self._event_queue.put(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Realtime reader stopped: %s", exc)
            await self._event_queue.put({"type": "error", "error": str(exc)})

    async def _wait_for_event_types(
        self, types: set[str], timeout: float
    ) -> dict[str, Any]:
        deadline = time.perf_counter() + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {types}")
            event = await asyncio.wait_for(self._event_queue.get(), timeout=remaining)
            if event.get("type") in types:
                return event
            if event.get("type") == "error":
                err = event.get("error") or event
                code = err.get("code") if isinstance(err, dict) else None
                msg = err.get("message") if isinstance(err, dict) else str(err)
                if code == "beta_api_shape_disabled":
                    raise RuntimeError(
                        "OpenAI Realtime rejected beta protocol. "
                        "Adapter must use GA (no OpenAI-Beta header). "
                        f"Details: {msg}"
                    )
                if code == "invalid_api_key":
                    raise RuntimeError(
                        "OpenAI rejected OPENAI_API_KEY (invalid_api_key)."
                    )
                raise RuntimeError(f"Realtime API error: {err}")
