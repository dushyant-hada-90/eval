from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional

import httpx

from utils.logging import get_logger

logger = get_logger(__name__)


class EventBus:
    """Simple in-process pub/sub for live dashboard updates."""

    def __init__(self, forward_url: Optional[str] = None) -> None:
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._lock = asyncio.Lock()
        self.history: list[dict[str, Any]] = []
        self._history_limit = 500
        self.forward_url = forward_url or os.getenv(
            "DASHBOARD_EVENTS_URL", "http://127.0.0.1:8000/api/events"
        )
        self.forward_enabled = os.getenv("DASHBOARD_FORWARD", "1") != "0"

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
        async with self._lock:
            self._subscribers.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    async def emit(
        self, event_type: str, *, forward: bool = True, **payload: Any
    ) -> dict[str, Any]:
        event = {
            "type": event_type,
            "ts": time.time(),
            **payload,
        }
        self.history.append(event)
        if len(self.history) > self._history_limit:
            self.history = self.history[-self._history_limit :]
        async with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

        if forward and self.forward_enabled and self.forward_url:
            # Best-effort forward so a separate dashboard process can watch CLI runs
            await self._forward(event)

        return event

    async def _forward(self, event: dict[str, Any]) -> None:
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                await client.post(self.forward_url, json=event)
        except Exception:
            # Dashboard may not be running; ignore silently
            logger.debug("Dashboard forward skipped (unreachable)")


_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
