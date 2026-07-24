from __future__ import annotations

from typing import Any

from providers.registry import realtime_registry

from .base import AbstractAgentAdapter
from .azure_realtime import AzureRealtimeAdapter
from .claude_realtime import ClaudeRealtimeAdapter
from .gemini_realtime import GeminiRealtimeAdapter
from .gpt_realtime import GPTRealtimeAdapter

realtime_registry.add(
    GPTRealtimeAdapter, "gpt_realtime", "openai_realtime", "gpt"
)
realtime_registry.add(
    GeminiRealtimeAdapter, "gemini_realtime", "gemini_live", "gemini"
)
realtime_registry.add(ClaudeRealtimeAdapter, "claude_realtime", "claude")
realtime_registry.add(AzureRealtimeAdapter, "azure_realtime", "azure")


def get_adapter(name: str, **kwargs: Any) -> AbstractAgentAdapter:
    return realtime_registry.get(name, **kwargs)


def list_adapters() -> list[str]:
    return realtime_registry.list()


__all__ = [
    "AbstractAgentAdapter",
    "GPTRealtimeAdapter",
    "GeminiRealtimeAdapter",
    "get_adapter",
    "list_adapters",
]
