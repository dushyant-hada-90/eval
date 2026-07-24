from __future__ import annotations

from typing import Any

from providers.registry import stt_registry

from .base import AbstractSTTAdapter, STTResult

from . import google as _google  # noqa: F401
from . import groq as _groq  # noqa: F401
from . import openai as _openai  # noqa: F401
from . import sarvam as _sarvam  # noqa: F401


def get_stt(name: str, **kwargs: Any) -> AbstractSTTAdapter:
    return stt_registry.get(name, **kwargs)


def list_stt() -> list[str]:
    return stt_registry.list()


__all__ = ["AbstractSTTAdapter", "STTResult", "get_stt", "list_stt"]
