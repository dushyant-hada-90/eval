from __future__ import annotations

from typing import Any

from providers.registry import tts_registry

from .base import AbstractTTSAdapter, TTSResult

from . import google as _google  # noqa: F401
from . import gradio as _gradio  # noqa: F401
from . import groq as _groq  # noqa: F401
from . import openai as _openai  # noqa: F401
from . import sarvam as _sarvam  # noqa: F401


def get_tts(name: str, **kwargs: Any) -> AbstractTTSAdapter:
    return tts_registry.get(name, **kwargs)


def list_tts() -> list[str]:
    return tts_registry.list()


__all__ = ["AbstractTTSAdapter", "TTSResult", "get_tts", "list_tts"]
