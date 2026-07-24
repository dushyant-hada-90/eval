from __future__ import annotations

from typing import Any, Optional

import httpx

from utils.config import settings
from utils.logging import get_logger

logger = get_logger(__name__)

_status: dict[str, Any] = {
    "ok": False,
    "checked": False,
    "provider": "groq",
    "message": "TTS health not checked yet.",
    "url": "",
}


def get_tts_status() -> dict[str, Any]:
    return dict(_status)


def preferred_live_tts(requested: str | None = None) -> str:
    """
    Resolve live-checkpoint TTS provider.
    Gradio only when health check passed; otherwise Groq (or an explicit non-gradio request).
    """
    req = (requested or "").strip().lower()
    if req and req not in {"gradio", "auto", ""}:
        return req
    if _status.get("ok") and settings.gradio_tts_url:
        return "gradio"
    return "groq"


async def check_gradio_tts_on_startup() -> dict[str, Any]:
    """Probe GRADIO_TTS_URL; fall back to Groq when missing or unhealthy."""
    global _status
    url = (settings.gradio_tts_url or "").rstrip("/")
    if not url:
        _status = {
            "ok": False,
            "checked": True,
            "provider": "groq",
            "message": "GRADIO_TTS_URL is not set - using Groq TTS.",
            "url": "",
        }
        logger.warning(_status["message"])
        return get_tts_status()

    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(f"{url}/config")
            if resp.status_code < 400:
                _status = {
                    "ok": True,
                    "checked": True,
                    "provider": "gradio",
                    "message": f"Gradio TTS reachable - using clone endpoint ({url}).",
                    "url": url,
                }
                logger.info(_status["message"])
                return get_tts_status()
            detail = f"HTTP {resp.status_code}"
    except Exception as exc:
        detail = str(exc)

    _status = {
        "ok": False,
        "checked": True,
        "provider": "groq",
        "message": (
            f"Gradio TTS unavailable ({detail}) - falling back to Groq TTS. "
            f"Checked URL: {url}"
        ),
        "url": url,
    }
    logger.warning(_status["message"])
    return get_tts_status()
