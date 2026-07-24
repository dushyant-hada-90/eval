from __future__ import annotations

import base64
import re
import time
from typing import Any
from urllib.parse import quote, urljoin

import httpx

from providers.registry import tts_registry
from utils.config import settings

from .base import AbstractTTSAdapter

# Gradio 3 → /run/predict; Gradio 4–6 → /gradio_api/run/predict
_DEFAULT_PREDICT_CANDIDATES = (
    "/gradio_api/run/predict",
    "/run/predict",
    "/api/predict",
)


@tts_registry.register("gradio")
class GradioTTSAdapter(AbstractTTSAdapter):
    """TTS via a Gradio share URL (POST …/run/predict → audio file)."""

    name = "gradio"
    default_model = "gradio-clone"
    default_voice = "cloned"
    native_sample_rate = 24000

    def __init__(
        self,
        model: str | None = None,
        voice: str | None = None,
        language: str | None = None,
        base_url: str | None = None,
        fn_index: int | None = None,
    ) -> None:
        super().__init__(
            model=model or self.default_model,
            voice=voice or self.default_voice,
            language=language,
        )
        raw = (base_url or settings.gradio_tts_url or "").rstrip("/")
        if not raw:
            raise RuntimeError(
                "GRADIO_TTS_URL is not set. Add it to .env "
                "(e.g. https://xxxxxx.gradio.live)."
            )
        self.base_url = raw
        self.fn_index = (
            fn_index if fn_index is not None else settings.gradio_tts_fn_index
        )
        # Empty / "auto" → discover from /config; else use explicit path
        configured = (settings.gradio_tts_path or "").strip()
        self._explicit_path = (
            None
            if not configured or configured.lower() in {"auto", "discover"}
            else (configured if configured.startswith("/") else f"/{configured}")
        )
        self._resolved_predict_url: str | None = None

    async def synthesize_wav(self, text: str) -> tuple[bytes, float]:
        if not text or not text.strip():
            raise ValueError("TTS text is empty")

        payload: dict[str, Any] = {
            "fn_index": self.fn_index,
            "data": [text.strip()],
        }
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
            body, predict_url = await self._post_predict(client, payload)
            audio = await _audio_bytes_from_gradio_response(
                body, self.base_url, client
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        if not audio:
            raise RuntimeError(
                "Gradio TTS returned no audio. "
                f"predict={predict_url} keys={list(body) if isinstance(body, dict) else type(body)}"
            )
        if audio[:4] != b"RIFF":
            if audio[:3] == b"ID3" or audio[:2] == b"\xff\xfb":
                raise RuntimeError(
                    "Gradio returned MP3; configure the Gradio app to return WAV."
                )
            from audio.pcm_converter import PCMConverter

            audio = PCMConverter.pcm_to_wav(audio, sample_rate=self.native_sample_rate)
        return audio, latency_ms

    async def _post_predict(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], str]:
        candidates = await self._predict_candidates(client)
        errors: list[str] = []
        for path in candidates:
            url = f"{self.base_url}{path}"
            resp = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code < 400:
                self._resolved_predict_url = url
                try:
                    body = resp.json()
                except Exception as exc:
                    raise RuntimeError(
                        f"Gradio TTS non-JSON from {url}: {resp.text[:200]}"
                    ) from exc
                if not isinstance(body, dict):
                    raise RuntimeError(f"Unexpected Gradio body from {url}")
                return body, url
            errors.append(f"{path}→{resp.status_code}:{resp.text[:120]}")

        raise RuntimeError(
            "Gradio TTS HTTP failed on all predict paths. "
            + " | ".join(errors)
        )

    async def _predict_candidates(self, client: httpx.AsyncClient) -> list[str]:
        if self._explicit_path:
            return [self._explicit_path]
        if self._resolved_predict_url:
            # Reuse path that worked earlier in this process
            prefix = self.base_url
            return [self._resolved_predict_url[len(prefix) :]]

        paths: list[str] = []
        try:
            cfg = await client.get(f"{self.base_url}/config")
            if cfg.status_code < 400:
                data = cfg.json()
                api_prefix = str(data.get("api_prefix") or "/gradio_api").rstrip("/")
                if not api_prefix.startswith("/"):
                    api_prefix = "/" + api_prefix
                paths.append(f"{api_prefix}/run/predict")
        except Exception:
            pass

        for p in _DEFAULT_PREDICT_CANDIDATES:
            if p not in paths:
                paths.append(p)
        return paths


async def _audio_bytes_from_gradio_response(
    body: Any, base_url: str, client: httpx.AsyncClient
) -> bytes:
    if not isinstance(body, dict):
        raise RuntimeError(f"Unexpected Gradio response type: {type(body)}")

    data = body.get("data")
    if data is None:
        raise RuntimeError(f"Gradio response missing 'data': {str(body)[:300]}")
    if isinstance(data, list):
        if not data:
            raise RuntimeError("Gradio 'data' list is empty")
        item = data[0]
    else:
        item = data

    if isinstance(item, str):
        decoded = _maybe_b64_audio(item)
        if decoded:
            return decoded
        return await _fetch_gradio_file(base_url, item, client)

    if isinstance(item, dict):
        for key in ("data", "value", "url"):
            val = item.get(key)
            if isinstance(val, str) and val.startswith("data:"):
                decoded = _maybe_b64_audio(val)
                if decoded:
                    return decoded
        # Prefer full URL when Gradio returns one
        for key in ("url", "path", "name"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return await _fetch_gradio_file(base_url, val.strip(), client)

    raise RuntimeError(f"Could not parse Gradio audio payload: {str(item)[:300]}")


def _maybe_b64_audio(value: str) -> bytes | None:
    if value.startswith("data:") and ";base64," in value:
        b64 = value.split(";base64,", 1)[1]
        return base64.b64decode(b64)
    if re.fullmatch(r"[A-Za-z0-9+/=\s]+", value) and len(value) > 200:
        try:
            raw = base64.b64decode(value, validate=False)
            if raw[:4] == b"RIFF" or len(raw) > 1000:
                return raw
        except Exception:
            return None
    return None


async def _fetch_gradio_file(
    base_url: str, path_or_url: str, client: httpx.AsyncClient
) -> bytes:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        url = path_or_url
    elif path_or_url.startswith("/file=") or path_or_url.startswith("/gradio_api/file="):
        url = urljoin(base_url + "/", path_or_url.lstrip("/"))
    elif path_or_url.startswith("file="):
        url = f"{base_url}/gradio_api/{path_or_url}"
    elif path_or_url.startswith("/"):
        if "/file=" in path_or_url or path_or_url.startswith("/gradio"):
            url = urljoin(base_url + "/", path_or_url.lstrip("/"))
        else:
            url = f"{base_url}/gradio_api/file={quote(path_or_url, safe='/:')}"
    else:
        url = f"{base_url}/gradio_api/file={quote(path_or_url, safe='/:')}"

    resp = await client.get(url)
    if resp.status_code >= 400:
        alts = [
            f"{base_url}/file={path_or_url}",
            f"{base_url}/gradio_api/file={path_or_url}",
        ]
        for alt in alts:
            resp = await client.get(alt)
            if resp.status_code < 400:
                break
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Failed to download Gradio audio ({resp.status_code}) from {url}"
        )
    return resp.content
