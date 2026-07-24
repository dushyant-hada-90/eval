from __future__ import annotations

from audio.metrics import cer, wer
from agents import list_adapters
from scenarios.loader import load_scenario
from stt import list_stt
from tts import list_tts
from pathlib import Path


def test_registries_include_base_providers():
    stt = set(list_stt())
    tts = set(list_tts())
    rt = set(list_adapters())
    for name in ("groq", "openai", "google", "sarvam"):
        assert name in stt
        assert name in tts
    assert "gradio" in tts
    assert "gemini_realtime" in rt
    assert "gpt_realtime" in rt


def test_gradio_response_base64_parse():
    import base64
    import asyncio
    from audio.pcm_converter import PCMConverter
    from tts.gradio import _audio_bytes_from_gradio_response

    pcm = b"\x00\x01" * 240
    wav = PCMConverter.pcm_to_wav(pcm, sample_rate=24000)
    b64 = base64.b64encode(wav).decode("ascii")
    body = {"data": [{"name": "out.wav", "data": f"data:audio/wav;base64,{b64}"}]}

    class _FakeClient:
        async def get(self, url):  # pragma: no cover
            raise AssertionError(f"should not fetch {url}")

    out = asyncio.run(
        _audio_bytes_from_gradio_response(
            body, "https://example.gradio.live", _FakeClient()
        )
    )
    assert out[:4] == b"RIFF"


def test_wer_perfect_and_error():
    assert wer("hello world", "hello world") == 0.0
    assert wer("hello world", "hello there") == 0.5
    assert cer("abc", "abc") == 0.0


def test_load_stt_and_tts_scenarios():
    root = Path(__file__).resolve().parent.parent / "scenarios"
    stt = load_scenario(root / "stt_basic.yaml")
    assert stt.eval_type == "stt"
    assert len(stt.test_items) >= 2
    v = stt.get_variation()
    assert v.test_script[0].prompt

    tts = load_scenario(root / "tts_basic.yaml")
    assert tts.eval_type == "tts"
    assert tts.tts_provider == "groq"
