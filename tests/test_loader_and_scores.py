from __future__ import annotations

from pathlib import Path

from audio.pcm_converter import PCMConverter
from scoring.groq import GroqScorer
from scenarios.loader import load_scenario


def test_load_sales_pitch_scenario():
    path = Path(__file__).resolve().parent.parent / "scenarios" / "sales_pitch_test.yaml"
    sc = load_scenario(path)
    assert sc.scenario_id == "sales_pitch_v1"
    assert sc.eval_type == "realtime"
    assert "voip" in sc.realtime_prompt.lower() or "sales" in sc.realtime_prompt.lower()
    v = sc.get_variation("neutral")
    assert len(v.test_script) >= 2
    assert v.test_script[0].intent == "express_need"


def test_parse_scores_fenced_json():
    raw = """Here you go:
```json
{"intent_alignment": 4, "questions_asked": 5, "tone": 3, "context_retention": 4, "reasoning": "ok"}
```
"""
    scores = GroqScorer.parse_scores(raw)
    assert scores["intent_alignment"] == 4
    assert scores["questions_asked"] == 5
    assert scores["avg_score"] == 4.0


def test_pcm_roundtrip():
    pcm = b"\x00\x00" * 2400
    wav = PCMConverter.pcm_to_wav(pcm, sample_rate=24000)
    out, rate = PCMConverter.wav_to_pcm(wav, target_rate=24000)
    assert rate == 24000
    assert len(out) == len(pcm)
    assert abs(PCMConverter.duration_ms(pcm, 24000) - 100.0) < 1.0
