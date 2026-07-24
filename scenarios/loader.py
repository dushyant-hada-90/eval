from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ScriptTurn:
    prompt: str
    intent: str = ""


@dataclass
class Variation:
    name: str
    test_script: list[ScriptTurn] = field(default_factory=list)
    persona: str = ""


@dataclass
class TestItem:
    text: str
    item_id: str = ""
    intent: str = ""


@dataclass
class ScenarioConfig:
    scenario_id: str
    eval_type: str = "realtime"  # realtime | stt | tts
    agent_name: str = "gemini_realtime"
    realtime_prompt: str = ""
    testing_prompt: str = ""
    stt_provider: str = "groq"
    stt_model: Optional[str] = None
    tts_provider: str = "groq"
    tts_model: Optional[str] = None
    tts_voice: Optional[str] = None
    # For STT eval: which TTS builds reference audio from text
    reference_tts_provider: str = "groq"
    reference_tts_model: Optional[str] = None
    reference_tts_voice: Optional[str] = None
    # For TTS eval: optional STT used for round-trip WER
    quality_stt_provider: Optional[str] = "groq"
    language: Optional[str] = None
    test_variations: list[Variation] = field(default_factory=list)
    test_script: list[ScriptTurn] = field(default_factory=list)
    test_items: list[TestItem] = field(default_factory=list)
    source_path: Optional[Path] = None

    def get_variation(self, name: Optional[str] = None) -> Variation:
        if self.test_variations:
            if name is None:
                return self.test_variations[0]
            for v in self.test_variations:
                if v.name == name:
                    return v
            raise ValueError(
                f"Variation '{name}' not found. Available: {[v.name for v in self.test_variations]}"
            )
        # STT/TTS items → synthetic variation
        if self.test_items:
            script = [
                ScriptTurn(prompt=i.text, intent=i.intent or i.item_id)
                for i in self.test_items
            ]
            return Variation(name=name or "default", test_script=script)
        return Variation(name=name or "default", test_script=self.test_script)


def _parse_turns(raw: list[Any]) -> list[ScriptTurn]:
    turns: list[ScriptTurn] = []
    for item in raw or []:
        if isinstance(item, str):
            turns.append(ScriptTurn(prompt=item))
        else:
            turns.append(
                ScriptTurn(
                    prompt=str(item.get("prompt", item.get("text", ""))),
                    intent=str(item.get("intent", "")),
                )
            )
    return turns


def _parse_items(raw: list[Any]) -> list[TestItem]:
    items: list[TestItem] = []
    for idx, item in enumerate(raw or [], start=1):
        if isinstance(item, str):
            items.append(TestItem(text=item, item_id=f"item_{idx}"))
        else:
            text = str(item.get("text", item.get("prompt", "")))
            items.append(
                TestItem(
                    text=text,
                    item_id=str(item.get("id", item.get("item_id", f"item_{idx}"))),
                    intent=str(item.get("intent", "")),
                )
            )
    return items


def load_scenario(path: str | Path) -> ScenarioConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    eval_type = str(data.get("eval_type", "realtime")).lower()
    if eval_type not in {"realtime", "stt", "tts"}:
        raise ValueError(f"Unsupported eval_type: {eval_type}")

    if eval_type == "realtime":
        if "realtime_prompt" not in data or "testing_prompt" not in data:
            raise ValueError(
                f"{p} (realtime) must define 'realtime_prompt' and 'testing_prompt'"
            )

    variations: list[Variation] = []
    for v in data.get("test_variations") or []:
        variations.append(
            Variation(
                name=str(v.get("name", "default")),
                persona=str(v.get("persona", "")),
                test_script=_parse_turns(v.get("test_script") or []),
            )
        )

    return ScenarioConfig(
        scenario_id=str(data.get("scenario_id", p.stem)),
        eval_type=eval_type,
        agent_name=str(data.get("agent_name", "gemini_realtime")),
        realtime_prompt=str(data.get("realtime_prompt", "")).strip(),
        testing_prompt=str(data.get("testing_prompt", "")).strip(),
        stt_provider=str(data.get("stt_provider", "groq")),
        stt_model=data.get("stt_model"),
        tts_provider=str(data.get("tts_provider", "groq")),
        tts_model=data.get("tts_model"),
        tts_voice=data.get("tts_voice"),
        reference_tts_provider=str(data.get("reference_tts_provider", "groq")),
        reference_tts_model=data.get("reference_tts_model"),
        reference_tts_voice=data.get("reference_tts_voice"),
        quality_stt_provider=data.get("quality_stt_provider", "groq"),
        language=data.get("language"),
        test_variations=variations,
        test_script=_parse_turns(data.get("test_script") or []),
        test_items=_parse_items(data.get("test_items") or []),
        source_path=p,
    )
