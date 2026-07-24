from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from audio.metrics import cer, wer
from audio.pcm_converter import PCMConverter
from db.models import Database
from latency.stats import aggregate_latencies
from scenarios.loader import ScenarioConfig, load_scenario
from stt import get_stt
from tts import get_tts
from utils.config import settings
from utils.logging import get_logger

from .events import EventBus, get_event_bus

logger = get_logger(__name__)


@dataclass
class ComponentEvalResult:
    evaluation_id: int
    eval_type: str
    scenario_id: str
    provider_name: str
    model_name: str
    total_turns: int
    avg_latency_ms: Optional[float]
    avg_wer: Optional[float] = None
    turns: list[dict[str, Any]] = field(default_factory=list)


class STTEvaluationEngine:
    """
    STT bench:
      reference text → reference TTS (not scored) → STT under test → latency + WER
    """

    def __init__(
        self,
        db: Optional[Database] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.db = db or Database(settings.db_path)
        self.bus = event_bus or get_event_bus()
        settings.recordings_dir.mkdir(parents=True, exist_ok=True)

    async def run_scenario_file(
        self,
        scenario_path: str | Path,
        provider: str | None = None,
        model: str | None = None,
        variation_name: str | None = None,
    ) -> ComponentEvalResult:
        scenario = load_scenario(scenario_path)
        return await self.run_scenario(
            scenario,
            provider=provider,
            model=model,
            variation_name=variation_name,
        )

    async def run_scenario(
        self,
        scenario: ScenarioConfig,
        provider: str | None = None,
        model: str | None = None,
        variation_name: str | None = None,
    ) -> ComponentEvalResult:
        stt_name = provider or scenario.stt_provider or settings.default_stt_provider
        stt = get_stt(
            stt_name,
            model=model or scenario.stt_model,
            language=scenario.language,
        )
        ref_tts = get_tts(
            scenario.reference_tts_provider or settings.default_tts_provider,
            model=scenario.reference_tts_model,
            voice=scenario.reference_tts_voice,
            language=scenario.language,
        )
        variation = scenario.get_variation(variation_name)
        items = variation.test_script
        if not items:
            raise ValueError("STT scenario has no test_items / test_script")

        evaluation_id = self.db.create_evaluation(
            scenario_id=scenario.scenario_id,
            agent_name=f"stt:{stt.name}",
            variation_name=variation.name,
            total_turns=len(items),
            eval_type="stt",
            provider_name=stt.name,
            model_name=stt.model,
        )
        await self.bus.emit(
            "run_started",
            evaluation_id=evaluation_id,
            eval_type="stt",
            scenario_id=scenario.scenario_id,
            provider_name=stt.name,
            model_name=stt.model,
            total_turns=len(items),
        )

        latencies: list[Optional[float]] = []
        wers: list[Optional[float]] = []
        turns_out: list[dict[str, Any]] = []

        try:
            for idx, step in enumerate(items, start=1):
                await self.bus.emit(
                    "turn_started",
                    evaluation_id=evaluation_id,
                    turn_number=idx,
                    test_prompt=step.prompt,
                )
                # Reference audio (excluded from STT latency)
                tts_res = await ref_tts.synthesize_pcm(step.prompt, target_sample_rate=16000)
                audio_rel = f"recordings/{evaluation_id}/turn_{idx}_ref.wav"
                audio_abs = settings.project_root / audio_rel
                audio_abs.parent.mkdir(parents=True, exist_ok=True)
                audio_abs.write_bytes(tts_res.wav_bytes)

                stt_res = await stt.transcribe_wav(tts_res.wav_bytes)
                w = wer(step.prompt, stt_res.text)
                c = cer(step.prompt, stt_res.text)
                latencies.append(stt_res.latency_ms)
                wers.append(w)

                await self.bus.emit(
                    "stt_result",
                    evaluation_id=evaluation_id,
                    turn_number=idx,
                    latency_ms=stt_res.latency_ms,
                    wer=w,
                    transcript=stt_res.text,
                )

                row = {
                    "turn_number": idx,
                    "user_intent": step.intent,
                    "latency_ms": stt_res.latency_ms,
                    "wer": w,
                    "cer": c,
                    "reference_text": step.prompt,
                    "agent_transcript": stt_res.text,
                    "test_prompt": step.prompt,
                    "test_audio_path": audio_rel,
                    "agent_audio_path": audio_rel,
                    "scoring_reasoning": f"STT {stt.name}/{stt.model}",
                }
                turn_id = self.db.insert_turn(evaluation_id, row)
                self.db.insert_audio(
                    turn_id,
                    "test_input",
                    audio_rel,
                    None,
                    PCMConverter.duration_ms(tts_res.pcm_bytes, tts_res.sample_rate),
                )
                turns_out.append({**row, "id": turn_id})
                await self.bus.emit(
                    "turn_complete",
                    evaluation_id=evaluation_id,
                    turn_number=idx,
                    latency_ms=stt_res.latency_ms,
                    wer=w,
                )

            lat_agg = aggregate_latencies(latencies)
            wer_agg = aggregate_latencies(wers)
            self.db.update_evaluation_summary(
                evaluation_id,
                total_turns=len(turns_out),
                avg_latency_ms=lat_agg["avg"],
                avg_wer=wer_agg["avg"],
            )
            result = ComponentEvalResult(
                evaluation_id=evaluation_id,
                eval_type="stt",
                scenario_id=scenario.scenario_id,
                provider_name=stt.name,
                model_name=stt.model,
                total_turns=len(turns_out),
                avg_latency_ms=lat_agg["avg"],
                avg_wer=wer_agg["avg"],
                turns=turns_out,
            )
            await self.bus.emit(
                "run_complete",
                evaluation_id=evaluation_id,
                eval_type="stt",
                avg_latency_ms=result.avg_latency_ms,
                avg_wer=result.avg_wer,
            )
            return result
        except Exception as exc:
            logger.exception("STT evaluation failed: %s", exc)
            await self.bus.emit("error", evaluation_id=evaluation_id, message=str(exc))
            raise
