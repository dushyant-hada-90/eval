from __future__ import annotations

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
from .stt_eval import ComponentEvalResult

logger = get_logger(__name__)


class TTSEvaluationEngine:
    """
    TTS bench:
      text → TTS under test (latency) → optional quality STT → WER vs original text
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
        voice: str | None = None,
        variation_name: str | None = None,
    ) -> ComponentEvalResult:
        scenario = load_scenario(scenario_path)
        return await self.run_scenario(
            scenario,
            provider=provider,
            model=model,
            voice=voice,
            variation_name=variation_name,
        )

    async def run_scenario(
        self,
        scenario: ScenarioConfig,
        provider: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        variation_name: str | None = None,
    ) -> ComponentEvalResult:
        tts_name = provider or scenario.tts_provider or settings.default_tts_provider
        tts = get_tts(
            tts_name,
            model=model or scenario.tts_model,
            voice=voice or scenario.tts_voice,
            language=scenario.language,
        )
        quality_stt = None
        if scenario.quality_stt_provider:
            quality_stt = get_stt(
                scenario.quality_stt_provider,
                language=scenario.language,
            )

        variation = scenario.get_variation(variation_name)
        items = variation.test_script
        if not items:
            raise ValueError("TTS scenario has no test_items / test_script")

        evaluation_id = self.db.create_evaluation(
            scenario_id=scenario.scenario_id,
            agent_name=f"tts:{tts.name}",
            variation_name=variation.name,
            total_turns=len(items),
            eval_type="tts",
            provider_name=tts.name,
            model_name=tts.model,
        )
        await self.bus.emit(
            "run_started",
            evaluation_id=evaluation_id,
            eval_type="tts",
            scenario_id=scenario.scenario_id,
            provider_name=tts.name,
            model_name=tts.model,
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
                tts_res = await tts.synthesize_pcm(step.prompt, target_sample_rate=24000)
                audio_rel = f"recordings/{evaluation_id}/turn_{idx}_tts.wav"
                audio_abs = settings.project_root / audio_rel
                audio_abs.parent.mkdir(parents=True, exist_ok=True)
                audio_abs.write_bytes(tts_res.wav_bytes)
                latencies.append(tts_res.latency_ms)

                transcript = ""
                w = None
                c = None
                if quality_stt is not None:
                    stt_res = await quality_stt.transcribe_wav(tts_res.wav_bytes)
                    transcript = stt_res.text
                    w = wer(step.prompt, transcript)
                    c = cer(step.prompt, transcript)
                    wers.append(w)

                await self.bus.emit(
                    "tts_result",
                    evaluation_id=evaluation_id,
                    turn_number=idx,
                    latency_ms=tts_res.latency_ms,
                    wer=w,
                    audio_path=audio_rel,
                )

                row = {
                    "turn_number": idx,
                    "user_intent": step.intent,
                    "latency_ms": tts_res.latency_ms,
                    "wer": w,
                    "cer": c,
                    "reference_text": step.prompt,
                    "agent_transcript": transcript,
                    "test_prompt": step.prompt,
                    "test_audio_path": audio_rel,
                    "agent_audio_path": audio_rel,
                    "scoring_reasoning": f"TTS {tts.name}/{tts.model} voice={tts.voice}",
                }
                turn_id = self.db.insert_turn(evaluation_id, row)
                self.db.insert_audio(
                    turn_id,
                    "agent_output",
                    audio_rel,
                    None,
                    PCMConverter.duration_ms(tts_res.pcm_bytes, tts_res.sample_rate),
                )
                turns_out.append({**row, "id": turn_id})
                await self.bus.emit(
                    "turn_complete",
                    evaluation_id=evaluation_id,
                    turn_number=idx,
                    latency_ms=tts_res.latency_ms,
                    wer=w,
                )

            lat_agg = aggregate_latencies(latencies)
            wer_agg = aggregate_latencies(wers) if wers else {
                "avg": None,
                "count": 0,
            }
            self.db.update_evaluation_summary(
                evaluation_id,
                total_turns=len(turns_out),
                avg_latency_ms=lat_agg["avg"],
                avg_wer=wer_agg["avg"],
            )
            result = ComponentEvalResult(
                evaluation_id=evaluation_id,
                eval_type="tts",
                scenario_id=scenario.scenario_id,
                provider_name=tts.name,
                model_name=tts.model,
                total_turns=len(turns_out),
                avg_latency_ms=lat_agg["avg"],
                avg_wer=wer_agg["avg"],
                turns=turns_out,
            )
            await self.bus.emit(
                "run_complete",
                evaluation_id=evaluation_id,
                eval_type="tts",
                avg_latency_ms=result.avg_latency_ms,
                avg_wer=result.avg_wer,
            )
            return result
        except Exception as exc:
            logger.exception("TTS evaluation failed: %s", exc)
            await self.bus.emit("error", evaluation_id=evaluation_id, message=str(exc))
            raise
