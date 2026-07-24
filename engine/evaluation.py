from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents import get_adapter
from agents.base import AbstractAgentAdapter
from audio.pcm_converter import PCMConverter
from db.models import Database
from latency.stats import aggregate_latencies
from latency.tracker import LatencyTracker
from scenarios.loader import ScenarioConfig, Variation, load_scenario
from scoring import GroqScorer
from stt import get_stt
from tts import get_tts
from utils.config import settings
from utils.logging import get_logger

from .events import EventBus, get_event_bus

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    evaluation_id: int
    scenario_id: str
    variation_name: str
    agent_name: str
    total_turns: int
    avg_ttf_ms: Optional[float]
    avg_ftl_ms: Optional[float]
    turns: list[dict[str, Any]] = field(default_factory=list)


class EvaluationEngine:
    """Realtime agent evaluation (agent latency only; STT/TTS are pluggable)."""

    def __init__(
        self,
        db: Optional[Database] = None,
        event_bus: Optional[EventBus] = None,
        stt_provider: str | None = None,
        tts_provider: str | None = None,
    ) -> None:
        self.db = db or Database(settings.db_path)
        self.bus = event_bus or get_event_bus()
        self.stt_provider = stt_provider or settings.default_stt_provider
        self.tts_provider = tts_provider or settings.default_tts_provider
        self.scorer = GroqScorer()
        settings.recordings_dir.mkdir(parents=True, exist_ok=True)

    async def run_scenario_file(
        self,
        scenario_path: str | Path,
        variation_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        stt_provider: Optional[str] = None,
        tts_provider: Optional[str] = None,
    ) -> EvaluationResult:
        scenario = load_scenario(scenario_path)
        if agent_name:
            scenario.agent_name = agent_name
        if stt_provider:
            scenario.stt_provider = stt_provider
        if tts_provider:
            scenario.tts_provider = tts_provider
        variation = scenario.get_variation(variation_name)
        adapter = get_adapter(scenario.agent_name)
        return await self.run_scenario(adapter, scenario, variation)

    async def run_scenario(
        self,
        agent_adapter: AbstractAgentAdapter,
        scenario_config: ScenarioConfig,
        variation: Variation,
    ) -> EvaluationResult:
        stt = get_stt(
            scenario_config.stt_provider or self.stt_provider,
            model=scenario_config.stt_model,
            language=scenario_config.language,
        )
        tts = get_tts(
            scenario_config.tts_provider or self.tts_provider,
            model=scenario_config.tts_model,
            voice=scenario_config.tts_voice,
            language=scenario_config.language,
        )

        tracker = LatencyTracker()
        turns_out: list[dict[str, Any]] = []
        history: list[dict[str, str]] = []
        ttf_values: list[Optional[float]] = []
        ftl_values: list[Optional[float]] = []

        evaluation_id = self.db.create_evaluation(
            scenario_id=scenario_config.scenario_id,
            agent_name=agent_adapter.name,
            variation_name=variation.name,
            total_turns=len(variation.test_script),
            eval_type="realtime",
            provider_name=agent_adapter.name,
            model_name=getattr(agent_adapter, "model", None),
        )

        await self.bus.emit(
            "run_started",
            evaluation_id=evaluation_id,
            eval_type="realtime",
            scenario_id=scenario_config.scenario_id,
            variation_name=variation.name,
            agent_name=agent_adapter.name,
            total_turns=len(variation.test_script),
        )

        try:
            startup_ts = await agent_adapter.start(scenario_config.realtime_prompt)
            tracker.agent_start_time = startup_ts
            await self.bus.emit(
                "agent_ready",
                evaluation_id=evaluation_id,
                agent_name=agent_adapter.name,
            )

            for idx, step in enumerate(variation.test_script, start=1):
                tracker.reset_turn()
                await self.bus.emit(
                    "turn_started",
                    evaluation_id=evaluation_id,
                    turn_number=idx,
                    intent=step.intent,
                    test_prompt=step.prompt,
                )

                in_rate = agent_adapter.input_sample_rate
                out_rate = agent_adapter.output_sample_rate

                tts_res = await tts.synthesize_pcm(
                    step.prompt, target_sample_rate=in_rate
                )
                pcm, test_wav, rate = (
                    tts_res.pcm_bytes,
                    tts_res.wav_bytes,
                    tts_res.sample_rate,
                )
                test_rel = f"recordings/{evaluation_id}/turn_{idx}_test.wav"
                test_abs = settings.project_root / test_rel
                test_abs.parent.mkdir(parents=True, exist_ok=True)
                test_abs.write_bytes(test_wav)
                test_dur = PCMConverter.duration_ms(pcm, rate)

                sent_ts = await agent_adapter.send_audio(pcm)
                tracker.tts_sent_time = sent_ts
                await self.bus.emit(
                    "tts_sent",
                    evaluation_id=evaluation_id,
                    turn_number=idx,
                )

                agent_chunks: list[bytes] = []
                async for chunk, first_ts in agent_adapter.receive_audio_stream():
                    if first_ts is not None and not tracker._first_token_recorded:
                        tracker.first_token_time = first_ts
                        tracker._first_token_recorded = True
                        emit_ttf = tracker.ttf_ms if idx == 1 else None
                        await self.bus.emit(
                            "first_token",
                            evaluation_id=evaluation_id,
                            turn_number=idx,
                            ttf_ms=emit_ttf,
                            ftl_ms=tracker.ftl_ms,
                        )
                    agent_chunks.append(chunk)

                agent_pcm = b"".join(agent_chunks)
                agent_wav = PCMConverter.pcm_to_wav(agent_pcm, sample_rate=out_rate)
                agent_rel = f"recordings/{evaluation_id}/turn_{idx}_agent.wav"
                agent_abs = settings.project_root / agent_rel
                agent_abs.write_bytes(agent_wav)
                agent_dur = PCMConverter.duration_ms(agent_pcm, out_rate)

                await self.bus.emit(
                    "agent_audio_ready",
                    evaluation_id=evaluation_id,
                    turn_number=idx,
                    duration_ms=agent_dur,
                    agent_audio_path=agent_rel,
                    test_audio_path=test_rel,
                )

                agent_transcript = ""
                if agent_pcm:
                    stt_res = await stt.transcribe_pcm(agent_pcm, out_rate)
                    agent_transcript = stt_res.text

                history.append({"role": "user", "content": step.prompt})
                history.append({"role": "agent", "content": agent_transcript})

                scores = await self.scorer.score_turn(
                    testing_prompt=scenario_config.testing_prompt,
                    test_prompt=step.prompt,
                    user_intent=step.intent,
                    agent_transcript=agent_transcript,
                    conversation_history=history,
                )
                await self.bus.emit(
                    "scored",
                    evaluation_id=evaluation_id,
                    turn_number=idx,
                    scores=scores,
                    agent_transcript=agent_transcript,
                )

                turn_ttf = tracker.ttf_ms if idx == 1 else None
                turn_row = {
                    "turn_number": idx,
                    "user_intent": step.intent,
                    "ttf_ms": turn_ttf,
                    "ftl_ms": tracker.ftl_ms,
                    "agent_transcript": agent_transcript,
                    "agent_audio_path": agent_rel,
                    "test_prompt": step.prompt,
                    "test_audio_path": test_rel,
                    "intent_alignment_score": scores.get("intent_alignment"),
                    "question_asking_score": scores.get("questions_asked"),
                    "tone_score": scores.get("tone"),
                    "context_retention_score": scores.get("context_retention"),
                    "avg_score": scores.get("avg_score"),
                    "scoring_reasoning": scores.get("reasoning"),
                }
                turn_id = self.db.insert_turn(evaluation_id, turn_row)
                self.db.insert_audio(
                    turn_id, "test_input", test_rel, None, test_dur
                )
                self.db.insert_audio(
                    turn_id, "agent_output", agent_rel, None, agent_dur
                )

                ttf_values.append(turn_ttf)
                ftl_values.append(tracker.ftl_ms)
                turns_out.append({**turn_row, "id": turn_id})

                await self.bus.emit(
                    "turn_complete",
                    evaluation_id=evaluation_id,
                    turn_number=idx,
                    ttf_ms=turn_ttf,
                    ftl_ms=tracker.ftl_ms,
                    avg_score=scores.get("avg_score"),
                )

            ttf_agg = aggregate_latencies(ttf_values)
            ftl_agg = aggregate_latencies(ftl_values)
            self.db.update_evaluation_summary(
                evaluation_id,
                total_turns=len(turns_out),
                avg_ttf_ms=ttf_agg["avg"],
                avg_ftl_ms=ftl_agg["avg"],
            )

            result = EvaluationResult(
                evaluation_id=evaluation_id,
                scenario_id=scenario_config.scenario_id,
                variation_name=variation.name,
                agent_name=agent_adapter.name,
                total_turns=len(turns_out),
                avg_ttf_ms=ttf_agg["avg"],
                avg_ftl_ms=ftl_agg["avg"],
                turns=turns_out,
            )
            await self.bus.emit(
                "run_complete",
                evaluation_id=evaluation_id,
                avg_ttf_ms=result.avg_ttf_ms,
                avg_ftl_ms=result.avg_ftl_ms,
                total_turns=result.total_turns,
            )
            return result
        except Exception as exc:
            logger.exception("Evaluation failed: %s", exc)
            await self.bus.emit(
                "error",
                evaluation_id=evaluation_id,
                message=str(exc),
            )
            raise
        finally:
            await agent_adapter.close()
