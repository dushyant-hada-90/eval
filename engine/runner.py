from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from scenarios.loader import load_scenario

from .evaluation import EvaluationEngine, EvaluationResult
from .stt_eval import ComponentEvalResult, STTEvaluationEngine
from .tts_eval import TTSEvaluationEngine

EvalResult = Union[EvaluationResult, ComponentEvalResult]


async def run_eval(
    scenario_path: str | Path,
    *,
    variation: Optional[str] = None,
    agent: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    voice: Optional[str] = None,
    stt: Optional[str] = None,
    tts: Optional[str] = None,
) -> EvalResult:
    """Dispatch to realtime / STT / TTS engine from scenario eval_type."""
    scenario = load_scenario(scenario_path)
    eval_type = scenario.eval_type

    if eval_type == "stt":
        return await STTEvaluationEngine().run_scenario(
            scenario,
            provider=provider or stt,
            model=model,
            variation_name=variation,
        )
    if eval_type == "tts":
        return await TTSEvaluationEngine().run_scenario(
            scenario,
            provider=provider or tts,
            model=model,
            voice=voice,
            variation_name=variation,
        )

    return await EvaluationEngine().run_scenario_file(
        scenario_path=scenario_path,
        variation_name=variation,
        agent_name=agent,
        stt_provider=stt,
        tts_provider=tts,
    )


def result_to_dict(result: EvalResult) -> dict[str, Any]:
    if isinstance(result, EvaluationResult):
        return {
            "eval_type": "realtime",
            "evaluation_id": result.evaluation_id,
            "scenario_id": result.scenario_id,
            "variation_name": result.variation_name,
            "agent_name": result.agent_name,
            "total_turns": result.total_turns,
            "avg_ttf_ms": result.avg_ttf_ms,
            "avg_ftl_ms": result.avg_ftl_ms,
        }
    return {
        "eval_type": result.eval_type,
        "evaluation_id": result.evaluation_id,
        "scenario_id": result.scenario_id,
        "provider_name": result.provider_name,
        "model_name": result.model_name,
        "total_turns": result.total_turns,
        "avg_latency_ms": result.avg_latency_ms,
        "avg_wer": result.avg_wer,
    }
