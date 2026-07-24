from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents import get_adapter
from agents.base import AbstractAgentAdapter
from audio.pcm_converter import PCMConverter
from db.models import Database
from engine.events import EventBus, get_event_bus
from latency.stats import aggregate_latencies
from latency.tracker import LatencyTracker
from scenarios.loader import ScenarioConfig, ScriptTurn, load_scenario
from scoring import GroqScorer
from stt import get_stt
from tts import get_tts
from utils.config import settings
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CheckpointSession:
    session_id: str
    scenario: ScenarioConfig
    agent: AbstractAgentAdapter
    bus: EventBus
    db: Database
    evaluation_id: Optional[int] = None
    tracker: LatencyTracker = field(default_factory=LatencyTracker)
    stt_name: str = "groq"
    tts_name: str = "groq"
    variation_name: str = "default"
    persona: str = ""
    script: list[ScriptTurn] = field(default_factory=list)
    script_index: int = 0
    history: list[dict[str, str]] = field(default_factory=list)
    state: str = "idle"
    turn_number: int = 0
    ttf_ms: Optional[float] = None
    ftl_ms: Optional[float] = None
    agent_transcript: str = ""
    llm_response: dict[str, Any] = field(default_factory=dict)
    next_user_utterance: str = ""
    conversation_should_end: bool = False
    end_after_agent_reply: bool = False
    agent_audio_path: Optional[str] = None
    tts_audio_path: Optional[str] = None
    error: Optional[str] = None
    stopwatch_started_at: Optional[float] = None
    # Pending user side for the *next* agent turn flush
    pending_user_prompt: str = ""
    pending_user_audio: Optional[str] = None
    pending_user_intent: str = ""
    persisted_turn_numbers: set[int] = field(default_factory=set)
    turn_ttfs: list[Optional[float]] = field(default_factory=list)
    turn_ftls: list[Optional[float]] = field(default_factory=list)
    _rec_dir: Path = field(default_factory=Path)

    def public_state(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "evaluation_id": self.evaluation_id,
            "state": self.state,
            "scenario_id": self.scenario.scenario_id,
            "agent_name": self.agent.name,
            "variation_name": self.variation_name,
            "persona": self.persona,
            "turn_number": self.turn_number,
            "ttf_ms": self.ttf_ms,
            "ftl_ms": self.ftl_ms,
            "agent_transcript": self.agent_transcript,
            "llm_response": self.llm_response,
            "next_user_utterance": self.next_user_utterance,
            "conversation_should_end": self.conversation_should_end,
            "agent_audio_path": self.agent_audio_path,
            "tts_audio_path": self.tts_audio_path,
            "tts_name": self.tts_name,
            "agent_audio_url": (
                f"/api/live/audio/{self.session_id}/agent"
                if self.agent_audio_path
                else None
            ),
            "tts_audio_url": (
                f"/api/live/audio/{self.session_id}/tts"
                if self.tts_audio_path
                else None
            ),
            "results_url": (
                f"/results?eval={self.evaluation_id}"
                if self.evaluation_id
                else "/results"
            ),
            "script_remaining": max(0, len(self.script) - self.script_index),
            "stopwatch_started_at": self.stopwatch_started_at,
            "error": self.error,
            "awaiting": _awaiting_label(self.state),
        }


def _awaiting_label(state: str) -> Optional[str]:
    return {
        "awaiting_transcript_approval": "approve_transcript",
        "awaiting_llm_approval": "approve_llm",
        "awaiting_tts_send": "send_tts",
    }.get(state)


class CheckpointSessionManager:
    """Interactive live session with human approvals + SQLite persistence."""

    def __init__(self, db: Optional[Database] = None) -> None:
        self._session: Optional[CheckpointSession] = None
        self.bus = get_event_bus()
        self.scorer = GroqScorer()
        self.db = db or Database(settings.db_path)

    @property
    def session(self) -> Optional[CheckpointSession]:
        return self._session

    async def start(
        self,
        scenario_path: str,
        *,
        variation: str | None = None,
        agent_name: str | None = None,
        stt_provider: str | None = None,
        tts_provider: str | None = None,
    ) -> dict[str, Any]:
        if self._session and self._session.state not in {"idle", "done", "error"}:
            await self.stop()

        scenario = load_scenario(scenario_path)
        if scenario.eval_type != "realtime":
            raise ValueError("Checkpoint live mode requires eval_type: realtime")
        if agent_name:
            scenario.agent_name = agent_name
        variation_obj = scenario.get_variation(variation)
        agent = get_adapter(scenario.agent_name)

        sid = uuid.uuid4().hex[:10]
        rec = settings.recordings_dir / f"live_{sid}"
        rec.mkdir(parents=True, exist_ok=True)

        # Gradio only when startup health check passed; else Groq (see utils.tts_runtime)
        from utils.tts_runtime import preferred_live_tts

        resolved_tts = preferred_live_tts(
            tts_provider or scenario.tts_provider or settings.default_tts_provider
        )

        evaluation_id = self.db.create_evaluation(
            scenario_id=scenario.scenario_id,
            agent_name=agent.name,
            variation_name=variation_obj.name,
            total_turns=0,
            eval_type="realtime_live",
            provider_name=agent.name,
            model_name=getattr(agent, "model", None),
        )

        sess = CheckpointSession(
            session_id=sid,
            scenario=scenario,
            agent=agent,
            bus=self.bus,
            db=self.db,
            evaluation_id=evaluation_id,
            stt_name=stt_provider
            or scenario.stt_provider
            or settings.default_stt_provider,
            tts_name=resolved_tts,
            variation_name=variation_obj.name,
            persona=variation_obj.persona
            or f"{variation_obj.name} prospect on a sales call",
            script=list(variation_obj.test_script),
            state="connecting",
            stopwatch_started_at=time.time(),
            _rec_dir=rec,
        )
        self._session = sess
        self._timeline(
            sess,
            "session_start",
            "Connection initiated",
            "connection",
            detail=f"{agent.name} · {scenario.scenario_id}/{variation_obj.name}",
        )
        await self._emit("checkpoint_started", sess)

        try:
            connect_t0 = time.time()
            startup = await agent.start(scenario.realtime_prompt)
            sess.tracker.agent_start_time = startup
            sess.state = "awaiting_agent"
            connect_ms = (time.time() - connect_t0) * 1000
            t_now = self._session_t_ms(sess)
            self._timeline(
                sess,
                "agent_ready",
                "Realtime session ready",
                "connection",
                t_ms=max(0.0, t_now - connect_ms),
                duration_ms=connect_ms,
                detail="WebSocket / Live session established",
            )
            await self._emit("checkpoint_agent_ready", sess)

            await agent.trigger_response(
                "Greet the user briefly and introduce yourself as the sales agent."
            )
            await self._collect_agent_audio(sess, measure_ttf=True)
            await self._transcribe_agent(sess)
            sess.state = "awaiting_transcript_approval"
            self._timeline(
                sess,
                "await_transcript",
                "Awaiting transcript approval",
                "human",
                detail="Human checkpoint",
            )
            await self._emit("checkpoint_transcript_ready", sess)
            return sess.public_state()
        except Exception as exc:
            logger.exception("Checkpoint start failed")
            sess.state = "error"
            sess.error = str(exc)
            self._timeline(
                sess, "error", "Error", "error", detail=str(exc)[:400]
            )
            await self._emit("checkpoint_error", sess, message=str(exc))
            self._finalize_summary(sess)
            await agent.close()
            raise

    async def approve_transcript(
        self, edited_transcript: str | None = None
    ) -> dict[str, Any]:
        sess = self._require("awaiting_transcript_approval")
        if edited_transcript is not None and edited_transcript.strip():
            sess.agent_transcript = edited_transcript.strip()

        sess.history.append({"role": "agent", "content": sess.agent_transcript})
        sess.state = "running_llm"
        llm_t0 = time.time()
        await self._emit("checkpoint_llm_started", sess)

        # Soft exchange budget from YAML length if present; speech is persona-driven only
        max_exchanges = len(sess.script) if sess.script else 8

        result = await self.scorer.checkpoint_next_turn(
            testing_prompt=sess.scenario.testing_prompt,
            agent_transcript=sess.agent_transcript,
            conversation_history=sess.history,
            persona=sess.persona,
            turn_number=sess.turn_number,
            max_exchanges=max_exchanges,
        )
        sess.llm_response = result
        sess.next_user_utterance = str(result.get("next_user_utterance") or "").strip()
        sess.conversation_should_end = bool(result.get("conversation_should_end"))

        # Persist this agent turn + scores now
        self._persist_current_agent_turn(sess, scores=result)

        sess.state = "awaiting_llm_approval"
        llm_ms = (time.time() - llm_t0) * 1000
        t_now = self._session_t_ms(sess)
        self._timeline(
            sess,
            "llm_done",
            "LLM scored + drafted next line",
            "system",
            t_ms=max(0.0, t_now - llm_ms),
            duration_ms=llm_ms,
            detail=(sess.next_user_utterance or "")[:160],
        )
        self._timeline(
            sess,
            "await_utterance",
            "Awaiting utterance approval",
            "human",
            detail="Human checkpoint",
        )
        await self._emit("checkpoint_llm_ready", sess)
        return sess.public_state()

    async def approve_llm(self, edited_utterance: str | None = None) -> dict[str, Any]:
        sess = self._require("awaiting_llm_approval")
        if edited_utterance is not None and edited_utterance.strip():
            sess.next_user_utterance = edited_utterance.strip()
        if not sess.next_user_utterance:
            raise ValueError("next_user_utterance is empty")

        sess.end_after_agent_reply = bool(sess.conversation_should_end)
        await self._run_tts(sess, sess.next_user_utterance)
        return sess.public_state()

    async def regenerate_tts(self, text: str | None = None) -> dict[str, Any]:
        """Re-run Gradio/provider TTS from edited text while waiting to send."""
        sess = self._require("awaiting_tts_send")
        utterance = (text if text is not None else sess.next_user_utterance) or ""
        utterance = utterance.strip()
        if not utterance:
            raise ValueError("TTS text is empty")
        sess.next_user_utterance = utterance
        await self._run_tts(sess, utterance)
        return sess.public_state()

    async def _run_tts(self, sess: CheckpointSession, text: str) -> None:
        sess.state = "running_tts"
        tts_t0 = time.time()
        await self._emit("checkpoint_tts_started", sess)
        try:
            tts = get_tts(
                sess.tts_name,
                model=sess.scenario.tts_model,
                voice=sess.scenario.tts_voice,
                language=sess.scenario.language,
            )
            tts_res = await tts.synthesize_pcm(
                text,
                target_sample_rate=sess.agent.input_sample_rate,
            )
            rel = (
                f"recordings/live_{sess.session_id}/"
                f"turn_{sess.turn_number}_tts.wav"
            )
            abs_path = settings.project_root / rel
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_bytes(tts_res.wav_bytes)
            sess.tts_audio_path = rel
            sess._tts_pcm = tts_res.pcm_bytes  # type: ignore[attr-defined]
            sess._tts_dur = PCMConverter.duration_ms(  # type: ignore[attr-defined]
                tts_res.pcm_bytes, tts_res.sample_rate
            )
            sess.state = "awaiting_tts_send"
            tts_ms = (time.time() - tts_t0) * 1000
            t_now = self._session_t_ms(sess)
            self._timeline(
                sess,
                "tts_gen",
                f"TTS generation ({sess.tts_name})",
                "system",
                t_ms=max(0.0, t_now - tts_ms),
                duration_ms=tts_ms,
                detail=f"Audio {sess._tts_dur:.0f} ms" if sess._tts_dur else text[:120],
            )
            self._timeline(
                sess,
                "await_tts_send",
                "TTS ready — awaiting send",
                "human",
                detail=text[:120],
            )
            await self._emit("checkpoint_tts_ready", sess)
        except Exception as exc:
            logger.exception("TTS generation failed")
            sess.state = "awaiting_tts_send" if sess.tts_audio_path else "awaiting_llm_approval"
            sess.error = str(exc)
            self._timeline(
                sess, "error", "TTS error", "error", detail=str(exc)[:400]
            )
            await self._emit("checkpoint_error", sess, message=str(exc))
            raise

    async def send_tts(self) -> dict[str, Any]:
        sess = self._require("awaiting_tts_send")
        pcm = getattr(sess, "_tts_pcm", None)
        if not pcm:
            raise RuntimeError("No TTS PCM buffered — regenerate TTS first")

        # This user line becomes the "test input" for the next agent turn
        sess.script_index += 1
        sess.pending_user_prompt = sess.next_user_utterance
        sess.pending_user_audio = sess.tts_audio_path
        sess.pending_user_intent = "caller_improvised"
        sess.history.append({"role": "user", "content": sess.next_user_utterance})

        sess.tracker.reset_turn()
        sess.state = "awaiting_agent"
        user_dur = getattr(sess, "_tts_dur", None)
        self._timeline(
            sess,
            "user_audio_sent",
            "Our audio sent to model",
            "user",
            duration_ms=user_dur,
            detail=(sess.next_user_utterance or "")[:160],
        )
        await self._emit("checkpoint_tts_sent", sess)

        sent_ts = await sess.agent.send_audio(pcm)
        sess.tracker.tts_sent_time = sent_ts
        await self._collect_agent_audio(sess, measure_ttf=False)
        await self._transcribe_agent(sess)

        if sess.end_after_agent_reply:
            # Persist final agent reply without waiting for another LLM score
            self._persist_current_agent_turn(sess, scores=None)
            sess.state = "done"
            self._finalize_summary(sess)
            try:
                await sess.agent.close()
            except Exception:
                pass
            self._timeline(
                sess, "session_complete", "Session complete", "connection"
            )
            await self._emit("checkpoint_complete", sess)
        else:
            sess.state = "awaiting_transcript_approval"
            self._timeline(
                sess,
                "await_transcript",
                "Awaiting transcript approval",
                "human",
            )
            await self._emit("checkpoint_transcript_ready", sess)
        return sess.public_state()

    async def stop(self) -> dict[str, Any]:
        sess = self._session
        if not sess:
            return {"state": "idle"}
        try:
            await sess.agent.close()
        except Exception:
            pass
        # Persist last agent turn if not yet scored/saved
        if (
            sess.turn_number
            and sess.turn_number not in sess.persisted_turn_numbers
            and sess.agent_transcript
        ):
            self._persist_current_agent_turn(sess, scores=sess.llm_response or None)
        sess.state = "done"
        self._finalize_summary(sess)
        self._timeline(sess, "session_stopped", "Session stopped", "connection")
        await self._emit("checkpoint_stopped", sess)
        out = sess.public_state()
        self._session = None
        return out

    def _require(self, expected: str) -> CheckpointSession:
        if not self._session:
            raise RuntimeError("No active checkpoint session")
        if self._session.state != expected:
            raise RuntimeError(
                f"Invalid state '{self._session.state}', expected '{expected}'"
            )
        return self._session

    def _persist_current_agent_turn(
        self, sess: CheckpointSession, scores: Optional[dict[str, Any]]
    ) -> None:
        if not sess.evaluation_id or not sess.turn_number:
            return
        if sess.turn_number in sess.persisted_turn_numbers:
            return

        turn_ttf = sess.ttf_ms if sess.turn_number == 1 else None
        turn_ftl = sess.ftl_ms if sess.turn_number > 1 else sess.ftl_ms
        # Greeting turn has no FTL (no TTS sent before it)
        if sess.turn_number == 1 and not sess.pending_user_audio:
            turn_ftl = None

        scores = scores or {}
        row = {
            "turn_number": sess.turn_number,
            "user_intent": sess.pending_user_intent or "",
            "ttf_ms": turn_ttf,
            "ftl_ms": turn_ftl,
            "agent_transcript": sess.agent_transcript,
            "agent_audio_path": sess.agent_audio_path,
            "test_prompt": sess.pending_user_prompt or "(agent greeting)",
            "test_audio_path": sess.pending_user_audio,
            "intent_alignment_score": scores.get("intent_alignment"),
            "question_asking_score": scores.get("questions_asked"),
            "tone_score": scores.get("tone"),
            "context_retention_score": scores.get("context_retention"),
            "avg_score": scores.get("avg_score"),
            "scoring_reasoning": scores.get("reasoning"),
        }
        turn_id = sess.db.insert_turn(sess.evaluation_id, row)

        if sess.pending_user_audio:
            sess.db.insert_audio(
                turn_id,
                "test_input",
                sess.pending_user_audio,
                None,
                getattr(sess, "_tts_dur", None),
            )
        if sess.agent_audio_path:
            pcm = getattr(sess, "_agent_pcm", b"")
            rate = getattr(sess, "_agent_rate", sess.agent.output_sample_rate)
            sess.db.insert_audio(
                turn_id,
                "agent_output",
                sess.agent_audio_path,
                None,
                PCMConverter.duration_ms(pcm, rate) if pcm else None,
            )

        sess.persisted_turn_numbers.add(sess.turn_number)
        sess.turn_ttfs.append(turn_ttf)
        sess.turn_ftls.append(turn_ftl)
        # Clear pending user after attaching to this turn
        sess.pending_user_prompt = ""
        sess.pending_user_audio = None
        sess.pending_user_intent = ""
        self._finalize_summary(sess)
        logger.info(
            "Persisted live turn eval=%s turn=%s turn_id=%s",
            sess.evaluation_id,
            sess.turn_number,
            turn_id,
        )

    def _finalize_summary(self, sess: CheckpointSession) -> None:
        if not sess.evaluation_id:
            return
        ttf_agg = aggregate_latencies(sess.turn_ttfs)
        ftl_agg = aggregate_latencies(sess.turn_ftls)
        sess.db.update_evaluation_summary(
            sess.evaluation_id,
            total_turns=len(sess.persisted_turn_numbers),
            avg_ttf_ms=ttf_agg["avg"],
            avg_ftl_ms=ftl_agg["avg"],
        )

    async def _collect_agent_audio(
        self, sess: CheckpointSession, *, measure_ttf: bool
    ) -> None:
        chunks: list[bytes] = []
        first_recorded = False
        wait_anchor = time.time()
        async for chunk, first_ts in sess.agent.receive_audio_stream():
            if first_ts is not None and not first_recorded:
                first_recorded = True
                sess.tracker.first_token_time = first_ts
                sess.tracker._first_token_recorded = True
                if measure_ttf:
                    sess.ttf_ms = sess.tracker.ttf_ms
                sess.ftl_ms = sess.tracker.ftl_ms
                wait_ms = (
                    sess.ttf_ms
                    if measure_ttf and sess.ttf_ms is not None
                    else sess.ftl_ms
                )
                if wait_ms is None:
                    wait_ms = (time.time() - wait_anchor) * 1000
                # Place the wait segment ending at "now"
                t_now = self._session_t_ms(sess)
                self._timeline(
                    sess,
                    "ttf_wait" if measure_ttf else "ftl_wait",
                    (
                        "Wait → model first audio (TTF)"
                        if measure_ttf
                        else "Wait → model reply (FTL)"
                    ),
                    "latency",
                    t_ms=max(0.0, t_now - float(wait_ms)),
                    duration_ms=float(wait_ms),
                    detail=f"{'TTF' if measure_ttf else 'FTL'} {float(wait_ms):.1f} ms",
                )
                self._timeline(
                    sess,
                    "agent_first_audio",
                    "Model started talking",
                    "agent",
                    detail=(
                        f"turn≈{sess.turn_number + 1} · "
                        f"{'TTF' if measure_ttf else 'FTL'} {float(wait_ms):.1f} ms"
                    ),
                )
                await self._emit(
                    "checkpoint_first_token",
                    sess,
                    ttf_ms=sess.ttf_ms,
                    ftl_ms=sess.ftl_ms,
                )
            chunks.append(chunk)

        sess.turn_number += 1
        pcm = b"".join(chunks)
        wav = PCMConverter.pcm_to_wav(pcm, sample_rate=sess.agent.output_sample_rate)
        rel = f"recordings/live_{sess.session_id}/turn_{sess.turn_number}_agent.wav"
        abs_path = settings.project_root / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(wav)
        sess.agent_audio_path = rel
        sess._agent_pcm = pcm  # type: ignore[attr-defined]
        sess._agent_rate = sess.agent.output_sample_rate  # type: ignore[attr-defined]
        agent_dur = PCMConverter.duration_ms(pcm, sess.agent.output_sample_rate)
        t_now = self._session_t_ms(sess)
        self._timeline(
            sess,
            "agent_audio",
            f"Agent speaking (turn {sess.turn_number})",
            "agent",
            t_ms=max(0.0, t_now - agent_dur),
            duration_ms=agent_dur,
            detail=f"{agent_dur:.0f} ms audio",
        )
        await self._emit("checkpoint_agent_audio_ready", sess)

    async def _transcribe_agent(self, sess: CheckpointSession) -> None:
        pcm = getattr(sess, "_agent_pcm", b"")
        rate = getattr(sess, "_agent_rate", sess.agent.output_sample_rate)
        if not pcm:
            sess.agent_transcript = ""
            return
        stt_t0 = time.time()
        stt = get_stt(sess.stt_name, model=sess.scenario.stt_model)
        result = await stt.transcribe_pcm(pcm, rate)
        sess.agent_transcript = result.text
        stt_ms = (time.time() - stt_t0) * 1000
        t_now = self._session_t_ms(sess)
        self._timeline(
            sess,
            "stt_done",
            "STT transcript ready",
            "system",
            t_ms=max(0.0, t_now - stt_ms),
            duration_ms=stt_ms,
            detail=(result.text or "")[:160],
        )
        await self._emit("checkpoint_stt_done", sess, transcript=result.text)

    def _session_t_ms(self, sess: CheckpointSession) -> float:
        if sess.stopwatch_started_at:
            return max(0.0, (time.time() - sess.stopwatch_started_at) * 1000)
        return 0.0

    def _timeline(
        self,
        sess: CheckpointSession,
        event_type: str,
        label: str,
        group_key: str,
        *,
        duration_ms: Optional[float] = None,
        detail: str = "",
        t_ms: Optional[float] = None,
    ) -> None:
        if not sess.evaluation_id:
            return
        try:
            sess.db.insert_session_event(
                sess.evaluation_id,
                event_type=event_type,
                label=label,
                group_key=group_key,
                t_ms=self._session_t_ms(sess) if t_ms is None else t_ms,
                duration_ms=duration_ms,
                turn_number=sess.turn_number or None,
                detail=detail or None,
            )
        except Exception:
            logger.exception("Failed to persist timeline event %s", event_type)

    async def _emit(
        self, event_type: str, sess: CheckpointSession, **extra: Any
    ) -> None:
        payload = sess.public_state()
        payload.update(extra)
        await self.bus.emit(event_type, forward=False, **payload)


_manager: Optional[CheckpointSessionManager] = None


def get_checkpoint_manager() -> CheckpointSessionManager:
    global _manager
    if _manager is None:
        _manager = CheckpointSessionManager()
    return _manager
