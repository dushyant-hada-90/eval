from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .schema import init_db


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = init_db(db_path)

    def close(self) -> None:
        self.conn.close()

    def create_evaluation(
        self,
        scenario_id: str,
        agent_name: str,
        variation_name: str,
        total_turns: int = 0,
        eval_type: str = "realtime",
        provider_name: str | None = None,
        model_name: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO evaluations (
                scenario_id, eval_type, agent_name, provider_name, model_name,
                variation_name, total_turns
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scenario_id,
                eval_type,
                agent_name,
                provider_name or agent_name,
                model_name,
                variation_name,
                total_turns,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def update_evaluation_summary(
        self,
        evaluation_id: int,
        total_turns: int,
        avg_ttf_ms: Optional[float] = None,
        avg_ftl_ms: Optional[float] = None,
        avg_latency_ms: Optional[float] = None,
        avg_wer: Optional[float] = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE evaluations
            SET total_turns = ?, avg_ttf_ms = ?, avg_ftl_ms = ?,
                avg_latency_ms = ?, avg_wer = ?
            WHERE id = ?
            """,
            (
                total_turns,
                avg_ttf_ms,
                avg_ftl_ms,
                avg_latency_ms,
                avg_wer,
                evaluation_id,
            ),
        )
        self.conn.commit()

    def insert_turn(self, evaluation_id: int, data: dict[str, Any]) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO turns (
                evaluation_id, turn_number, user_intent,
                ttf_ms, ftl_ms, latency_ms, wer, cer, reference_text,
                agent_transcript, agent_audio_path,
                test_prompt, test_audio_path,
                intent_alignment_score, question_asking_score,
                tone_score, context_retention_score,
                avg_score, scoring_reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation_id,
                data.get("turn_number"),
                data.get("user_intent"),
                data.get("ttf_ms"),
                data.get("ftl_ms"),
                data.get("latency_ms"),
                data.get("wer"),
                data.get("cer"),
                data.get("reference_text"),
                data.get("agent_transcript"),
                data.get("agent_audio_path"),
                data.get("test_prompt"),
                data.get("test_audio_path"),
                data.get("intent_alignment_score"),
                data.get("question_asking_score"),
                data.get("tone_score"),
                data.get("context_retention_score"),
                data.get("avg_score"),
                data.get("scoring_reasoning"),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def insert_audio(
        self,
        turn_id: int,
        audio_type: str,
        wav_path: str,
        pcm_path: Optional[str],
        duration_ms: Optional[float],
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO audio_recordings (turn_id, audio_type, wav_path, pcm_path, duration_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            (turn_id, audio_type, wav_path, pcm_path, duration_ms),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_evaluations(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM evaluations
            ORDER BY run_timestamp DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_evaluation(self, evaluation_id: int) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
        ).fetchone()
        return dict(row) if row else None

    def delete_evaluation(self, evaluation_id: int) -> bool:
        """Delete evaluation, its turns, and audio_recordings rows. Returns False if missing."""
        if not self.get_evaluation(evaluation_id):
            return False
        turn_ids = [
            int(r[0])
            for r in self.conn.execute(
                "SELECT id FROM turns WHERE evaluation_id = ?",
                (evaluation_id,),
            ).fetchall()
        ]
        if turn_ids:
            placeholders = ",".join("?" * len(turn_ids))
            self.conn.execute(
                f"DELETE FROM audio_recordings WHERE turn_id IN ({placeholders})",
                turn_ids,
            )
        self.conn.execute(
            "DELETE FROM turns WHERE evaluation_id = ?", (evaluation_id,)
        )
        self.conn.execute(
            "DELETE FROM session_events WHERE evaluation_id = ?", (evaluation_id,)
        )
        self.conn.execute("DELETE FROM evaluations WHERE id = ?", (evaluation_id,))
        self.conn.commit()
        return True

    def insert_session_event(
        self,
        evaluation_id: int,
        *,
        event_type: str,
        label: str,
        group_key: str,
        t_ms: float,
        duration_ms: Optional[float] = None,
        turn_number: Optional[int] = None,
        detail: Optional[str] = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO session_events (
                evaluation_id, turn_number, event_type, label, group_key,
                t_ms, duration_ms, detail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation_id,
                turn_number,
                event_type,
                label,
                group_key,
                t_ms,
                duration_ms,
                detail,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_session_events(self, evaluation_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM session_events
            WHERE evaluation_id = ?
            ORDER BY t_ms ASC, id ASC
            """,
            (evaluation_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_audio_for_evaluation(self, evaluation_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT a.*, t.turn_number
            FROM audio_recordings a
            JOIN turns t ON t.id = a.turn_id
            WHERE t.evaluation_id = ?
            ORDER BY t.turn_number ASC, a.id ASC
            """,
            (evaluation_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_turns(self, evaluation_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM turns
            WHERE evaluation_id = ?
            ORDER BY turn_number ASC
            """,
            (evaluation_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_turn(self, turn_id: int) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM turns WHERE id = ?", (turn_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_audio_for_turn(
        self, turn_id: int, audio_type: str
    ) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            """
            SELECT * FROM audio_recordings
            WHERE turn_id = ? AND audio_type = ?
            ORDER BY id DESC LIMIT 1
            """,
            (turn_id, audio_type),
        ).fetchone()
        return dict(row) if row else None

    def all_turns_flat(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
                e.id AS evaluation_id,
                e.scenario_id,
                e.eval_type,
                e.agent_name,
                e.provider_name,
                e.model_name,
                e.variation_name,
                e.run_timestamp,
                t.id AS turn_id,
                t.turn_number,
                t.user_intent,
                t.ttf_ms,
                t.ftl_ms,
                t.latency_ms,
                t.wer,
                t.cer,
                t.reference_text,
                t.agent_transcript,
                t.test_prompt,
                t.intent_alignment_score,
                t.question_asking_score,
                t.tone_score,
                t.context_retention_score,
                t.avg_score,
                t.scoring_reasoning
            FROM turns t
            JOIN evaluations e ON e.id = t.evaluation_id
            ORDER BY e.id DESC, t.turn_number ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]
