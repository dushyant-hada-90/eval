from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY,
    scenario_id TEXT,
    eval_type TEXT DEFAULT 'realtime',
    agent_name TEXT,
    provider_name TEXT,
    model_name TEXT,
    variation_name TEXT,
    run_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    total_turns INTEGER,
    avg_ttf_ms REAL,
    avg_ftl_ms REAL,
    avg_latency_ms REAL,
    avg_wer REAL
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    evaluation_id INTEGER,
    turn_number INTEGER,
    user_intent TEXT,
    ttf_ms REAL,
    ftl_ms REAL,
    latency_ms REAL,
    wer REAL,
    cer REAL,
    reference_text TEXT,
    agent_transcript TEXT,
    agent_audio_path TEXT,
    test_prompt TEXT,
    test_audio_path TEXT,
    intent_alignment_score INTEGER,
    question_asking_score INTEGER,
    tone_score INTEGER,
    context_retention_score INTEGER,
    avg_score REAL,
    scoring_reasoning TEXT,
    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
);

CREATE TABLE IF NOT EXISTS audio_recordings (
    id INTEGER PRIMARY KEY,
    turn_id INTEGER,
    audio_type TEXT,
    wav_path TEXT,
    pcm_path TEXT,
    duration_ms REAL,
    FOREIGN KEY (turn_id) REFERENCES turns(id)
);

CREATE TABLE IF NOT EXISTS session_events (
    id INTEGER PRIMARY KEY,
    evaluation_id INTEGER NOT NULL,
    turn_number INTEGER,
    event_type TEXT NOT NULL,
    label TEXT NOT NULL,
    group_key TEXT NOT NULL,
    t_ms REAL NOT NULL,
    duration_ms REAL,
    detail TEXT,
    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
);
CREATE INDEX IF NOT EXISTS idx_session_events_eval
    ON session_events(evaluation_id, t_ms, id);
"""

_MIGRATIONS = [
    ("evaluations", "eval_type", "TEXT DEFAULT 'realtime'"),
    ("evaluations", "provider_name", "TEXT"),
    ("evaluations", "model_name", "TEXT"),
    ("evaluations", "avg_latency_ms", "REAL"),
    ("evaluations", "avg_wer", "REAL"),
    ("turns", "latency_ms", "REAL"),
    ("turns", "wer", "REAL"),
    ("turns", "cer", "REAL"),
    ("turns", "reference_text", "TEXT"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, coltype in _MIGRATIONS:
        cols = {
            r[1]
            for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    _migrate(conn)
    conn.commit()
    return conn
