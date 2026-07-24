from __future__ import annotations

import time

from latency.stats import aggregate_latencies
from latency.tracker import LatencyTracker


def test_ttf_and_ftl_ms():
    t = LatencyTracker()
    t.record_agent_startup()
    time.sleep(0.01)
    t.record_tts_sent()
    time.sleep(0.02)
    t.record_first_token()

    assert t.ttf_ms is not None
    assert t.ftl_ms is not None
    assert t.ttf_ms >= 25  # ~30ms with sleep slack
    assert t.ftl_ms >= 15
    # TTF includes startup→token; FTL is subset after TTS sent
    assert t.ttf_ms > t.ftl_ms


def test_first_token_only_once():
    t = LatencyTracker()
    t.record_agent_startup()
    t.record_tts_sent()
    first = t.record_first_token()
    second = t.record_first_token()
    assert first is not None
    assert second is None
    assert t._first_token_recorded is True


def test_reset_turn_keeps_startup():
    t = LatencyTracker()
    t.record_agent_startup()
    start = t.agent_start_time
    t.record_tts_sent()
    t.record_first_token()
    t.reset_turn()
    assert t.agent_start_time == start
    assert t.tts_sent_time is None
    assert t.first_token_time is None
    assert t.ttf_ms is None
    assert t.ftl_ms is None


def test_missing_timestamps_return_none():
    t = LatencyTracker()
    assert t.ttf_ms is None
    assert t.ftl_ms is None
    t.record_agent_startup()
    assert t.ttf_ms is None
    t.record_first_token()
    assert t.ttf_ms is not None
    assert t.ftl_ms is None


def test_aggregate_latencies():
    stats = aggregate_latencies([10.0, 20.0, 30.0, None])
    assert stats["count"] == 3
    assert stats["avg"] == 20.0
    assert stats["min"] == 10.0
    assert stats["max"] == 30.0
    assert stats["p50"] == 20.0
