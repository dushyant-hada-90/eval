from __future__ import annotations

from typing import Iterable, Optional


def _percentile(sorted_vals: list[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def aggregate_latencies(values: Iterable[Optional[float]]) -> dict[str, Optional[float]]:
    nums = sorted(v for v in values if v is not None)
    if not nums:
        return {"avg": None, "p50": None, "p95": None, "min": None, "max": None, "count": 0}
    return {
        "avg": sum(nums) / len(nums),
        "p50": _percentile(nums, 50),
        "p95": _percentile(nums, 95),
        "min": nums[0],
        "max": nums[-1],
        "count": len(nums),
    }
