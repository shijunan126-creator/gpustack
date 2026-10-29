"""Distribution summaries in the shape GPUStack reads benchmark reports in.

GPUStack stores what a runner reports as a guidellm ``GenerativeBenchmark``
(``gpustack/worker/schemas/benchmark_runner.py``) and charts every latency from
its ``DistributionSummary`` — mean, median, min, max, standard deviation and the
p25..p99 percentiles. aisbench's own CSV carries neither p25 nor p95, and p95 is
what half of the SLO thresholds are evaluated on, so the runner computes the
summaries itself from the per-request timings aisbench records.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

PERCENTILES = {
    "p001": 0.1,
    "p01": 1.0,
    "p05": 5.0,
    "p10": 10.0,
    "p25": 25.0,
    "p50": 50.0,
    "p75": 75.0,
    "p90": 90.0,
    "p95": 95.0,
    "p99": 99.0,
    "p999": 99.9,
}


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile of already sorted values (numpy's default)."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    weight = rank - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def summarize(values: Iterable[float]) -> Dict[str, object]:
    """One ``DistributionSummary``. An empty input yields zeros with count 0,
    which is how guidellm reports a status class that had no requests."""
    data: List[float] = sorted(float(v) for v in values)
    if not data:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0,
            "std_dev": 0.0,
            "percentiles": {key: 0.0 for key in PERCENTILES},
        }
    mean = sum(data) / len(data)
    variance = sum((v - mean) ** 2 for v in data) / len(data)
    return {
        "mean": mean,
        "median": percentile(data, 50.0),
        "min": data[0],
        "max": data[-1],
        "count": len(data),
        "std_dev": math.sqrt(variance),
        "percentiles": {key: percentile(data, pct) for key, pct in PERCENTILES.items()},
    }


def status_summary(
    successful: Iterable[float],
    errored: Iterable[float] = (),
    incomplete: Iterable[float] = (),
) -> Dict[str, object]:
    """A ``StatusDistributionSummary``: one summary per request status plus the
    union of all of them."""
    ok = list(successful)
    bad = list(errored)
    cut = list(incomplete)
    return {
        "successful": summarize(ok),
        "errored": summarize(bad),
        "incomplete": summarize(cut),
        "total": summarize(ok + bad + cut),
    }


def with_mean(summary: Dict[str, object], mean: Optional[float]) -> Dict[str, object]:
    """Replace the mean of every status of a status summary.

    Rates are not means of per-request values: a throughput is a count over the
    measured window, which is how both guidellm and aisbench define it. The
    per-second samples still give the spread; only the headline number is the
    window rate.
    """
    if mean is None:
        return summary
    for status in ("successful", "total"):
        part = summary.get(status)
        if isinstance(part, dict) and part.get("count"):
            part["mean"] = mean
    return summary


def per_second_series(
    events: Sequence[tuple],
    start: float,
    end: float,
) -> List[float]:
    """Bucket ``(timestamp, weight)`` events into one-second bins over
    ``[start, end]`` and return the per-bin totals.

    The bins give a throughput its spread over the run; the final, partial bin
    is scaled to a full second so a short tail does not read as a stall.
    """
    span = end - start
    if span <= 0:
        return [float(sum(weight for _, weight in events))] if events else []
    bins = max(1, math.ceil(span))
    totals = [0.0] * bins
    for ts, weight in events:
        index = min(bins - 1, max(0, int(ts - start)))
        totals[index] += weight
    last_width = span - (bins - 1)
    if 0 < last_width < 1:
        totals[-1] = totals[-1] / last_width
    return totals


def concurrency_series(
    intervals: Sequence[tuple],
    start: float,
    end: float,
    resolution: float = 0.1,
) -> List[float]:
    """In-flight request counts sampled every ``resolution`` seconds."""
    span = end - start
    if span <= 0 or not intervals:
        return [float(len(intervals))] if intervals else []
    steps = max(1, int(span / resolution))
    counts = [0.0] * (steps + 1)
    for begin, finish in intervals:
        first = max(0, int(math.ceil((begin - start) / resolution)))
        last = min(steps, int((finish - start) / resolution))
        for i in range(first, last + 1):
            counts[i] += 1
    return counts
