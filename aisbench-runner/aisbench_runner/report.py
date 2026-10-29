"""Turn aisbench's per-request records into the report GPUStack reads.

GPUStack's worker loads every result file with
``GenerativeBenchmarksReport.load_file`` (``gpustack/worker/schemas/
benchmark_runner.py``), a trimmed copy of guidellm's report schema, and the UI
charts straight out of ``raw_metrics.benchmarks[0].metrics``. Writing that
shape is what lets this runner replace benchmark-runner without the server,
the analysis or the UI knowing which engine measured the point.

Definitions follow aisbench's own summarizer (``summarizers/default_perf.py``):

* ``time_points[0]`` is when the request was sent, every later entry is the
  arrival of one response chunk.
* TTFT is the first chunk minus the send time; end-to-end latency is the last
  chunk minus the send time.
* TPOT is decode-only, ``(latency - ttft) / (output_tokens - 1)``, which is the
  quantity GPUStack calls TPOT and guidellm calls ``inter_token_latency``.
* ITL is every gap between consecutive chunks.

Throughputs are counts over the measured window (first send to last chunk of
the measured requests), the definition aisbench and guidellm share.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from aisbench_runner import stats

# Errored requests copied into the report so GPUStack can show what failed.
MAX_ERROR_SAMPLES = 20


@dataclass
class RequestRecord:
    """One request as aisbench recorded it."""

    success: bool
    time_points: List[float] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""

    @property
    def start(self) -> Optional[float]:
        return self.time_points[0] if self.time_points else None

    @property
    def end(self) -> Optional[float]:
        return self.time_points[-1] if self.time_points else None

    @property
    def measurable(self) -> bool:
        """A response chunk arrived, so TTFT and latency are defined."""
        return self.success and len(self.time_points) >= 2

    @property
    def latency(self) -> float:
        return self.time_points[-1] - self.time_points[0]

    @property
    def ttft(self) -> float:
        return self.time_points[1] - self.time_points[0]

    @property
    def decode_time_per_token(self) -> Optional[float]:
        if self.output_tokens <= 1:
            return None
        return (self.latency - self.ttft) / (self.output_tokens - 1)

    @property
    def chunk_gaps(self) -> List[float]:
        points = self.time_points[1:]
        return [b - a for a, b in zip(points, points[1:])]


@dataclass
class Strategy:
    """The load a point ran at, as guidellm names it.

    ``kind`` is ``concurrent`` (``streams`` in flight), ``constant``
    (``rate`` requests per second) or ``throughput`` (the saturation probe:
    as fast as ``max_concurrency`` streams allow, with no load value).
    """

    kind: str
    streams: Optional[int] = None
    rate: Optional[float] = None
    max_concurrency: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type_": self.kind}
        if self.streams is not None:
            out["streams"] = self.streams
        if self.rate is not None:
            out["rate"] = self.rate
        if self.max_concurrency is not None:
            out["max_concurrency"] = self.max_concurrency
        return out


def trim(records: Sequence[RequestRecord], warmup, cooldown) -> List[RequestRecord]:
    """Drop the warmup head and cooldown tail, in send order.

    ``warmup`` / ``cooldown`` follow guidellm's scalar convention (below 1 a
    fraction of the requests, 1 and above a count), which is what GPUStack
    stores. Trimming by request count removes the same population from a busy
    point as from an idle one.
    """
    ordered = sorted((r for r in records if r.start is not None), key=lambda r: r.start)
    head = _trim_count(len(ordered), warmup)
    tail = _trim_count(len(ordered), cooldown)
    if head + tail >= len(ordered):
        return ordered
    return ordered[head : len(ordered) - tail]


def _trim_count(total: int, value) -> int:
    if value is None or total == 0:
        return 0
    value = float(value)
    if value <= 0:
        return 0
    if value < 1:
        return int(total * value)
    return int(value)


def build_benchmark(
    records: Sequence[RequestRecord],
    strategy: Strategy,
    *,
    requested: int,
    stop_reason: str = "max_requests",
    warmup=None,
    cooldown=None,
    clock_offset: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """One guidellm-shaped ``GenerativeBenchmark`` for one measured point.

    ``requested`` is the number of requests the point was sized for and
    ``stop_reason`` the constraint that ended it (``max_requests`` when every
    request ran, ``max_seconds`` when the duration cap cut it short,
    ``runner_aborted`` when the runner stopped aisbench itself), which
    GPUStack surfaces as the point's termination. ``clock_offset`` converts
    aisbench's ``perf_counter`` timestamps to Unix time.

    Returns None only when no request was recorded at all, which GPUStack
    reads as a failed point. A point whose requests all failed is still
    reported, with its error counts: dropping it left a hole in the curve
    where a 100%-error point had been measured.
    """
    if not records:
        return None
    measured = trim(records, warmup, cooldown)
    # A request that never got a time point (a refused connection, one cut
    # short by the kill) has no place in send order, so the warmup trim cannot
    # see it. It is still an error, and counting it is what keeps a point with
    # a third of its requests refused from reading as 100% successful.
    untimed = [r for r in records if r.start is None]
    counted = list(measured) + untimed
    offset = (
        clock_offset if clock_offset is not None else time.time() - time.perf_counter()
    )

    ok = [r for r in measured if r.measurable]
    bad = [r for r in counted if not r.measurable]

    # The measured window comes from the timed requests only: an untimed one
    # has no start to place on it. A point with no timed request at all still
    # reports, with a zero-length window and zero rates, so its failures are
    # visible instead of the point going missing.
    if measured:
        start = min(r.start for r in measured)
        end = max(r.end for r in measured)
        duration = max(end - start, 1e-9)
    else:
        start = end = time.perf_counter()
        duration = 0.0

    def rate(numerator: float) -> float:
        return numerator / duration if duration > 0 else 0.0

    def tokens_per_second(select) -> float:
        return rate(sum(select(r) for r in ok))

    rps_mean = rate(len(ok))
    output_tps = tokens_per_second(lambda r: r.output_tokens)
    input_tps = tokens_per_second(lambda r: r.input_tokens)
    total_tps = output_tps + input_tps

    completions = [(r.end, 1.0) for r in ok]
    output_events = [(r.end, float(r.output_tokens)) for r in ok]
    input_events = [(r.end, float(r.input_tokens)) for r in ok]
    total_events = [(r.end, float(r.input_tokens + r.output_tokens)) for r in ok]

    concurrency = stats.concurrency_series(
        [(r.start, r.end) for r in measured], start, end
    )
    concurrency_mean = rate(sum(r.latency for r in measured))

    def iterations(r: RequestRecord) -> int:
        return max(1, len(r.time_points) - 1)

    metrics = {
        "request_totals": {
            "successful": len(ok),
            "errored": len(bad),
            "incomplete": 0,
            "total": len(counted),
        },
        "requests_per_second": stats.with_mean(
            stats.status_summary(stats.per_second_series(completions, start, end)),
            rps_mean,
        ),
        "request_concurrency": stats.with_mean(
            stats.status_summary(concurrency), concurrency_mean
        ),
        "request_latency": stats.status_summary(
            [r.latency for r in ok], [r.latency for r in bad if r.time_points]
        ),
        "request_streaming_iterations_count": stats.status_summary(
            [iterations(r) for r in ok]
        ),
        "prompt_token_count": stats.status_summary(
            [r.input_tokens for r in ok], [r.input_tokens for r in bad]
        ),
        "output_token_count": stats.status_summary([r.output_tokens for r in ok]),
        "total_token_count": stats.status_summary(
            [r.input_tokens + r.output_tokens for r in ok]
        ),
        "time_to_first_token_ms": stats.status_summary([r.ttft * 1000 for r in ok]),
        "time_per_output_token_ms": stats.status_summary(
            [r.latency * 1000 / r.output_tokens for r in ok if r.output_tokens > 0]
        ),
        "inter_token_latency_ms": stats.status_summary(
            [
                r.decode_time_per_token * 1000
                for r in ok
                if r.decode_time_per_token is not None
            ]
        ),
        "prompt_tokens_per_second": stats.with_mean(
            stats.status_summary(stats.per_second_series(input_events, start, end)),
            input_tps,
        ),
        "output_tokens_per_second": stats.with_mean(
            stats.status_summary(stats.per_second_series(output_events, start, end)),
            output_tps,
        ),
        "tokens_per_second": stats.with_mean(
            stats.status_summary(stats.per_second_series(total_events, start, end)),
            total_tps,
        ),
        "output_tokens_per_iteration": stats.status_summary(
            [r.output_tokens / iterations(r) for r in ok]
        ),
        "iter_tokens_per_iteration": stats.status_summary(
            [(r.input_tokens + r.output_tokens) / iterations(r) for r in ok]
        ),
        "inter_token_latency_per_chunk_ms": stats.status_summary(
            [gap * 1000 for r in ok for gap in r.chunk_gaps]
        ),
    }

    unix_start = start + offset
    unix_end = end + offset
    return {
        "config": {"run_index": 0, "strategy": strategy.to_dict()},
        "scheduler_metrics": {
            "start_time": unix_start,
            "request_start_time": unix_start,
            "measure_start_time": unix_start,
            "measure_end_time": unix_end,
            "request_end_time": unix_end,
            "end_time": unix_end,
        },
        "metrics": metrics,
        "start_time": unix_start,
        "end_time": unix_end,
        "duration": duration,
        "requests_truncated": {
            "successful": [],
            "errored": [_error_sample(r, offset) for r in bad[:MAX_ERROR_SAMPLES]],
            "incomplete": [],
        },
        "scheduler_state": _scheduler_state(requested, len(records), stop_reason),
    }


def _error_sample(record: RequestRecord, offset: float) -> Dict[str, Any]:
    started = record.start + offset if record.start is not None else None
    return {
        "type_": "generative_request_stats",
        "request_id": str(uuid.uuid4()),
        "request_type": "chat_completions",
        "info": {
            "status": "errored",
            "error": record.error or "request failed without an error message",
            "timings": {"request_start": started},
        },
        "input_metrics": {"text_tokens": record.input_tokens or None},
        "output_metrics": {"text_tokens": record.output_tokens or None},
    }


def _scheduler_state(requested: int, processed: int, stop_reason: str) -> Dict:
    """The part of guidellm's scheduler state GPUStack reads a point's
    termination from: which constraint stopped it, and how far it got."""
    return {
        "end_processing_constraints": {
            stop_reason: {
                "request_processing": "stop",
                "metadata": {
                    "processed_requests": processed,
                    "max_requests": requested,
                },
            }
        },
        "scheduler_constraints": {
            "max_requests": {"metadata": {"max_requests": requested}}
        },
    }


def write_report(path: str, benchmarks: List[Optional[Dict[str, Any]]]) -> None:
    """Write a report holding the given benchmarks, atomically.

    GPUStack polls the output directory while a ramp runs and reads every
    point file it finds, so a half-written file must never be visible under
    the final name.
    """
    report = {"benchmarks": [b for b in benchmarks if b is not None]}
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f)
    os.replace(tmp, path)


def summary_line(benchmark: Optional[Dict[str, Any]]) -> str:
    """One line for the container log, so a point can be read without the UI."""
    if benchmark is None:
        return "no request completed"
    m = benchmark["metrics"]
    totals = m["request_totals"]
    return (
        f"requests ok={totals['successful']} failed={totals['errored']} | "
        f"{m['requests_per_second']['successful']['mean']:.2f} req/s, "
        f"{m['output_tokens_per_second']['successful']['mean']:.1f} out tok/s | "
        f"TTFT avg {m['time_to_first_token_ms']['successful']['mean']:.1f} ms "
        f"p99 {m['time_to_first_token_ms']['successful']['percentiles']['p99']:.1f} ms | "
        f"TPOT avg {m['inter_token_latency_ms']['successful']['mean']:.2f} ms | "
        f"latency avg {m['request_latency']['successful']['mean']:.3f} s"
    )
