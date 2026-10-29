"""The three run shapes GPUStack asks for, each point measured by aisbench.

* **auto-tune** — the adaptive ramp behind both presets ("Max Throughput" on
  the rate axis, "Meet Latency SLO" on the concurrency axis with thresholds).
  The search is benchmark-runner's own engine (``ramp.py``); only the point
  measurement changed.
* **stages** — one point per user-given load value.
* **single** — one point.

Output files are benchmark-runner's, named and shaped as GPUStack's
``worker/benchmark/artifacts.py`` expects: ``{base}__p{i}.json`` per ramp
point, ``{base}__satprobe.json`` for the saturation probe,
``{base}__ramp.json`` for why the ramp stopped, ``{base}__stage{i}.json`` and
``{base}__curve.json`` for stages, ``{base}.json`` for a single run. ``{base}``
is the benchmark id.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from aisbench_runner import ramp
from aisbench_runner.datasets import (
    DatasetError,
    RandomSpec,
    Row,
    ShareGPTSource,
    TextBuilder,
    ensure_lengths,
    parse_random_spec,
)
from aisbench_runner.engine import STOP_MAX_REQUESTS, Engine, PointRun, Target
from aisbench_runner.progress import ServerProgress
from aisbench_runner.report import Strategy, build_benchmark, summary_line, write_report

# Requests sized for a point that only has a duration cap, per unit of load:
# enough that the cap, not the dataset, ends it on any realistic server.
_DURATION_ROWS_PER_STREAM_SECOND = 0.5
# Never generate more rows than this for one point.
_MAX_ROWS = 20000

CURVE_OUTCOME_VERSION = 1


@dataclass
class RunOptions:
    target: str
    processor: str
    output_dir: str
    output_base: str
    model: Optional[str] = None
    trust_remote_code: bool = False
    api_key: str = ""
    verify_tls: bool = True
    mode: str = "single"
    axis: str = "rate"
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    multiplier: Optional[float] = None
    min_requests: Optional[int] = None
    max_points: Optional[int] = None
    max_total_seconds: Optional[float] = None
    slo: Dict[str, float] = field(default_factory=dict)
    probe_saturation: bool = True
    stages: List[Dict[str, Any]] = field(default_factory=list)
    profile: str = "concurrent"
    rate: Optional[float] = None
    data: Optional[str] = None
    sharegpt_min_input: Optional[int] = None
    sharegpt_max_input: Optional[int] = None
    sharegpt_output_tokens: Optional[int] = None
    random_seed: Optional[int] = None
    seed_increment: bool = True
    max_requests: Optional[int] = None
    max_seconds: Optional[float] = None
    warmup: Optional[float] = None
    cooldown: Optional[float] = None
    progress_url: Optional[str] = None
    progress_auth: Optional[str] = None
    progress_ca_cert: Optional[str] = None
    progress_insecure: bool = False
    ignored_options: List[str] = field(default_factory=list)


def log(message: str) -> None:
    print(f"[aisbench-runner] {message}", flush=True)


def _namespace(value: Any) -> Any:
    """A report dict as attributes, which is how the ramp's ``_normalize``
    reads a benchmark."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespace(v) for v in value]
    return value


def point_metrics(
    benchmark: Optional[Dict[str, Any]], knob: float, index: int
) -> Optional[ramp.PointMetrics]:
    if benchmark is None:
        return None
    return ramp._normalize(_namespace(benchmark), knob, index)


class Benchmark:
    """State shared by every point of one run: the target, the tokenizer,
    the prompt source and the progress channel."""

    def __init__(self, opts: RunOptions):
        self.opts = opts
        self.progress = (
            ServerProgress(
                opts.progress_url,
                opts.progress_auth,
                ca_cert=opts.progress_ca_cert,
                insecure_skip_tls_verify=opts.progress_insecure,
            )
            if opts.progress_url
            else None
        )
        tokenizer = _load_tokenizer(opts.processor, opts.trust_remote_code)
        self.builder = TextBuilder(tokenizer)
        self.random: Optional[RandomSpec] = None
        self.sharegpt: Optional[ShareGPTSource] = None
        if opts.data and os.path.isfile(opts.data):
            self.sharegpt = ShareGPTSource(
                opts.data,
                self.builder,
                input_min=opts.sharegpt_min_input,
                input_max=opts.sharegpt_max_input,
                output_tokens=opts.sharegpt_output_tokens,
            )
        elif opts.data:
            self.random = parse_random_spec(opts.data)
        else:
            raise DatasetError("--data is required")
        self.engine = Engine(
            Target(
                url=opts.target,
                tokenizer=opts.processor,
                model=opts.model,
                api_key=opts.api_key,
                trust_remote_code=opts.trust_remote_code,
                verify_tls=opts.verify_tls,
            ),
            work_root=os.path.join(opts.output_dir, f"{opts.output_base}.aisbench"),
            builder=self.builder,
            keep_work_dirs=os.environ.get("AISBENCH_RUNNER_KEEP_WORK_DIRS") == "1",
        )

    def path(self, suffix: str) -> str:
        return os.path.join(self.opts.output_dir, f"{self.opts.output_base}{suffix}")

    def rows(self, count: int, seed: int) -> List[Row]:
        if self.sharegpt is not None:
            return self.sharegpt.take(count)
        assert self.random is not None
        return ensure_lengths(self.random, self.builder, count, seed)

    def measure(
        self,
        name: str,
        strategy: Strategy,
        *,
        count: int,
        seed: int,
        max_seconds: Optional[float],
        sized_by_duration: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Run one point and return its report entry (None when nothing was
        recorded)."""
        rows = self.rows(count, seed)
        if strategy.kind == "constant":
            concurrency, request_rate = len(rows), float(strategy.rate or 0.0)
        else:
            concurrency = int(strategy.streams or strategy.max_concurrency or 1)
            request_rate = 0.0
        log(
            f"{name}: {strategy.kind} "
            + (
                f"{strategy.rate} req/s"
                if strategy.kind == "constant"
                else f"{concurrency} streams"
            )
            + f", {len(rows)} requests"
            + (f", cap {max_seconds:.0f}s" if max_seconds else "")
        )
        run: PointRun = self.engine.run(
            name,
            rows,
            concurrency=concurrency,
            request_rate=request_rate,
            max_seconds=max_seconds,
            on_progress=self.progress.report_fraction if self.progress else None,
        )
        stop = run.stop_reason
        if stop == STOP_MAX_REQUESTS and sized_by_duration:
            stop = "requests_exhausted"
        benchmark = build_benchmark(
            run.records,
            strategy,
            requested=run.requested,
            stop_reason=stop,
            warmup=self.opts.warmup,
            cooldown=self.opts.cooldown,
            clock_offset=run.clock_offset,
        )
        if run.error:
            log(f"{name}: {run.error}")
        log(f"{name}: {summary_line(benchmark)}")
        return benchmark

    def request_count(
        self,
        knob: float,
        kind: str,
        max_requests: Optional[int],
        max_seconds: Optional[float],
    ) -> tuple:
        """(rows, sized_by_duration) for a point that was given a request
        count, a duration, both or neither."""
        if max_requests:
            return int(max_requests), False
        if max_seconds:
            if kind == "constant":
                return (
                    min(_MAX_ROWS, max(1, math.ceil(knob * max_seconds * 1.1) + 10)),
                    True,
                )
            per_stream = max(
                20, math.ceil(max_seconds * _DURATION_ROWS_PER_STREAM_SECOND)
            )
            return min(_MAX_ROWS, max(1, int(knob)) * per_stream), True
        multiplier = 30 if kind == "constant" else 10
        return max(100, round(knob * multiplier)), False


def _load_tokenizer(path: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    if not os.path.exists(path):
        raise DatasetError(
            f"tokenizer path {path} does not exist in the benchmark container; the "
            "model directory must be mounted at the same path as on the worker"
        )
    return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)


def _strategy(axis: str, knob: float) -> Strategy:
    if axis == "concurrency":
        return Strategy("concurrent", streams=int(knob))
    return Strategy("constant", rate=float(knob))


# ── auto-tune ────────────────────────────────────────────────────────────────


def run_auto_tune(bench: Benchmark) -> None:
    opts = bench.opts
    cfg = ramp.AutoTuneConfig(
        axis=opts.axis,
        **{
            key: value
            for key, value in (
                ("lower_bound", opts.lower_bound),
                ("upper_bound", opts.upper_bound),
                ("multiplier", opts.multiplier),
                ("min_requests", opts.min_requests),
                ("max_points", opts.max_points),
                ("max_total_seconds", opts.max_total_seconds),
            )
            if value is not None
        },
        **opts.slo,
        random_seed_base=42 if opts.random_seed is None else int(opts.random_seed),
        seed_increment=opts.seed_increment,
        probe_saturation=opts.probe_saturation,
    )
    for flag, value in (
        ("--lower-bound", cfg.lower_bound),
        ("--upper-bound", cfg.upper_bound),
        ("--min-requests", cfg.min_requests),
        ("--max-points", cfg.max_points),
        ("--max-total-seconds", cfg.max_total_seconds),
    ):
        if value <= 0:
            raise ValueError(f"{flag} must be greater than 0 (got {value})")
    if cfg.lower_bound > cfg.upper_bound:
        raise ValueError(
            f"--lower-bound ({cfg.lower_bound}) must not exceed --upper-bound ({cfg.upper_bound})"
        )

    started = time.monotonic()

    def remaining() -> float:
        return max(
            ramp._MIN_POINT_SECONDS,
            cfg.max_total_seconds - (time.monotonic() - started),
        )

    def measure_point(knob: float, index: int) -> Optional[ramp.PointMetrics]:
        count = max(cfg.min_requests, round(knob * cfg.resolved_multiplier))
        seed = (
            cfg.random_seed_base + index if cfg.seed_increment else cfg.random_seed_base
        )
        benchmark = bench.measure(
            f"point {index}",
            _strategy(cfg.axis, knob),
            count=count,
            seed=seed,
            max_seconds=remaining(),
        )
        write_report(bench.path(f"__p{index}.json"), [benchmark])
        return point_metrics(benchmark, knob, index)

    def measure_probe() -> float:
        benchmark = bench.measure(
            "saturation probe",
            Strategy("throughput", max_concurrency=ramp.PROBE_MAX_CONCURRENCY),
            count=ramp.PROBE_MAX_REQUESTS,
            seed=cfg.random_seed_base,
            max_seconds=min(ramp.PROBE_SECONDS, remaining()),
        )
        write_report(bench.path("__satprobe.json"), [benchmark])
        metrics = point_metrics(benchmark, 0.0, 0)
        return metrics.achieved_rate if metrics else 0.0

    async def run_point(knob: float, index: int) -> Optional[ramp.PointMetrics]:
        return await asyncio.to_thread(measure_point, knob, index)

    async def saturation_probe() -> float:
        return await asyncio.to_thread(measure_probe)

    log(
        f"auto-tune: axis={cfg.axis} target={cfg.target} bounds=[{cfg.lower_bound},"
        f"{cfg.upper_bound}] multiplier={cfg.resolved_multiplier} "
        f"min_requests={cfg.min_requests} max_points={cfg.max_points} "
        f"max_total_seconds={cfg.max_total_seconds}"
    )
    outcome = asyncio.run(
        ramp.run_ramp(
            cfg=cfg,
            base_kwargs={},
            output_base=opts.output_base,
            server_progress=bench.progress,
            run_point=run_point,
            saturation_probe=saturation_probe,
        )
    )
    _write_json(bench.path("__ramp.json"), outcome.to_dict())
    log(
        f"auto-tune finished: {len(outcome.points)} point(s), bracket="
        f"{outcome.bracket_reason} stop={outcome.stop_reason} stopped_at={outcome.stopped_at}"
    )


# ── stages / single ──────────────────────────────────────────────────────────


def run_stages(bench: Benchmark) -> None:
    opts = bench.opts
    started = time.monotonic()
    points: List[ramp.PointMetrics] = []
    for i, stage in enumerate(opts.stages):
        if not isinstance(stage, dict) or "rate" not in stage:
            raise ValueError(f"stage {i} is missing the required 'rate' key")
        knob = float(stage["rate"])
        if bench.progress is not None:
            bench.progress.run_index = i
            bench.progress.run_total = len(opts.stages)
        strategy = _strategy(opts.axis, knob)
        count, by_duration = bench.request_count(
            knob, strategy.kind, stage.get("max_requests"), stage.get("max_seconds")
        )
        # A stage list varies the load and holds the data fixed unless a seed
        # was given, which then increments per stage (benchmark-runner's rule).
        if opts.random_seed is None:
            seed = 42
        else:
            seed = opts.random_seed + i if opts.seed_increment else opts.random_seed
        benchmark = bench.measure(
            f"stage {i}",
            strategy,
            count=count,
            seed=seed,
            max_seconds=stage.get("max_seconds"),
            sized_by_duration=by_duration,
        )
        write_report(bench.path(f"__stage{i}.json"), [benchmark])
        metrics = point_metrics(benchmark, knob, i)
        if metrics is not None:
            points.append(metrics)
    if points:
        _write_json(
            bench.path("__curve.json"),
            {
                "version": CURVE_OUTCOME_VERSION,
                "points": [asdict(p) for p in points],
                "axis": opts.axis,
                "target": "stages",
                "stop_reason": "stages_completed",
                "bracket_reason": "",
                "points_measured": len(points),
                "slo_bracket": None,
                "slo_thresholds": {},
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "probe_ceiling": None,
            },
        )
    if bench.progress is not None:
        bench.progress.send(100.0)


def run_single(bench: Benchmark) -> None:
    opts = bench.opts
    if opts.rate is None:
        raise ValueError("--rate is required with --profile")
    strategy = (
        Strategy("concurrent", streams=int(opts.rate))
        if opts.profile == "concurrent"
        else Strategy("constant", rate=float(opts.rate))
    )
    count, by_duration = bench.request_count(
        float(opts.rate), strategy.kind, opts.max_requests, opts.max_seconds
    )
    benchmark = bench.measure(
        "run",
        strategy,
        count=count,
        seed=42 if opts.random_seed is None else opts.random_seed,
        max_seconds=opts.max_seconds,
        sized_by_duration=by_duration,
    )
    write_report(bench.path(".json"), [benchmark])
    if bench.progress is not None:
        bench.progress.send(100.0)


def _write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def run_benchmark(opts: RunOptions) -> int:
    for flag in opts.ignored_options:
        log(f"{flag} is not supported by the aisbench engine and is ignored")
    try:
        os.makedirs(opts.output_dir, exist_ok=True)
        bench = Benchmark(opts)
        {"auto": run_auto_tune, "stages": run_stages, "single": run_single}[opts.mode](
            bench
        )
    except (DatasetError, ValueError) as e:
        log(f"error: {e}")
        return 2
    except Exception:  # noqa: BLE001 - the traceback is the only diagnosis
        traceback.print_exc()
        return 1
    finally:
        # Point work directories are removed as they finish; drop their parent
        # once empty so reruns of a benchmark id do not accumulate them.
        try:
            os.rmdir(os.path.join(opts.output_dir, f"{opts.output_base}.aisbench"))
        except OSError:
            pass
    return 0
