"""``benchmark-runner benchmark run``, with aisbench measuring every point.

The command line is benchmark-runner's, because GPUStack's worker builds it
(``gpustack/worker/benchmark/runner.py::_build_command_args``) and swapping
the image is all it takes to switch engines. Every option GPUStack sends is
accepted. The few that guidellm enforces while a run is in flight and
aisbench cannot (``--max-errors``, ``--max-error-rate``,
``--detect-saturation``) are reported and ignored rather than refused, so a
benchmark cloned from an older run still starts.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List, Optional, Sequence

from aisbench_runner.runner import RunOptions, run_benchmark

_SLO_FLAGS = (
    ("slo_avg_ttft_ms", ["--slo-avg-ttft-ms"]),
    ("slo_p95_ttft_ms", ["--slo-p95-ttft-ms"]),
    ("slo_p99_ttft_ms", ["--slo-p99-ttft-ms"]),
    ("slo_avg_itl_ms", ["--slo-avg-itl-ms", "--slo-avg-tpot-ms"]),
    ("slo_p95_itl_ms", ["--slo-p95-itl-ms", "--slo-p95-tpot-ms"]),
    ("slo_p99_itl_ms", ["--slo-p99-itl-ms", "--slo-p99-tpot-ms"]),
    ("slo_avg_latency_ms", ["--slo-avg-latency-ms"]),
    ("slo_p95_latency_ms", ["--slo-p95-latency-ms"]),
    ("slo_p99_latency_ms", ["--slo-p99-latency-ms"]),
)


def _json(value: str) -> Any:
    return json.loads(value)


def _timeout(value: Any) -> Optional[float]:
    """The backend kwargs' per-request allowance, when one was stated.

    Anything unparseable is treated as no statement: the runner falls back
    to its own stall bound rather than refusing to start over a hand-edited
    config."""
    if value is None or value == "":
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def _phase(value: str) -> Optional[float]:
    """``--warmup`` / ``--cooldown``: a bare number, or GPUStack's
    ``{"percent": p}`` / ``{"value": n}`` object."""
    text = value.strip()
    if not text.startswith("{"):
        return float(text)
    data = json.loads(text)
    if "percent" in data:
        return float(data["percent"])
    if "value" in data:
        return float(data["value"])
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark-runner")
    sub = parser.add_subparsers(dest="group", required=True)
    bench = sub.add_parser("benchmark").add_subparsers(dest="command", required=True)
    run = bench.add_parser("run")

    run.add_argument("--target", required=True)
    run.add_argument("--model")
    run.add_argument("--processor", required=True)
    run.add_argument("--processor-args", type=_json)
    run.add_argument("--backend", default="openai_http")
    run.add_argument("--backend-kwargs", "--backend-args", type=_json)

    run.add_argument("--auto-tune", action="store_true")
    run.add_argument("--axis", choices=["rate", "concurrency"], default="rate")
    run.add_argument("--lower-bound", type=float)
    run.add_argument("--upper-bound", type=float)
    run.add_argument("--multiplier", type=float)
    run.add_argument("--min-requests", type=int)
    run.add_argument("--max-points", type=int)
    run.add_argument("--max-total-seconds", type=float)
    for dest, flags in _SLO_FLAGS:
        run.add_argument(*flags, dest=dest, type=float)
    run.add_argument(
        "--probe-saturation", dest="probe_saturation", action="store_true", default=True
    )
    run.add_argument(
        "--no-probe-saturation", dest="probe_saturation", action="store_false"
    )

    run.add_argument("--stages", type=_json)
    run.add_argument("--profile", "--rate-type", choices=["concurrent", "constant"])
    run.add_argument("--rate", type=float)

    run.add_argument("--data")
    run.add_argument("--sharegpt-min-input-tokens", type=int)
    run.add_argument("--sharegpt-max-input-tokens", type=int)
    run.add_argument("--sharegpt-max-output-tokens", type=int)
    run.add_argument("--random-seed", type=int)
    run.add_argument(
        "--seed-increment", dest="seed_increment", action="store_true", default=True
    )
    run.add_argument("--no-seed-increment", dest="seed_increment", action="store_false")

    run.add_argument("--max-requests", type=int)
    run.add_argument("--max-seconds", type=float)
    run.add_argument("--warmup", "--warmup-percent", type=_phase)
    run.add_argument("--cooldown", "--cooldown-percent", type=_phase)
    run.add_argument("--max-errors", type=int)
    run.add_argument("--max-error-rate", type=float)
    run.add_argument(
        "--detect-saturation", "--default-over-saturation", action="store_true"
    )
    run.add_argument("--sample-requests", "--output-sampling", type=int)

    run.add_argument("--output-dir", required=True)
    run.add_argument("--outputs", default="benchmarks.dual_json")
    run.add_argument("--progress-url")
    run.add_argument("--progress-auth")
    run.add_argument("--progress-ca-cert")
    run.add_argument("--progress-insecure-skip-tls-verify", action="store_true")
    return parser


def options_from_args(args: argparse.Namespace) -> RunOptions:
    backend = dict(args.backend_kwargs or {})
    processor_args = dict(args.processor_args or {})
    modes = [
        name
        for name, chosen in (
            ("--auto-tune", args.auto_tune),
            ("--stages", bool(args.stages)),
            ("--profile", args.profile is not None),
        )
        if chosen
    ]
    if len(modes) > 1:
        raise SystemExit(
            f"{modes[0]} is mutually exclusive with {', '.join(modes[1:])}"
        )
    output = args.outputs
    output_base = output.rpartition(".")[0] if "." in output else output
    slo = {
        dest: getattr(args, dest)
        for dest, _ in _SLO_FLAGS
        if getattr(args, dest) is not None
    }

    ignored: List[str] = []
    if args.max_errors is not None:
        ignored.append("--max-errors")
    if args.max_error_rate is not None:
        ignored.append("--max-error-rate")
    if args.detect_saturation:
        ignored.append("--detect-saturation")

    return RunOptions(
        target=args.target,
        model=args.model,
        processor=args.processor,
        trust_remote_code=bool(processor_args.get("trust_remote_code")),
        api_key=str(backend.get("api_key") or ""),
        verify_tls=bool(backend.get("verify", True)),
        request_timeout=_timeout(backend.get("timeout")),
        mode="auto" if args.auto_tune else ("stages" if args.stages else "single"),
        axis=args.axis,
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
        multiplier=args.multiplier,
        min_requests=args.min_requests,
        max_points=args.max_points,
        max_total_seconds=args.max_total_seconds,
        slo=slo,
        probe_saturation=args.probe_saturation,
        stages=list(args.stages or []),
        profile=args.profile or "concurrent",
        rate=args.rate,
        data=args.data,
        sharegpt_min_input=args.sharegpt_min_input_tokens,
        sharegpt_max_input=args.sharegpt_max_input_tokens,
        sharegpt_output_tokens=args.sharegpt_max_output_tokens,
        random_seed=args.random_seed,
        seed_increment=args.seed_increment,
        max_requests=args.max_requests,
        max_seconds=args.max_seconds,
        warmup=args.warmup,
        cooldown=args.cooldown,
        output_dir=args.output_dir,
        output_base=output_base,
        progress_url=args.progress_url,
        progress_auth=args.progress_auth,
        progress_ca_cert=args.progress_ca_cert,
        progress_insecure=args.progress_insecure_skip_tls_verify,
        ignored_options=ignored,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run_benchmark(options_from_args(args))


def entrypoint() -> None:
    sys.exit(main())
