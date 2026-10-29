"""The reports this runner writes must load the way GPUStack's worker loads
benchmark-runner's, and must carry the numbers the requests imply."""

import json
import os

import pytest

from aisbench_runner import ramp
from aisbench_runner.report import (
    RequestRecord,
    Strategy,
    build_benchmark,
    trim,
    write_report,
)
from aisbench_runner.runner import point_metrics
from gpustack.worker.schemas.benchmark_runner import GenerativeBenchmarksReport


def request(
    start, ttft, chunks, gap, output_tokens=None, input_tokens=100, success=True
):
    points = [start, start + ttft] + [
        start + ttft + gap * (i + 1) for i in range(chunks - 1)
    ]
    return RequestRecord(
        success=success,
        time_points=points,
        input_tokens=input_tokens,
        output_tokens=output_tokens if output_tokens is not None else chunks,
    )


def ten_requests():
    # Two streams, five requests each, back to back: 50 ms TTFT, 10 ms
    # between chunks, 21 output tokens -> 250 ms per request.
    return [request(1000.0 + 0.25 * i, 0.05, 21, 0.01) for i in range(5)] + [
        request(1000.0 + 0.25 * i, 0.05, 21, 0.01) for i in range(5)
    ]


def load(tmp_path, benchmark):
    path = os.path.join(tmp_path, "1__p0.json")
    write_report(path, [benchmark])
    return GenerativeBenchmarksReport.load_file(path)


def test_report_loads_through_gpustacks_schema(tmp_path):
    bm = build_benchmark(
        ten_requests(),
        Strategy("concurrent", streams=2),
        requested=10,
        clock_offset=0.0,
    )
    report = load(tmp_path, bm)

    results = report.to_results(input_tokens=100)
    assert len(results) == 1
    point = results[0]
    assert point["strategy_type"] == "concurrent"
    assert point["rate"] == 2.0
    assert point["request_total"] == 10
    assert point["request_successful"] == 10
    assert point["time_to_first_token_mean"] == pytest.approx(50.0)
    # Decode-only TPOT: (250 - 50) ms over 20 tokens after the first.
    assert point["inter_token_latency_mean"] == pytest.approx(10.0)
    assert point["request_latency_mean"] == pytest.approx(0.25)
    # Window: 1000.0 .. 1001.25 -> 10 requests in 1.25 s.
    assert point["requests_per_second_mean"] == pytest.approx(8.0)
    assert point["output_tokens_per_second_mean"] == pytest.approx(210 / 1.25)
    assert point["itl_per_chunk_mean"] == pytest.approx(10.0)
    assert report.to_metrics() is not None


def test_the_termination_reads_back(tmp_path):
    bm = build_benchmark(
        ten_requests(),
        Strategy("constant", rate=8.0),
        requested=40,
        stop_reason="max_seconds",
        clock_offset=0.0,
    )
    point = load(tmp_path, bm).to_results()[0]
    assert point["rate"] == 8.0
    assert point["raw_metrics"]["termination"] == {
        "reason": "max_seconds",
        "requested": 40,
        "processed": 10,
    }


def test_failed_requests_are_counted_and_sampled(tmp_path):
    records = ten_requests()
    records.append(
        RequestRecord(success=False, time_points=[1001.0], error="Connection refused")
    )
    bm = build_benchmark(
        records, Strategy("concurrent", streams=2), requested=11, clock_offset=0.0
    )
    report = load(tmp_path, bm)
    point = report.to_results()[0]
    assert point["request_errored"] == 1
    assert point["request_successful"] == 10
    errored = report.benchmarks[0].requests_truncated.errored
    assert errored[0].info.error == "Connection refused"


def test_no_request_is_no_benchmark():
    assert build_benchmark([], Strategy("concurrent", streams=1), requested=5) is None


def test_the_ramp_reads_the_point_it_was_given():
    bm = build_benchmark(
        ten_requests(),
        Strategy("concurrent", streams=2),
        requested=10,
        clock_offset=0.0,
    )
    m = point_metrics(bm, 2.0, 0)
    assert isinstance(m, ramp.PointMetrics)
    assert m.success == 1.0
    assert m.ttft_ms == pytest.approx(50.0)
    assert m.itl_ms == pytest.approx(10.0)
    assert m.latency_ms == pytest.approx(250.0)
    assert m.achieved_rate == pytest.approx(8.0)
    assert m.output_tps == pytest.approx(168.0)


def test_a_single_chunk_response_has_no_decode_time(tmp_path):
    records = [request(1000.0 + i, 0.05, 1, 0.0, output_tokens=1) for i in range(3)]
    bm = build_benchmark(
        records, Strategy("concurrent", streams=1), requested=3, clock_offset=0.0
    )
    point = load(tmp_path, bm).to_results()[0]
    assert point["inter_token_latency_mean"] == 0.0
    assert point["itl_per_chunk_mean"] is None


@pytest.mark.parametrize(
    "warmup, cooldown, kept",
    [(None, None, 10), (0.2, None, 8), (2, 3, 5), (0.5, 0.5, 10)],
)
def test_trim_removes_requests_by_send_order(warmup, cooldown, kept):
    records = [request(1000.0 + i, 0.05, 3, 0.01) for i in range(10)]
    out = trim(records, warmup, cooldown)
    assert len(out) == kept
    assert [r.start for r in out] == sorted(r.start for r in out)


def test_written_report_is_valid_json(tmp_path):
    path = os.path.join(tmp_path, "x.json")
    write_report(path, [None])
    with open(path) as f:
        assert json.load(f) == {"benchmarks": []}
    assert not os.path.exists(path + ".tmp")


def test_a_request_without_timings_still_counts_as_an_error(tmp_path):
    """A refused connection has no time points, so the warmup trim cannot
    place it; it is still one of the requests the point ran."""
    records = ten_requests()
    records.append(RequestRecord(success=False, error="Connection refused"))
    bm = build_benchmark(
        records, Strategy("concurrent", streams=2), requested=11, clock_offset=0.0
    )
    report = load(tmp_path, bm)
    point = report.to_results()[0]
    assert point["request_total"] == 11
    assert point["request_successful"] == 10
    assert point["request_errored"] == 1
    errored = report.benchmarks[0].requests_truncated.errored
    assert errored[0].info.error == "Connection refused"


def test_a_point_whose_requests_all_failed_is_still_reported(tmp_path):
    """All-error points carry the failure, rather than vanishing from the
    curve as a point nobody measured."""
    records = [
        RequestRecord(success=False, error="Connection refused") for _ in range(3)
    ]
    bm = build_benchmark(
        records, Strategy("constant", rate=4.0), requested=3, clock_offset=0.0
    )
    assert bm is not None
    point = load(tmp_path, bm).to_results()[0]
    assert point["request_total"] == 3
    assert point["request_successful"] == 0
    assert point["request_errored"] == 3
    assert point["requests_per_second_mean"] == 0.0
    assert point["raw_metrics"]["duration"] == 0.0
