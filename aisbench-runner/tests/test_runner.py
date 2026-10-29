"""What one point may cost, and how long it may go quiet."""

import time

from aisbench_runner.engine import (
    STOP_MAX_REQUESTS,
    STOP_RUNNER_ABORTED,
    Engine,
    _Watch,
)
from aisbench_runner.runner import (
    _MAX_DATASET_TOKENS,
    _MAX_ROWS,
    _stall_timeout,
    dataset_row_limit,
)


def test_a_file_dataset_is_bounded_by_the_row_cap_only():
    # Nothing states what one of its rows costs; the file is the bound.
    assert dataset_row_limit(None) == _MAX_ROWS


def test_a_cheap_row_keeps_the_row_cap():
    assert dataset_row_limit(1024 + 128) == _MAX_ROWS


def test_an_expensive_row_is_bounded_by_the_token_budget():
    per_row = 8192 + 1024
    assert dataset_row_limit(per_row) == _MAX_DATASET_TOKENS // per_row
    assert dataset_row_limit(per_row) < _MAX_ROWS


def test_the_stall_bound_covers_the_workers_request_allowance():
    # Below the allowance the default bound still decides: a request that
    # quick is not allowed to stall the point.
    assert _stall_timeout(None) == 900.0
    assert _stall_timeout(60) == 900.0
    assert _stall_timeout(3600) == 3600.0


def _engine(stall_timeout: float = 900.0):
    # The verdict reads the stall bound and nothing else off the engine.
    engine = Engine.__new__(Engine)
    engine.stall_timeout = stall_timeout
    return engine


def test_a_failed_inference_task_ends_the_point_as_an_abort():
    """Reported as its own reason: read back as max_requests, a point cut
    short by a failed task would look like one that ran all it was given."""
    watch = _Watch(launched=time.monotonic())
    watch.task_failed = True

    message, reason, error = _engine()._verdict(watch, total=10, max_seconds=None)

    assert reason == STOP_RUNNER_ABORTED
    assert "inference task failed" in message
    assert error


def test_a_stalled_point_ends_as_an_abort_not_as_completed():
    watch = _Watch(launched=time.monotonic())
    watch.last_activity = time.monotonic() - 901

    _, reason, error = _engine(stall_timeout=900.0)._verdict(
        watch, total=10, max_seconds=None
    )

    assert reason == STOP_RUNNER_ABORTED
    assert error


def test_a_finished_point_keeps_max_requests_as_its_reason():
    """Waiting for aisbench to exit after its last record is not an abort."""
    watch = _Watch(launched=time.monotonic())
    watch.done = 10
    watch.last_activity = time.monotonic() - 61

    verdict = _engine()._verdict(watch, total=10, max_seconds=None)

    assert verdict is not None
    assert verdict[1] == STOP_MAX_REQUESTS
