"""Reading an uploaded benchmark dataset, before it is stored."""

import json

import pytest

from gpustack.schemas.benchmark_dataset import BenchmarkDatasetFormat
from gpustack.utils.benchmark_dataset import (
    MAX_BENCHMARK_DATASET_BYTES,
    DatasetContentError,
    inspect_dataset,
    validate_name,
)


def _jsonl(rows):
    return "\n".join(json.dumps(row) for row in rows)


def test_a_question_answer_file_is_counted_through_its_answers():
    content = _jsonl(
        [
            {"question": "2+2?", "answer": "#### 4"},
            {"question": "3+3?"},
            {"prompt": "4+4?", "reference": "#### 8"},
        ]
    )
    summary = inspect_dataset(content)
    assert summary.format == BenchmarkDatasetFormat.QA_JSONL
    assert summary.sample_count == 3
    assert summary.answer_count == 2
    assert summary.has_answers is True
    assert len(summary.sha256) == 64


def test_a_conversation_file_is_recognised_and_carries_no_answers():
    content = json.dumps(
        [
            {
                "conversations": [
                    {"from": "human", "value": "hi"},
                    {"from": "gpt", "value": "hello"},
                ]
            },
            {"conversations": [{"from": "human", "value": "no reply"}]},
        ]
    )
    summary = inspect_dataset(content)
    assert summary.format == BenchmarkDatasetFormat.SHAREGPT
    assert summary.sample_count == 1
    assert summary.has_answers is False


def test_a_broken_line_names_the_line_and_the_field():
    content = _jsonl([{"question": "ok"}, {"prompt": "fine"}]) + "\n{not json"
    with pytest.raises(DatasetContentError, match="line 3"):
        inspect_dataset(content)

    with pytest.raises(DatasetContentError, match="line 2"):
        inspect_dataset(_jsonl([{"question": "ok"}, {"text": "no question"}]))


def test_a_declared_format_is_checked_not_trusted():
    content = _jsonl([{"question": "q", "answer": "#### 1"}])
    with pytest.raises(DatasetContentError, match="jsonl"):
        inspect_dataset(content, BenchmarkDatasetFormat.SHAREGPT)


def test_an_empty_or_oversized_file_is_refused():
    with pytest.raises(DatasetContentError, match="empty"):
        inspect_dataset("   ")
    oversized = "x" * (MAX_BENCHMARK_DATASET_BYTES + 1)
    with pytest.raises(DatasetContentError, match="larger than"):
        inspect_dataset(oversized)


def test_a_name_is_required_and_bounded():
    assert validate_name("  my corpus  ") == "my corpus"
    with pytest.raises(DatasetContentError):
        validate_name("")
    with pytest.raises(DatasetContentError):
        validate_name("x" * 65)
