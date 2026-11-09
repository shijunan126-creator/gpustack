"""Reading a dataset a user uploaded, before it is stored.

The benchmark runner understands two file shapes — a JSON array of ShareGPT
conversations and a JSONL of question/answer rows — and this module is where an
upload is held to one of them. It runs on the server, so a malformed file is
rejected at upload time with a message about the line that is wrong, instead of
failing a run an hour later inside the benchmark container.

The row count and the number of answerable rows are computed here too: the
first is what the UI shows next to a dataset, and the second is what decides
whether the dataset may be used for an accuracy run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, List, Optional

from gpustack.schemas.benchmark_dataset import (
    BenchmarkDatasetFormat,
)

# One upload may not exceed this. A benchmark dataset is a few thousand prompts,
# not a training corpus; the cap keeps a single row from growing the database by
# hundreds of megabytes and gives a clear error instead.
MAX_BENCHMARK_DATASET_BYTES = 8 * 1024 * 1024

MAX_BENCHMARK_DATASET_NAME_LENGTH = 64

_QUESTION_FIELDS = ("question", "prompt", "input")
_ANSWER_FIELDS = ("answer", "reference", "output")


class DatasetContentError(ValueError):
    """The uploaded text is not a dataset this build can run."""


@dataclass
class DatasetSummary:
    format: BenchmarkDatasetFormat
    sample_count: int
    answer_count: int
    size_bytes: int
    sha256: str

    @property
    def has_answers(self) -> bool:
        return self.answer_count > 0


def _text_field(row: Any, names: tuple) -> Optional[str]:
    if not isinstance(row, dict):
        return None
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _has_conversation(row: Any) -> bool:
    """A ShareGPT row: a human turn followed by an assistant turn."""
    if not isinstance(row, dict):
        return False
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        return False
    first, second = conversations[0], conversations[1]
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    if first.get("from") not in ("human", "user"):
        return False
    if second.get("from") not in ("gpt", "assistant"):
        return False
    return bool(first.get("value")) and bool(second.get("value"))


def _inspect_sharegpt(rows: List[Any]) -> DatasetSummary:
    usable = sum(1 for row in rows if _has_conversation(row))
    if not usable:
        raise DatasetContentError(
            "no conversation with a human turn followed by an assistant turn "
            "was found; each row needs [{\"from\": \"human\", ...}, "
            "{\"from\": \"gpt\", ...}]"
        )
    return DatasetSummary(
        format=BenchmarkDatasetFormat.SHAREGPT,
        sample_count=usable,
        answer_count=0,
        size_bytes=0,
        sha256="",
    )


def _inspect_jsonl(lines: List[str], summary: DatasetSummary) -> DatasetSummary:
    samples = 0
    answers = 0
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise DatasetContentError(f"line {number} is not valid JSON: {e}")
        question = _text_field(row, _QUESTION_FIELDS)
        if question is None:
            raise DatasetContentError(
                f"line {number} has no question; each row needs one of "
                f"{', '.join(_QUESTION_FIELDS)}"
            )
        samples += 1
        if _text_field(row, _ANSWER_FIELDS) is not None:
            answers += 1
    if not samples:
        raise DatasetContentError("the file holds no rows")
    summary.sample_count = samples
    summary.answer_count = answers
    return summary


def inspect_dataset(
    content: str, declared: Optional[BenchmarkDatasetFormat] = None
) -> DatasetSummary:
    """Validate an uploaded dataset and describe what it holds.

    A declared format is honored but still checked; without one the shape is
    inferred, JSON array first because a JSONL file of one object per line is
    also a valid JSON array only by accident.
    """
    if not content or not content.strip():
        raise DatasetContentError("the file is empty")
    raw = content.encode("utf-8")
    if len(raw) > MAX_BENCHMARK_DATASET_BYTES:
        raise DatasetContentError(
            f"the dataset is larger than {MAX_BENCHMARK_DATASET_BYTES // (1024 * 1024)} MB"
        )

    summary = DatasetSummary(
        format=BenchmarkDatasetFormat.QA_JSONL,
        sample_count=0,
        answer_count=0,
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )

    text = content.lstrip("\ufeff").strip()
    if declared == BenchmarkDatasetFormat.SHAREGPT:
        try:
            rows = json.loads(text)
        except json.JSONDecodeError as e:
            raise DatasetContentError(f"the file is not valid JSON: {e}")
        if not isinstance(rows, list):
            raise DatasetContentError("a ShareGPT dataset must be a JSON array")
        return _inspect_sharegpt(rows)

    if text.startswith("["):
        try:
            rows = json.loads(text)
        except json.JSONDecodeError as e:
            # A JSONL file whose first row is an array-shaped object still
            # reads as an array here; fall through to the line parser.
            if declared is not None:
                raise DatasetContentError(f"the file is not valid JSON: {e}")
            rows = None
        if isinstance(rows, list):
            if declared == BenchmarkDatasetFormat.QA_JSONL:
                raise DatasetContentError(
                    "a question/answer dataset must have one JSON object per line"
                )
            return _inspect_sharegpt(rows)

    summary = _inspect_jsonl(text.splitlines(), summary)
    if declared is not None and declared != summary.format:
        raise DatasetContentError(
            f"the file looks like a {summary.format.value} dataset, not "
            f"{declared.value}"
        )
    return summary


def validate_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        raise DatasetContentError("a dataset name is required")
    if len(cleaned) > MAX_BENCHMARK_DATASET_NAME_LENGTH:
        raise DatasetContentError(
            f"the name is longer than {MAX_BENCHMARK_DATASET_NAME_LENGTH} characters"
        )
    return cleaned
