"""Accuracy scoring: the answer a model gave, and whether it was right."""

import json
import os

import pytest

from aisbench_runner import accuracy
from aisbench_runner.datasets import QAItem


@pytest.mark.parametrize(
    "prediction,expected",
    [
        ("... so the answer is 18.\n\nanswer:18", "18"),
        ("reasoning\n#### 1,234", "1234"),
        ("first 2 then 7, so 7", "7"),
        ("no numbers here", None),
        ("", None),
    ],
)
def test_the_final_answer_is_extracted(prediction, expected):
    assert accuracy.extract_answer(prediction) == expected


def test_the_explicit_marker_wins_over_a_later_number():
    # A reasoning model often restates the question's numbers after its answer.
    text = "We need 16 eggs minus 3 and 4.\nanswer:18\nQuestion: 16 - 3 - 4"
    assert accuracy.extract_answer(text) == "18"


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("She sells 18 eggs. #### 18", "18"),
        ("no marker, but 42", "42"),
        ("unparseable", None),
    ],
)
def test_the_reference_is_the_number_after_the_marker(answer, expected):
    assert accuracy.parse_reference(answer) == expected


@pytest.mark.parametrize(
    "extracted,reference,correct",
    [
        ("18", "18", True),
        ("18.0", "18", True),
        ("1,234", "1234", True),
        ("17", "18", False),
        (None, "18", False),
        ("18", None, False),
    ],
)
def test_comparison_is_numeric(extracted, reference, correct):
    assert accuracy.is_correct(extracted, reference) is correct


def test_the_report_counts_errors_and_missing_answers_separately():
    options = accuracy.AccuracyOptions(
        target="http://server:8000", data="gsm8k.jsonl", output_dir="/tmp", output_base="7"
    )
    results = [
        accuracy.SampleResult(0, "q0", "answer:1", "1", "1", True),
        accuracy.SampleResult(1, "q1", "answer:2", "2", "3", False),
        accuracy.SampleResult(2, "q2", "no number", None, "4", False),
        accuracy.SampleResult(3, "q3", "", None, "5", False, error="HTTP 500"),
    ]
    report = accuracy.build_report(options, results, 12.5)
    assert report["total"] == 4
    assert report["correct"] == 1
    assert report["invalid"] == 2
    assert report["errored"] == 1
    assert report["score"] == 25.0
    assert report["partial"] is False


@pytest.mark.parametrize(
    "target,expected",
    [
        ("http://host:8000", "http://host:8000/v1/chat/completions"),
        ("http://host:8000/", "http://host:8000/v1/chat/completions"),
        ("https://host/v1", "https://host/v1/chat/completions"),
        ("https://host/gpustack", "https://host/gpustack/v1/chat/completions"),
    ],
)
def test_the_endpoint_follows_the_base_url(target, expected):
    assert accuracy.chat_completions_url(target) == expected


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_a_run_scores_every_question_and_writes_the_report(tmp_path, monkeypatch):
    data = tmp_path / "qa.jsonl"
    _write_jsonl(
        data,
        [
            {"question": "2+2?", "answer": "#### 4"},
            {"question": "3+3?", "answer": "#### 6"},
        ],
    )

    def fake_ask(session, opts, question):
        return "answer:4" if question == "2+2?" else "answer:5"

    monkeypatch.setattr(accuracy, "_ask", fake_ask)
    opts = accuracy.AccuracyOptions(
        target="http://server:8000",
        data=str(data),
        output_dir=str(tmp_path),
        output_base="9",
        concurrency=2,
    )
    assert accuracy.run_accuracy(opts) == 0
    report = json.loads((tmp_path / "9__accuracy.json").read_text(encoding="utf-8"))
    assert report["total"] == 2
    assert report["correct"] == 1
    assert report["score"] == 50.0
    assert [s["index"] for s in report["samples"]] == [0, 1]


def test_a_failed_request_is_counted_not_raised(tmp_path, monkeypatch):
    data = tmp_path / "qa.jsonl"
    _write_jsonl(data, [{"question": "q", "answer": "#### 1"}])

    def boom(session, opts, question):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(accuracy, "_ask", boom)
    opts = accuracy.AccuracyOptions(
        target="http://server:8000",
        data=str(data),
        output_dir=str(tmp_path),
        output_base="3",
    )
    assert accuracy.run_accuracy(opts) == 0
    report = json.loads((tmp_path / "3__accuracy.json").read_text(encoding="utf-8"))
    assert report["errored"] == 1
    assert report["correct"] == 0


def test_a_dataset_without_answers_is_refused_with_a_message(tmp_path):
    data = tmp_path / "qa.jsonl"
    _write_jsonl(data, [{"question": "q only"}])
    opts = accuracy.AccuracyOptions(
        target="http://server:8000",
        data=str(data),
        output_dir=str(tmp_path),
        output_base="4",
    )
    assert accuracy.run_accuracy(opts) == 2
    report = json.loads((tmp_path / "4__accuracy.json").read_text(encoding="utf-8"))
    assert "question/answer" in report["failure"]


def test_the_builtin_gsm8k_dataset_ships_with_the_package():
    from aisbench_runner.datasets import builtin_gsm8k_path, resolve_data_path

    path = resolve_data_path("builtin:gsm8k")
    assert path == builtin_gsm8k_path()
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1319
    assert all("question" in row and "####" in row["answer"] for row in rows)


def test_qa_items_skip_rows_without_answers(tmp_path):
    from aisbench_runner.datasets import load_qa_items

    data = tmp_path / "qa.jsonl"
    _write_jsonl(
        data,
        [
            {"question": "a", "answer": "#### 1"},
            {"question": "b"},
            {"prompt": "c", "reference": "2"},
        ],
    )
    items = load_qa_items(str(data))
    assert [item.question for item in items] == ["a", "c"]
    assert isinstance(items[0], QAItem)
