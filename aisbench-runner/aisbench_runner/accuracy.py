"""Accuracy: ask every question, extract its answer, compare with the reference.

This is the evaluation half of the benchmark container — the half that cares
what the model *says* rather than how fast it says it. It shares the dataset
loader, the progress channel and the target resolution with the performance
run, and replaces the load search with a single pass over the questions.

GSM8K's protocol is the one implemented: the model is asked to reason and to
end with ``answer:<number>``, the last number of the response is taken as its
answer, and it is compared numerically with the number after ``####`` in the
reference. The prompt wording and the comparison rule are copied from
AISBench's own GSM8K config (``gsm8k_gen_0_shot_cot_chat_prompt`` and
``Gsm8kEvaluator``) so a score here is comparable with one from the aisbench
CLI against the same server.

Requests go out over plain HTTP rather than through ``ais_bench``: the
inference task aisbench builds is shaped for timed, high-concurrency load and
drops the response text once it has timed it, while scoring needs that text.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from aisbench_runner.datasets import (
    DatasetError,
    load_qa_items,
    resolve_data_path,
)
from aisbench_runner.progress import ServerProgress

REPORT_VERSION = 1

# The response is kept as evidence for a wrong answer, but a runaway generation
# must not turn one sample into megabytes of state.
_MAX_TEXT_CHARS = 8192

# Longest answer string recorded. Enough for a number with its unit; the point
# is to show what was compared, not to keep the whole response twice.
_MAX_ANSWER_CHARS = 64

# Progress and partial results are written on this many completed samples. A
# run that is stopped or times out keeps everything measured so far.
_REPORT_EVERY = 25

# GSM8K's reference answers embed the final value after "####".
_REFERENCE_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
# GSM8K's prompt asks for this marker; preferring it keeps the extractor from
# picking a number out of the reasoning when the model did state its answer.
_ANSWER_MARKER_RE = re.compile(
    r"answer\s*[:：]\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Copied from ais_bench.benchmark.configs.datasets.gsm8k:
# gsm8k_gen_0_shot_cot_chat_prompt. Keeping the wording identical is what makes
# the number comparable, so do not reword it without saying so in a benchmark's
# notes.
PROMPT_TEMPLATE = (
    "Answer the following question.The last line of the response should follow "
    'this format: "answer:$ANSWER" (without quotes), where ANSWER is a number. '
    "Let's think step by step.\n\nQuestion: {question}"
)


@dataclass
class AccuracyOptions:
    """Everything one accuracy pass needs, resolved from the command line."""

    target: str
    data: str
    output_dir: str
    output_base: str
    model: Optional[str] = None
    api_key: str = ""
    verify_tls: bool = True
    request_timeout: Optional[float] = None
    max_samples: Optional[int] = None
    concurrency: int = 16
    max_out_len: int = 2048
    progress: Optional[ServerProgress] = None


@dataclass
class SampleResult:
    index: int
    question: str
    prediction: str = ""
    extracted: Optional[str] = None
    reference: Optional[str] = None
    correct: bool = False
    latency_ms: Optional[float] = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "question": self.question,
            "prediction": self.prediction,
            "extracted": self.extracted,
            "reference": self.reference,
            "correct": self.correct,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


def chat_completions_url(target: str) -> str:
    """aisbench's endpoint, from the same base URL GPUStack gives it."""
    base = (target or "").strip().rstrip("/")
    if not base:
        raise DatasetError("--target is required")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _number(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", ""))
    except (TypeError, ValueError):
        return None


def extract_answer(text: str) -> Optional[str]:
    """The model's final answer, as the string it wrote.

    The explicit marker wins; otherwise the last number in the response is
    taken, which is what AISBench's ``gsm8k_postprocess`` does and what a model
    trained without the marker still satisfies.
    """
    if not text:
        return None
    body = text.split("Question:")[0]
    for pattern in (_REFERENCE_RE, _ANSWER_MARKER_RE):
        matches = pattern.findall(body)
        if matches:
            return matches[-1].replace(",", "")
    matches = _NUMBER_RE.findall(body)
    if matches:
        return matches[-1].replace(",", "")
    return None


def parse_reference(answer: str) -> Optional[str]:
    """The reference number of a GSM8K answer (``... #### 18``)."""
    if not answer:
        return None
    match = _REFERENCE_RE.search(answer)
    if match:
        return match.group(1).replace(",", "")
    matches = _NUMBER_RE.findall(answer)
    if matches:
        return matches[-1].replace(",", "")
    return None


def is_correct(extracted: Optional[str], reference: Optional[str]) -> bool:
    """Numeric comparison, the same tolerance AISBench's evaluator uses."""
    left, right = _number(extracted or ""), _number(reference or "")
    if left is None or right is None:
        return False
    return abs(left - right) < 1e-6


def _thread_session() -> requests.Session:
    """One pooled session per worker thread, created on first use."""
    existing = getattr(_SESSION_LOCAL, "session", None)
    if existing is None:
        existing = requests.Session()
        _SESSION_LOCAL.session = existing
    return existing


_SESSION_LOCAL = threading.local()


def _ask(session: requests.Session, opts: AccuracyOptions, question: str) -> str:
    headers = {"Content-Type": "application/json"}
    if opts.api_key:
        headers["Authorization"] = f"Bearer {opts.api_key}"
    payload: Dict[str, Any] = {
        "messages": [
            {"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}
        ],
        "max_tokens": opts.max_out_len,
        "temperature": 0.0,
        "stream": False,
    }
    if opts.model:
        payload["model"] = opts.model

    timeout = opts.request_timeout or 3600
    response = session.post(
        chat_completions_url(opts.target),
        json=payload,
        headers=headers,
        timeout=(10, timeout),
        verify=opts.verify_tls,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("response carries no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("response carries no content")
    return content


def _measure_one(
    opts: AccuracyOptions, index: int, question: str, reference: str
) -> SampleResult:
    result = SampleResult(
        index=index,
        question=question,
        reference=reference,
    )
    started = time.monotonic()
    try:
        prediction = _ask(_thread_session(), opts, question)
    except Exception as e:  # noqa: BLE001 - one failure must not stop the run
        result.error = str(e)[:500]
    else:
        result.prediction = prediction[:_MAX_TEXT_CHARS]
        result.extracted = extract_answer(prediction)
        if result.extracted is not None:
            result.extracted = result.extracted[:_MAX_ANSWER_CHARS]
        result.correct = is_correct(result.extracted, reference)
    result.latency_ms = round((time.monotonic() - started) * 1000, 1)
    return result


def build_report(
    opts: AccuracyOptions,
    results: List[SampleResult],
    elapsed_seconds: float,
    *,
    partial: bool = False,
    failure: str = "",
) -> Dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    errored = sum(1 for r in results if r.error)
    answered = sum(1 for r in results if r.extracted is not None)
    return {
        "version": REPORT_VERSION,
        "kind": "accuracy",
        "dataset": os.path.basename(opts.data) if opts.data else "",
        "model": opts.model or "",
        "target": opts.target,
        "score": round(correct / total * 100, 2) if total else 0.0,
        "correct": correct,
        "total": total,
        "answered": answered,
        # Answered without a number. Counted apart from the requests that never
        # got an answer at all, which `errored` reports: the two are different
        # failures and the UI shows them as such.
        "invalid": max(0, total - answered - errored),
        "errored": errored,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "partial": partial,
        "failure": failure,
        "samples": [r.to_dict() for r in results],
    }


def _write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def report_path(opts: AccuracyOptions) -> str:
    return os.path.join(opts.output_dir, f"{opts.output_base}__accuracy.json")


def run_accuracy(opts: AccuracyOptions) -> int:
    """Score every question once and write the report. Exit status is the run's."""
    from aisbench_runner.runner import log

    try:
        path = resolve_data_path(opts.data)
        items = load_qa_items(path, limit=opts.max_samples)
    except DatasetError as e:
        log(f"error: {e}")
        _write_json(
            report_path(opts),
            build_report(opts, [], 0.0, partial=True, failure=str(e)),
        )
        return 2

    total = len(items)
    concurrency = max(1, int(opts.concurrency or 1))
    log(
        f"accuracy: {total} question(s), concurrency {concurrency}, "
        f"dataset {os.path.basename(path)}"
    )
    if opts.progress is not None:
        opts.progress.run_index = 0
        opts.progress.run_total = 1

    started = time.monotonic()
    results: List[SampleResult] = []
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _measure_one, opts, index, item.question, parse_reference(item.answer)
            )
            for index, item in enumerate(items)
        ]
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if opts.progress is not None:
                opts.progress.report_fraction(done / max(1, total))
            if done % _REPORT_EVERY == 0:
                # Partial writes keep a long run's measurements even if it is
                # stopped mid-way; the terminal write below replaces them.
                ordered = sorted(results, key=lambda r: r.index)
                _write_json(
                    report_path(opts),
                    build_report(
                        opts,
                        ordered,
                        time.monotonic() - started,
                        partial=True,
                    ),
                )
                log(f"accuracy: {done}/{total} answered")

    results.sort(key=lambda r: r.index)
    report = build_report(opts, results, time.monotonic() - started)
    _write_json(report_path(opts), report)
    if opts.progress is not None:
        opts.progress.send(100.0)
    log(
        f"accuracy finished: {report['correct']}/{report['total']} correct "
        f"({report['score']}%), {report['invalid']} without an answer, "
        f"{report['errored']} errored"
    )
    return 0 if report["total"] else 1


def read_report(path: str) -> Optional[Dict[str, Any]]:
    """A written accuracy report, or None when the run never produced one."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def summarize(report: Dict[str, Any]) -> str:
    return (
        f"{report.get('correct', 0)}/{report.get('total', 0)} correct "
        f"({report.get('score', 0)}%)"
    )


def sample_rows(report: Dict[str, Any]) -> List[Tuple[Any, ...]]:
    """The per-sample tuples the worker uploads, in report order."""
    rows = []
    for sample in report.get("samples") or []:
        rows.append(
            (
                sample.get("index", 0),
                (sample.get("question") or "")[:_MAX_TEXT_CHARS],
                (sample.get("prediction") or "")[:_MAX_TEXT_CHARS],
                sample.get("extracted"),
                sample.get("reference"),
                bool(sample.get("correct")),
                sample.get("latency_ms"),
                (sample.get("error") or "")[:500],
            )
        )
    return rows
