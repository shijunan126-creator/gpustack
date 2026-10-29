"""Run one measured point with aisbench and read its requests back.

A point is one ``ais_bench <config> -m perf`` process over a dataset this
runner wrote. Three things about aisbench decide how it is driven, all found
by running it rather than by reading its docs:

* ``retry`` is the number of ATTEMPTS, not of retries: ``retry=0`` sends no
  request at all and reports every one as failed with an empty reason. Points
  run with one attempt — a retried request would fold a failure and a second
  try into one latency.
* It cannot be stopped early. SIGINT is ignored and the run continues to the
  last row, so a duration cap is enforced by killing the process group. The
  requests that finished before that are still on disk: aisbench appends each
  completed request to ``performances/<model>/tmp/tmp_<db>.jsonl`` and its
  timings to the matching SQLite file, and the summarizer itself falls back to
  those when a run was killed.
* A path with a space in it breaks the command line it starts its inference
  task with; the task dies at once and the parent then waits forever. The
  same parent hang follows any failed task, and occasionally a finished one,
  so a failed task is acted on as soon as it is logged, and a run whose
  requests have all been recorded is not waited on past a short grace.
* Its exit status does not say whether anything was measured, and its error
  messages often point away from the cause (an unreachable server surfaces as
  ``different structure of perf data``). The per-request records are read
  regardless of the exit status, and the failure summary table aisbench logs
  is what a failed point reports.

aisbench's per-request ``time_points`` are ``time.perf_counter()`` values;
Linux backs that with ``CLOCK_MONOTONIC``, which every process shares, so the
runner converts them to Unix time with its own offset.
"""

from __future__ import annotations

import glob
import io
import json
import logging
import os
import queue
import re
import shutil
import signal
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.parse import urlparse

from aisbench_runner.datasets import Row, TextBuilder, write_rows
from aisbench_runner.report import RequestRecord

logger = logging.getLogger(__name__)

# aisbench validates batch_size into (0, 100000].
MAX_BATCH_SIZE = 100000

# Printed by the inference task right before it starts sending; the duration
# cap is measured from here, not from process start, which costs 10-20 s of
# imports and tokenizer loading that no request is waiting on.
_SENDING_MARKER = "Total data num"
# Seconds to wait for the marker before the cap starts counting anyway.
_SENDING_MARKER_TIMEOUT = 180.0

# Seconds a run may stay silent after its last request was recorded before it
# is stopped; everything that is measured is on disk by then.
_DONE_GRACE_SECONDS = 60.0

_EXP_FOLDER_RE = re.compile(r"Current exp folder:\s*(\S+)")
# Logged by the runner when the inference task exits non-zero.
_TASK_FAILED_RE = re.compile(r"RUNNER-TASK-\d+|task failed")
_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_FAILED_SUMMARY_RE = re.compile(
    r"failed reasons summary:(.*?)(?:\n\s*\n|\Z)", re.DOTALL
)
_ERR_LINE_RE = re.compile(r"^[A-Za-z_.]*(?:Error|Exception)\b.*$", re.MULTILINE)

STOP_MAX_REQUESTS = "max_requests"
STOP_MAX_SECONDS = "max_seconds"


@dataclass
class Target:
    """The endpoint under test and how to talk to it."""

    url: str
    tokenizer: str
    model: Optional[str] = None
    api_key: str = ""
    trust_remote_code: bool = False
    verify_tls: bool = True


@dataclass
class PointRun:
    records: List[RequestRecord]
    requested: int
    stop_reason: str
    clock_offset: float
    error: Optional[str] = None
    log_tail: List[str] = field(default_factory=list)


def connection_fields(url: str) -> Dict[str, Any]:
    """aisbench's model connection fields for a base URL.

    aisbench joins ``v1/chat/completions`` onto the base itself, so a trailing
    ``/v1`` must go or the path doubles. A base without a path is given as
    host and port; one with a path prefix (a gateway behind a proxy) can only
    be given as ``url``, which aisbench then uses instead of host and port.
    """
    raw = url.strip()
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    if not parsed.hostname:
        raise ValueError(f"cannot parse target URL {url!r}")
    scheme = (parsed.scheme or "http").lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    path = re.sub(r"/v1$", "", (parsed.path or "").rstrip("/")).rstrip("/")
    fields: Dict[str, Any] = {"enable_ssl": scheme == "https"}
    if path:
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        fields["url"] = f"{scheme}://{host}:{port}{path}/"
    else:
        fields["url"] = ""
        fields["host_ip"] = parsed.hostname
        fields["host_port"] = port
    return fields


def build_config(
    target: Target,
    dataset_path: str,
    work_dir: str,
    concurrency: int,
    request_rate: float,
    max_out_len: int,
) -> str:
    """Source of a self-contained aisbench config for one point.

    Written as literals, not as calls: mmengine loads configs lazily and
    calling an imported name at the top level of a config raises. The dataset
    entry is therefore built here (``make_qa_gen_config`` with a fixed
    ``{question}`` template — the default template would prepend "Question:"
    and append the answer placeholder to every prompt) and serialized.

    No ``infer`` block: given ``attr="service"`` the CLI picks the API
    inference task itself, and a hand-written one is how the wrong task class
    gets chosen.
    """
    from ais_bench.benchmark.datasets.custom import make_qa_gen_config, stringfy_types
    from ais_bench.benchmark.utils.logging.logger import AISLogger

    dataset = stringfy_types(
        make_qa_gen_config(
            {
                "path": dataset_path,
                "abbr": "bench",
                "input_columns": ["question", "max_out_len"],
                "output_column": "answer",
                "options": [],
                "data_type": "qa",
                "infer_method": "gen",
                "meta_path": None,
                "template": "{question}",
            },
            AISLogger(),
        )
    )
    model: Dict[str, Any] = {
        "attr": "service",
        "abbr": "target",
        "path": target.tokenizer,
        "model": target.model or "",
        "stream": True,
        "max_out_len": max_out_len,
        "retry": 1,
        "api_key": target.api_key or "",
        "trust_remote_code": target.trust_remote_code,
        "generation_kwargs": {"ignore_eos": True},
        "batch_size": max(1, min(MAX_BATCH_SIZE, concurrency)),
        "request_rate": request_rate,
        **connection_fields(target.url),
    }
    model_items = ", ".join(f"{json.dumps(k)}: {v!r}" for k, v in model.items())
    return "\n".join(
        [
            '"""Generated by aisbench-runner for one benchmark point."""',
            "from mmengine.config import read_base",
            "from ais_bench.benchmark.models import VLLMCustomAPIChat",
            "",
            "with read_base():",
            "    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer",
            "",
            f"datasets = [{dataset!r}]",
            f"models = [{{'type': VLLMCustomAPIChat, {model_items}}}]",
            f"work_dir = {json.dumps(work_dir)}",
            "",
        ]
    )


class Engine:
    """Runs points against one target, one aisbench process at a time."""

    def __init__(
        self,
        target: Target,
        work_root: str,
        builder: TextBuilder,
        stall_timeout: float = 900.0,
        keep_work_dirs: bool = False,
        echo: Callable[[str], None] = print,
    ):
        self.target = target
        self.work_root = work_root
        self.builder = builder
        self.stall_timeout = stall_timeout
        self.keep_work_dirs = keep_work_dirs
        self.echo = echo

    def run(
        self,
        name: str,
        rows: Sequence[Row],
        *,
        concurrency: int,
        request_rate: float = 0.0,
        max_seconds: Optional[float] = None,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> PointRun:
        """Measure one point over ``rows``.

        ``concurrency`` bounds the requests in flight and ``request_rate``
        paces sending (0 sends as fast as the concurrency allows).
        ``max_seconds`` caps the time spent sending; the point then keeps
        whatever completed.
        """
        point_dir = os.path.join(self.work_root, _UNSAFE_NAME_RE.sub("_", name))
        shutil.rmtree(point_dir, ignore_errors=True)
        os.makedirs(point_dir, exist_ok=True)
        dataset_path = os.path.join(point_dir, "data.jsonl")
        write_rows(dataset_path, rows)
        config_path = os.path.join(point_dir, "config.py")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(
                build_config(
                    self.target,
                    dataset_path,
                    os.path.join(point_dir, "out"),
                    concurrency=concurrency,
                    request_rate=float(request_rate),
                    max_out_len=max(r.output_tokens for r in rows),
                )
            )
        try:
            result = self._execute(
                point_dir, config_path, rows, max_seconds, on_progress
            )
        finally:
            if not self.keep_work_dirs:
                shutil.rmtree(point_dir, ignore_errors=True)
        return result

    def _execute(
        self,
        point_dir: str,
        config_path: str,
        rows: Sequence[Row],
        max_seconds: Optional[float],
        on_progress: Optional[Callable[[float], None]],
    ) -> PointRun:
        env = dict(os.environ)
        if not self.target.verify_tls:
            env["AISBENCH_RUNNER_TLS_INSECURE"] = "1"
        cmd = ["ais_bench", config_path, "-m", "perf", "--num-warmups", "0"]
        self.echo(f"$ {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            cwd=point_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        lines: "queue.Queue[Optional[bytes]]" = queue.Queue()

        def _reader() -> None:
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, b""):
                lines.put(line)
            lines.put(None)

        threading.Thread(target=_reader, daemon=True).start()

        out_dir = os.path.join(point_dir, "out")
        watch = _Watch(launched=time.monotonic())
        while True:
            try:
                raw = lines.get(timeout=1.0)
            except queue.Empty:
                raw = b""
            if raw is None:
                watch.stream_open = False
            elif raw:
                text = raw.decode("utf-8", errors="replace").rstrip()
                self.echo(text)
                watch.read_line(text)

            done = watch.observe(out_dir)
            if done is not None and on_progress is not None:
                on_progress(min(1.0, done / max(1, len(rows))))

            if not watch.stream_open and proc.poll() is not None:
                break
            verdict = self._verdict(watch, len(rows), max_seconds)
            if verdict is not None:
                message, watch.stop_reason, watch.error = verdict
                self.echo(f"[aisbench-runner] {message}")
                _kill_group(proc)
                break

        proc.wait()
        run_dir = watch.run_dir or _latest_run_dir(out_dir)
        records = load_records(run_dir, rows, self.builder) if run_dir else []
        error = watch.error
        if not records and error is None:
            reason = failure_reason(run_dir) if run_dir else None
            error = (
                reason
                or f"aisbench exited with status {proc.returncode} without recording a request"
            )
        return PointRun(
            records=records,
            requested=len(rows),
            stop_reason=watch.stop_reason,
            clock_offset=time.time() - time.perf_counter(),
            error=error,
            log_tail=watch.tail,
        )

    def _verdict(
        self, watch: "_Watch", total: int, max_seconds: Optional[float]
    ) -> Optional[tuple]:
        """Why to stop aisbench now, as (message, stop reason, error), or None
        to keep waiting."""
        now = time.monotonic()
        if watch.task_failed:
            return (
                "the inference task failed; stopping aisbench",
                STOP_MAX_REQUESTS,
                None,
            )
        if watch.done >= total and now - watch.last_activity > _DONE_GRACE_SECONDS:
            return (
                "every request is recorded but aisbench has not exited; stopping it",
                STOP_MAX_REQUESTS,
                None,
            )
        # Once every request is recorded the remaining time is aisbench's own
        # summarizing, which the cap is not about.
        if (
            max_seconds is not None
            and watch.sending_at is not None
            and watch.done < total
            and now - watch.sending_at > max_seconds
        ):
            return (
                f"duration cap of {max_seconds:.0f}s reached; stopping the point",
                STOP_MAX_SECONDS,
                None,
            )
        if self.stall_timeout and now - watch.last_activity > self.stall_timeout:
            error = (
                f"aisbench made no progress for {self.stall_timeout:.0f}s and was "
                "stopped; the target may have stopped answering"
            )
            return error, STOP_MAX_REQUESTS, error
        return None


@dataclass
class _Watch:
    """What the runner has seen of one aisbench process so far."""

    launched: float
    run_dir: Optional[str] = None
    tail: List[str] = field(default_factory=list)
    sending_at: Optional[float] = None
    last_activity: float = 0.0
    done: int = -1
    stop_reason: str = STOP_MAX_REQUESTS
    error: Optional[str] = None
    stream_open: bool = True
    task_failed: bool = False

    def __post_init__(self) -> None:
        self.last_activity = self.launched

    def read_line(self, text: str) -> None:
        self.last_activity = time.monotonic()
        self.tail = (self.tail + [text])[-40:]
        match = _EXP_FOLDER_RE.search(text)
        if match:
            self.run_dir = match.group(1)
        if _TASK_FAILED_RE.search(text):
            self.task_failed = True

    def observe(self, out_dir: str) -> Optional[int]:
        """Refresh from the files aisbench writes; returns the completed
        request count when it changed."""
        now = time.monotonic()
        if self.sending_at is None and (
            _infer_log_has(out_dir, _SENDING_MARKER)
            or now - self.launched > _SENDING_MARKER_TIMEOUT
        ):
            self.sending_at = now
        done = _completed_count(out_dir)
        if done == self.done:
            return None
        self.done = done
        self.last_activity = now
        return done


def _kill_group(proc: subprocess.Popen) -> None:
    """aisbench forks its inference workers; killing only the parent leaves
    them sending requests at the target."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _infer_log_has(out_dir: str, marker: str) -> bool:
    for path in glob.glob(os.path.join(out_dir, "*", "logs", "infer", "*", "*.out")):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if marker in f.read():
                    return True
        except OSError:
            continue
    return False


def _completed_count(out_dir: str) -> int:
    """Requests aisbench has finished so far, from the files it appends to."""
    total = 0
    for pattern in ("tmp/*.jsonl", "*_details.jsonl"):
        paths = glob.glob(os.path.join(out_dir, "*", "performances", "*", pattern))
        count = 0
        for path in paths:
            try:
                with open(path, "rb") as f:
                    count += sum(
                        chunk.count(b"\n")
                        for chunk in iter(lambda: f.read(1 << 20), b"")
                    )
            except OSError:
                continue
        total = max(total, count)
    return total


def _latest_run_dir(out_dir: str) -> Optional[str]:
    runs = sorted(
        d
        for d in glob.glob(os.path.join(out_dir, "*"))
        if re.fullmatch(r"\d{8}_\d{6}", os.path.basename(d))
    )
    return runs[-1] if runs else None


def _load_arrays(db_path: str) -> Dict[int, List[float]]:
    import numpy as np

    arrays: Dict[int, List[float]] = {}
    if not os.path.exists(db_path):
        return arrays
    conn = sqlite3.connect(db_path)
    try:
        for row_id, blob in conn.execute("SELECT id, arr_blob FROM numpy_store"):
            try:
                arrays[row_id] = [
                    float(v) for v in np.load(io.BytesIO(blob), allow_pickle=False)
                ]
            except Exception:
                continue
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return arrays


def load_records(
    run_dir: str, rows: Sequence[Row], builder: TextBuilder
) -> List[RequestRecord]:
    """Every request of a finished or killed run.

    The final ``*_details.jsonl`` and ``db_data/`` exist once a run completes;
    a killed run only has the ``tmp/`` copies. Token counts prefer the
    server's usage report (which counts the chat template too) and fall back
    to the dataset row's length and the tokenizer.
    """
    perf_root = os.path.join(run_dir, "performances")
    details = glob.glob(os.path.join(perf_root, "*", "*_details.jsonl"))
    entries: List[tuple] = []
    if details:
        for path in details:
            model_dir = os.path.dirname(path)
            entries.extend((model_dir, "db_data", line) for line in _read_lines(path))
    else:
        for path in glob.glob(os.path.join(perf_root, "*", "tmp", "*.jsonl")):
            model_dir = os.path.dirname(os.path.dirname(path))
            entries.extend((model_dir, "tmp", line) for line in _read_lines(path))

    arrays_by_db: Dict[str, Dict[int, List[float]]] = {}
    records: List[RequestRecord] = []
    for model_dir, kind, data in entries:
        db_name = data.get("db_name") or ""
        key = os.path.join(model_dir, kind, db_name)
        if key not in arrays_by_db:
            if kind == "db_data":
                path = os.path.join(model_dir, "db_data", db_name)
                if not os.path.exists(path):
                    path = os.path.join(model_dir, "tmp", f"tmp_{db_name}")
            else:
                path = os.path.join(model_dir, "tmp", f"tmp_{db_name}")
            arrays_by_db[key] = _load_arrays(path)
        ref = (
            (data.get("time_points") or {}).get("__db_ref__")
            if isinstance(data.get("time_points"), dict)
            else None
        )
        points = arrays_by_db[key].get(ref, []) if ref is not None else []
        index = data.get("id")
        row = rows[index] if isinstance(index, int) and 0 <= index < len(rows) else None
        input_tokens = int(data.get("input_tokens") or 0) or (
            row.prompt_tokens if row else 0
        )
        output_tokens = int(data.get("output_tokens") or 0)
        if output_tokens <= 0 and data.get("prediction"):
            output_tokens = builder.count(str(data["prediction"]))
        records.append(
            RequestRecord(
                success=bool(data.get("success")) and bool(points),
                time_points=points,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=str(data.get("error_info") or ""),
            )
        )
    return records


def _read_lines(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # The last line of a killed run can be cut mid-write.
                    continue
    except OSError:
        pass
    return out


def failure_reason(run_dir: Optional[str]) -> Optional[str]:
    """The cause aisbench logged for a failed run: its failure summary table
    when it printed one, else the last exception line of a task log."""
    if not run_dir:
        return None
    texts: List[tuple] = []
    for path in sorted(
        glob.glob(os.path.join(run_dir, "logs", "**", "*.out"), recursive=True)
    ):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                texts.append((os.path.basename(path), f.read()))
        except OSError:
            continue
    for _, text in texts:
        summary = _failed_summary(text)
        if summary:
            return summary
    for name, text in texts:
        hits = _ERR_LINE_RE.findall(text)
        if hits:
            return f"{name}: {hits[-1].strip()[:300]}"
    return None


def _failed_summary(text: str) -> Optional[str]:
    match = _FAILED_SUMMARY_RE.search(text)
    if not match:
        return None
    reasons = []
    for raw in match.group(1).splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
        if len(cells) >= 2 and cells[-1].isdigit():
            reasons.append(f"{' | '.join(cells[:-1])} (x{cells[-1]})")
        if len(reasons) >= 3:
            break
    return "; ".join(reasons)[:400] if reasons else None
