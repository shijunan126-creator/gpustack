"""Benchmark prompts, written as the rows of an aisbench ``CustomDataset``.

Every point runs on a dataset this module writes: one JSON line per request,
``{"question": <prompt>, "answer": "none", "max_out_len": <output tokens>}``.
aisbench honours a per-row ``max_out_len`` over the model's, and with
``ignore_eos`` that fixes each request's output length, so a point measures
exactly the context the benchmark asked for.

Three sources:

* **Random** 鈥?GPUStack's ``--data prompt_tokens=...,output_tokens=...``. The
  lengths are drawn the way guidellm draws them (``IntegerRangeSampler``:
  constant, uniform between min and max, or a clipped Gaussian when a standard
  deviation is given), and ``prefix_buckets`` prepend shared prefixes the same
  way, so a benchmark configured for guidellm means the same thing here.
* **ShareGPT** 鈥?the first human/assistant turn of each conversation, filtered
  by prompt length; the output length is the reference answer's unless a
  fixed maximum is configured.
* **QA JSONL** 鈥?one ``{"question": ..., "answer": ...}`` object per line, the
  shape GSM8K ships in. A performance run uses the questions as prompts; an
  accuracy run compares the model's answer with the reference one.

Random prompt text is built from the tokenizer's own vocabulary: whole-word
tokens that encode back to themselves, joined with spaces, so a prompt of N
words is N tokens without an encode/decode round trip per row. A sample is
re-tokenized to confirm that before a dataset is used; a tokenizer where the
words do not add up falls back to guidellm's encode-and-truncate per row.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Upper bound aisbench's config validation puts on max_out_len.
MAX_OUT_LEN = 131072

# Prompts re-tokenized to confirm the word lengths add up.
LENGTH_CHECK_SAMPLES = 16
# Mean relative length error the word builder may have before the per-row
# encode-and-truncate path is used instead.
LENGTH_TOLERANCE = 0.02

_WORD_RE = re.compile(r"^ [a-z]{2,12}$")


class DatasetError(ValueError):
    """The dataset cannot be built; the message is shown to the user."""


@dataclass
class PrefixBucket:
    prefix_tokens: int
    prefix_count: int = 1
    bucket_weight: int = 100


@dataclass
class RandomSpec:
    """guidellm's synthetic text options, as GPUStack sends them."""

    prompt_tokens: int
    output_tokens: int = 1
    prompt_tokens_stdev: Optional[int] = None
    prompt_tokens_min: Optional[int] = None
    prompt_tokens_max: Optional[int] = None
    output_tokens_stdev: Optional[int] = None
    output_tokens_min: Optional[int] = None
    output_tokens_max: Optional[int] = None
    turns: int = 1
    prefix_buckets: List[PrefixBucket] = field(default_factory=list)


@dataclass
class Row:
    prompt: str
    output_tokens: int
    prompt_tokens: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "question": self.prompt,
                "answer": "none",
                "max_out_len": self.output_tokens,
            },
            ensure_ascii=False,
        )


def parse_random_spec(data: str) -> RandomSpec:
    """Parse ``--data``: ``k=v,k=v`` pairs, or a JSON object when the config
    carries nested values (``prefix_buckets``)."""
    text = data.strip()
    if text.startswith("{"):
        raw: Dict[str, Any] = json.loads(text)
    else:
        raw = {}
        for pair in text.split(","):
            key, sep, value = pair.partition("=")
            if not sep:
                raise DatasetError(f"cannot parse --data entry '{pair}'")
            raw[key.strip()] = value.strip()

    def opt_int(name: str) -> Optional[int]:
        value = raw.pop(name, None)
        if value is None or value == "":
            return None
        return int(float(value))

    prompt_tokens = opt_int("prompt_tokens")
    if prompt_tokens is None or prompt_tokens < 1:
        raise DatasetError("--data needs prompt_tokens >= 1")
    if raw.get("output_tokens") in (None, ""):
        raise DatasetError("--data needs output_tokens")
    buckets = [
        PrefixBucket(
            prefix_tokens=int(b.get("prefix_tokens", 0)),
            prefix_count=max(1, int(b.get("prefix_count", 1))),
            bucket_weight=int(b.get("bucket_weight", 100)),
        )
        for b in (raw.pop("prefix_buckets", None) or [])
    ]
    spec = RandomSpec(
        prompt_tokens=prompt_tokens,
        output_tokens=opt_int("output_tokens"),
        prompt_tokens_stdev=opt_int("prompt_tokens_stdev"),
        prompt_tokens_min=opt_int("prompt_tokens_min"),
        prompt_tokens_max=opt_int("prompt_tokens_max"),
        output_tokens_stdev=opt_int("output_tokens_stdev"),
        output_tokens_min=opt_int("output_tokens_min"),
        output_tokens_max=opt_int("output_tokens_max"),
        turns=opt_int("turns") or 1,
        prefix_buckets=[b for b in buckets if b.prefix_tokens > 0],
    )
    if spec.turns > 1:
        raise DatasetError(
            "multi-turn conversations (turns > 1) are not supported by the "
            "aisbench benchmark runner"
        )
    if raw:
        raise DatasetError(f"unsupported --data options: {', '.join(sorted(raw))}")
    return spec


def integer_sampler(
    average: int,
    stdev: Optional[int],
    min_value: Optional[int],
    max_value: Optional[int],
    seed: int,
) -> Iterator[int]:
    """guidellm's ``IntegerRangeSampler``: constant, uniform or clipped Gaussian."""
    rng = random.Random(seed)
    low = min_value
    if low is None:
        low = max(1, average - 5 * stdev if stdev else average)
    high = max_value
    if high is None:
        high = average + 5 * stdev if stdev else average
    if low == high:
        while True:
            yield low
    elif not stdev:
        while True:
            yield rng.randint(low, high)
    else:
        while True:
            yield round(max(low, min(high, rng.gauss(average, stdev))))


class TextBuilder:
    """Builds text of a given token length from one tokenizer."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.words = self._single_token_words()
        self.exact = len(self.words) >= 200

    def _single_token_words(self, wanted: int = 4000, scan: int = 120_000) -> List[str]:
        """Vocabulary entries that are one lowercase word with a leading space
        and encode back to exactly that one token."""
        tok = self.tokenizer
        size = min(scan, getattr(tok, "vocab_size", 0) or len(tok))
        words: List[str] = []
        for token_id in range(size):
            try:
                text = tok.decode([token_id])
            except Exception:
                continue
            if not _WORD_RE.match(text):
                continue
            if tok.encode(text, add_special_tokens=False) != [token_id]:
                continue
            words.append(text)
            if len(words) >= wanted:
                break
        return words

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def text(self, tokens: int, rng: random.Random) -> str:
        if tokens <= 0:
            return ""
        if self.exact:
            return "".join(rng.choice(self.words) for _ in range(tokens))
        return self._fit(tokens, rng)

    def _fit(self, tokens: int, rng: random.Random) -> str:
        """guidellm's approach: oversized text, encoded, truncated, decoded."""
        pool = self.words or [" token", " text", " data", " model", " bench"]
        ids: List[int] = []
        attempt = 0
        while len(ids) < tokens:
            attempt += 1
            text = "".join(
                rng.choice(pool) for _ in range(int(tokens * 1.5 * attempt) + 8)
            )
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.decode(ids[:tokens], skip_special_tokens=True)

    def check(self, rows: Sequence[Row]) -> float:
        """Mean relative error between target and actual prompt length."""
        sample = list(rows[:LENGTH_CHECK_SAMPLES])
        if not sample:
            return 0.0
        errors = [
            abs(self.count(r.prompt) - r.prompt_tokens) / max(1, r.prompt_tokens)
            for r in sample
        ]
        return sum(errors) / len(errors)


def _prefix_choices(
    builder: TextBuilder, buckets: Sequence[PrefixBucket], rng: random.Random
) -> List[str]:
    """guidellm's prefix pool: each bucket's prefixes repeated so that drawing
    uniformly from the pool honours the bucket weights."""
    if not buckets:
        return [""]
    lcm = math.lcm(*(b.prefix_count for b in buckets))
    pool: List[str] = []
    for bucket in buckets:
        repeats = max(1, lcm * bucket.bucket_weight // bucket.prefix_count)
        prefixes = [
            builder.text(bucket.prefix_tokens, rng) for _ in range(bucket.prefix_count)
        ]
        pool.extend(prefixes * repeats)
    return pool


def random_rows(
    spec: RandomSpec, builder: TextBuilder, count: int, seed: int
) -> List[Row]:
    """``count`` synthetic requests, reproducible from ``seed``."""
    prompt_lengths = integer_sampler(
        spec.prompt_tokens,
        spec.prompt_tokens_stdev,
        spec.prompt_tokens_min,
        spec.prompt_tokens_max,
        seed,
    )
    output_lengths = integer_sampler(
        spec.output_tokens,
        spec.output_tokens_stdev,
        spec.output_tokens_min,
        spec.output_tokens_max,
        seed + 1,
    )
    rng = random.Random(seed + 3)
    prefixes = _prefix_choices(builder, spec.prefix_buckets, rng)

    rows: List[Row] = []
    for _ in range(count):
        prompt_tokens = max(1, next(prompt_lengths))
        output_tokens = next(output_lengths)
        prefix = rng.choice(prefixes)
        body = builder.text(prompt_tokens, rng)
        rows.append(
            Row(
                prompt=prefix + body,
                output_tokens=min(MAX_OUT_LEN, max(1, output_tokens)),
                prompt_tokens=prompt_tokens + (builder.count(prefix) if prefix else 0),
            )
        )
    return rows


def ensure_lengths(
    spec: RandomSpec, builder: TextBuilder, count: int, seed: int
) -> List[Row]:
    """Random rows whose prompt lengths are confirmed against the tokenizer.

    The word builder is checked on a sample; when it misses by more than the
    tolerance the rows are rebuilt with the slower per-row truncation, so a
    point never runs on prompts of a length nobody asked for.
    """
    rows = random_rows(spec, builder, count, seed)
    if builder.exact:
        error = builder.check(rows)
        if error > LENGTH_TOLERANCE:
            logger.warning(
                "Prompt lengths from whole-word tokens are off by %.1f%% for this "
                "tokenizer; building prompts by truncation instead.",
                error * 100,
            )
            builder.exact = False
            rows = random_rows(spec, builder, count, seed)
    return rows


# 鈹€鈹€ ShareGPT 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


def _first_turn(sample: Dict[str, Any]) -> Optional[tuple]:
    conversations = sample.get("conversations")
    if not conversations or len(conversations) < 2:
        return None
    first, second = conversations[0], conversations[1]
    if first.get("from") not in ("human", "user"):
        return None
    if second.get("from") not in ("gpt", "assistant"):
        return None
    prompt, completion = first.get("value"), second.get("value")
    if not prompt or not completion:
        return None
    return prompt, completion


class ShareGPTSource:
    """ShareGPT requests in file order (the order guidellm reads a file
    dataset in), filtered by prompt length.

    The file is parsed once and tokenized only as far as the largest point
    needs: a ramp asks for more rows at every step, and re-reading a 600 MB
    file per point would cost more than the point.
    """

    def __init__(
        self,
        path: str,
        builder: TextBuilder,
        input_min: Optional[int] = None,
        input_max: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ):
        if not os.path.exists(path):
            raise DatasetError(f"ShareGPT dataset not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            self._samples = json.load(f)
        self._builder = builder
        self._input_min = input_min
        self._input_max = input_max
        self._output_tokens = output_tokens
        self._cursor = 0
        self.rows: List[Row] = []

    def take(self, count: int) -> List[Row]:
        """The first ``count`` matching requests, or all of them when the
        file has fewer."""
        while len(self.rows) < count and self._cursor < len(self._samples):
            sample = self._samples[self._cursor]
            self._cursor += 1
            row = self._row(sample)
            if row is not None:
                self.rows.append(row)
        if not self.rows:
            raise DatasetError(
                "no ShareGPT conversation matches the configured prompt length range"
            )
        return self.rows[:count]

    def _row(self, sample: Dict[str, Any]) -> Optional[Row]:
        turn = _first_turn(sample)
        if turn is None:
            return None
        prompt, completion = turn
        prompt_tokens = self._builder.count(prompt)
        if self._input_min is not None and prompt_tokens < self._input_min:
            return None
        if self._input_max is not None and prompt_tokens > self._input_max:
            return None
        out = (
            self._output_tokens
            if self._output_tokens is not None
            else self._builder.count(completion)
        )
        if out < 1:
            return None
        return Row(
            prompt=prompt,
            output_tokens=min(MAX_OUT_LEN, out),
            prompt_tokens=prompt_tokens,
        )


def write_rows(path: str, rows: Sequence[Row]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.to_json())
            f.write("\n")


# ── QA JSONL ─────────────────────────────────────────────────────────────────

# A built-in dataset is named, not pathed: ``builtin:gsm8k`` is the copy that
# ships inside this package, ``builtin:sharegpt`` the data the runner image
# already carries. Naming them keeps a benchmark row portable across images and
# hosts, where an absolute path would not be.
BUILTIN_PREFIX = "builtin:"

GSM8K_BUILTIN = "gsm8k"
SHAREGPT_BUILTIN = "sharegpt"

_SHAREGPT_DEFAULT_PATH = (
    "/workspace/benchmark-runner/sharegpt_data/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)

DATA_FORMAT_RANDOM = "random"
DATA_FORMAT_SHAREGPT = "sharegpt"
DATA_FORMAT_QA = "qa_jsonl"

# A QA JSONL row may spell its two fields any of these ways; GSM8K uses the
# first of each.
_QUESTION_FIELDS = ("question", "prompt", "input")
_ANSWER_FIELDS = ("answer", "reference", "output")


def builtin_gsm8k_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k.jsonl"
    )


def resolve_data_path(value: str) -> str:
    """The file a ``--data`` value names, resolving the built-in aliases."""
    text = (value or "").strip()
    if not text.startswith(BUILTIN_PREFIX):
        return text
    name = text[len(BUILTIN_PREFIX) :].strip().lower()
    if name == GSM8K_BUILTIN:
        path = builtin_gsm8k_path()
        if not os.path.exists(path):
            raise DatasetError(
                "the built-in GSM8K dataset is missing from this runner image "
                f"({path}); rebuild aisbench-runner"
            )
        return path
    if name == SHAREGPT_BUILTIN:
        directory = os.environ.get(
            "BENCHMARK_SHAREGPT_DIR", os.path.dirname(_SHAREGPT_DEFAULT_PATH)
        )
        return os.path.join(directory, os.path.basename(_SHAREGPT_DEFAULT_PATH))
    raise DatasetError(f"unknown built-in dataset '{text}'")


def detect_data_format(value: str, declared: Optional[str] = None) -> str:
    """Which family a ``--data`` value belongs to.

    A declared format wins; the fallbacks exist so hand-written command lines
    (and rows that predate the format column) keep working.
    """
    if declared and declared != "auto":
        return declared
    text = (value or "").strip()
    if text.startswith(BUILTIN_PREFIX):
        name = text[len(BUILTIN_PREFIX) :].strip().lower()
        if name == SHAREGPT_BUILTIN:
            return DATA_FORMAT_SHAREGPT
        return DATA_FORMAT_QA
    if text.startswith("{") or "prompt_tokens=" in text:
        return DATA_FORMAT_RANDOM
    return DATA_FORMAT_QA if text.endswith(".jsonl") else DATA_FORMAT_SHAREGPT


def _first_field(sample: Dict[str, Any], names: Sequence[str]) -> Optional[str]:
    for name in names:
        value = sample.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return None


@dataclass
class QAItem:
    """One question and the reference answer it should produce."""

    question: str
    answer: str


def load_qa_items(path: str, limit: Optional[int] = None) -> List[QAItem]:
    """Every question of a QA JSONL file that also carries a reference answer.

    Rows without an answer cannot be scored, so they are left out rather than
    counted wrong: a mixed file still measures its answerable part.
    """
    if not os.path.exists(path):
        raise DatasetError(f"dataset not found at {path}")
    items: List[QAItem] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(sample, dict):
                continue
            question = _first_field(sample, _QUESTION_FIELDS)
            answer = _first_field(sample, _ANSWER_FIELDS)
            if question is None or answer is None:
                continue
            items.append(QAItem(question=question, answer=answer))
            if limit is not None and len(items) >= limit:
                break
    if not items:
        raise DatasetError(
            f"no question/answer pairs in {path}; an accuracy run needs both "
            "fields on every row"
        )
    return items


class QASource:
    """QA JSONL requests in file order, the shape GSM8K ships in.

    The rows are the questions; a performance run only needs those. The file is
    read once per run, which is also when its lengths are first needed.
    """

    def __init__(
        self,
        path: str,
        builder: TextBuilder,
        output_tokens: Optional[int] = None,
        input_min: Optional[int] = None,
        input_max: Optional[int] = None,
    ):
        if not os.path.exists(path):
            raise DatasetError(f"dataset not found at {path}")
        self._builder = builder
        self._output_tokens = output_tokens
        self._input_min = input_min
        self._input_max = input_max
        self.rows: List[Row] = []
        self._load(path)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(sample, dict):
                    continue
                question = _first_field(sample, _QUESTION_FIELDS)
                if question is None:
                    continue
                prompt_tokens = self._builder.count(question)
                if self._input_min is not None and prompt_tokens < self._input_min:
                    continue
                if self._input_max is not None and prompt_tokens > self._input_max:
                    continue
                answer = _first_field(sample, _ANSWER_FIELDS)
                if self._output_tokens is not None:
                    output_tokens = self._output_tokens
                elif answer is not None:
                    output_tokens = self._builder.count(answer)
                else:
                    output_tokens = 256
                self.rows.append(
                    Row(
                        prompt=question,
                        output_tokens=min(MAX_OUT_LEN, max(1, output_tokens)),
                        prompt_tokens=prompt_tokens,
                    )
                )
        if not self.rows:
            raise DatasetError(
                f"no usable questions in {path} within the configured prompt "
                "length range"
            )

    def take(self, count: int) -> List[Row]:
        """The first ``count`` questions, or every one the file holds."""
        return self.rows[:count]

    def tokens_per_row(self) -> int:
        """Average prompt+output tokens of one row, for the dataset budget."""
        prompts = sum(r.prompt_tokens for r in self.rows)
        outputs = sum(r.output_tokens for r in self.rows)
        return max(1, (prompts + outputs) // len(self.rows))


# ── composed prompts ─────────────────────────────────────────────────────────

# Tokens inserted between the shared prefix and the rest of a prompt. They are
# what makes each request differ: without them every request carrying the same
# prefix would be byte-identical, and a server may deduplicate or merge them —
# the measurement would then be of the dedup, not of the prefix cache.
UNIQUE_TOKENS = 3


def load_corpus_texts(path: str) -> List[str]:
    """The prompt texts of a QA dataset, as material to compose prompts from."""
    return [item.question for item in load_qa_items(path)]


def _fit_text(builder: TextBuilder, text: str, target: int) -> str:
    """Trim or repeat ``text`` until it is about ``target`` tokens.

    Exact equality is not achievable: encode→decode is lossy for rare tokens, so
    the result lands within a token or two of the target. The measured length is
    what the run reports, so the small deviation is visible rather than hidden.
    """
    if target <= 0:
        return ""
    tokenizer = builder.tokenizer
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return ""
    if len(ids) > target:
        ids = ids[:target]
    else:
        repeats = (target + len(ids) - 1) // len(ids)
        ids = (ids * repeats)[:target]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _unique_tokens(tokenizer: Any, rng: random.Random, count: int) -> str:
    """``count`` distinct random tokens, decoded as one piece of text.

    Decoded together rather than one id at a time: a tokenizer's spacing lives
    in the tokens themselves, and concatenating individually decoded pieces can
    merge into a different token at the seam.
    """
    vocab = len(tokenizer)
    ids: List[int] = []
    seen: set = set()
    guard = 0
    while len(ids) < count and guard < count * 50:
        guard += 1
        token_id = rng.randrange(vocab)
        if token_id in seen:
            continue
        seen.add(token_id)
        ids.append(token_id)
    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except Exception:  # noqa: BLE001 - a broken vocabulary entry is not fatal
        return ""


class ComposedSource:
    """Prompts of a chosen length, assembled out of a corpus.

    A real corpus has whatever lengths it happens to have, so it cannot drive a
    32k-context run; and a prefix-cache run needs requests that share their head
    and differ after it. Both are built here from the corpus's own text:

    * the prefix is corpus text repeated to the bucket's prefix length,
    * ``UNIQUE_TOKENS`` random tokens differ per request,
    * the rest is corpus text fitted to the remaining length.

    The length distribution reuses the Random dataset's knobs (mean / stdev /
    min / max), so a run configured for synthetic prompts keeps its shape here.
    """

    def __init__(
        self,
        corpus: List[str],
        builder: TextBuilder,
        spec: RandomSpec,
        seed: int,
    ):
        self._corpus = [text for text in corpus if text.strip()]
        if not self._corpus:
            raise DatasetError("the dataset carries no prompt text to compose from")
        self._builder = builder
        self._spec = spec
        self._rng = random.Random(seed)
        self._lengths = integer_sampler(
            spec.prompt_tokens,
            spec.prompt_tokens_stdev,
            spec.prompt_tokens_min,
            spec.prompt_tokens_max,
            seed,
        )
        self._rows: List[Row] = []
        # One prefix pool per bucket, built once: the pool is what makes a
        # controlled share of requests share the same head.
        self._pools: List[List[str]] = []
        for bucket_index, bucket in enumerate(spec.prefix_buckets):
            pool = []
            for i in range(max(1, bucket.prefix_count)):
                # Offset by the bucket index so two buckets are two different
                # heads; within a bucket, `i` gives the pool its distinct heads.
                source = self._corpus[
                    (bucket_index * 101 + i * 7919) % len(self._corpus)
                ]
                pool.append(_fit_text(builder, source, bucket.prefix_tokens))
            self._pools.append(pool)
        self._cursor = 0

    def take(self, count: int) -> List[Row]:
        while len(self._rows) < count:
            self._rows.append(self._build(len(self._rows)))
        return self._rows[:count]

    def _pick_bucket(self) -> Optional[int]:
        buckets = self._spec.prefix_buckets
        if not buckets:
            return None
        weights = [max(0, b.bucket_weight) for b in buckets]
        total = sum(weights) or len(buckets)
        point = self._rng.uniform(0, total)
        running = 0.0
        for index, weight in enumerate(weights):
            running += weight
            if point <= running:
                return index
        return len(buckets) - 1

    def _build(self, index: int) -> Row:
        target = max(1, next(self._lengths))
        bucket_index = self._pick_bucket()
        prefix = ""
        prefix_len = 0
        if bucket_index is not None:
            bucket = self._spec.prefix_buckets[bucket_index]
            prefix_len = min(
                bucket.prefix_tokens, max(0, target - UNIQUE_TOKENS)
            )
            if prefix_len > 0:
                pool = self._pools[bucket_index]
                base = pool[index % len(pool)]
                prefix = (
                    base
                    if prefix_len == bucket.prefix_tokens
                    else _fit_text(self._builder, base, prefix_len)
                )
        unique = ""
        if prefix:
            unique = _unique_tokens(
                self._builder.tokenizer, self._rng, UNIQUE_TOKENS
            )
        suffix_len = max(0, target - prefix_len - (UNIQUE_TOKENS if prefix else 0))
        suffix = ""
        if suffix_len > 0:
            source = self._corpus[(index * 37) % len(self._corpus)]
            suffix = _fit_text(self._builder, source, suffix_len)
        prompt = prefix + unique + suffix
        return Row(
            prompt=prompt,
            output_tokens=max(1, self._spec.output_tokens),
            prompt_tokens=self._builder.count(prompt),
        )

    def tokens_per_row(self) -> int:
        return max(1, self._spec.prompt_tokens + self._spec.output_tokens)
