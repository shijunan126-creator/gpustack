"""Benchmark prompts, written as the rows of an aisbench ``CustomDataset``.

Every point runs on a dataset this module writes: one JSON line per request,
``{"question": <prompt>, "answer": "none", "max_out_len": <output tokens>}``.
aisbench honours a per-row ``max_out_len`` over the model's, and with
``ignore_eos`` that fixes each request's output length, so a point measures
exactly the context the benchmark asked for.

Two sources:

* **Random** — GPUStack's ``--data prompt_tokens=...,output_tokens=...``. The
  lengths are drawn the way guidellm draws them (``IntegerRangeSampler``:
  constant, uniform between min and max, or a clipped Gaussian when a standard
  deviation is given), and ``prefix_buckets`` prepend shared prefixes the same
  way, so a benchmark configured for guidellm means the same thing here.
* **ShareGPT** — the first human/assistant turn of each conversation, filtered
  by prompt length; the output length is the reference answer's unless a
  fixed maximum is configured.

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


# ── ShareGPT ─────────────────────────────────────────────────────────────────


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
