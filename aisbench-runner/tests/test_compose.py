"""Composing prompts of a chosen length (and prefix) out of a corpus."""

import json

import pytest

from aisbench_runner.datasets import (
    ComposedSource,
    PrefixBucket,
    RandomSpec,
    TextBuilder,
    load_corpus_texts,
)

WORDS = [f"w{i}" for i in range(2000)]
INDEX = {word: i for i, word in enumerate(WORDS)}


class _FakeTokenizer:
    """Word-level tokenizer: one token per space-separated word."""

    def __len__(self):
        return len(WORDS)

    def encode(self, text, add_special_tokens=False):
        return [INDEX[word] for word in text.split() if word in INDEX]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(WORDS[i] for i in ids)


@pytest.fixture()
def builder():
    return TextBuilder(_FakeTokenizer())


def _corpus(tmp_path, rows=6):
    path = tmp_path / "qa.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(rows):
            words = " ".join(f"w{j}" for j in range(i * 40, i * 40 + 40))
            f.write(json.dumps({"question": words, "answer": "#### 1"}) + "\n")
    return load_corpus_texts(str(path))


def _source(builder, corpus, **kwargs):
    spec = RandomSpec(
        prompt_tokens=kwargs.pop("prompt_tokens", 50),
        output_tokens=kwargs.pop("output_tokens", 8),
        **kwargs,
    )
    return ComposedSource(corpus, builder, spec, seed=7)


def test_rows_are_composed_to_the_requested_length(builder, tmp_path):
    source = _source(builder, _corpus(tmp_path))
    rows = source.take(5)
    assert len(rows) == 5
    for row in rows:
        # decode/encode is lossless for this tokenizer, so the length is exact.
        assert row.prompt_tokens == 50
        assert row.output_tokens == 8


def test_a_shared_prefix_is_actually_shared(builder, tmp_path):
    source = _source(
        builder,
        _corpus(tmp_path),
        prompt_tokens=60,
        prefix_buckets=[PrefixBucket(prefix_tokens=20, prefix_count=1)],
    )
    rows = source.take(4)
    heads = {" ".join(row.prompt.split()[:20]) for row in rows}
    assert len(heads) == 1, "the prefix must be the same text in every request"
    # ... while the requests themselves differ, so a server cannot deduplicate
    # them and claim the prefix cache was hit.
    assert len({row.prompt for row in rows}) == len(rows)


def test_a_second_prefix_bucket_gives_a_second_head(builder, tmp_path):
    source = _source(
        builder,
        _corpus(tmp_path),
        prompt_tokens=60,
        prefix_buckets=[
            PrefixBucket(prefix_tokens=20, prefix_count=1, bucket_weight=1),
            PrefixBucket(prefix_tokens=20, prefix_count=1, bucket_weight=1),
        ],
    )
    heads = {" ".join(row.prompt.split()[:20]) for row in source.take(12)}
    assert len(heads) == 2


def test_the_length_distribution_is_honoured(builder, tmp_path):
    source = _source(
        builder,
        _corpus(tmp_path),
        prompt_tokens=100,
        prompt_tokens_min=20,
        prompt_tokens_max=30,
    )
    lengths = [row.prompt_tokens for row in source.take(8)]
    assert all(20 <= length <= 30 for length in lengths)
    assert len(set(lengths)) > 1


def test_a_corpus_without_text_is_refused(builder, tmp_path):
    from aisbench_runner.datasets import DatasetError

    with pytest.raises(DatasetError):
        ComposedSource([], builder, RandomSpec(prompt_tokens=10), seed=1)
