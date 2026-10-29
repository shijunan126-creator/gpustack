import json
import os

import pytest

from aisbench_runner.datasets import (
    DatasetError,
    PrefixBucket,
    RandomSpec,
    ShareGPTSource,
    TextBuilder,
    ensure_lengths,
    integer_sampler,
    parse_random_spec,
    random_rows,
)

TOKENIZER = os.environ.get("AISBENCH_RUNNER_TEST_TOKENIZER", "/tmp/tok")


@pytest.fixture(scope="module")
def builder():
    if not os.path.exists(os.path.join(TOKENIZER, "tokenizer.json")):
        pytest.skip(f"no tokenizer at {TOKENIZER}")
    from transformers import AutoTokenizer

    return TextBuilder(AutoTokenizer.from_pretrained(TOKENIZER))


def test_parse_the_flat_form_gpustack_sends():
    spec = parse_random_spec(
        "prompt_tokens=1024,output_tokens=128,prompt_tokens_stdev=10,output_tokens_max=200"
    )
    assert spec.prompt_tokens == 1024
    assert spec.output_tokens == 128
    assert spec.prompt_tokens_stdev == 10
    assert spec.output_tokens_max == 200


def test_parse_the_json_form_with_prefix_buckets():
    spec = parse_random_spec(
        json.dumps(
            {
                "prompt_tokens": 512,
                "output_tokens": 64,
                "prefix_buckets": [
                    {"prefix_tokens": 256, "prefix_count": 2, "bucket_weight": 70},
                    {"prefix_tokens": 0, "prefix_count": 1, "bucket_weight": 30},
                ],
            }
        )
    )
    assert spec.prefix_buckets == [PrefixBucket(256, 2, 70)]


@pytest.mark.parametrize(
    "data, message",
    [
        ("output_tokens=10", "prompt_tokens"),
        ("prompt_tokens=10", "output_tokens"),
        ("prompt_tokens=10,output_tokens=5,turns=2", "multi-turn"),
        ("prompt_tokens=10,output_tokens=5,bogus=1", "bogus"),
    ],
)
def test_parse_refuses_what_it_cannot_honour(data, message):
    with pytest.raises(DatasetError, match=message):
        parse_random_spec(data)


def test_sampler_matches_guidellms_shapes():
    assert {next(integer_sampler(100, None, None, None, 1)) for _ in range(5)} == {100}
    uniform = integer_sampler(100, None, 50, 150, 1)
    values = [next(uniform) for _ in range(500)]
    assert min(values) >= 50 and max(values) <= 150 and len(set(values)) > 50
    gauss = integer_sampler(100, 10, None, None, 1)
    values = [next(gauss) for _ in range(2000)]
    assert 98 < sum(values) / len(values) < 102
    assert min(values) >= 50 and max(values) <= 150


@pytest.mark.parametrize("length", [1, 128, 1024, 8192])
def test_prompts_have_the_requested_token_count(builder, length):
    spec = RandomSpec(prompt_tokens=length, output_tokens=32)
    rows = ensure_lengths(spec, builder, 8, seed=7)
    for row in rows:
        assert builder.count(row.prompt) == length
        assert row.prompt_tokens == length
        assert row.output_tokens == 32


def test_the_seed_decides_the_prompts(builder):
    spec = RandomSpec(prompt_tokens=64, output_tokens=8)
    a = [r.prompt for r in random_rows(spec, builder, 4, seed=1)]
    b = [r.prompt for r in random_rows(spec, builder, 4, seed=1)]
    c = [r.prompt for r in random_rows(spec, builder, 4, seed=2)]
    assert a == b
    assert a != c


def test_prefix_buckets_share_prefixes(builder):
    spec = RandomSpec(
        prompt_tokens=64,
        output_tokens=8,
        prefix_buckets=[
            PrefixBucket(prefix_tokens=128, prefix_count=2, bucket_weight=100)
        ],
    )
    rows = random_rows(spec, builder, 40, seed=3)
    heads = {r.prompt[:200] for r in rows}
    assert len(heads) == 2
    for row in rows:
        assert abs(builder.count(row.prompt) - 192) <= 2


def test_sharegpt_filters_by_prompt_length(builder, tmp_path):
    convo = lambda q, a: {  # noqa: E731
        "conversations": [{"from": "human", "value": q}, {"from": "gpt", "value": a}]
    }
    data = [
        convo("short", "ok"),
        convo("a much longer question " * 20, "answer " * 30),
        {"conversations": [{"from": "gpt", "value": "no human turn"}]},
        convo("another long question here " * 20, "reply " * 10),
    ]
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(data))
    source = ShareGPTSource(str(path), builder, input_min=50)
    rows = source.take(10)
    assert len(rows) == 2
    assert all(r.prompt_tokens >= 50 for r in rows)
    fixed = ShareGPTSource(str(path), builder, output_tokens=16).take(1)
    assert fixed[0].output_tokens == 16
    with pytest.raises(DatasetError):
        ShareGPTSource(str(path), builder, input_min=100000).take(1)
