## What changed

Three benchmark features that belong together, plus the two knobs the baseline
preset needs:

**1. Datasets are rows, not constants**

A new `benchmark_datasets` table holds what a run measures. Three built-ins are
seeded at startup (`GSM8K`, `ShareGPT`, `Random`), and a corpus can be uploaded
from the UI: `GET/POST/DELETE /v2/benchmark-datasets` and
`GET /v2/benchmark-datasets/{id}/content` (the worker downloads the bytes and
writes them into the directory the container already mounts). Uploads are
validated at the boundary — format, row count, whether the rows carry answers.

A benchmark now records the dataset it measured (`dataset_id`, `dataset_format`,
`dataset_sha256`) and freezes a `snapshot.dataset` block, so a report survives
the dataset being edited or deleted.

**2. GSM8K is the default corpus, and it is composed, not used verbatim**

GSM8K's own questions are short; a run needs a *chosen* length and, for cache
work, a shared prefix. Selecting a QA dataset now composes prompts out of it:

```
prompt = prefix(prefix_tokens, from the corpus)
       + 3 unique random tokens
       + suffix(corpus text, fitted to input_len - prefix - 3)
```

The three unique tokens are what keep requests distinct, so the measurement is
of the server's prefix cache and not of request deduplication. The length
distribution reuses the Random dataset's knobs (mean / stdev / min / max) and
the shared-prefix buckets reuse the existing UI.

**3. Accuracy runs**

`kind=accuracy` runs one pass over a question/answer dataset: ask, extract the
final number, compare with the reference (`#### N`), store one row per question
(`benchmark_accuracy_samples`) and a score on the parent. Prompt wording and the
comparison rule are copied from AISBench's own GSM8K evaluator, so a score here
is comparable with one from the aisbench CLI.

**4. Baseline preset and the two knobs it needs**

`Baseline` is now the default preset: 8 concurrent requests, 100 prompts, 512
output tokens, 1024-token composed input, temperature 0.01, one warmup request —
the aisbench gateway's own defaults. New `temperature` column, and a stage may
carry a `request_rate` cap alongside its concurrency (aisbench takes both).

The create form renders a fixed-load preset as a **Fixed load** card (concurrency
/ requests / rate cap) rather than a stage sweep; auto-tune presets keep the
ramp card.

## Why

`dataset_name` used to be one of two hard-coded strings, so adding a corpus meant
a code change, and GSM8K could only ever be 1,319 short prompts. Composition
makes the corpus material for long-context and prefix-cache runs, which is what
the baseline numbers are actually compared on.

## Verification

- Server: 237 benchmark-related tests pass; new tests for dataset upload parsing.
- Runner: 42 tests pass, including new composition tests (exact length, shared
  prefix, two buckets, distribution) and accuracy scoring/extraction.
- Live, against a running Qwen3.8-27B deployment on 2× Ascend 910B2:
  - composed prompts of 2048 tokens with a 512-token shared head: both prompts
    exactly 2048 tokens, first 512 identical, rows differ afterwards;
  - accuracy run over 8 GSM8K questions: 7/8 = 87.5%, per-question rows stored;
  - composed baseline with `stages=[{rate:2,max_requests:4,request_rate:1}]` and
    `temperature=0.01`: 4/4 successful, runner log `concurrent 2 streams @ 1 req/s`.

## Migration

Two revisions, both guarded and reversible:

- `c1f2a3b4d5e6` — `benchmark_datasets`, `benchmark_accuracy_samples`, and the
  run's kind / dataset / accuracy columns;
- `d2e3f4a5b6c7` — `benchmarks.temperature`.

Existing runs stay performance runs (`kind` defaults to `perf`).

## Notes

- The runner image is bumped to `v0.1.3` (new `benchmark accuracy` subcommand and
  the composer); a worker builds it from the image's build context as before.
- `sharegpt` datasets are not composed yet — only `qa_jsonl` corpora are. The
  reference project only supports GSM8K for the same reason: every other corpus
  spells its text fields and layout differently.
