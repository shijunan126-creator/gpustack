# aisbench-runner

The benchmark container GPUStack starts for every run on the Benchmarks page,
with each measured point executed by [AISBench](https://github.com/AISBench/benchmark)
instead of guidellm.

It is a drop-in replacement for `gpustack/benchmark-runner`:

- **Same command line.** The worker builds `benchmark-runner benchmark run ...`
  (`gpustack/worker/benchmark/runner.py`); every option it sends is accepted.
- **Same output files.** `{id}__p{i}.json` per ramp point, `{id}__satprobe.json`,
  `{id}__ramp.json`, `{id}__stage{i}.json`, `{id}__curve.json` and `{id}.json`,
  in the guidellm report shape the worker loads with
  `GenerativeBenchmarksReport.load_file`. The server, the analysis and the UI
  do not know which engine measured a point.
- **Same load search.** Both presets (Max Throughput, Meet Latency SLO) run
  benchmark-runner's own adaptive ramp, vendored in `ramp.py` with its tests.
  Only measuring a point changed.

## How a point is measured

1. `datasets.py` writes the requests as an aisbench `CustomDataset`: one row
   per request with an exact prompt length and a per-row `max_out_len`
   (`ignore_eos` holds the output to it). Random prompts follow guidellm's
   length distributions and shared-prefix buckets; ShareGPT is filtered by
   prompt length.
2. `engine.py` writes a self-contained aisbench config and runs
   `ais_bench <config> -m perf` at a fixed concurrency (`batch_size`) or
   request rate (`request_rate`).
3. The per-request timings aisbench records are read back and summarized in
   `report.py`: TTFT, decode-only TPOT, end-to-end latency, chunk gaps,
   throughputs, all with mean / median / p25..p99.

## aisbench behaviour this depends on

Found by running it; each is handled in `engine.py`.

| Behaviour | Handling |
|---|---|
| `retry` counts attempts: `retry=0` sends nothing and fails every request with an empty reason | points run with `retry=1` |
| SIGINT is ignored; a run cannot be stopped early | duration caps kill the process group; requests finished before that are read from `tmp/` |
| A space in the config path breaks the inference task's command line; the parent then waits forever | work directories are named without spaces |
| After a failed (or occasionally a finished) task the parent can hang | a logged `task failed` stops the run at once; a run whose requests are all recorded gets 60 s to exit |
| Refuses to start when host memory use + dataset exceeds 80% | threshold made configurable in the image (`AISBENCH_MEMORY_THRESHOLD_PERCENT`, 95) |
| Exit status says nothing about whether anything was measured | records are read regardless; aisbench's failure summary is what a failed point reports |

## Not supported

Multi-turn conversations (`turns > 1`, refused by the server), and the
in-flight constraints guidellm enforces (`--max-errors`, `--max-error-rate`,
`--detect-saturation`), which are logged and ignored.

## Build and test

```bash
deploy/build-benchmark-runner.sh          # aisbench-runner:<__aisbench_runner_version__>
cd aisbench-runner && PYTHONPATH=.:.. pytest tests
```

The dataset tests need a tokenizer directory (`AISBENCH_RUNNER_TEST_TOKENIZER`,
default `/tmp/tok`) and are skipped without one.
