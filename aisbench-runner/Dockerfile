# GPUStack benchmark runner that measures with aisbench.
#
#   docker build -f aisbench-runner/Dockerfile -t gpustack-aisbench-runner:<tag> .
#
# Built from the repository root. GPUStack's worker starts the image with
# benchmark-runner's arguments (it sets none of the entrypoint), so the
# entrypoint has to be the runner itself.

# Huawei publishes the base for amd64 and arm64 (pick with --platform), in an
# Ubuntu and an openEuler flavour:
#   swr.cn-south-1.myhuaweicloud.com/ascendhub/aisbench_benchmark:v3.1-20260630-master-ubuntu22.04-py310
#   swr.cn-south-1.myhuaweicloud.com/ascendhub/aisbench_benchmark:v3.1-20260630-master-openeuler24.03-py311
# Only the Ubuntu one has been built and run here.
ARG AISBENCH_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/aisbench_benchmark:v3.1-20260630-master-ubuntu22.04-py310
FROM ${AISBENCH_IMAGE}

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

# aisbench refuses to start a run while host memory usage plus the dataset
# exceeds 80%. The check reads the whole host, and an inference server holding
# its weights in host memory routinely sits above that, so the threshold is
# made configurable and raised; the guard against a dataset that cannot fit
# stays. The edit is asserted so a changed aisbench fails the build here
# instead of benchmarks failing on a busy machine.
RUN f=/benchmark/ais_bench/benchmark/tasks/openicl_api_infer.py \
    && grep -q '^import os$' "$f" \
    && test "$(grep -c 'threshold_percent=80)' "$f")" = 1 \
    && sed -i 's/threshold_percent=80)/threshold_percent=int(os.environ.get("AISBENCH_MEMORY_THRESHOLD_PERCENT", "80")))/' "$f" \
    && grep -q 'AISBENCH_MEMORY_THRESHOLD_PERCENT' "$f"
ENV AISBENCH_MEMORY_THRESHOLD_PERCENT=95

# The ShareGPT dataset, at the path GPUStack sends by default
# (GPUSTACK_BENCHMARK_DATASET_SHAREGPT_PATH). Point SHAREGPT_URL at a mirror
# where huggingface.co is not reachable.
ARG SHAREGPT_URL=https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
RUN mkdir -p /workspace/benchmark-runner/sharegpt_data \
    && curl -fsSL --retry 3 -o /workspace/benchmark-runner/sharegpt_data/ShareGPT_V3_unfiltered_cleaned_split.json "${SHAREGPT_URL}" \
    && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert len(d) > 1000, len(d)" \
        /workspace/benchmark-runner/sharegpt_data/ShareGPT_V3_unfiltered_cleaned_split.json

# Installed by copying rather than with pip: the image's setuptools predates
# pyproject metadata, and the package has no dependency the image lacks.
COPY aisbench-runner/aisbench_runner /opt/aisbench-runner/aisbench_runner
COPY aisbench-runner/aisbench_runner_tls.pth /opt/aisbench-runner/
RUN site="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && cp -r /opt/aisbench-runner/aisbench_runner "${site}/" \
    && cp /opt/aisbench-runner/aisbench_runner_tls.pth "${site}/" \
    && printf '#!/usr/bin/env python3\nfrom aisbench_runner.cli import entrypoint\n\nentrypoint()\n' > /usr/local/bin/benchmark-runner \
    && chmod +x /usr/local/bin/benchmark-runner \
    && benchmark-runner benchmark run --help > /dev/null \
    && python3 -c "import aisbench_runner.engine, aisbench_runner.runner"

ARG RUNNER_COMMIT=unknown
LABEL io.github.shijunan126-creator.gpustack.commit="${RUNNER_COMMIT}"

WORKDIR /workspace
ENTRYPOINT ["benchmark-runner"]
CMD []
