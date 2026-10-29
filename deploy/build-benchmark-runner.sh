#!/usr/bin/env bash
# Build the benchmark runner image (aisbench-runner/) GPUStack starts for every
# benchmark.
#
#   deploy/build-benchmark-runner.sh [extra docker build args...]
#
# The tag is the one the worker asks for by default, aisbench-runner:<version>
# with the version from gpustack/__init__.py, so a worker that has the image
# loaded needs no configuration. Pass --platform linux/arm64 for Kunpeng hosts.
set -euo pipefail

cd "$(dirname "$0")/.."

version="$(sed -n "s/^__aisbench_runner_version__ = '\(.*\)'$/\1/p" gpustack/__init__.py)"
if [[ -z "${version}" ]]; then
  echo "error: __aisbench_runner_version__ not found in gpustack/__init__.py" >&2
  exit 1
fi
commit="$(git rev-parse --short HEAD)"
if [[ -n "$(git status --porcelain -- aisbench-runner)" ]]; then
  commit="${commit}-dirty"
fi
tag="aisbench-runner:${version}"

docker build -f aisbench-runner/Dockerfile --build-arg "RUNNER_COMMIT=${commit}" -t "${tag}" "$@" .
echo "built ${tag}"
