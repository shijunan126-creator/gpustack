#!/usr/bin/env bash
# Build the fork's image from this repository.
#
#   deploy/build.sh [image-tag] [extra docker build args...]
#
# The default tag carries the commit, so every build is traceable and a rollback
# is just running the previous tag.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "${script_dir}/.."

commit="$(git rev-parse --short HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "warning: working tree has uncommitted changes; the image will not match ${commit}" >&2
  commit="${commit}-dirty"
fi
tag="${1:-gpustack-custom:v2.3.0-${commit}}"
shift || true

docker build -f deploy/Dockerfile --build-arg "FORK_COMMIT=${commit}" -t "${tag}" "$@" .

# A worker finds its own container for mirrored deployment by matching the OCI
# labels the base image was built with (their values are baked into the image as
# GPUSTACK_RUNTIME_DOCKER_MIRRORED_NAME_FILTER_LABELS). Rewriting one of them in
# deploy/Dockerfile silently costs every inference container its GPU, so the
# invariant is checked against the built image rather than trusted.
"${script_dir}/check-image-labels.sh" "${tag}"

echo "built ${tag}"
