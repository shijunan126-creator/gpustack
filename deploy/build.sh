#!/usr/bin/env bash
# Build the fork's image from this repository.
#
#   deploy/build.sh [image-tag] [extra docker build args...]
#
# The default tag carries the commit, so every build is traceable and a rollback
# is just running the previous tag.
set -euo pipefail

cd "$(dirname "$0")/.."

commit="$(git rev-parse --short HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "warning: working tree has uncommitted changes; the image will not match ${commit}" >&2
  commit="${commit}-dirty"
fi
tag="${1:-gpustack-custom:v2.3.0-${commit}}"
shift || true

docker build -f deploy/Dockerfile --build-arg "FORK_COMMIT=${commit}" -t "${tag}" "$@" .
echo "built ${tag}"
