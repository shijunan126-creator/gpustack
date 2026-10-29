#!/usr/bin/env bash
# Fail when an image's org.opencontainers.image.* labels no longer match the
# values baked into it as GPUSTACK_RUNTIME_DOCKER_MIRRORED_NAME_FILTER_LABELS.
#
#   deploy/check-image-labels.sh <image-tag>
#
# A worker finds its own container -- the one whose GPU runtime and devices it
# mirrors onto the inference containers it creates -- by matching those labels.
# deploy/build.sh runs this against the image it just built, because rewriting
# one label in deploy/Dockerfile silently leaves every inference container
# without a GPU.
set -euo pipefail

tag="${1:-}"
if [[ -z "${tag}" ]]; then
  echo "usage: $0 <image-tag>" >&2
  exit 2
fi

filter_labels="$(docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${tag}" \
  | grep -m1 '^GPUSTACK_RUNTIME_DOCKER_MIRRORED_NAME_FILTER_LABELS=' || true)"
if [[ -z "${filter_labels}" ]]; then
  echo "error: ${tag} carries no GPUSTACK_RUNTIME_DOCKER_MIRRORED_NAME_FILTER_LABELS; the worker could not find its own container" >&2
  exit 1
fi

IFS=';' read -r -a label_pairs <<< "${filter_labels#*=}"
for pair in "${label_pairs[@]}"; do
  if [[ -z "${pair}" ]]; then
    continue
  fi
  key="${pair%%=*}"
  expected="${pair#*=}"
  actual="$(docker image inspect --format "{{index .Config.Labels \"${key}\"}}" "${tag}")"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "error: label ${key} is '${actual}' but the image's mirrored-deployment filter expects '${expected}'." >&2
    echo "       deploy/Dockerfile must not rewrite the base image's org.opencontainers.* labels." >&2
    exit 1
  fi
done

echo "labels ok: ${tag}"
