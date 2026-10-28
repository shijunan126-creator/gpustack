# Image for this fork: the official GPUStack image with the fork's backend
# sources and frontend build laid over it. Build from the repository root:
#
#   docker build -f deploy/Dockerfile -t gpustack-custom:<tag> .
#
# Everything the official build produces and the fork does not touch (bundled
# runtimes and tools, the offline docs site, the packaged Helm chart, Python
# dependencies) is taken from the base image as is. That keeps the image small
# to build, but it also means a change to dependencies in pyproject.toml is NOT
# picked up here; see deploy/README.md.
ARG GPUSTACK_VERSION=v2.3.0
ARG NODE_IMAGE=node:22-slim

# --- frontend ---------------------------------------------------------------
FROM ${NODE_IMAGE} AS ui
# Override for a mirror, e.g. https://registry.npmmirror.com
ARG NPM_REGISTRY=https://registry.npmjs.org
ENV HUSKY=0 CI=true COREPACK_NPM_REGISTRY=${NPM_REGISTRY}
RUN corepack enable
WORKDIR /ui
COPY ui/ ./
RUN pnpm install --frozen-lockfile --registry "${NPM_REGISTRY}" \
    && pnpm build \
    && test -f dist/index.html

# --- final image ------------------------------------------------------------
FROM gpustack/gpustack:${GPUSTACK_VERSION}

ARG PKG=/usr/local/lib/python3.11/dist-packages/gpustack
# Recorded on the image to tell which commit of this repository it was built from.
ARG FORK_COMMIT=unknown
LABEL org.opencontainers.image.revision="${FORK_COMMIT}"

# The release build stamps the version into gpustack/__init__.py; the repository
# holds a placeholder. Keep the stamp of the base image.
RUN grep -E "^__(version|git_commit)__ = " "${PKG}/__init__.py" > /tmp/version-stamp

# Backend: make the package's Python sources match the repository exactly, so a
# file the fork deleted does not linger from the base image.
RUN find "${PKG}" -type f \( -name '*.py' -o -name '*.pyc' \) -not -path "${PKG}/ui/*" -delete
COPY gpustack/ ${PKG}/
RUN sed -i -E "/^__(version|git_commit)__ = /d" "${PKG}/__init__.py" \
    && cat /tmp/version-stamp "${PKG}/__init__.py" > /tmp/init.py \
    && mv /tmp/init.py "${PKG}/__init__.py" && rm /tmp/version-stamp

# Frontend: the UI directory is replaced wholesale. The backend ships its own
# static files into ui/static (catalog icons from the repository's static/, the
# packaged Helm chart from the base image); the frontend build has neither.
RUN mkdir /tmp/charts && cp -a "${PKG}/ui/static/charts/." /tmp/charts/ \
    && rm -rf "${PKG}/ui"
COPY --from=ui /ui/dist ${PKG}/ui
COPY static/ ${PKG}/ui/static/
RUN mkdir -p "${PKG}/ui/static/charts" \
    && cp -a /tmp/charts/. "${PKG}/ui/static/charts/" && rm -rf /tmp/charts

# Fail the build rather than the server: compile, resolve every import (lazy
# ones included), and check the static files the frontend relies on.
COPY hack/check_imports.py /tmp/check_imports.py
RUN python -m compileall -q "${PKG}" \
    && cd "$(dirname "${PKG}")" && python /tmp/check_imports.py \
    && rm /tmp/check_imports.py \
    && test "$(ls ${PKG}/ui/static/catalog_icons | wc -l)" -ge 47 \
    && test -n "$(ls ${PKG}/ui/static/charts)" \
    && test -f "${PKG}/ui/index.html" \
    && python -c "import gpustack; assert gpustack.__version__ != '0.0.0', 'version stamp lost'"
