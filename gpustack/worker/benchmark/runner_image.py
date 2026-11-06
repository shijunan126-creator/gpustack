"""Build the benchmark runner image on the worker's own Docker host.

The runner image (aisbench-runner/) is published to no registry. Rather than
asking every deployment to build and carry it by hand, the GPUStack image ships
its build context and the worker builds the image the first time it finds it
missing: in the background at startup, and again before a benchmark if an
earlier build failed. A benchmark that arrives while a build is under way waits
for it instead of failing on a pull that can never succeed.

Only the default image name is built. A worker pointed at its own image with
--benchmark-image-repo is left to pull it as before.
"""

import logging
import os
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Callable, Deque, Dict, Optional

import docker
from gpustack_runtime import envs as runtime_envs

from gpustack import __aisbench_runner_version__, envs
from gpustack.config import registration
from gpustack.config.config import Config
from gpustack.utils.config import apply_registry_override_to_image

logger = logging.getLogger(__name__)

DEFAULT_RUNNER_IMAGE = f"aisbench-runner:{__aisbench_runner_version__}"
RUNNER_DOCKERFILE = "aisbench-runner/Dockerfile"
# Written next to the build context when the GPUStack image is built: the
# commit the runner sources came from, recorded on the runner image's label.
RUNNER_COMMIT_FILE = "COMMIT"
# A failed build's message carries the last lines of its output; the full
# output is in the build log.
FAILURE_TAIL_LINES = 5
# Seconds a single read from the Docker API may block. A build step such as the
# dataset download prints nothing for minutes, which the client's 60-second
# default would take for a dead connection.
DOCKER_API_TIMEOUT = 3600


class RunnerImageState(str, Enum):
    UNMANAGED = "unmanaged"
    """The worker does not build this image; benchmarks pull it as configured."""
    PRESENT = "present"
    BUILDING = "building"
    FAILED = "failed"


def _docker_client() -> docker.DockerClient:
    environment = os.environ.copy()
    if runtime_envs.GPUSTACK_RUNTIME_DOCKER_HOST:
        environment["DOCKER_HOST"] = runtime_envs.GPUSTACK_RUNTIME_DOCKER_HOST
    return docker.from_env(environment=environment, timeout=DOCKER_API_TIMEOUT)


class RunnerImageBuilder:
    """Keeps the default benchmark runner image present on this worker.

    Thread-safe: `ensure` is called from the worker's startup and from the
    benchmark queue, and the build itself runs on a thread of its own.
    """

    def __init__(
        self,
        cfg: Config,
        build_log_path: str,
        client_factory: Callable[[], docker.DockerClient] = _docker_client,
        context_dir: Optional[str] = None,
    ):
        self._config = cfg
        self._build_log_path = build_log_path
        self._client_factory = client_factory
        self._context_dir = Path(context_dir or envs.BENCHMARK_RUNNER_BUILD_CONTEXT)
        self._lock = threading.Lock()
        self._state: Optional[RunnerImageState] = None
        self._error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def target_image(self) -> Optional[str]:
        """The image name benchmarks run, when this worker is the one to build it.

        Returns:
            The fully resolved image name, or None when auto build is off, the
            image was configured by hand, or the build context is not shipped.
        """
        if not envs.BENCHMARK_RUNNER_AUTO_BUILD:
            return None
        if self._config.benchmark_image_repo != DEFAULT_RUNNER_IMAGE:
            return None
        if not (self._context_dir / RUNNER_DOCKERFILE).is_file():
            return None
        # Resolved exactly the way the runner resolves it, so the tag built here
        # is the name the benchmark container asks for and finds locally.
        fallback_registry = registration.determine_default_registry(
            self._config.system_default_container_registry
        )
        return apply_registry_override_to_image(
            self._config, self._config.benchmark_image_repo, fallback_registry
        )

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def state(self) -> RunnerImageState:
        """The last known state, without touching Docker."""
        with self._lock:
            return self._state or RunnerImageState.UNMANAGED

    def ensure(self) -> RunnerImageState:
        """Start a build if the image is missing and none is under way.

        A failed build is retried by the next call, so a transient network
        failure costs one benchmark rather than every one until a restart.

        Returns:
            The state after the check: PRESENT, BUILDING, or UNMANAGED when this
            worker does not build the image or cannot reach Docker. Never
            FAILED -- a failure starts a new attempt.
        """
        image = self.target_image()
        if image is None:
            return self._set_state(RunnerImageState.UNMANAGED)

        with self._lock:
            if self._state == RunnerImageState.BUILDING:
                return RunnerImageState.BUILDING

        try:
            client = self._client_factory()
        except docker.errors.DockerException as e:
            logger.info(
                f"Not building benchmark runner image {image}: Docker is not "
                f"reachable ({e})"
            )
            return self._set_state(RunnerImageState.UNMANAGED)

        try:
            client.images.get(image)
            return self._set_state(RunnerImageState.PRESENT)
        except docker.errors.ImageNotFound:
            pass
        except docker.errors.APIError as e:
            logger.warning(f"Failed to look up benchmark runner image {image}: {e}")
            return self._set_state(RunnerImageState.UNMANAGED)

        with self._lock:
            if self._state == RunnerImageState.BUILDING:
                return RunnerImageState.BUILDING
            self._state = RunnerImageState.BUILDING
            self._error = None
            self._thread = threading.Thread(
                target=self._build,
                args=(client, image),
                name="benchmark-runner-image-build",
                daemon=True,
            )
            self._thread.start()
        return RunnerImageState.BUILDING

    def wait(self, timeout: Optional[float] = None) -> RunnerImageState:
        """Block until the current build, if any, ends."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return self.state()

    def _set_state(self, state: RunnerImageState) -> RunnerImageState:
        with self._lock:
            # A build under way owns the state until it ends.
            if self._state != RunnerImageState.BUILDING:
                self._state = state
            return self._state

    def build_args(self) -> Dict[str, str]:
        args = {}
        commit_file = self._context_dir / RUNNER_COMMIT_FILE
        if commit_file.is_file():
            commit = commit_file.read_text(encoding="utf-8").strip()
            if commit:
                args["RUNNER_COMMIT"] = commit
        if envs.BENCHMARK_RUNNER_BASE_IMAGE:
            args["AISBENCH_IMAGE"] = envs.BENCHMARK_RUNNER_BASE_IMAGE
        if envs.BENCHMARK_RUNNER_SHAREGPT_URL:
            args["SHAREGPT_URL"] = envs.BENCHMARK_RUNNER_SHAREGPT_URL
        return args

    def _build(self, client: docker.DockerClient, image: str) -> None:
        started = time.monotonic()
        tail: Deque[str] = deque(maxlen=FAILURE_TAIL_LINES)
        error: Optional[str] = None
        logger.info(
            f"Building benchmark runner image {image} from {self._context_dir}; "
            f"output goes to {self._build_log_path}"
        )
        try:
            os.makedirs(os.path.dirname(self._build_log_path), exist_ok=True)
            with open(self._build_log_path, "w", encoding="utf-8") as log:
                stream = client.api.build(
                    path=str(self._context_dir),
                    dockerfile=RUNNER_DOCKERFILE,
                    tag=image,
                    buildargs=self.build_args(),
                    rm=True,
                    forcerm=True,
                    decode=True,
                )
                for chunk in stream:
                    error = self._record_chunk(chunk, log, tail) or error
                    if error:
                        break
        except Exception as e:
            error = str(e)

        if error is None:
            # The daemon can end the stream without an error chunk and without
            # tagging; the image being there is what counts.
            try:
                client.images.get(image)
            except Exception as e:
                error = f"the build ended but {image} is not present ({e})"

        elapsed = int(time.monotonic() - started)
        with self._lock:
            if error is None:
                self._state = RunnerImageState.PRESENT
                self._error = None
            else:
                context = " | ".join(line for line in tail if line)
                self._state = RunnerImageState.FAILED
                self._error = f"{error}" + (
                    f" (last build output: {context})" if context else ""
                )
            message = self._error
        if error is None:
            logger.info(f"Built benchmark runner image {image} in {elapsed}s")
        else:
            logger.error(
                f"Failed to build benchmark runner image {image} after {elapsed}s: "
                f"{message}; see {self._build_log_path}"
            )

    @staticmethod
    def _record_chunk(chunk: dict, log, tail: Deque[str]) -> Optional[str]:
        """Write one build stream chunk to the log; return its error, if any."""
        if "error" in chunk:
            message = str(chunk.get("error") or "").strip()
            log.write(f"ERROR: {message}\n")
            log.flush()
            return message or "the build failed"
        if chunk.get("progress"):
            # A base image layer's download ticking along; its start and end
            # arrive as plain status lines.
            return None
        text = chunk.get("stream") or chunk.get("status") or ""
        if text:
            log.write(text if text.endswith("\n") else text + "\n")
            log.flush()
            for line in text.splitlines():
                if line.strip():
                    tail.append(line.strip())
        return None
