"""The worker keeps the benchmark runner image present by building it itself."""

import asyncio
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import docker
import pytest

import gpustack.worker.benchmark_manager as bm
from gpustack.schemas.benchmark import BenchmarkStateEnum
from gpustack.worker.benchmark import runner_image
from gpustack.worker.benchmark.runner_image import (
    DEFAULT_RUNNER_IMAGE,
    RunnerImageBuilder,
    RunnerImageState,
)
from gpustack.worker.benchmark_manager import BenchmarkManager


class _FakeImages:
    def __init__(self, present):
        self.present = set(present)

    def get(self, name):
        if name in self.present:
            return SimpleNamespace(tags=[name])
        raise docker.errors.ImageNotFound(name)


class _FakeAPI:
    """Docker's build endpoint: streams the given chunks, then tags the image
    unless one of them was an error. `gate` holds the build open."""

    def __init__(self, images, chunks, gate=None):
        self.images = images
        self.chunks = chunks
        self.gate = gate
        self.calls = []

    def build(self, **kwargs):
        self.calls.append(kwargs)
        return self._stream(kwargs["tag"])

    def _stream(self, tag):
        if self.gate is not None:
            assert self.gate.wait(5)
        yield from self.chunks
        if not any("error" in chunk for chunk in self.chunks):
            self.images.present.add(tag)


def _fake_client(present=(), chunks=None, gate=None):
    images = _FakeImages(present)
    api = _FakeAPI(images, chunks if chunks is not None else [{"stream": "ok\n"}], gate)
    return SimpleNamespace(images=images, api=api)


def _context(tmp_path: Path, commit="abc1234") -> Path:
    context = tmp_path / "context"
    (context / "aisbench-runner").mkdir(parents=True)
    (context / "aisbench-runner" / "Dockerfile").write_text("FROM x\n")
    (context / "COMMIT").write_text(f"{commit}\n")
    return context


def _builder(tmp_path, client, image_repo=DEFAULT_RUNNER_IMAGE, registry=None):
    cfg = SimpleNamespace(
        benchmark_image_repo=image_repo,
        system_default_container_registry=registry,
    )
    return RunnerImageBuilder(
        cfg,
        build_log_path=str(tmp_path / "logs" / "runner-image-build.log"),
        client_factory=lambda: client,
        context_dir=str(_context(tmp_path)),
    )


@pytest.fixture(autouse=True)
def _no_registry_probe(monkeypatch):
    # The fallback registry is found by probing Docker Hub; the runner image's
    # name does not start with "gpustack", so it never takes part.
    monkeypatch.setattr(
        runner_image.registration, "determine_default_registry", lambda _: None
    )
    monkeypatch.setattr(runner_image.envs, "BENCHMARK_RUNNER_AUTO_BUILD", True)
    monkeypatch.setattr(runner_image.envs, "BENCHMARK_RUNNER_BASE_IMAGE", "")
    monkeypatch.setattr(runner_image.envs, "BENCHMARK_RUNNER_SHAREGPT_URL", "")


@pytest.mark.parametrize(
    "registry, expected",
    [
        (None, DEFAULT_RUNNER_IMAGE),
        # Built under the name the runner will ask for, so it is found locally
        # instead of pulled from a registry that does not have it.
        ("registry.local:5000", f"registry.local:5000/{DEFAULT_RUNNER_IMAGE}"),
    ],
)
def test_the_image_is_built_under_the_name_benchmarks_run(tmp_path, registry, expected):
    builder = _builder(tmp_path, _fake_client(), registry=registry)

    assert builder.target_image() == expected


def test_an_image_configured_by_hand_is_not_built(tmp_path):
    client = _fake_client()
    builder = _builder(tmp_path, client, image_repo="my.registry/runner:v9")

    assert builder.ensure() == RunnerImageState.UNMANAGED
    assert client.api.calls == []


def test_nothing_is_built_without_the_shipped_context(tmp_path):
    client = _fake_client()
    builder = _builder(tmp_path, client)
    (tmp_path / "context" / "aisbench-runner" / "Dockerfile").unlink()

    assert builder.ensure() == RunnerImageState.UNMANAGED
    assert client.api.calls == []


def test_auto_build_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_image.envs, "BENCHMARK_RUNNER_AUTO_BUILD", False)
    client = _fake_client()

    assert _builder(tmp_path, client).ensure() == RunnerImageState.UNMANAGED
    assert client.api.calls == []


def test_a_present_image_is_left_alone(tmp_path):
    client = _fake_client(present=[DEFAULT_RUNNER_IMAGE])

    assert _builder(tmp_path, client).ensure() == RunnerImageState.PRESENT
    assert client.api.calls == []


def test_a_missing_image_is_built_from_the_shipped_context(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner_image.envs, "BENCHMARK_RUNNER_SHAREGPT_URL", "https://mirror/sharegpt"
    )
    client = _fake_client(chunks=[{"stream": "Step 1/9 : FROM x\n"}])
    builder = _builder(tmp_path, client)

    assert builder.ensure() == RunnerImageState.BUILDING
    assert builder.wait(5) == RunnerImageState.PRESENT

    (call,) = client.api.calls
    assert call["path"] == str(tmp_path / "context")
    assert call["dockerfile"] == "aisbench-runner/Dockerfile"
    assert call["tag"] == DEFAULT_RUNNER_IMAGE
    assert call["buildargs"] == {
        "RUNNER_COMMIT": "abc1234",
        "SHAREGPT_URL": "https://mirror/sharegpt",
    }
    log = (tmp_path / "logs" / "runner-image-build.log").read_text()
    assert "Step 1/9 : FROM x" in log


def test_one_build_serves_every_caller_while_it_runs(tmp_path):
    gate = threading.Event()
    client = _fake_client(gate=gate)
    builder = _builder(tmp_path, client)

    states = [builder.ensure() for _ in range(3)]
    gate.set()

    assert states == [RunnerImageState.BUILDING] * 3
    assert builder.wait(5) == RunnerImageState.PRESENT
    assert len(client.api.calls) == 1


def test_a_failed_build_reports_why_and_is_retried(tmp_path):
    client = _fake_client(
        chunks=[
            {"stream": "Step 4/9 : RUN curl ...\n"},
            {"status": "Downloading", "progress": "[=>  ] 1MB/9MB"},
            {"error": "curl: (6) Could not resolve host: huggingface.co"},
        ]
    )
    builder = _builder(tmp_path, client)

    builder.ensure()
    assert builder.wait(5) == RunnerImageState.FAILED
    assert "Could not resolve host: huggingface.co" in builder.error
    assert "Step 4/9" in builder.error
    log = (tmp_path / "logs" / "runner-image-build.log").read_text()
    # Layer download progress would bury the build's own output.
    assert "1MB/9MB" not in log

    client.api.chunks = [{"stream": "ok\n"}]
    assert builder.ensure() == RunnerImageState.BUILDING
    assert builder.wait(5) == RunnerImageState.PRESENT
    assert builder.error is None
    assert len(client.api.calls) == 2


def test_an_unreachable_docker_leaves_the_image_unmanaged(tmp_path):
    def unreachable():
        raise docker.errors.DockerException("no docker socket")

    builder = _builder(tmp_path, _fake_client())
    builder._client_factory = unreachable

    assert builder.ensure() == RunnerImageState.UNMANAGED


class _StubBuilder:
    """A runner image whose state walks through the given sequence."""

    def __init__(self, ensured, states, error=None):
        self._ensured = ensured
        self._states = deque(states)
        self.error = error

    def ensure(self):
        return self._ensured

    def target_image(self):
        return DEFAULT_RUNNER_IMAGE

    def state(self):
        return self._states.popleft() if len(self._states) > 1 else self._states[0]


def _waiting_manager(builder, monkeypatch):
    monkeypatch.setattr(bm, "RUNNER_IMAGE_WAIT_POLL_SECONDS", 0)
    mgr = object.__new__(BenchmarkManager)
    mgr._runner_image = builder
    mgr._canceled_ids = set()
    mgr.patches = []

    async def update(benchmark_id, **kwargs):
        mgr.patches.append(kwargs)
        return True

    mgr._update_benchmark_state = update
    return mgr


_BENCHMARK = SimpleNamespace(id=7, name="qwen-throughput")


@pytest.mark.asyncio
async def test_a_benchmark_starts_at_once_when_the_image_is_there(monkeypatch):
    mgr = _waiting_manager(
        _StubBuilder(RunnerImageState.PRESENT, [RunnerImageState.PRESENT]),
        monkeypatch,
    )

    assert await mgr._wait_for_runner_image(_BENCHMARK) is True
    assert mgr.patches == []


@pytest.mark.asyncio
async def test_a_benchmark_waits_for_the_build_and_says_so(monkeypatch):
    mgr = _waiting_manager(
        _StubBuilder(
            RunnerImageState.BUILDING,
            [RunnerImageState.BUILDING] * 3 + [RunnerImageState.PRESENT],
        ),
        monkeypatch,
    )

    assert await mgr._wait_for_runner_image(_BENCHMARK) is True
    assert DEFAULT_RUNNER_IMAGE in mgr.patches[0]["state_message"]
    assert "state" not in mgr.patches[0]
    # The waiting note does not outlive the wait.
    assert mgr.patches[-1] == {"state_message": ""}


@pytest.mark.asyncio
async def test_a_failed_build_fails_the_benchmark_with_its_reason(monkeypatch):
    mgr = _waiting_manager(
        _StubBuilder(
            RunnerImageState.BUILDING,
            [RunnerImageState.BUILDING, RunnerImageState.FAILED],
            error="pull access denied for swr.example/aisbench",
        ),
        monkeypatch,
    )

    assert await mgr._wait_for_runner_image(_BENCHMARK) is False
    final = mgr.patches[-1]
    assert final["state"] == BenchmarkStateEnum.ERROR
    assert "pull access denied for swr.example/aisbench" in final["state_message"]


@pytest.mark.asyncio
async def test_a_benchmark_canceled_while_waiting_never_starts(monkeypatch):
    mgr = _waiting_manager(
        _StubBuilder(RunnerImageState.BUILDING, [RunnerImageState.BUILDING]),
        monkeypatch,
    )
    waiting = asyncio.create_task(mgr._wait_for_runner_image(_BENCHMARK))
    await asyncio.sleep(0.05)
    mgr._canceled_ids.add(_BENCHMARK.id)

    assert await asyncio.wait_for(waiting, timeout=5) is False
    assert _BENCHMARK.id not in mgr._canceled_ids


@pytest.mark.parametrize(
    "recorded, expected",
    [
        (
            "Failed to pull image aisbench-runner:v0.1.1: pull access denied",
            "Failed to start benchmark: Failed to pull image "
            "aisbench-runner:v0.1.1: pull access denied",
        ),
        (None, bm.BENCHMARK_EXITED_MESSAGE),
        ("", bm.BENCHMARK_EXITED_MESSAGE),
    ],
)
def test_a_run_that_never_started_fails_with_its_provisioning_error(
    tmp_path, recorded, expected
):
    mgr = object.__new__(BenchmarkManager)
    mgr._benchmark_log_dir = str(tmp_path)
    if recorded is not None:
        Path(bm.provisioning_error_path(str(tmp_path / "7.log"))).write_text(
            recorded, encoding="utf-8"
        )

    assert mgr._failure_message(_BENCHMARK) == expected
