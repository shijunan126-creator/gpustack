import types

import pytest

import gpustack.worker.backends.vllm_ascend as module
from gpustack.schemas.models import (
    VLLM_FAMILY,
    BackendEnum,
    catalog_backend_name,
    is_vllm_family,
)
from gpustack.worker.backends.vllm import VLLMServer
from gpustack.worker.backends.vllm_ascend import VLLMAscendServer
from gpustack.worker.serve_manager import _SERVER_CLASS_MAPPING

TAGS = [
    "v0.30.0rc2",
    "v0.30.0rc2-a3",
    "v0.30.0rc2-310p",
    "v0.30.0rc1",
    "v0.29.0-a3",
    "nightly-main",
]


def make_server(*, device, version=None, image=None, inference_backend=None):
    server = VLLMAscendServer.__new__(VLLMAscendServer)
    server._model = types.SimpleNamespace(
        image_name=image, backend=BackendEnum.VLLM_ASCEND.value, backend_version=version
    )
    server.inference_backend = inference_backend
    server._get_device_info = lambda: device
    return server


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(module, "get_repository_tags", lambda *_, **__: TAGS)
    monkeypatch.setattr(
        module.envs, "VLLM_ASCEND_IMAGE_REPOSITORY", "quay.io/ascend/vllm-ascend"
    )


def test_backend_is_a_vllm_server_registered_under_its_own_name():
    assert issubclass(VLLMAscendServer, VLLMServer)
    assert _SERVER_CLASS_MAPPING[BackendEnum.VLLM_ASCEND] is VLLMAscendServer


def test_vllm_family_and_catalog_names():
    assert set(VLLM_FAMILY) == {BackendEnum.VLLM, BackendEnum.VLLM_ASCEND}
    assert is_vllm_family("vLLM-Ascend") and is_vllm_family("vLLM")
    assert not is_vllm_family("SGLang") and not is_vllm_family(None)
    assert catalog_backend_name("vLLM-Ascend") == "vLLM"
    assert catalog_backend_name("SGLang") == "SGLang"


@pytest.mark.parametrize(
    "arch_family, version, expected",
    [
        ("Ascend910B3", "0.30.0rc2", "quay.io/ascend/vllm-ascend:v0.30.0rc2"),
        ("Ascend910_9391", "0.30.0rc2", "quay.io/ascend/vllm-ascend:v0.30.0rc2-a3"),
        ("Ascend310P3", "v0.30.0rc2", "quay.io/ascend/vllm-ascend:v0.30.0rc2-310p"),
        ("Ascend950PR", "0.30.0rc2", "quay.io/ascend/vllm-ascend:v0.30.0rc2-a5"),
        # An SoC the runtime cannot map falls back to the 910B image.
        (None, "0.30.0rc2", "quay.io/ascend/vllm-ascend:v0.30.0rc2"),
    ],
)
def test_image_is_named_from_version_and_npu_family(arch_family, version, expected):
    server = make_server(device=("cann", "9.2", arch_family), version=version)
    assert server._resolve_image() == (expected, version.lstrip("v"))


def test_without_a_pinned_version_the_newest_release_for_the_family_is_used():
    a3 = make_server(device=("cann", "9.2", "Ascend910_9391"))
    assert a3._resolve_image() == (
        "quay.io/ascend/vllm-ascend:v0.30.0rc2-a3",
        "0.30.0rc2",
    )
    # 310P has only 0.30.0rc2 here; the 0.29.0 image is A3-only.
    p310 = make_server(device=("cann", "9.2", "Ascend310P1"))
    assert p310._resolve_image()[1] == "0.30.0rc2"


def test_a_repository_override_moves_every_image_to_the_internal_registry(monkeypatch):
    monkeypatch.setattr(
        module.envs,
        "VLLM_ASCEND_IMAGE_REPOSITORY",
        "http://registry.lan:5000/ascend/vllm-ascend",
    )
    server = make_server(device=("cann", "9.2", "Ascend910_9391"), version="0.30.0rc2")
    image, _ = server._resolve_image()
    assert image == "registry.lan:5000/ascend/vllm-ascend:v0.30.0rc2-a3"


def test_an_image_pinned_on_the_model_wins_and_is_not_rewritten():
    server = make_server(
        device=("cann", "9.2", "Ascend910B2"),
        version="0.30.0rc2",
        image="harbor.lan/team/vllm-ascend:custom",
    )
    assert server._resolve_image() == ("harbor.lan/team/vllm-ascend:custom", None)


def test_a_version_configured_with_its_own_image_wins_over_the_convention():
    backend = types.SimpleNamespace(
        get_image_name=lambda v: ("harbor.lan/vllm-ascend:patched", v)
    )
    server = make_server(
        device=("cann", "9.2", "Ascend910B2"),
        version="0.30.0rc2-custom",
        inference_backend=backend,
    )
    assert server._resolve_image() == (
        "harbor.lan/vllm-ascend:patched",
        "0.30.0rc2-custom",
    )


def test_no_image_is_resolved_off_ascend_devices():
    assert make_server(device=("cuda", "12.8", None))._resolve_image() == (None, None)


@pytest.mark.parametrize("arch_family", ["Ascend910A", "Ascend310B1"])
def test_a_family_without_a_published_image_says_so(arch_family):
    server = make_server(device=("cann", "9.2", arch_family), version="0.30.0rc2")
    with pytest.raises(ValueError, match="publishes no image"):
        server._resolve_image()


def test_no_release_for_the_family_is_an_error_not_a_guess():
    server = make_server(device=("cann", "9.2", "Ascend950PR"))
    with pytest.raises(ValueError, match="No vLLM-Ascend release"):
        server._resolve_image()
