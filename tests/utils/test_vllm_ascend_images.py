import json
from pathlib import Path

import pytest
from gpustack_runner.runner import build_service_runners

from gpustack.schemas.runner_source import merged_runners
from gpustack.utils import vllm_ascend_images as images
from gpustack.utils.vllm_ascend_images import (
    ReleaseTag,
    image_for,
    latest_version,
    normalize_version,
    parse_release_tag,
    release_versions,
    synthesize_runners,
    tag_for,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "vllm_ascend_quay_tags.json"


@pytest.fixture(scope="module")
def quay_tags():
    return json.loads(FIXTURE.read_text())


@pytest.mark.parametrize(
    "tag, expected",
    [
        ("v0.30.0rc2", ReleaseTag("0.30.0rc2", "910b", False)),
        ("v0.30.0rc2-a3", ReleaseTag("0.30.0rc2", "a3", False)),
        ("v0.30.0rc2-310p", ReleaseTag("0.30.0rc2", "310p", False)),
        ("v0.30.0rc2-a5", ReleaseTag("0.30.0rc2", "950", False)),
        ("v0.30.0rc2-a3-openeuler", ReleaseTag("0.30.0rc2", "a3", True)),
        ("v0.30.0rc1-openeuler", ReleaseTag("0.30.0rc1", "910b", True)),
        ("v0.23.0.post1-a3", ReleaseTag("0.23.0.post1", "a3", False)),
        ("v0.20.0", ReleaseTag("0.20.0", "910b", False)),
        ("nightly-main", None),
        ("nightly-releases-v0.30.0-a3", None),
        ("minimax-m3.1-flash-a5", None),
        ("v0.30", None),
        ("0.30.0rc2", None),
    ],
)
def test_parse_release_tag(tag, expected):
    assert parse_release_tag(tag) == expected


def test_every_release_tag_in_the_registry_listing_round_trips(quay_tags):
    parsed = [(t, parse_release_tag(t)) for t in quay_tags]
    releases = [(t, p) for t, p in parsed if p]
    assert releases, "fixture has no release tags"
    for tag, p in releases:
        assert tag_for(p.version, p.variant, p.openeuler) == tag


def test_release_versions_are_newest_first_with_their_variants(quay_tags):
    versions = release_versions(quay_tags)
    order = list(versions)
    assert (
        order.index("0.30.0rc2") < order.index("0.30.0rc1") < order.index("0.27.1rc1")
    )
    assert order.index("0.27.1rc1") < order.index("0.23.0.post1")
    assert versions["0.30.0rc2"] == ["910b", "a3", "310p", "950"]


def test_release_versions_leave_out_openeuler_unless_asked():
    tags = ["v1.0.0-openeuler", "v1.0.0-a3-openeuler"]
    assert release_versions(tags) == {}
    assert release_versions(tags, include_openeuler=True) == {"1.0.0": ["910b", "a3"]}


def test_a_prerelease_ranks_below_its_final_release():
    tags = ["v1.0.0rc1", "v1.0.0", "v0.9.0"]
    assert list(release_versions(tags)) == ["1.0.0", "1.0.0rc1", "0.9.0"]


def test_latest_version_is_per_variant():
    tags = ["v1.1.0", "v1.1.0-a3", "v1.0.0-310p", "v1.0.0"]
    assert latest_version(tags, "910b") == "1.1.0"
    assert latest_version(tags, "310p") == "1.0.0"
    assert latest_version(tags, "950") is None


@pytest.mark.parametrize(
    "variant, suffix",
    [("910b", ""), ("a3", "-a3"), ("310p", "-310p"), ("950", "-a5")],
)
def test_tag_for_appends_the_soc_suffix(variant, suffix):
    assert tag_for("0.30.0rc2", variant) == f"v0.30.0rc2{suffix}"


@pytest.mark.parametrize("variant", ["910", "310b", "unknown"])
def test_tag_for_rejects_a_family_without_an_image(variant):
    with pytest.raises(ValueError, match="No vLLM-Ascend image"):
        tag_for("0.30.0rc2", variant)


@pytest.mark.parametrize("given", ["v0.30.0rc2", "0.30.0rc2", "V0.30.0rc2"])
def test_a_leading_v_on_the_version_is_ignored(given):
    assert normalize_version(given) == "0.30.0rc2"
    assert tag_for(given, "a3") == "v0.30.0rc2-a3"


@pytest.mark.parametrize(
    "repository, expected",
    [
        ("quay.io/ascend/vllm-ascend", "quay.io/ascend/vllm-ascend:v1.0.0-a3"),
        (
            "registry.lan:5000/ascend/vllm-ascend/",
            "registry.lan:5000/ascend/vllm-ascend:v1.0.0-a3",
        ),
        (
            "http://registry.lan:5000/ascend/vllm-ascend",
            "registry.lan:5000/ascend/vllm-ascend:v1.0.0-a3",
        ),
    ],
)
def test_image_for_names_a_runtime_reference_without_a_scheme(repository, expected):
    assert image_for(repository, "1.0.0", "a3") == expected


def test_synthesized_runners_feed_the_catalog_grouping(quay_tags):
    runners = synthesize_runners(quay_tags, "quay.io/ascend/vllm-ascend")
    assert {r.service for r in runners} == {"vllm-ascend"}
    assert {r.platform for r in runners} == {"linux/amd64", "linux/arm64"}

    grouped = build_service_runners(runners)
    assert [s.service for s in grouped] == ["vllm-ascend"]
    versions = [v.version for v in grouped[0].versions]
    assert versions[0] == "0.30.0rc2"
    assert versions == list(dict.fromkeys(versions))


def test_the_catalog_merge_serves_vllm_ascend_from_the_registry(monkeypatch, quay_tags):
    monkeypatch.setattr(images, "get_repository_tags", lambda *_, **__: quay_tags)
    runners = merged_runners(
        [], backend="cann", service="vllm-ascend", backend_variant="a3"
    )
    assert runners
    assert {r.backend_variant for r in runners} == {"a3"}
    assert all(r.docker_image.endswith("-a3") for r in runners)


def test_the_catalog_merge_ignores_runner_overrides_for_vllm_ascend(
    monkeypatch, quay_tags
):
    monkeypatch.setattr(images, "get_repository_tags", lambda *_, **__: quay_tags)
    sentinel = object()  # an override row would break the conversion if it were read
    runners = merged_runners([sentinel], backend="cann", service="vllm-ascend")
    assert runners and all(r.service == "vllm-ascend" for r in runners)


def test_an_unreachable_registry_yields_no_runners_instead_of_failing(monkeypatch):
    def unreachable(*_, **__):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(images, "list_tags", unreachable)
    monkeypatch.setattr(images, "_tags_cache", {})
    assert (
        images.get_repository_tags("registry.lan/ascend/vllm-ascend", strict=False)
        == []
    )
    with pytest.raises(ConnectionError):
        images.get_repository_tags("registry.lan/other/name")


def test_the_last_good_listing_survives_a_registry_outage(monkeypatch):
    calls = {"n": 0}

    def flaky(*_, **__):
        calls["n"] += 1
        if calls["n"] == 1:
            return ["v1.0.0"]
        raise ConnectionError("down")

    monkeypatch.setattr(images, "list_tags", flaky)
    monkeypatch.setattr(images, "_tags_cache", {})
    repo = "registry.lan/ascend/vllm-ascend"
    assert images.get_repository_tags(repo) == ["v1.0.0"]
    assert images.get_repository_tags(repo, refresh=True) == ["v1.0.0"]
    assert calls["n"] == 2
