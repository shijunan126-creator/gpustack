"""Image naming for the vLLM-Ascend backend.

The upstream project publishes one multi-arch image per release and Ascend SoC
family, tagged ``v<version>[-<soc>][-openeuler]``: no SoC suffix is Atlas A2
(910B), ``-a3`` is A3, ``-310p`` is 310P and ``-a5`` is A5. Because the tag is
derived from the version and the device, the image needs no catalog entry: a
repository that follows the same convention (an internal mirror) resolves the
same way.
"""

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from gpustack_runner.runner import Runner
from packaging.version import InvalidVersion, Version

from gpustack import envs
from gpustack.utils.registry_client import image_repository, list_tags

logger = logging.getLogger(__name__)

DEFAULT_REPOSITORY = "quay.io/ascend/vllm-ascend"

# The runner-catalog service name of the backend: ``BackendEnum.VLLM_ASCEND``
# lower-cased, which is how the catalog is queried.
SERVICE_NAME = "vllm-ascend"

# CANN variant (see ``get_ascend_cann_variant``) -> tag suffix. A variant that
# is absent has no published image: 910 (Atlas 800T A1) and 310B.
VARIANT_TAG_SUFFIX: Dict[str, str] = {
    "910b": "",
    "a3": "-a3",
    "310p": "-310p",
    "950": "-a5",
}

_SUFFIX_VARIANT = {
    suffix.lstrip("-") or "910b": v for v, suffix in VARIANT_TAG_SUFFIX.items()
}

_RELEASE_TAG = re.compile(
    r"^v(?P<version>\d+\.\d+\.\d+(?:\.post\d+)?(?:(?:a|b|rc)\d+)?)"
    r"(?:-(?P<soc>a3|310p|a5))?"
    r"(?P<openeuler>-openeuler)?$"
)


@dataclass(frozen=True)
class ReleaseTag:
    version: str
    variant: str
    openeuler: bool


def parse_release_tag(tag: str) -> Optional[ReleaseTag]:
    """Parse a release tag; nightly, model-specific and other tags give None."""
    m = _RELEASE_TAG.match(tag)
    if not m:
        return None
    soc = m.group("soc") or "910b"
    return ReleaseTag(
        version=m.group("version"),
        variant=_SUFFIX_VARIANT[soc],
        openeuler=bool(m.group("openeuler")),
    )


def normalize_version(version: str) -> str:
    """The version as GPUStack stores it: the tag's version without the ``v``."""
    return version[1:] if version[:1] in ("v", "V") else version


def tag_for(version: str, variant: str, openeuler: bool = False) -> str:
    suffix = VARIANT_TAG_SUFFIX.get(variant)
    if suffix is None:
        raise ValueError(
            f"No vLLM-Ascend image is published for the Ascend variant '{variant}'. "
            f"Supported variants: {', '.join(VARIANT_TAG_SUFFIX)}."
        )
    return f"v{normalize_version(version)}{suffix}{'-openeuler' if openeuler else ''}"


def image_for(
    repository: str, version: str, variant: str, openeuler: bool = False
) -> str:
    return f"{image_repository(repository)}:{tag_for(version, variant, openeuler)}"


def _sort_key(version: str) -> Tuple[int, Version]:
    try:
        return (1, Version(version))
    except InvalidVersion:
        return (0, Version("0"))


def release_versions(
    tags: Iterable[str], *, include_openeuler: bool = False
) -> Dict[str, List[str]]:
    """Release versions found in ``tags``, newest first, each with the variants
    it is published for. Pre-releases sort below the release they precede."""
    found: Dict[str, set] = {}
    for tag in tags:
        parsed = parse_release_tag(tag)
        if parsed is None or (parsed.openeuler and not include_openeuler):
            continue
        found.setdefault(parsed.version, set()).add(parsed.variant)
    order = sorted(found, key=_sort_key, reverse=True)
    variant_order = list(VARIANT_TAG_SUFFIX)
    return {v: sorted(found[v], key=variant_order.index) for v in order}


def latest_version(tags: Iterable[str], variant: str) -> Optional[str]:
    """Newest version published for ``variant``. A pre-release ranks below the
    final release of the same number and above every earlier release."""
    for version, variants in release_versions(tags).items():
        if variant in variants:
            return version
    return None


_tags_cache: Dict[str, Tuple[float, List[str]]] = {}
_tags_cache_lock = threading.Lock()


# How long an unreachable registry is left alone before being asked again, when
# the caller prefers an empty listing to an error.
_FAILURE_RETRY_SECONDS = 60


def get_repository_tags(
    repository: Optional[str] = None,
    *,
    refresh: bool = False,
    strict: bool = True,
) -> List[str]:
    """Tags of the configured vLLM-Ascend repository, cached for a few minutes.

    A registry that cannot be reached falls back to the last listing it
    returned. With none to fall back to, the error propagates when ``strict``;
    otherwise the result is an empty listing, remembered briefly so a down
    registry is not retried on every call.
    """
    repository = repository or envs.VLLM_ASCEND_IMAGE_REPOSITORY
    now = time.monotonic()
    with _tags_cache_lock:
        cached = _tags_cache.get(repository)
    if cached and not refresh and now - cached[0] < envs.VLLM_ASCEND_TAGS_CACHE_SECONDS:
        return cached[1]
    try:
        tags = list_tags(
            repository,
            username=envs.VLLM_ASCEND_REGISTRY_USERNAME,
            password=envs.VLLM_ASCEND_REGISTRY_PASSWORD,
        )
    except Exception as e:
        if cached:
            logger.warning(
                "Listing tags of %s failed (%s); using the listing from %.0fs ago",
                repository,
                e,
                now - cached[0],
            )
            return cached[1]
        if strict:
            raise
        logger.warning("Listing tags of %s failed: %s", repository, e)
        with _tags_cache_lock:
            _tags_cache[repository] = (
                now - envs.VLLM_ASCEND_TAGS_CACHE_SECONDS + _FAILURE_RETRY_SECONDS,
                [],
            )
        return []
    with _tags_cache_lock:
        _tags_cache[repository] = (now, tags)
    return tags


_PLATFORMS = ("linux/amd64", "linux/arm64")


def synthesize_runners(tags: Iterable[str], repository: str) -> List[Runner]:
    """Runner records for the releases in ``tags``, shaped like the packaged
    gpustack-runner catalog so the version lists and compatibility filters that
    read the catalog work on them unchanged. The CANN version is not part of the
    tag, so it is left empty."""
    runners: List[Runner] = []
    for version, variants in release_versions(tags).items():
        for variant in variants:
            image = image_for(repository, version, variant)
            for platform in _PLATFORMS:
                runners.append(
                    Runner(
                        backend="cann",
                        backend_version="",
                        original_backend_version="",
                        backend_variant=variant,
                        service=SERVICE_NAME,
                        service_version=version,
                        platform=platform,
                        docker_image=image,
                        deprecated=False,
                        dependencies=None,
                    )
                )
    return runners


def list_runners_from_registry() -> List[Runner]:
    """Synthesized runners for the configured repository; empty while the
    registry cannot be reached."""
    repository = envs.VLLM_ASCEND_IMAGE_REPOSITORY
    return synthesize_runners(get_repository_tags(repository, strict=False), repository)
