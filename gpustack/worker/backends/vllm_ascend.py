import logging
from typing import Optional

from gpustack_runtime.detector.ascend import get_ascend_cann_variant

from gpustack import envs
from gpustack.schemas.models import Model
from gpustack.utils.vllm_ascend_images import (
    VARIANT_TAG_SUFFIX,
    get_repository_tags,
    image_for,
    latest_version,
    normalize_version,
)
from gpustack.worker.backends.base import _INHERIT
from gpustack.worker.backends.vllm import VLLMServer

logger = logging.getLogger(__name__)

# The CANN variant assumed when the SoC is not recognised, matching the runner
# catalog's fallback for Ascend devices.
_DEFAULT_VARIANT = "910b"


class VLLMAscendServer(VLLMServer):
    """vLLM on Ascend NPUs.

    Everything about launching the engine is vLLM's. Only where the image comes
    from differs: it is named from the release version and the NPU family
    against ``GPUSTACK_VLLM_ASCEND_IMAGE_REPOSITORY``, instead of being looked
    up in the gpustack-runner catalog.
    """

    def _resolve_image(
        self,
        backend: Optional[str] = None,
        spec: Optional[Model] = None,
        inference_backend=_INHERIT,
    ):
        """Resolve the image for the vLLM-Ascend backend.

        Precedence:
        1) Image pinned on the model (``image_name``).
        2) Image configured for the chosen version on the backend row.
        3) ``<repository>:v<version>[-<soc>]``, with the newest release published
           for this NPU family when the model does not pin a version.

        Returns:
            (image_name, backend_version), or (None, None) off Ascend devices.
        """
        model = spec if spec is not None else self._model
        if inference_backend is _INHERIT:
            inference_backend = self.inference_backend

        if model.image_name:
            return model.image_name, None

        if model and inference_backend:
            image_name, target_version = inference_backend.get_image_name(
                model.backend_version
            )
            if image_name and target_version:
                return image_name, target_version

        device_backend, _, arch_family = self._get_device_info()
        if device_backend != "cann":
            return None, None

        variant = get_ascend_cann_variant(arch_family) if arch_family else None
        variant = variant or _DEFAULT_VARIANT
        if variant not in VARIANT_TAG_SUFFIX:
            raise ValueError(
                f"vLLM-Ascend publishes no image for the Ascend '{variant}' family "
                f"(SoC {arch_family}). Pin an image on the deployment instead."
            )

        repository = envs.VLLM_ASCEND_IMAGE_REPOSITORY
        version = normalize_version(model.backend_version or "")
        if not version:
            version = latest_version(get_repository_tags(repository), variant)
            if not version:
                raise ValueError(
                    f"No vLLM-Ascend release is published for the '{variant}' "
                    f"family in {repository}."
                )
        logger.info(
            "vLLM-Ascend image: repository=%s version=%s variant=%s",
            repository,
            version,
            variant,
        )
        return image_for(repository, version, variant), version
