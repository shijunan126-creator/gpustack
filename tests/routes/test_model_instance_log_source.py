"""The log source a viewer asks for reaches the worker that holds the logs."""

import pytest

from gpustack.routes.model_instances import _build_serve_log_params
from gpustack.schemas.models import ModelInstanceStateEnum
from tests.utils.model import new_model_instance


def _instance():
    return new_model_instance(
        1, "qwen-abc", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )


@pytest.mark.parametrize(
    "source, forwarded",
    [
        # The default is left off, so a worker that predates the parameter
        # still answers the request exactly as it always did.
        ("all", None),
        ("engine", "engine"),
    ],
)
def test_the_log_source_is_forwarded_to_the_worker(source, forwarded):
    params = _build_serve_log_params(_instance(), follow=True, source=source)

    assert params.get("source") == forwarded
    assert params["follow"] is True
