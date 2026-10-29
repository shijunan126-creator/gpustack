import logging
import re

import aiohttp
from fastapi import APIRouter, Query

from gpustack.api.exceptions import (
    ForbiddenException,
    InternalServerErrorException,
    InvalidException,
    NotFoundException,
)
from gpustack.api.tenant import assert_resource_visible
from gpustack.client.worker_filesystem_client import WorkerFilesystemClient
from gpustack.schemas.filesystem import DirectoryListing
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.deps import SessionDep, TenantContextDep

logger = logging.getLogger(__name__)

router = APIRouter()

_STATUS = re.compile(r"status=(\d{3})")


@router.get("/{worker_id}/directories", response_model=DirectoryListing)
async def list_worker_directory(
    ctx: TenantContextDep,
    session: SessionDep,
    worker_id: int,
    path: str = Query("/", description="Directory on the worker to list"),
):
    """List the sub-directories and GGUF files of a directory on a worker.

    Backs the directory picker of the local-path model source. The caller must
    be able to see the worker; only names are returned, never file contents.
    """
    worker = await Worker.one_by_id(session, worker_id)
    assert_resource_visible(ctx, worker, not_found_message="worker not found")
    if worker.state != WorkerStateEnum.READY:
        raise InvalidException(message=f"Worker {worker.name} is not ready")

    try:
        async with WorkerFilesystemClient() as client:
            return await client.list_directory(worker, path)
    except aiohttp.ClientError as e:
        match = _STATUS.search(str(e))
        status = int(match.group(1)) if match else None
        if status == 404:
            raise NotFoundException(message=f"Path not found on {worker.name}: {path}")
        if status == 403:
            raise ForbiddenException(
                message=f"Permission denied on {worker.name}: {path}"
            )
        if status == 400:
            raise InvalidException(message=f"Not a directory on {worker.name}: {path}")
        logger.warning(f"Failed to list {path} on worker {worker.name}: {e}")
        raise InternalServerErrorException(
            message=f"Failed to list the directory on {worker.name}"
        )
