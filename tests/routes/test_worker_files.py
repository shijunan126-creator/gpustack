"""The directory picker of the local-path source: what a worker lists, and who
may ask for it."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from gpustack.api.exceptions import (
    ForbiddenException,
    InvalidException,
    NotFoundException,
)
from gpustack.routes import worker_files
from gpustack.routes.worker import filesystem
from gpustack.schemas.filesystem import DirectoryListing
from gpustack.schemas.workers import WorkerStateEnum


def _touch(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_listing_names_the_model_directories(tmp_path):
    _touch(tmp_path / "qwen" / "config.json", "{}")
    _touch(tmp_path / "qwen" / "model.safetensors")
    _touch(tmp_path / "plain" / "notes.txt")
    _touch(tmp_path / "small.gguf")
    _touch(tmp_path / "readme.md")
    _touch(tmp_path / ".hidden" / "config.json")

    listing = filesystem.list_directory(str(tmp_path))

    assert [(e.name, e.is_dir, e.is_model) for e in listing.entries] == [
        ("plain", True, False),
        ("qwen", True, True),
        ("small.gguf", False, True),
    ]
    assert listing.path == str(tmp_path.resolve())
    assert listing.parent == str(tmp_path.resolve().parent)
    assert listing.is_model is True  # holds a .gguf file itself


def test_a_model_directory_is_recognised_by_its_files(tmp_path):
    _touch(tmp_path / "config.json", "{}")
    assert filesystem.list_directory(str(tmp_path)).is_model is True
    empty = tmp_path / "empty"
    empty.mkdir()
    assert filesystem.list_directory(str(empty)).is_model is False


def test_the_root_has_no_parent():
    assert filesystem.list_directory("/").parent is None


def test_the_listing_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(filesystem, "MAX_LISTED_ENTRIES", 3)
    for i in range(5):
        (tmp_path / f"d{i}").mkdir()
    listing = filesystem.list_directory(str(tmp_path))
    assert len(listing.entries) == 3
    assert listing.truncated is True


@pytest.mark.parametrize(
    "make, status",
    [
        (lambda p: str(p / "missing"), 404),
        (lambda p: str(_file(p)), 400),
    ],
)
def test_a_path_that_cannot_be_listed_is_refused(tmp_path, make, status):
    with pytest.raises(filesystem.HTTPException) as err:
        filesystem.list_directory(make(tmp_path))
    assert err.value.status_code == status


def _file(root):
    path = root / "a.txt"
    path.write_text("x")
    return path


def _worker(state=WorkerStateEnum.READY):
    return SimpleNamespace(id=3, name="w3", state=state)


async def _call(error=None, worker=None, visible=True):
    client = AsyncMock()
    if error:
        client.list_directory.side_effect = error
    else:
        client.list_directory.return_value = DirectoryListing(path="/data")
    client.__aenter__.return_value = client
    with (
        patch.object(worker_files.Worker, "one_by_id", AsyncMock(return_value=worker)),
        patch.object(worker_files, "WorkerFilesystemClient", return_value=client),
        patch.object(
            worker_files,
            "assert_resource_visible",
            side_effect=None if visible else NotFoundException(message="x"),
        ) as check,
    ):
        result = await worker_files.list_worker_directory(
            ctx=SimpleNamespace(), session=SimpleNamespace(), worker_id=3, path="/data"
        )
    return result, check, client


@pytest.mark.asyncio
async def test_a_visible_ready_worker_is_listed():
    result, check, client = await _call(worker=_worker())
    assert result.path == "/data"
    check.assert_called_once()
    client.list_directory.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_worker_the_caller_cannot_see_is_not_found():
    with pytest.raises(NotFoundException):
        await _call(worker=_worker(), visible=False)


@pytest.mark.asyncio
async def test_a_worker_that_is_not_ready_is_refused():
    with pytest.raises(InvalidException):
        await _call(worker=_worker(WorkerStateEnum.NOT_READY))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, exception",
    [(404, NotFoundException), (403, ForbiddenException), (400, InvalidException)],
)
async def test_the_workers_answer_keeps_its_meaning(status, exception):
    error = aiohttp.ClientError(f"status={status}, error=boom")
    with pytest.raises(exception):
        await _call(error=error, worker=_worker())
