"""Benchmark datasets: the corpora a run can measure.

A dataset is a row rather than a constant so an operator can add one from the
UI. Built-ins (Random, ShareGPT, GSM8K) are seeded at startup and cannot be
deleted; an uploaded corpus belongs to the Org that created it, like every other
benchmark resource, and its bytes are stored in the row so a worker on another
host can fetch them through the API it already talks to.

Content is returned by its own endpoint rather than on the row: a listing must
not carry megabytes of prompts, and only the worker that is about to run needs
the bytes.
"""

from typing import List, Optional

from fastapi import APIRouter
from sqlalchemy import or_
from sqlmodel import col

from gpustack.api.exceptions import (
    AlreadyExistsException,
    BadRequestException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.api.tenant import (
    TenantContext,
    bypass_tenant_filter,
    tenant_list_conditions,
)
from gpustack.schemas.benchmark_dataset import (
    BenchmarkDataset,
    BenchmarkDatasetCreate,
    BenchmarkDatasetPublic,
    BenchmarkDatasetsPublic,
    BenchmarkDatasetSource,
)
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.utils.benchmark_dataset import (
    DatasetContentError,
    inspect_dataset,
    validate_name,
)

router = APIRouter()


def _visible_dataset(ctx: TenantContext, dataset: BenchmarkDataset) -> bool:
    """Built-ins are shared; an upload is visible inside its Org."""
    if dataset.source == BenchmarkDatasetSource.BUILTIN:
        return True
    if dataset.owner_principal_id is None:
        return True
    return bypass_tenant_filter(ctx) or (
        ctx.current_principal_id is not None
        and dataset.owner_principal_id == ctx.current_principal_id
    )


def assert_dataset_visible(ctx: TenantContext, dataset: Optional[BenchmarkDataset]):
    if dataset is None or not _visible_dataset(ctx, dataset):
        raise NotFoundException(message="Dataset not found")
    return dataset


def _dataset_conditions(ctx: TenantContext) -> List:
    """Rows the caller may list: their Org's uploads plus every built-in.

    ``tenant_list_conditions`` alone filters by owner and would hide the
    built-ins, whose owner is NULL because they belong to nobody.
    """
    conditions = tenant_list_conditions(ctx, BenchmarkDataset)
    if not conditions:
        return []
    return [or_(*conditions, col(BenchmarkDataset.owner_principal_id).is_(None))]


@router.get("", response_model=BenchmarkDatasetsPublic)
async def get_benchmark_datasets(
    session: SessionDep,
    ctx: TenantContextDep,
):
    datasets = await BenchmarkDataset.all_by_fields(
        session, fields={}, extra_conditions=_dataset_conditions(ctx)
    )
    # Built-ins first: they are what a new benchmark usually measures, and a
    # selector that opens on the built-ins reads as "these are the defaults".
    ordered = sorted(
        datasets,
        key=lambda d: (
            0 if d.source == BenchmarkDatasetSource.BUILTIN else 1,
            (d.name or "").lower(),
        ),
    )
    return BenchmarkDatasetsPublic(
        items=[BenchmarkDatasetPublic.model_validate(d) for d in ordered]
    )


@router.get("/{id}", response_model=BenchmarkDatasetPublic)
async def get_benchmark_dataset(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    dataset = await BenchmarkDataset.one_by_id(session, id)
    assert_dataset_visible(ctx, dataset)
    return dataset


@router.get("/{id}/content")
async def get_benchmark_dataset_content(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    """The uploaded bytes, for the worker that is about to run the benchmark.

    A text response rather than JSON: the file may be a JSONL whose blank lines
    matter, and wrapping it in a JSON string would need the worker to undo the
    escaping before writing it back.
    """
    from fastapi.responses import PlainTextResponse

    dataset = await BenchmarkDataset.one_by_id(session, id)
    assert_dataset_visible(ctx, dataset)
    if dataset.content is None:
        raise NotFoundException(
            message=f"Dataset '{dataset.name}' carries no uploaded content"
        )
    return PlainTextResponse(
        content=dataset.content, media_type="application/x-ndjson"
    )


@router.post("", response_model=BenchmarkDatasetPublic)
async def create_benchmark_dataset(
    session: SessionDep,
    ctx: TenantContextDep,
    dataset_in: BenchmarkDatasetCreate,
):
    try:
        name = validate_name(dataset_in.name)
        summary = inspect_dataset(dataset_in.content, dataset_in.format)
    except DatasetContentError as e:
        raise BadRequestException(message=str(e))

    existing = await BenchmarkDataset.one_by_field(session, "name", name)
    if existing is not None:
        raise AlreadyExistsException(
            message=f"Dataset with name '{name}' already exists."
        )

    dataset = BenchmarkDataset(
        name=name,
        description=(dataset_in.description or "").strip() or None,
        format=summary.format,
        source=BenchmarkDatasetSource.UPLOAD,
        path=None,
        sample_count=summary.sample_count,
        answer_count=summary.answer_count,
        has_answers=summary.has_answers,
        size_bytes=summary.size_bytes,
        sha256=summary.sha256,
        content=dataset_in.content,
        owner_principal_id=ctx.current_principal_id,
    )
    try:
        await BenchmarkDataset.create(session, dataset)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create dataset: {e}"
        )
    return dataset


@router.delete("/{id}")
async def delete_benchmark_dataset(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    dataset = await BenchmarkDataset.one_by_id(session, id)
    assert_dataset_visible(ctx, dataset)
    if dataset.source == BenchmarkDatasetSource.BUILTIN:
        raise BadRequestException(
            message=f"Built-in dataset '{dataset.name}' cannot be deleted."
        )
    # A run that already used it keeps its own snapshot of the name and bytes;
    # a run that has not started will fail with a message naming the dataset.
    try:
        await dataset.delete(session)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete dataset: {e}"
        )


@router.get("/name/{name}", response_model=BenchmarkDatasetPublic)
async def get_benchmark_dataset_by_name(
    session: SessionDep,
    ctx: TenantContextDep,
    name: str,
):
    """Look a dataset up by the name a benchmark row references."""
    dataset = await BenchmarkDataset.one_by_field(session, "name", name)
    assert_dataset_visible(ctx, dataset)
    return dataset
