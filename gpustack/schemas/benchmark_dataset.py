"""Named benchmark datasets.

A benchmark's prompts come from one of these rows rather than from a constant
in the code: the built-ins (synthetic Random, ShareGPT, GSM8K) are seeded
rows, and an uploaded corpus is another one. That is what makes "add a
dataset" a data change instead of a code change, and it also lets a benchmark
record which corpus it measured — the report outlives the dataset.

``content`` is the uploaded text itself for an ``upload`` row, kept so a worker
can fetch it without the server and the worker sharing a filesystem. Built-in
rows carry a ``path`` alias (``builtin:gsm8k``) instead, because their data is
already inside the runner image.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel
from sqlalchemy import Column
from sqlmodel import Field, SQLModel, Text
from sqlmodel.sql.sqltypes import AutoString

from gpustack.mixins import BaseModelMixin

# Names the seed functions and the UI agree on. A benchmark row references a
# dataset by name, so these are part of the API surface.
BUILTIN_DATASET_NAME_RANDOM = "Random"
BUILTIN_DATASET_NAME_SHAREGPT = "ShareGPT"
BUILTIN_DATASET_NAME_GSM8K = "GSM8K"

# The corpus a new benchmark measures unless the caller picks another one. GSM8K
# is the baseline the R&D workflow compares runs against, so it is also the
# default the create form starts from.
DEFAULT_BENCHMARK_DATASET_NAME = BUILTIN_DATASET_NAME_GSM8K


class BenchmarkDatasetFormat(str, Enum):
    """How the runner reads the file."""

    RANDOM = "random"
    """Generated prompts: no file, lengths come from the benchmark's config."""

    SHAREGPT = "sharegpt"
    """A JSON array of conversations; the first human/assistant turn is used."""

    QA_JSONL = "qa_jsonl"
    """One question (and optionally its answer) per line."""

    def __str__(self):
        return self.value


class BenchmarkDatasetSource(str, Enum):
    BUILTIN = "builtin"
    UPLOAD = "upload"

    def __str__(self):
        return self.value


class BenchmarkDatasetBase(SQLModel):
    name: str = Field(index=True, unique=True)
    description: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    format: BenchmarkDatasetFormat = Field(
        default=BenchmarkDatasetFormat.QA_JSONL, sa_type=AutoString
    )
    source: BenchmarkDatasetSource = Field(
        default=BenchmarkDatasetSource.UPLOAD, sa_type=AutoString
    )
    # Built-in alias the runner resolves inside the container
    # (``builtin:gsm8k``). Empty for an uploaded dataset, whose bytes live in
    # ``content`` and are written to disk by the worker.
    path: Optional[str] = Field(default=None)

    sample_count: Optional[int] = Field(default=None)
    answer_count: Optional[int] = Field(default=None)
    has_answers: bool = Field(default=False)

    size_bytes: Optional[int] = Field(default=None)
    sha256: Optional[str] = Field(default=None)


class BenchmarkDataset(BenchmarkDatasetBase, BaseModelMixin, table=True):
    __tablename__ = "benchmark_datasets"

    id: Optional[int] = Field(default=None, primary_key=True)
    # The uploaded file. Absent for a built-in dataset, whose data ships in the
    # runner image; ``path`` names it there.
    content: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    # NULL for a built-in row, which every Org can see. An upload belongs to the
    # Org that created it, like every other benchmark resource.
    owner_principal_id: Optional[int] = Field(default=None, index=True)


class BenchmarkDatasetCreate(BaseModel):
    """The request body of an upload.

    ``content`` carries the file's text rather than a multipart part: a
    benchmark dataset is a few hundred kilobytes of JSON, and sending it as a
    JSON string keeps the upload on the same client path as every other call.
    """

    name: str
    description: Optional[str] = None
    format: Optional[BenchmarkDatasetFormat] = None
    content: str


class BenchmarkDatasetPublic(BenchmarkDatasetBase):
    id: int
    owner_principal_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class BenchmarkDatasetsPublic(BaseModel):
    items: List[BenchmarkDatasetPublic] = []


class BenchmarkDatasetSnapshot(BaseModel):
    """What a run measured, frozen on the benchmark row at creation.

    The dataset row can be deleted later; the report still has to say which
    corpus produced it, and which bytes.
    """

    id: Optional[int] = None
    name: str
    format: BenchmarkDatasetFormat = BenchmarkDatasetFormat.QA_JSONL
    sha256: Optional[str] = None
    sample_count: Optional[int] = None
    has_answers: bool = False
