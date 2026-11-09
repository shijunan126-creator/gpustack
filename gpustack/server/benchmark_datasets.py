"""Seed the datasets every installation ships with.

Three rows, created once and never deleted: the synthetic Random generator, the
ShareGPT corpus the runner image carries, and GSM8K — the default, because it
is the corpus the baseline runs are compared on and because it is the only
built-in whose answers make it usable for an accuracy run.

Seeding is create-only. A built-in row is a definition, not user data, but
rewriting an existing row on every start would also overwrite whatever an
operator changed in it, and there is nothing here worth that.
"""

import logging
from typing import Any, Dict, List

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.benchmark_dataset import (
    BUILTIN_DATASET_NAME_GSM8K,
    BUILTIN_DATASET_NAME_RANDOM,
    BUILTIN_DATASET_NAME_SHAREGPT,
    BenchmarkDataset,
    BenchmarkDatasetFormat,
    BenchmarkDatasetSource,
)

logger = logging.getLogger(__name__)

# GSM8K's test split, the one an accuracy run measures. The file itself lives
# in the runner image (``builtin:gsm8k``), so this count cannot be read from
# here; it is the length of the file the image is built with.
GSM8K_TEST_SAMPLES = 1319

BUILTIN_DATASETS: List[Dict[str, Any]] = [
    {
        "name": BUILTIN_DATASET_NAME_GSM8K,
        "description": (
            "GSM8K grade-school math word problems (1,319 questions, each with "
            "a reference answer). The default: use it for performance runs and "
            "for accuracy runs alike."
        ),
        "format": BenchmarkDatasetFormat.QA_JSONL,
        "path": "builtin:gsm8k",
        "sample_count": GSM8K_TEST_SAMPLES,
        "answer_count": GSM8K_TEST_SAMPLES,
        "has_answers": True,
    },
    {
        "name": BUILTIN_DATASET_NAME_SHAREGPT,
        "description": (
            "ShareGPT conversations; the first human/assistant turn of each is "
            "used. Prompt lengths follow real chat traffic."
        ),
        "format": BenchmarkDatasetFormat.SHAREGPT,
        "path": "builtin:sharegpt",
        "sample_count": None,
        "answer_count": None,
        "has_answers": False,
    },
    {
        "name": BUILTIN_DATASET_NAME_RANDOM,
        "description": (
            "Synthetic prompts generated at the input/output token lengths the "
            "run configures."
        ),
        "format": BenchmarkDatasetFormat.RANDOM,
        "path": None,
        "sample_count": None,
        "answer_count": None,
        "has_answers": False,
    },
]


async def sync_builtin_benchmark_datasets(session: AsyncSession):
    for spec in BUILTIN_DATASETS:
        existing = await BenchmarkDataset.one_by_field(
            session, "name", spec["name"]
        )
        if existing is not None:
            continue
        try:
            await BenchmarkDataset.create(
                session,
                BenchmarkDataset(
                    **spec,
                    source=BenchmarkDatasetSource.BUILTIN,
                    owner_principal_id=None,
                ),
            )
        except Exception as e:
            # A name collision with a user upload must not stop startup: the
            # upload wins, and a benchmark naming it still runs — as the
            # uploaded corpus rather than the built-in one.
            logger.warning(
                f"Could not seed built-in benchmark dataset '{spec['name']}': {e}"
            )
