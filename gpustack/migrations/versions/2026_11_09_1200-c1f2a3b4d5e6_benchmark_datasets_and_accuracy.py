"""Benchmark datasets and accuracy runs.

A benchmark's prompts now come from a ``benchmark_datasets`` row — the three
built-ins are seeded at startup — instead of from one of two constants, and a
run can measure answer accuracy as well as throughput. That needs the dataset
table, the run's kind and accuracy columns, and a table for the per-question
verdicts.

Existing runs stay performance runs: ``kind`` defaults to ``perf`` and their
``dataset_name`` is left exactly as it was.

Revision ID: c1f2a3b4d5e6
Revises: a7e2c9d4f681
Create Date: 2026-11-09 12:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes

import gpustack.schemas.common

from gpustack.migrations.utils import column_exists, table_exists

revision: str = 'c1f2a3b4d5e6'
down_revision: Union[str, None] = 'a7e2c9d4f681'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _add_benchmark_columns() -> None:
    columns = [
        ('kind', sqlmodel.sql.sqltypes.AutoString(), False, 'perf'),
        ('dataset_id', sa.Integer(), True, None),
        ('dataset_format', sqlmodel.sql.sqltypes.AutoString(), True, None),
        ('dataset_sha256', sqlmodel.sql.sqltypes.AutoString(), True, None),
        ('max_samples', sa.Integer(), True, None),
        ('accuracy_concurrency', sa.Integer(), True, None),
        ('accuracy_max_out_len', sa.Integer(), True, None),
        ('accuracy_score', sa.Float(), True, None),
        ('accuracy_total', sa.Integer(), True, None),
        ('accuracy_correct', sa.Integer(), True, None),
        ('accuracy_invalid', sa.Integer(), True, None),
        ('accuracy_errored', sa.Integer(), True, None),
        ('accuracy_elapsed_seconds', sa.Float(), True, None),
    ]
    for name, type_, nullable, default in columns:
        if column_exists('benchmarks', name):
            continue
        op.add_column(
            'benchmarks',
            sa.Column(name, type_, nullable=nullable, server_default=default),
        )


def _create_datasets_table() -> None:
    if table_exists('benchmark_datasets'):
        return
    op.create_table(
        'benchmark_datasets',
        sa.Column(
            'created_at', gpustack.schemas.common.UTCDateTime(), nullable=False
        ),
        sa.Column(
            'updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False
        ),
        sa.Column(
            'deleted_at', gpustack.schemas.common.UTCDateTime(), nullable=True
        ),
        sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column(
            'format',
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default='qa_jsonl',
        ),
        sa.Column(
            'source',
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default='upload',
        ),
        sa.Column('path', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('sample_count', sa.Integer(), nullable=True),
        sa.Column('answer_count', sa.Integer(), nullable=True),
        sa.Column(
            'has_answers', sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column('size_bytes', sa.Integer(), nullable=True),
        sa.Column('sha256', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('owner_principal_id', sa.Integer(), nullable=True),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['owner_principal_id'], ['principals.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_benchmark_datasets_name', 'benchmark_datasets', ['name'], unique=True
    )
    op.create_index(
        'ix_benchmark_datasets_owner_principal_id',
        'benchmark_datasets',
        ['owner_principal_id'],
    )


def _create_accuracy_samples_table() -> None:
    if table_exists('benchmark_accuracy_samples'):
        return
    op.create_table(
        'benchmark_accuracy_samples',
        sa.Column(
            'created_at', gpustack.schemas.common.UTCDateTime(), nullable=False
        ),
        sa.Column(
            'updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False
        ),
        sa.Column(
            'deleted_at', gpustack.schemas.common.UTCDateTime(), nullable=True
        ),
        sa.Column('seq', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('question', sa.Text(), nullable=True),
        sa.Column('prediction', sa.Text(), nullable=True),
        sa.Column('extracted', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('reference', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column(
            'correct', sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column('latency_ms', sa.Float(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('benchmark_id', sa.Integer(), nullable=False),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ['benchmark_id'], ['benchmarks.id'], ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_benchmark_accuracy_samples_benchmark_id',
        'benchmark_accuracy_samples',
        ['benchmark_id'],
    )


def upgrade() -> None:
    if table_exists('benchmarks'):
        _add_benchmark_columns()
    _create_datasets_table()
    _create_accuracy_samples_table()


def downgrade() -> None:
    if table_exists('benchmark_accuracy_samples'):
        op.drop_index(
            'ix_benchmark_accuracy_samples_benchmark_id',
            table_name='benchmark_accuracy_samples',
        )
        op.drop_table('benchmark_accuracy_samples')
    if table_exists('benchmark_datasets'):
        op.drop_index(
            'ix_benchmark_datasets_owner_principal_id',
            table_name='benchmark_datasets',
        )
        op.drop_index('ix_benchmark_datasets_name', table_name='benchmark_datasets')
        op.drop_table('benchmark_datasets')
    for name in (
        'kind',
        'dataset_id',
        'dataset_format',
        'dataset_sha256',
        'max_samples',
        'accuracy_concurrency',
        'accuracy_max_out_len',
        'accuracy_score',
        'accuracy_total',
        'accuracy_correct',
        'accuracy_invalid',
        'accuracy_errored',
        'accuracy_elapsed_seconds',
    ):
        if table_exists('benchmarks') and column_exists('benchmarks', name):
            op.drop_column('benchmarks', name)
