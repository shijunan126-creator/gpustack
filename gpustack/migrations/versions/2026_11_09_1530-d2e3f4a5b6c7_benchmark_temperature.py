"""A benchmark's sampling temperature.

The baseline preset asks for the reference project's 0.01, so the column has to
exist for the value to reach the runner. Nullable: a run that never states one
leaves the server's own default in place.

Revision ID: d2e3f4a5b6c7
Revises: c1f2a3b4d5e6
Create Date: 2026-11-09 15:30:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.migrations.utils import column_exists, table_exists

revision: str = 'd2e3f4a5b6c7'
down_revision: Union[str, None] = 'c1f2a3b4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if table_exists('benchmarks') and not column_exists('benchmarks', 'temperature'):
        op.add_column(
            'benchmarks',
            sa.Column('temperature', sa.Float(), nullable=True),
        )


def downgrade() -> None:
    if table_exists('benchmarks') and column_exists('benchmarks', 'temperature'):
        op.drop_column('benchmarks', 'temperature')
