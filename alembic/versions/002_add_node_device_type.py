"""Add device_type column to nodes table.

Revision ID: 002_add_node_device_type
Revises: 001_baseline
Create Date: 2026-03-18
"""

from alembic import op
import sqlalchemy as sa

revision = "002_add_node_device_type"
down_revision = "001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    columns = [c["name"] for c in sa.inspect(conn).get_columns("nodes")]
    if "device_type" not in columns:
        # Drop any leftover enum from a prior failed migration, then recreate
        # with the correct lowercase values that match our Python model.
        conn.execute(sa.text("DROP TYPE IF EXISTS devicetype"))
        conn.execute(sa.text(
            "CREATE TYPE devicetype AS ENUM ('DESKTOP', 'MOBILE')"
        ))
        conn.execute(sa.text(
            "ALTER TABLE nodes ADD COLUMN device_type devicetype DEFAULT 'DESKTOP'::devicetype"
        ))


def downgrade() -> None:
    op.drop_column("nodes", "device_type")
