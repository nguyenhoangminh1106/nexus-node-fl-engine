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
    # Only add if it doesn't already exist (fresh DBs get it from baseline).
    conn = op.get_bind()
    columns = [c["name"] for c in sa.inspect(conn).get_columns("nodes")]
    if "device_type" not in columns:
        devicetype_enum = sa.Enum("desktop", "mobile", name="devicetype")
        devicetype_enum.create(conn, checkfirst=True)
        op.add_column(
            "nodes",
            sa.Column("device_type", devicetype_enum, server_default="desktop"),
        )


def downgrade() -> None:
    op.drop_column("nodes", "device_type")
