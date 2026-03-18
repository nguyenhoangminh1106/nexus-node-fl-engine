"""Baseline schema — creates all tables from scratch or stamps existing DB.

Revision ID: 001_baseline
Revises:
Create Date: 2026-03-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # If the tables already exist (created by the old create_all approach),
    # skip creation so this migration is a safe no-op on existing databases.
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing = inspector.get_table_names()

    if "organizations" not in existing:
        op.create_table(
            "organizations",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("name", sa.String(255), unique=True, nullable=False),
            sa.Column("api_key_hash", sa.String(255), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        )

    if "jobs" not in existing:
        op.create_table(
            "jobs",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("org_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("organizations.id"), nullable=False),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("status", sa.Enum("PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", name="jobstatus"), default="PENDING"),
            sa.Column("model_type", sa.String(100), nullable=False, server_default="plantnet"),
            sa.Column("num_classes", sa.Integer, nullable=False, server_default="38"),
            sa.Column("total_rounds", sa.Integer, nullable=False, server_default="10"),
            sa.Column("current_round", sa.Integer, nullable=False, server_default="0"),
            sa.Column("min_nodes", sa.Integer, nullable=False, server_default="2"),
            sa.Column("config", sa.JSON, nullable=True),
            sa.Column("dataset_path", sa.Text, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        )

    if "rounds" not in existing:
        op.create_table(
            "rounds",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("jobs.id"), nullable=False),
            sa.Column("round_num", sa.Integer, nullable=False),
            sa.Column("status", sa.Enum("WAITING", "TRAINING", "AGGREGATING", "COMPLETED", name="roundstatus"), default="WAITING"),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        )

    if "nodes" not in existing:
        op.create_table(
            "nodes",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("api_key_hash", sa.String(255), nullable=False),
            sa.Column("status", sa.Enum("ONLINE", "OFFLINE", "TRAINING", name="nodestatus"), default="OFFLINE"),
            sa.Column("region", sa.String(100), nullable=True),
            sa.Column("hardware_info", sa.JSON, nullable=True),
            sa.Column("trust_score", sa.Float, default=0.5),
            sa.Column("rounds_completed", sa.Integer, default=0),
            sa.Column("rounds_failed", sa.Integer, default=0),
            sa.Column("last_heartbeat", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        )

    if "node_assignments" not in existing:
        op.create_table(
            "node_assignments",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("jobs.id"), nullable=False),
            sa.Column("node_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("nodes.id"), nullable=False),
            sa.Column("assigned_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        )

    if "submissions" not in existing:
        op.create_table(
            "submissions",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("round_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("rounds.id"), nullable=False),
            sa.Column("node_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("nodes.id"), nullable=False),
            sa.Column("n_samples", sa.Integer, nullable=False),
            sa.Column("weight_path", sa.Text, nullable=False),
            sa.Column("submitted_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        )

    if "checkpoints" not in existing:
        op.create_table(
            "checkpoints",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("jobs.id"), nullable=False),
            sa.Column("round_num", sa.Integer, nullable=False),
            sa.Column("path", sa.Text, nullable=False),
            sa.Column("accuracy", sa.Float, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        )


def downgrade() -> None:
    op.drop_table("checkpoints")
    op.drop_table("submissions")
    op.drop_table("node_assignments")
    op.drop_table("nodes")
    op.drop_table("rounds")
    op.drop_table("jobs")
    op.drop_table("organizations")
