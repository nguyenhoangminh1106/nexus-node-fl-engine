"""SQLAlchemy ORM models for Nexus Node FL Engine."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── Enums ────────────────────────────────────────────────────────────────────


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RoundStatus(str, enum.Enum):
    WAITING = "waiting"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"


class NodeStatus(str, enum.Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    TRAINING = "training"


class DeviceType(str, enum.Enum):
    DESKTOP = "desktop"
    MOBILE = "mobile"


# ── Organizations ────────────────────────────────────────────────────────────


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    api_key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    jobs: Mapped[list["Job"]] = relationship(back_populates="organization", cascade="all, delete-orphan")


# ── Jobs ─────────────────────────────────────────────────────────────────────


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.PENDING)

    # Model config
    model_type: Mapped[str] = mapped_column(String(100), nullable=False, default="plantnet")
    num_classes: Mapped[int] = mapped_column(Integer, nullable=False, default=38)

    # FL config
    total_rounds: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    current_round: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    min_nodes: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Dataset
    dataset_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    organization: Mapped["Organization"] = relationship(back_populates="jobs")
    rounds: Mapped[list["Round"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    checkpoints: Mapped[list["Checkpoint"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    assignments: Mapped[list["NodeAssignment"]] = relationship(back_populates="job", cascade="all, delete-orphan")


# ── Rounds ───────────────────────────────────────────────────────────────────


class Round(Base):
    __tablename__ = "rounds"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"), nullable=False)
    round_num: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[RoundStatus] = mapped_column(Enum(RoundStatus), default=RoundStatus.WAITING)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    job: Mapped["Job"] = relationship(back_populates="rounds")
    submissions: Mapped[list["Submission"]] = relationship(back_populates="round", cascade="all, delete-orphan")


# ── Nodes ────────────────────────────────────────────────────────────────────


class Node(Base):
    __tablename__ = "nodes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    api_key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[NodeStatus] = mapped_column(Enum(NodeStatus), default=NodeStatus.OFFLINE)
    device_type: Mapped[DeviceType] = mapped_column(Enum(DeviceType), default=DeviceType.DESKTOP)
    region: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Hardware reported by node
    hardware_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Trust & reputation
    trust_score: Mapped[float] = mapped_column(Float, default=0.5)
    rounds_completed: Mapped[int] = mapped_column(Integer, default=0)
    rounds_failed: Mapped[int] = mapped_column(Integer, default=0)

    last_heartbeat: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    submissions: Mapped[list["Submission"]] = relationship(back_populates="node")
    assignments: Mapped[list["NodeAssignment"]] = relationship(back_populates="node")


# ── Node Assignments (which nodes are assigned to which jobs) ────────────────


class NodeAssignment(Base):
    __tablename__ = "node_assignments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"), nullable=False)
    node_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("nodes.id"), nullable=False)
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped["Job"] = relationship(back_populates="assignments")
    node: Mapped["Node"] = relationship(back_populates="assignments")


# ── Submissions ──────────────────────────────────────────────────────────────


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    round_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("rounds.id"), nullable=False)
    node_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("nodes.id"), nullable=False)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    weight_path: Mapped[str] = mapped_column(Text, nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    round: Mapped["Round"] = relationship(back_populates="submissions")
    node: Mapped["Node"] = relationship(back_populates="submissions")


# ── Checkpoints ──────────────────────────────────────────────────────────────


class Checkpoint(Base):
    __tablename__ = "checkpoints"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"), nullable=False)
    round_num: Mapped[int] = mapped_column(Integer, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped["Job"] = relationship(back_populates="checkpoints")
