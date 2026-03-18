"""Node endpoints — heartbeat, task polling, weight submission, stats."""

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.api.deps import get_current_node
from nexus.core import orchestrator
from nexus.db.models import (
    Job,
    JobStatus,
    Node,
    NodeAssignment,
    NodeStatus,
    Round,
    RoundStatus,
    Submission,
)
from nexus.db.session import get_db

router = APIRouter(prefix="/nodes", tags=["nodes"])


# ── Schemas ──────────────────────────────────────────────────────────────────


class HeartbeatRequest(BaseModel):
    hardware_info: dict | None = None
    status: str = "online"


class HeartbeatResponse(BaseModel):
    node_id: uuid.UUID
    status: str
    server_time: str


class TaskResponse(BaseModel):
    job_id: uuid.UUID
    job_name: str
    round_num: int
    model_type: str
    num_classes: int
    config: dict | None


class SubmitWeightsRequest(BaseModel):
    round_num: int
    n_samples: int
    weights: str  # base64-encoded state_dict


class NodeStatsResponse(BaseModel):
    node_id: uuid.UUID
    name: str
    device_type: str
    trust_score: float
    rounds_completed: int
    rounds_failed: int
    status: str


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("/heartbeat", response_model=HeartbeatResponse)
async def heartbeat(
    body: HeartbeatRequest,
    node: Node = Depends(get_current_node),
    db: AsyncSession = Depends(get_db),
):
    """Report node health. Should be called every 30 seconds."""
    node.last_heartbeat = datetime.now(UTC)
    if body.hardware_info:
        node.hardware_info = body.hardware_info
    node.status = NodeStatus(body.status) if body.status in NodeStatus.__members__ else NodeStatus.ONLINE
    await db.commit()
    return HeartbeatResponse(
        node_id=node.id,
        status=node.status.value,
        server_time=datetime.now(UTC).isoformat(),
    )


@router.get("/task", response_model=TaskResponse | None)
async def poll_task(
    node: Node = Depends(get_current_node),
    db: AsyncSession = Depends(get_db),
):
    """Poll for an assigned task. Returns the current training task or null.

    A task exists when the node is assigned to a running job that has a
    round in WAITING or TRAINING status that the node hasn't submitted to yet.
    """
    # Find jobs this node is assigned to
    result = await db.execute(
        select(NodeAssignment).where(NodeAssignment.node_id == node.id)
    )
    assignments = result.scalars().all()

    for assignment in assignments:
        job = await db.get(Job, assignment.job_id)
        if job is None or job.status != JobStatus.RUNNING:
            continue

        # Find the current round
        result = await db.execute(
            select(Round).where(
                Round.job_id == job.id,
                Round.round_num == job.current_round,
                Round.status.in_([RoundStatus.WAITING, RoundStatus.TRAINING]),
            )
        )
        current_round = result.scalar_one_or_none()
        if current_round is None:
            continue

        # Check if node already submitted for this round
        result = await db.execute(
            select(Submission).where(
                Submission.round_id == current_round.id,
                Submission.node_id == node.id,
            )
        )
        if result.scalar_one_or_none() is not None:
            continue  # Already submitted

        return TaskResponse(
            job_id=job.id,
            job_name=job.name,
            round_num=job.current_round,
            model_type=job.model_type,
            num_classes=job.num_classes,
            config=job.config,
        )

    return None


@router.get("/task/{job_id}/model")
async def get_task_model(
    job_id: uuid.UUID,
    node: Node = Depends(get_current_node),
    db: AsyncSession = Depends(get_db),
):
    """Download the current global model for a job (base64-encoded)."""
    # Verify node is assigned to this job
    result = await db.execute(
        select(NodeAssignment).where(
            NodeAssignment.job_id == job_id,
            NodeAssignment.node_id == node.id,
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=403, detail="Node not assigned to this job")

    job = await db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    model_b64, round_num = await orchestrator.get_current_model(db, job_id)
    return {
        "model": model_b64,
        "round": round_num,
        "job_id": str(job_id),
        "model_type": job.model_type,
        "num_classes": job.num_classes,
    }


@router.post("/task/{job_id}/submit")
async def submit_task_weights(
    job_id: uuid.UUID,
    body: SubmitWeightsRequest,
    node: Node = Depends(get_current_node),
    db: AsyncSession = Depends(get_db),
):
    """Submit trained weights for a job round."""
    # Verify node is assigned
    result = await db.execute(
        select(NodeAssignment).where(
            NodeAssignment.job_id == job_id,
            NodeAssignment.node_id == node.id,
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=403, detail="Node not assigned to this job")

    resp = await orchestrator.submit_weights(
        db=db,
        job_id=job_id,
        node_id=node.id,
        round_num=body.round_num,
        n_samples=body.n_samples,
        weights_b64=body.weights,
    )
    return resp


@router.get("/stats", response_model=NodeStatsResponse)
async def get_stats(
    node: Node = Depends(get_current_node),
):
    """Get this node's stats and reputation."""
    return NodeStatsResponse(
        node_id=node.id,
        name=node.name,
        device_type=node.device_type.value,
        trust_score=node.trust_score,
        rounds_completed=node.rounds_completed,
        rounds_failed=node.rounds_failed,
        status=node.status.value,
    )
