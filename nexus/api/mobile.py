"""Mobile mining endpoints — lightweight API for phone-based compute nodes.

Mobile nodes use the same auth and job system as desktop nodes. These
endpoints add mobile-specific features:

  - /checkin: report device conditions (battery, wifi, charging) and get task
  - /model/onnx: download model in ONNX format (for TFLite/CoreML conversion)

For weight submission, mobile nodes use the same /nodes/task/{job}/submit
endpoint as desktop nodes — weights are base64-encoded PyTorch state_dicts
regardless of what format the device trained in.
"""

import base64
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.api.deps import get_current_node
from nexus.core import orchestrator
from nexus.core.conversion import get_or_create_onnx
from nexus.db.models import (
    Checkpoint,
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

router = APIRouter(prefix="/mobile", tags=["mobile"])


# ── Schemas ──────────────────────────────────────────────────────────────────


class MobileCheckinRequest(BaseModel):
    battery_pct: int
    is_charging: bool
    is_wifi: bool
    hardware_info: dict | None = None


class MobileCheckinResponse(BaseModel):
    has_task: bool
    job_id: uuid.UUID | None = None
    job_name: str | None = None
    round_num: int | None = None
    model_type: str | None = None
    num_classes: int | None = None
    config: dict | None = None
    message: str | None = None


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("/checkin", response_model=MobileCheckinResponse)
async def mobile_checkin(
    body: MobileCheckinRequest,
    node: Node = Depends(get_current_node),
    db: AsyncSession = Depends(get_db),
):
    """Mobile check-in: report device conditions, get task if eligible.

    The server checks battery, wifi, and charging status before assigning
    a task. This prevents training on low battery or metered connections.
    """
    # Update heartbeat
    node.last_heartbeat = datetime.now(UTC)
    node.status = NodeStatus.ONLINE
    if body.hardware_info:
        node.hardware_info = body.hardware_info
    await db.commit()

    # Check conditions
    if body.battery_pct < 30:
        return MobileCheckinResponse(has_task=False, message="Battery too low (need >30%)")
    if not body.is_wifi:
        return MobileCheckinResponse(has_task=False, message="WiFi required for training")
    if not body.is_charging:
        return MobileCheckinResponse(has_task=False, message="Device must be charging")

    # Find available task (same logic as /nodes/task)
    result = await db.execute(
        select(NodeAssignment).where(NodeAssignment.node_id == node.id)
    )
    assignments = result.scalars().all()

    for assignment in assignments:
        job = await db.get(Job, assignment.job_id)
        if job is None or job.status != JobStatus.RUNNING:
            continue

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

        # Check if already submitted
        result = await db.execute(
            select(Submission).where(
                Submission.round_id == current_round.id,
                Submission.node_id == node.id,
            )
        )
        if result.scalar_one_or_none() is not None:
            continue

        return MobileCheckinResponse(
            has_task=True,
            job_id=job.id,
            job_name=job.name,
            round_num=job.current_round,
            model_type=job.model_type,
            num_classes=job.num_classes,
            config=job.config,
        )

    return MobileCheckinResponse(has_task=False, message="No tasks available")


@router.get("/model/{job_id}/onnx")
async def get_model_onnx(
    job_id: uuid.UUID,
    node: Node = Depends(get_current_node),
    db: AsyncSession = Depends(get_db),
):
    """Download the current global model in ONNX format.

    Mobile apps use this to get the model in a format they can convert
    to TFLite (Android) or Core ML (iOS). The ONNX file is cached in S3
    so conversion only happens once per round.
    """
    # Verify node is assigned
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

    # Get latest checkpoint
    result = await db.execute(
        select(Checkpoint)
        .where(Checkpoint.job_id == job_id)
        .order_by(Checkpoint.round_num.desc())
        .limit(1)
    )
    checkpoint = result.scalar_one_or_none()
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="No checkpoint found")

    onnx_bytes = get_or_create_onnx(
        job_id=job_id,
        round_num=checkpoint.round_num,
        model_type=job.model_type,
        num_classes=job.num_classes,
        pytorch_checkpoint_path=checkpoint.path,
    )

    return Response(
        content=onnx_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=model_round_{checkpoint.round_num}.onnx",
            "X-Round-Num": str(checkpoint.round_num),
            "X-Model-Type": job.model_type,
            "X-Num-Classes": str(job.num_classes),
        },
    )
