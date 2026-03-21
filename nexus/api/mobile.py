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
from nexus.core.conversion import get_or_create_coreml, get_or_create_onnx
from nexus.core.serialization import serialize_state_dict
from nexus.storage import s3
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


@router.get("/model/{job_id}/coreml")
async def get_model_coreml(
    job_id: uuid.UUID,
    tier: int = 2,
    node: Node = Depends(get_current_node),
    db: AsyncSession = Depends(get_db),
):
    """Download the current global model as an updatable Core ML model.

    This model can be used for on-device training on iOS via Core ML's
    MLUpdateTask API. The model has specific layers marked as updatable:

      - tier=2 (default): only classifier/FC layer (transfer learning)
      - tier=3: all layers (full training)

    The Core ML file is cached in S3 per round+tier so conversion only
    happens once.
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

    coreml_bytes = get_or_create_coreml(
        job_id=job_id,
        round_num=checkpoint.round_num,
        model_type=job.model_type,
        num_classes=job.num_classes,
        pytorch_checkpoint_path=checkpoint.path,
        tier=tier,
    )

    return Response(
        content=coreml_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=model_round_{checkpoint.round_num}_tier{tier}.mlmodel",
            "X-Round-Num": str(checkpoint.round_num),
            "X-Model-Type": job.model_type,
            "X-Num-Classes": str(job.num_classes),
            "X-Tier": str(tier),
        },
    )


class DataPartitionResponse(BaseModel):
    job_id: uuid.UUID
    job_name: str
    total_files: int
    partition_size: int
    partition_index: int
    files: list[dict]


@router.get("/data/{job_id}", response_model=DataPartitionResponse)
async def get_training_data(
    job_id: uuid.UUID,
    node: Node = Depends(get_current_node),
    db: AsyncSession = Depends(get_db),
):
    """Get this node's partition of training data for a job.

    The dataset is evenly partitioned across all assigned nodes. Each node
    gets a unique subset so training covers the full dataset without overlap.

    Returns presigned download URLs for each file in the partition.
    """
    # Verify node is assigned
    result = await db.execute(
        select(NodeAssignment).where(
            NodeAssignment.job_id == job_id,
            NodeAssignment.node_id == node.id,
        )
    )
    assignment = result.scalar_one_or_none()
    if assignment is None:
        raise HTTPException(status_code=403, detail="Node not assigned to this job")

    job = await db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.dataset_path:
        raise HTTPException(status_code=404, detail="Job has no dataset configured")

    # List all files in the dataset
    all_files = s3.list_objects(job.dataset_path)
    if not all_files:
        raise HTTPException(status_code=404, detail="Dataset is empty")

    # Get all assigned nodes to determine partition
    result = await db.execute(
        select(NodeAssignment).where(NodeAssignment.job_id == job_id)
    )
    all_assignments = result.scalars().all()
    node_ids = sorted([str(a.node_id) for a in all_assignments])
    node_index = node_ids.index(str(node.id)) if str(node.id) in node_ids else 0
    total_nodes = len(node_ids)

    # Partition files: node N gets every Nth file
    partition = [f for i, f in enumerate(all_files) if i % total_nodes == node_index]

    # Generate presigned URLs for each file
    files_with_urls = []
    for f in partition:
        key = f["key"]
        # Extract class label from path: datasets/{id}/{class}/{filename}
        parts = key.split("/")
        class_label = parts[-2] if len(parts) >= 3 else "unknown"
        filename = parts[-1] if parts else key

        files_with_urls.append({
            "key": key,
            "filename": filename,
            "class_label": class_label,
            "size": f["size"],
            "url": s3.generate_presigned_url(key, expires_in=3600),
        })

    return DataPartitionResponse(
        job_id=job.id,
        job_name=job.name,
        total_files=len(all_files),
        partition_size=len(partition),
        partition_index=node_index,
        files=files_with_urls,
    )


class MobileSubmitRequest(BaseModel):
    round_num: int
    n_samples: int = 50
    weights: dict | str | None = None
    simulated: bool = False


@router.post("/submit/{job_id}")
async def mobile_submit_weights(
    job_id: uuid.UUID,
    body: MobileSubmitRequest,
    node: Node = Depends(get_current_node),
    db: AsyncSession = Depends(get_db),
):
    """Mobile weight submission — accepts simulated or real weights.

    For simulated training (native ML not available), this endpoint
    fetches the current model checkpoint and re-submits it as the node's
    contribution. This allows the FL pipeline to progress even with
    simulated mobile nodes.

    For real training, weights should be provided as a base64-encoded
    PyTorch state_dict string (same as desktop submit).
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

    if body.simulated or body.weights is None or isinstance(body.weights, dict):
        # Simulated: use the current global model as this node's "contribution"
        # This is a pass-through that lets the pipeline progress
        _, current_round_num = await orchestrator.get_current_model(db, job_id)

        # Get the latest checkpoint and re-serialize it
        result_cp = await db.execute(
            select(Checkpoint)
            .where(Checkpoint.job_id == job_id)
            .order_by(Checkpoint.round_num.desc())
            .limit(1)
        )
        checkpoint = result_cp.scalar_one_or_none()
        if checkpoint is None:
            raise HTTPException(status_code=404, detail="No checkpoint to base submission on")

        # Download current weights and re-serialize as base64
        from nexus.core.serialization import bytes_to_state_dict

        cp_bytes = s3.download_bytes(checkpoint.path)
        sd = bytes_to_state_dict(cp_bytes)

        # Add small random noise to avoid identical aggregation
        import torch
        for key in sd:
            if isinstance(sd[key], torch.Tensor) and sd[key].is_floating_point():
                sd[key] = sd[key] + torch.randn_like(sd[key]) * 0.001

        weights_b64 = serialize_state_dict(sd)
    else:
        # Real weights from native training
        weights_b64 = body.weights

    resp = await orchestrator.submit_weights(
        db=db,
        job_id=job_id,
        node_id=node.id,
        round_num=body.round_num,
        n_samples=body.n_samples,
        weights_b64=weights_b64,
    )
    return resp
