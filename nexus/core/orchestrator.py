"""FL Orchestrator — manages the lifecycle of training jobs and rounds.

This is the brain of the system. It handles:
  - Starting new rounds and assigning nodes
  - Receiving weight submissions from nodes
  - Triggering FedAvg when all submissions arrive
  - Saving checkpoints after each round
  - Advancing jobs through their round sequence
"""

import uuid
from datetime import UTC, datetime

import structlog
import torch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from nexus.core import model_registry
from nexus.core.fedavg import weighted_fedavg
from nexus.core.serialization import (
    bytes_to_state_dict,
    deserialize_state_dict,
    serialize_state_dict,
    state_dict_to_bytes,
)
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
from nexus.storage import s3

logger = structlog.get_logger()


def _checkpoint_key(job_id: uuid.UUID, round_num: int) -> str:
    return f"checkpoints/{job_id}/round_{round_num}.pth"


def _submission_key(job_id: uuid.UUID, round_num: int, node_id: uuid.UUID) -> str:
    return f"submissions/{job_id}/round_{round_num}/{node_id}.pth"


async def start_job(db: AsyncSession, job: Job) -> Job:
    """Initialize a job: build the model, save initial checkpoint, create round 0."""
    model_def = model_registry.get(job.model_type)
    model = model_def.build(num_classes=job.num_classes, pretrained=True)

    # Save initial model as round -1 checkpoint (the starting point)
    initial_sd = model.state_dict()
    key = _checkpoint_key(job.id, -1)
    s3.upload_bytes(key, state_dict_to_bytes(initial_sd))

    checkpoint = Checkpoint(job_id=job.id, round_num=-1, path=key)
    db.add(checkpoint)

    # Create the first round
    round_0 = Round(job_id=job.id, round_num=0, status=RoundStatus.WAITING)
    db.add(round_0)

    job.status = JobStatus.RUNNING
    job.current_round = 0
    await db.commit()

    logger.info("job_started", job_id=str(job.id), model=job.model_type, rounds=job.total_rounds)
    return job


async def get_current_model(db: AsyncSession, job_id: uuid.UUID) -> tuple[str, int]:
    """Get the latest checkpoint for a job. Returns (base64_model, round_num)."""
    result = await db.execute(
        select(Checkpoint)
        .where(Checkpoint.job_id == job_id)
        .order_by(Checkpoint.round_num.desc())
        .limit(1)
    )
    checkpoint = result.scalar_one_or_none()
    if checkpoint is None:
        raise ValueError(f"No checkpoint found for job {job_id}")

    data = s3.download_bytes(checkpoint.path)
    sd = bytes_to_state_dict(data)
    return serialize_state_dict(sd), checkpoint.round_num


async def submit_weights(
    db: AsyncSession,
    job_id: uuid.UUID,
    node_id: uuid.UUID,
    round_num: int,
    n_samples: int,
    weights_b64: str,
) -> dict:
    """Accept a weight submission from a node and check if FedAvg should run."""
    # Load the job with its current round
    job = await db.get(Job, job_id)
    if job is None:
        return {"status": "error", "reason": "Job not found"}
    if job.status != JobStatus.RUNNING:
        return {"status": "ignored", "reason": f"Job status is {job.status.value}"}
    if round_num != job.current_round:
        return {
            "status": "ignored",
            "reason": f"Round mismatch: job is at round {job.current_round}, got {round_num}",
        }

    # Find the current round record
    result = await db.execute(
        select(Round).where(Round.job_id == job_id, Round.round_num == round_num)
    )
    current_round = result.scalar_one_or_none()
    if current_round is None:
        return {"status": "error", "reason": f"Round {round_num} not found"}

    # Deserialize and store weights in S3
    sd = deserialize_state_dict(weights_b64)
    key = _submission_key(job_id, round_num, node_id)
    s3.upload_bytes(key, state_dict_to_bytes(sd))

    # Record submission
    submission = Submission(
        round_id=current_round.id,
        node_id=node_id,
        n_samples=n_samples,
        weight_path=key,
    )
    db.add(submission)

    if current_round.status == RoundStatus.WAITING:
        current_round.status = RoundStatus.TRAINING
        current_round.started_at = datetime.now(UTC)

    await db.commit()

    logger.info(
        "weights_submitted",
        job_id=str(job_id),
        node_id=str(node_id),
        round=round_num,
        n_samples=n_samples,
    )

    # Check if we should run FedAvg
    await _maybe_aggregate(db, job, current_round)

    return {"status": "accepted", "round": job.current_round}


async def _maybe_aggregate(db: AsyncSession, job: Job, current_round: Round) -> None:
    """Check if all assigned nodes have submitted, and run FedAvg if so."""
    # Count assignments for this job
    result = await db.execute(
        select(NodeAssignment).where(NodeAssignment.job_id == job.id)
    )
    assignments = result.scalars().all()
    assigned_node_ids = {a.node_id for a in assignments}

    # Count submissions for this round
    result = await db.execute(
        select(Submission).where(Submission.round_id == current_round.id)
    )
    submissions = result.scalars().all()
    submitted_node_ids = {s.node_id for s in submissions}

    if not assigned_node_ids.issubset(submitted_node_ids):
        remaining = assigned_node_ids - submitted_node_ids
        logger.debug(
            "waiting_for_submissions",
            round=current_round.round_num,
            remaining=len(remaining),
        )
        return

    # All nodes submitted — run FedAvg
    current_round.status = RoundStatus.AGGREGATING
    await db.commit()

    logger.info("fedavg_starting", job_id=str(job.id), round=current_round.round_num)

    # Load all submissions from S3
    fedavg_inputs: list[tuple[dict, int]] = []
    for sub in submissions:
        data = s3.download_bytes(sub.weight_path)
        sd = bytes_to_state_dict(data)
        fedavg_inputs.append((sd, sub.n_samples))

    # Run weighted FedAvg
    aggregated_sd = weighted_fedavg(fedavg_inputs)

    # Save the aggregated model as this round's checkpoint
    ckpt_key = _checkpoint_key(job.id, current_round.round_num)
    s3.upload_bytes(ckpt_key, state_dict_to_bytes(aggregated_sd))

    checkpoint = Checkpoint(
        job_id=job.id,
        round_num=current_round.round_num,
        path=ckpt_key,
    )
    db.add(checkpoint)

    current_round.status = RoundStatus.COMPLETED
    current_round.completed_at = datetime.now(UTC)

    # Advance job to next round or complete
    next_round_num = current_round.round_num + 1
    if next_round_num >= job.total_rounds:
        job.status = JobStatus.COMPLETED
        job.current_round = current_round.round_num
        logger.info("job_completed", job_id=str(job.id), total_rounds=job.total_rounds)
    else:
        job.current_round = next_round_num
        next_round = Round(job_id=job.id, round_num=next_round_num, status=RoundStatus.WAITING)
        db.add(next_round)
        logger.info("round_advanced", job_id=str(job.id), new_round=next_round_num)

    # Update node stats
    for sub in submissions:
        node = await db.get(Node, sub.node_id)
        if node:
            node.rounds_completed += 1

    await db.commit()

    # Clean up submission weights from S3 (checkpoint is the durable artifact)
    for sub in submissions:
        try:
            s3.delete_object(sub.weight_path)
        except Exception:
            pass


async def assign_nodes(db: AsyncSession, job: Job, node_ids: list[uuid.UUID]) -> list[NodeAssignment]:
    """Assign specific nodes to a job."""
    assignments = []
    for node_id in node_ids:
        assignment = NodeAssignment(job_id=job.id, node_id=node_id)
        db.add(assignment)
        assignments.append(assignment)
    await db.commit()
    logger.info("nodes_assigned", job_id=str(job.id), count=len(node_ids))
    return assignments
