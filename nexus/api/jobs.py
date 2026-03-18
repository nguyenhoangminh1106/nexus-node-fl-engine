"""Job management endpoints — create, list, get, cancel training jobs."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from nexus.api.deps import get_current_org
from nexus.core import model_registry, orchestrator
from nexus.db.models import Job, JobStatus, Organization, Round
from nexus.db.session import get_db

router = APIRouter(prefix="/jobs", tags=["jobs"])


# ── Schemas ──────────────────────────────────────────────────────────────────


class CreateJobRequest(BaseModel):
    name: str
    model_type: str = "plantnet"
    num_classes: int = 38
    total_rounds: int = 10
    min_nodes: int = 2
    config: dict | None = None
    dataset_path: str | None = None
    node_ids: list[uuid.UUID] | None = None


class JobResponse(BaseModel):
    id: uuid.UUID
    name: str
    status: str
    model_type: str
    num_classes: int
    total_rounds: int
    current_round: int
    min_nodes: int
    config: dict | None
    dataset_path: str | None

    model_config = {"from_attributes": True}


class JobDetailResponse(JobResponse):
    rounds: list["RoundResponse"]


class RoundResponse(BaseModel):
    id: uuid.UUID
    round_num: int
    status: str
    started_at: str | None
    completed_at: str | None
    submission_count: int

    model_config = {"from_attributes": True}


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("", response_model=JobResponse)
async def create_job(
    body: CreateJobRequest,
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Create a new federated training job.

    Validates the model_type exists in the registry, creates the job,
    optionally assigns nodes, and starts the first round.
    """
    # Validate model type
    try:
        model_registry.get(body.model_type)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = Job(
        org_id=org.id,
        name=body.name,
        model_type=body.model_type,
        num_classes=body.num_classes,
        total_rounds=body.total_rounds,
        min_nodes=body.min_nodes,
        config=body.config,
        dataset_path=body.dataset_path,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Assign nodes if provided
    if body.node_ids:
        await orchestrator.assign_nodes(db, job, body.node_ids)

    # Start the job (build model, save initial checkpoint, create round 0)
    job = await orchestrator.start_job(db, job)

    return job


@router.get("", response_model=list[JobResponse])
async def list_jobs(
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """List all training jobs for the authenticated organization."""
    result = await db.execute(
        select(Job).where(Job.org_id == org.id).order_by(Job.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{job_id}", response_model=JobDetailResponse)
async def get_job(
    job_id: uuid.UUID,
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Get detailed job info including all rounds and their submission counts."""
    result = await db.execute(
        select(Job)
        .where(Job.id == job_id, Job.org_id == org.id)
        .options(selectinload(Job.rounds).selectinload(Round.submissions))
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    rounds = []
    for r in sorted(job.rounds, key=lambda x: x.round_num):
        rounds.append(RoundResponse(
            id=r.id,
            round_num=r.round_num,
            status=r.status.value,
            started_at=r.started_at.isoformat() if r.started_at else None,
            completed_at=r.completed_at.isoformat() if r.completed_at else None,
            submission_count=len(r.submissions),
        ))

    return JobDetailResponse(
        id=job.id,
        name=job.name,
        status=job.status.value,
        model_type=job.model_type,
        num_classes=job.num_classes,
        total_rounds=job.total_rounds,
        current_round=job.current_round,
        min_nodes=job.min_nodes,
        config=job.config,
        dataset_path=job.dataset_path,
        rounds=rounds,
    )


@router.delete("/{job_id}")
async def cancel_job(
    job_id: uuid.UUID,
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Cancel a running job."""
    job = await db.get(Job, job_id)
    if job is None or job.org_id != org.id:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job already completed")

    job.status = JobStatus.CANCELLED
    await db.commit()
    return {"status": "cancelled", "job_id": str(job_id)}
