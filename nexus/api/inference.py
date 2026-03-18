"""Inference endpoint — list and download trained checkpoints."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.api.deps import get_current_org
from nexus.config import settings
from nexus.db.models import Checkpoint, Job, Organization
from nexus.db.session import get_db
from nexus.storage import s3

router = APIRouter(prefix="/inference", tags=["inference"])


class CheckpointResponse(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID
    round_num: int
    accuracy: float | None
    created_at: str

    model_config = {"from_attributes": True}


@router.get("/checkpoints/{job_id}", response_model=list[CheckpointResponse])
async def list_checkpoints(
    job_id: uuid.UUID,
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """List all saved checkpoints for a job."""
    job = await db.get(Job, job_id)
    if job is None or job.org_id != org.id:
        raise HTTPException(status_code=404, detail="Job not found")

    result = await db.execute(
        select(Checkpoint)
        .where(Checkpoint.job_id == job_id)
        .order_by(Checkpoint.round_num)
    )
    checkpoints = result.scalars().all()
    return [
        CheckpointResponse(
            id=c.id,
            job_id=c.job_id,
            round_num=c.round_num,
            accuracy=c.accuracy,
            created_at=c.created_at.isoformat(),
        )
        for c in checkpoints
    ]


@router.get("/checkpoints/{job_id}/download")
async def download_checkpoint(
    job_id: uuid.UUID,
    round_num: int | None = None,
    org: Organization = Depends(get_current_org),
    db: AsyncSession = Depends(get_db),
):
    """Download a checkpoint.

    If S3_PUBLIC_URL is configured: returns a presigned URL (client downloads
    directly from S3 — efficient for large models).

    If S3_PUBLIC_URL is not set: streams the file through the API server
    (works everywhere but uses server bandwidth).

    If round_num is omitted, returns the latest checkpoint.
    """
    job = await db.get(Job, job_id)
    if job is None or job.org_id != org.id:
        raise HTTPException(status_code=404, detail="Job not found")

    query = select(Checkpoint).where(Checkpoint.job_id == job_id)
    if round_num is not None:
        query = query.where(Checkpoint.round_num == round_num)
    else:
        query = query.order_by(Checkpoint.round_num.desc()).limit(1)

    result = await db.execute(query)
    checkpoint = result.scalar_one_or_none()
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    # If public S3 URL is configured, return presigned URL
    if settings.s3_public_url:
        url = s3.generate_presigned_url(checkpoint.path, expires_in=3600)
        return {
            "download_url": url,
            "round_num": checkpoint.round_num,
            "expires_in": 3600,
        }

    # Otherwise, stream the file through the API
    def _stream():
        data = s3.download_bytes(checkpoint.path)
        yield data

    return StreamingResponse(
        _stream(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=checkpoint_round_{checkpoint.round_num}.pth",
            "X-Round-Num": str(checkpoint.round_num),
            "X-Job-Id": str(job_id),
        },
    )
