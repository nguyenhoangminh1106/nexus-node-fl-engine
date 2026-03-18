"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.db.session import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Liveness probe — always returns OK if the process is running."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(db: AsyncSession = Depends(get_db)):
    """Readiness probe — checks DB connectivity."""
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ready", "db": "ok"}
    except Exception as e:
        return {"status": "not_ready", "db": str(e)}
