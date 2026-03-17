"""Seed the database with the first admin organization on startup."""

import secrets

from passlib.hash import bcrypt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.config import settings
from nexus.db.models import Organization

import structlog

logger = structlog.get_logger()


async def seed_initial_org(db: AsyncSession) -> None:
    """Create the seed org if the organizations table is empty."""
    result = await db.execute(select(Organization).limit(1))
    if result.scalar_one_or_none() is not None:
        return  # Already seeded

    api_key = f"nxo_{secrets.token_urlsafe(32)}"
    org = Organization(
        name=settings.seed_org_name,
        api_key_hash=bcrypt.hash(api_key),
    )
    db.add(org)
    await db.commit()

    logger.info(
        "seed_org_created",
        name=settings.seed_org_name,
        api_key=api_key,
    )
    print(f"\n{'='*60}")
    print(f"  SEED ORGANIZATION CREATED")
    print(f"  Name    : {settings.seed_org_name}")
    print(f"  API Key : {api_key}")
    print(f"  (save this — it won't be shown again)")
    print(f"{'='*60}\n")
