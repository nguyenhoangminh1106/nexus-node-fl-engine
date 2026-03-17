"""Shared FastAPI dependencies: DB sessions, authentication, etc."""

import uuid
from collections.abc import AsyncGenerator

from fastapi import Depends, Header, HTTPException, status
from passlib.hash import bcrypt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.db.models import Node, Organization
from nexus.db.session import get_db


async def get_session(session: AsyncSession = Depends(get_db)) -> AsyncGenerator[AsyncSession, None]:
    """Alias for get_db — makes it clear at call sites."""
    yield session


async def _resolve_api_key(
    db: AsyncSession, api_key: str, model_cls, label: str,
):
    """Look up an entity by trying the API key against all hashed keys.

    This is intentionally simple (scan + bcrypt.verify) and fine for a
    small-to-medium number of orgs/nodes. For large scale, switch to a
    prefix-based lookup (store first 8 chars of the key unhashed).
    """
    result = await db.execute(select(model_cls))
    entities = result.scalars().all()
    for entity in entities:
        if bcrypt.verify(api_key, entity.api_key_hash):
            return entity
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=f"Invalid {label} API key",
    )


async def get_current_org(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> Organization:
    """Authenticate a B2B caller (T3 app) via API key."""
    return await _resolve_api_key(db, x_api_key, Organization, "organization")


async def get_current_node(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> Node:
    """Authenticate a compute node via API key."""
    return await _resolve_api_key(db, x_api_key, Node, "node")
