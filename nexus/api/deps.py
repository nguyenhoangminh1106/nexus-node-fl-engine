"""Shared FastAPI dependencies: DB sessions, authentication, etc."""

import hashlib
import hmac
from collections.abc import AsyncGenerator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.db.models import Node, Organization
from nexus.db.session import get_db


def hash_api_key(api_key: str) -> str:
    """SHA-256 hash an API key. Fine for high-entropy random tokens."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, stored_hash: str) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    return hmac.compare_digest(hash_api_key(api_key), stored_hash)


async def get_session(session: AsyncSession = Depends(get_db)) -> AsyncGenerator[AsyncSession, None]:
    """Alias for get_db — makes it clear at call sites."""
    yield session


async def _resolve_api_key(
    db: AsyncSession, api_key: str, model_cls, label: str,
):
    """Look up an entity by matching the API key hash."""
    key_hash = hash_api_key(api_key)
    result = await db.execute(
        select(model_cls).where(model_cls.api_key_hash == key_hash)
    )
    entity = result.scalar_one_or_none()
    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid {label} API key",
        )
    return entity


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
