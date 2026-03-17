"""Auth endpoints — register organizations and compute nodes."""

import secrets
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.api.deps import hash_api_key
from nexus.db.models import Node, Organization
from nexus.db.session import get_db

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Schemas ──────────────────────────────────────────────────────────────────


class RegisterOrgRequest(BaseModel):
    name: str


class RegisterOrgResponse(BaseModel):
    id: uuid.UUID
    name: str
    api_key: str  # Only returned once at creation


class RegisterNodeRequest(BaseModel):
    name: str
    region: str | None = None
    hardware_info: dict | None = None


class RegisterNodeResponse(BaseModel):
    id: uuid.UUID
    name: str
    api_key: str  # Only returned once at creation


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("/register-org", response_model=RegisterOrgResponse)
async def register_org(body: RegisterOrgRequest, db: AsyncSession = Depends(get_db)):
    """Create a new organization and return its API key.

    The API key is shown only once — store it securely.
    """
    api_key = f"nxo_{secrets.token_urlsafe(32)}"
    org = Organization(
        name=body.name,
        api_key_hash=hash_api_key(api_key),
    )
    db.add(org)
    await db.commit()
    await db.refresh(org)
    return RegisterOrgResponse(id=org.id, name=org.name, api_key=api_key)


@router.post("/register-node", response_model=RegisterNodeResponse)
async def register_node(body: RegisterNodeRequest, db: AsyncSession = Depends(get_db)):
    """Register a new compute node and return its API key.

    Nodes use this key for all subsequent API calls (heartbeat, task polling,
    weight submission).
    """
    api_key = f"nxn_{secrets.token_urlsafe(32)}"
    node = Node(
        name=body.name,
        api_key_hash=hash_api_key(api_key),
        region=body.region,
        hardware_info=body.hardware_info,
    )
    db.add(node)
    await db.commit()
    await db.refresh(node)
    return RegisterNodeResponse(id=node.id, name=node.name, api_key=api_key)
