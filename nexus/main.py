"""Nexus Node FL Engine — FastAPI application entrypoint."""

from contextlib import asynccontextmanager

import logging

import structlog
from fastapi import FastAPI

from nexus.api import auth, health, inference, jobs, mobile, nodes
from nexus.config import settings
from nexus.db.migrate import run_migrations
from nexus.db.seed import seed_initial_org
from nexus.db.session import async_session_factory, engine
from nexus.storage import s3

_LOG_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        _LOG_LEVELS.get(settings.log_level.lower(), logging.INFO)
    ),
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("starting", host=settings.host, port=settings.port)

    # Apply database migrations
    run_migrations()
    logger.info("database_ready")

    # Ensure S3 bucket exists
    try:
        s3.ensure_bucket()
        logger.info("storage_ready", bucket=settings.s3_bucket)
    except Exception as e:
        logger.warning("storage_not_available", error=str(e))

    # Seed initial org
    async with async_session_factory() as session:
        await seed_initial_org(session)

    yield

    # Shutdown
    await engine.dispose()
    logger.info("shutdown_complete")


app = FastAPI(
    title="Nexus Node FL Engine",
    description="Federated Learning orchestration service for distributed AI training",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount routers
app.include_router(health.router)
app.include_router(auth.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(nodes.router, prefix="/api/v1")
app.include_router(inference.router, prefix="/api/v1")
app.include_router(mobile.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "service": "nexus-node-fl-engine",
        "version": "0.1.0",
        "docs": "/docs",
    }
